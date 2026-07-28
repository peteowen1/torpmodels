# xScore power rating — feature builder for the in-model swap test
# ================================================================
# FABLE-MATCH-FEATURES-PLAN.md §6.3 action 3.
#
# WS1 (§6.1) established, as STANDALONE margin predictors on an identical
# 695-match set (2023-2026):
#   production Elo (win-based + MOV multiplier)   MAE 27.15  cor 0.524
#   xScore power rating                           MAE 26.38  cor 0.559
#   xScore fast+slow pair                         MAE 26.34  cor 0.561
# and every variant renders production's Elo redundant (beta(V0) ~ 0.05-0.09,
# p < 1e-7). This file packages the winner as a joinable model feature so the
# claim can be retested where it actually matters: inside the GAM/XGB chain,
# as a replacement for `elo_diff`.
#
# Expectation, set honestly in advance: WS1's reverse test found the new
# ratings add almost nothing ON TOP OF C6 (delta r2 0.0015, p 0.21) and that
# blending them with C6 makes it worse. So the in-model swap is expected to be
# a small win at best. It is run because a feature the model consumes is a
# different thing from a prediction blended post-hoc — the GAM can use a
# cleaner signal to learn different smooths — not because a large gain is
# anticipated.
#
# WHY xScore RATHER THAN SCORE
# ----------------------------
# The scoreboard margin is a noisy realisation of team quality: AFL conversion
# variance is large, so a team can dominate territory and shots and still lose.
# Updating the rating on EXPECTED score strips out that conversion noise, so
# the rating re-converges faster after genuine form changes and wanders less
# after flukes. No competitor can build this — xScore is torp's own.
#
# LEAK SAFETY
#   Ratings are strictly past-only by construction (a match's rating is a
#   function only of matches before it), so the table can be built over all
#   seasons at once, exactly like production's elo_diff.
#   k is FIXED at the value WS1's per-season tuner selected using pre-2025
#   data only (plan G6) — deliberately NOT the value 2026 preferred, which
#   would be tuned on the window it is scored on.

# Fixed on pre-2025 data by WS1's expanding-window tuner (which chose 0.08 for
# the 2025 test season, i.e. fitted on 2021-2024). Do not "update" this to the
# value a later season prefers without moving the test window too.
XRATING_K_SLOW <- 0.08
XRATING_K_FAST <- 0.22
XRATING_HGA    <- 8      # points, home advantage in rating space
XRATING_CARRY  <- 0.75   # season-boundary regression to mean (matches ELO_CARRYOVER)

# .xrating_match_table ----

#' One row per completed match, home perspective, ordered by time
#' @keywords internal
.xrating_match_table <- function(team_mdl_df) {
  h <- team_mdl_df[team_mdl_df$team_type == "home" & !is.na(team_mdl_df$win), ]
  out <- data.frame(
    match_id = h$match_id,
    date     = as.Date(h$utc_start_time),
    season   = h$season.x,
    round    = h$round_number.x,
    home     = as.character(h$team_name.x),
    away     = as.character(h$team_name.y),
    margin   = h$score_diff,
    xmargin  = h$xscore_diff,
    stringsAsFactors = FALSE
  )
  n_na <- sum(is.na(out$xmargin))
  if (n_na > 0) {
    cli::cli_alert_warning("{n_na} of {nrow(out)} matches lack xmargin - falling back to actual margin there")
    out$xmargin[is.na(out$xmargin)] <- out$margin[is.na(out$xmargin)]
  }
  out[order(out$season, out$round, out$date, out$match_id), ]
}

# .xrating_sequential ----

#' Sequential points-space rating updated on expected-score margin error
#'
#' Returns PER-TEAM pre-match ratings (two rows per match), the same shape as
#' production's `build_team_elo()$by_match`. Per-team rather than just the
#' difference because unplayed fixtures need a per-team current-rating fallback,
#' and a team's rating is emphatically NOT half the match difference — it is its
#' own accumulated history (home +30 / away +10 gives a difference of 20, not a
#' rating of 10 each).
#'
#' @return list(by_match = data.frame(match_id, team_name, rating_pre),
#'   current = data.frame(team_name, rating_current))
#' @keywords internal
.xrating_sequential <- function(m, k, hga = XRATING_HGA, carryover = XRATING_CARRY) {
  teams <- sort(unique(c(m$home, m$away)))
  r <- stats::setNames(rep(0, length(teams)), teams)
  last_season <- stats::setNames(rep(NA_integer_, length(teams)), teams)
  pre_h <- pre_a <- numeric(nrow(m))

  for (i in seq_len(nrow(m))) {
    h <- m$home[i]; a <- m$away[i]; s <- m$season[i]
    if (!is.na(last_season[[h]]) && last_season[[h]] < s) r[[h]] <- carryover * r[[h]]
    if (!is.na(last_season[[a]]) && last_season[[a]] < s) r[[a]] <- carryover * r[[a]]

    pre_h[i] <- r[[h]]; pre_a[i] <- r[[a]]
    err <- m$xmargin[i] - ((r[[h]] - r[[a]]) + hga)
    r[[h]] <- r[[h]] + k * err
    r[[a]] <- r[[a]] - k * err
    last_season[[h]] <- s; last_season[[a]] <- s
  }

  list(
    by_match = data.frame(
      match_id    = rep(m$match_id, 2),
      team_name   = c(m$home, m$away),
      rating_pre  = c(pre_h, pre_a),
      stringsAsFactors = FALSE
    ),
    current = data.frame(team_name = names(r), rating_current = unname(r),
                         stringsAsFactors = FALSE)
  )
}

# add_xrating_diff ----

#' Join xScore-rating difference feature(s) onto team_mdl_df
#'
#' Mirrors production's `join_elo_diff_to_team_mdl_df()` conventions: symmetric
#' (this team minus opponent, no HGA baked in — `team_type_fac` already carries
#' it), and unplayed fixtures fall back to each team's CURRENT rating so the
#' feature is defined for the rows predictions are actually needed on.
#'
#' @param team_mdl_df Complete model dataset
#' @param pair If TRUE, add both `xelo_slow_diff` and `xelo_fast_diff` (the
#'   form-vs-class decomposition — WS1's marginally best standalone variant).
#'   If FALSE, add a single `xelo_diff` at the slow k.
#' @return team_mdl_df with the new column(s)
#' @keywords internal
add_xrating_diff <- function(team_mdl_df, pair = FALSE) {
  m <- .xrating_match_table(team_mdl_df)

  specs <- if (pair) {
    list(xelo_slow_diff = XRATING_K_SLOW, xelo_fast_diff = XRATING_K_FAST)
  } else {
    list(xelo_diff = XRATING_K_SLOW)
  }

  tx <- as.character(team_mdl_df$team_name.x)
  ty <- as.character(team_mdl_df$team_name.y)

  for (nm in names(specs)) {
    rt <- .xrating_sequential(m, k = specs[[nm]])
    # String keys rather than factor joins — factor level mismatches join
    # silently and wrongly (the hazard elo_lib.R documents).
    key <- stats::setNames(rt$by_match$rating_pre,
                           paste(rt$by_match$match_id, rt$by_match$team_name))
    cur <- stats::setNames(rt$current$rating_current, rt$current$team_name)

    rx <- unname(key[paste(team_mdl_df$match_id, tx)])
    ry <- unname(key[paste(team_mdl_df$match_id, ty)])
    rx[is.na(rx)] <- unname(cur[tx[is.na(rx)]])
    ry[is.na(ry)] <- unname(cur[ty[is.na(ry)]])
    rx[is.na(rx)] <- 0
    ry[is.na(ry)] <- 0

    team_mdl_df[[nm]] <- rx - ry
  }

  n_fixture <- sum(is.na(team_mdl_df$win))
  cli::cli_alert_info("add_xrating_diff: added {paste(names(specs), collapse=', ')} ({n_fixture} unplayed row{?s} used current-rating fallback)")
  team_mdl_df
}
