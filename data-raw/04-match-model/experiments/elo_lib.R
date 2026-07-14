# elo_lib.R — Team-Elo builder for FABLE-MATCH-MAE-PLAN.md WS2
# ================================================================
# Sequential, pre-match (leak-safe-by-construction) team Elo rating, built
# from team_mdl_df's own match universe (not a fresh torp::load_results()
# pull) so team names line up exactly with team_name.x/team_name.y and the
# match set matches what the rolling harness trains/tests on (same
# MATCH_MIN_DATA_SEASON/MATCH_MIN_DATA_ROUND floor already applied upstream
# by .build_team_mdl_df()). Because each match's elo_pre is a function only
# of strictly-earlier matches (sorted by date), the whole table can be
# computed once, globally, and joined onto team_mdl_df without leakage --
# no need to recompute inside run_rolling_eval()'s per-round loop.
#
# Assumes the caller has already devtools::load_all()'d torp and
# library()'d dplyr, exactly as ws2_team_elo.R does before sourcing this
# file.

# .matches_from_team_mdl_df ----

#' Build a one-row-per-match table from team_mdl_df for Elo construction
#'
#' @param team_mdl_df Complete model dataset from build_team_mdl_df()
#' @return data.frame(match_id, date, season, round, home_team, away_team,
#'   home_margin), completed matches only, sorted by date then match_id
#' @keywords internal
.matches_from_team_mdl_df <- function(team_mdl_df) {
  out <- team_mdl_df[team_mdl_df$team_type == "home" & !is.na(team_mdl_df$win), ]
  data.frame(
    match_id    = out$match_id,
    date        = as.Date(out$utc_start_time),
    season      = out$season.x,
    round       = out$round_number.x,
    home_team   = as.character(out$team_name.x),
    away_team   = as.character(out$team_name.y),
    home_margin = out$score_diff,
    stringsAsFactors = FALSE
  ) |>
    dplyr::arrange(date, match_id)
}

# build_team_elo ----

#' Sequential team-Elo rating, pre-match ratings only
#'
#' Standard Elo with home-ground advantage and a 538-style margin-of-victory
#' multiplier that dampens updates when the pre-match rating gap is already
#' large (prevents runaway ratings for dominant teams). Applies partial
#' regression-to-mean at each team's first match of a new season.
#'
#' @param matches data.frame with columns match_id, date, season, home_team,
#'   away_team, home_margin (home_score - away_score), sorted by date
#'   ascending (see .matches_from_team_mdl_df()).
#' @param k Update rate (higher = faster-moving ratings)
#' @param hga Home-ground advantage, in Elo points, added to the home team's
#'   rating before computing expected result
#' @param carryover Fraction of a team's rating retained across a season
#'   boundary (1 = no regression, 0 = full reset to 1500)
#' @param mov_mult Logical; apply the log-margin x anti-runaway multiplier
#'   (538-style) to the update magnitude. FALSE = plain win/loss/draw Elo.
#' @return data.table(match_id, team_name, elo_pre) -- two rows per match
#'   (home team + away team), elo_pre = rating immediately BEFORE that match
#' @keywords internal
build_team_elo <- function(matches, k = 20, hga = 35, carryover = 0.75, mov_mult = TRUE) {
  stopifnot(all(c("match_id", "date", "season", "home_team", "away_team", "home_margin") %in% names(matches)))
  matches <- matches[order(matches$date, matches$match_id), ]

  teams <- sort(unique(c(matches$home_team, matches$away_team)))
  elo <- stats::setNames(rep(1500, length(teams)), teams)
  last_season <- stats::setNames(rep(NA_integer_, length(teams)), teams)

  n <- nrow(matches)
  pre_home <- numeric(n)
  pre_away <- numeric(n)

  for (i in seq_len(n)) {
    h <- matches$home_team[i]
    a <- matches$away_team[i]
    s <- matches$season[i]

    # Season-boundary regression-to-mean, applied lazily on first sight of
    # each team in a new season (equivalent to applying it once per team at
    # season start, since ratings are otherwise untouched between matches).
    if (!is.na(last_season[[h]]) && last_season[[h]] < s) {
      elo[[h]] <- carryover * elo[[h]] + (1 - carryover) * 1500
    }
    if (!is.na(last_season[[a]]) && last_season[[a]] < s) {
      elo[[a]] <- carryover * elo[[a]] + (1 - carryover) * 1500
    }

    elo_h <- elo[[h]]
    elo_a <- elo[[a]]
    pre_home[i] <- elo_h
    pre_away[i] <- elo_a

    exp_home <- 1 / (1 + 10^(-((elo_h + hga) - elo_a) / 400))
    m <- matches$home_margin[i]
    result <- if (is.na(m)) 0.5 else if (m > 0) 1 else if (m < 0) 0 else 0.5

    mov <- if (isTRUE(mov_mult) && !is.na(m) && m != 0) {
      log(abs(m) + 1) * (2.2 / (0.001 * abs(elo_h - elo_a) + 2.2))
    } else {
      1
    }

    delta <- k * mov * (result - exp_home)
    elo[[h]] <- elo_h + delta
    elo[[a]] <- elo_a - delta

    last_season[[h]] <- s
    last_season[[a]] <- s
  }

  data.table::data.table(
    match_id  = rep(matches$match_id, 2),
    team_name = c(matches$home_team, matches$away_team),
    elo_pre   = c(pre_home, pre_away)
  )
}

# fit_elo_margin_scale ----

#' Fit the points-per-Elo scale (and HGA-adjusted intercept-free slope) that
#' converts (home_elo_pre - away_elo_pre) into a predicted home margin.
#'
#' `pred_margin = predict(fit, newdata = data.frame(elo_diff_hga = elo_diff + hga))`
#'
#' @param elo_diff_home Vector of (home elo_pre - away elo_pre)
#' @param hga Home-ground-advantage constant (same value used to build `elo_diff_home`'s table)
#' @param home_margin Actual home margin (home_score - away_score)
#' @return An lm object with a single coefficient on `elo_diff_hga`
#' @keywords internal
fit_elo_margin_scale <- function(elo_diff_home, hga, home_margin) {
  df <- data.frame(elo_diff_hga = elo_diff_home + hga, home_margin = home_margin)
  stats::lm(home_margin ~ elo_diff_hga + 0, data = df)
}

# elo_pred_win ----

#' Elo win-probability formula (logistic on rating gap incl. HGA)
#' @keywords internal
elo_pred_win <- function(elo_diff_home, hga) {
  1 / (1 + 10^(-(elo_diff_home + hga) / 400))
}

# join_elo_diff_to_team_mdl_df ----

#' Join per-team elo_pre onto team_mdl_df's long (team-per-match) format and
#' compute a symmetric `elo_diff` feature (this team's elo_pre minus
#' opponent's elo_pre -- same convention as epr_diff/torp_diff, no HGA
#' baked in; HGA is already carried by the existing team_type_fac term).
#'
#' @param team_mdl_df Complete model dataset (has match_id, team_name.x, team_name.y)
#' @param elo_table Output of build_team_elo(): data.table(match_id, team_name, elo_pre)
#' @return team_mdl_df with an added numeric `elo_diff` column (0 for the
#'   very first match a team appears in, by construction -- not NA)
#' @keywords internal
join_elo_diff_to_team_mdl_df <- function(team_mdl_df, elo_table) {
  elo_x <- elo_table
  names(elo_x) <- c("match_id", "team_name_x_chr", "elo_pre_x")
  elo_y <- elo_table
  names(elo_y) <- c("match_id", "team_name_y_chr", "elo_pre_y")

  team_mdl_df$team_name_x_chr <- as.character(team_mdl_df$team_name.x)
  team_mdl_df$team_name_y_chr <- as.character(team_mdl_df$team_name.y)

  out <- team_mdl_df |>
    dplyr::left_join(as.data.frame(elo_x), by = c("match_id", "team_name_x_chr")) |>
    dplyr::left_join(as.data.frame(elo_y), by = c("match_id", "team_name_y_chr")) |>
    dplyr::mutate(elo_diff = elo_pre_x - elo_pre_y) |>
    dplyr::select(-team_name_x_chr, -team_name_y_chr)

  n_na <- sum(is.na(out$elo_diff))
  if (n_na > 0) {
    cli::cli_warn("join_elo_diff_to_team_mdl_df: {n_na} row{?s} have NA elo_diff after join (team-name mismatch?)")
  }
  out
}

# tune_team_elo ----

#' Grid search Elo hyperparameters on a pre-test-season match set (plan G6)
#'
#' For each (k, hga, carryover) combo: builds the Elo table over `matches`
#' (already restricted to pre-test seasons by the caller), fits the
#' points-per-Elo scale on the SAME matches (single scalar, negligible
#' overfit risk), and scores in-sample MAE. Because elo_pre is inherently
#' point-in-time (never uses future matches), this in-sample MAE is already
#' an honest walk-forward measure -- it is not the same kind of leakage as
#' in-sample GAM/XGB scoring.
#'
#' @param matches Pre-test-season match table (see .matches_from_team_mdl_df())
#' @param k_grid,hga_grid,carryover_grid Numeric vectors to cross
#' @param mov_mult Passed through to build_team_elo()
#' @return data.frame of every combo with its MAE, sorted best-first
#' @keywords internal
tune_team_elo <- function(matches, k_grid = c(15, 20, 30), hga_grid = c(25, 35, 45),
                           carryover_grid = c(0.6, 0.75, 0.9), mov_mult = TRUE) {
  grid <- expand.grid(k = k_grid, hga = hga_grid, carryover = carryover_grid)
  grid$mae <- NA_real_

  for (i in seq_len(nrow(grid))) {
    et <- build_team_elo(matches, k = grid$k[i], hga = grid$hga[i],
                          carryover = grid$carryover[i], mov_mult = mov_mult)
    et_home <- et[et$team_name %in% matches$home_team, ]

    # Rebuild home/away elo_pre per match_id directly (avoids factor joins)
    ex <- stats::setNames(et$elo_pre, paste(et$match_id, et$team_name))
    elo_h <- ex[paste(matches$match_id, matches$home_team)]
    elo_a <- ex[paste(matches$match_id, matches$away_team)]
    elo_diff_home <- unname(elo_h - elo_a)

    fit <- fit_elo_margin_scale(elo_diff_home, grid$hga[i], matches$home_margin)
    pred <- stats::predict(fit)
    grid$mae[i] <- mean(abs(pred - matches$home_margin))
  }

  grid[order(grid$mae), ]
}
