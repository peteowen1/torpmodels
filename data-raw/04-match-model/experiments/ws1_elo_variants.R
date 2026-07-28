# WS1 — structurally different team-rating estimators
# ===================================================
# FABLE-MATCH-FEATURES-PLAN.md WS1 (promoted to main event by §0a).
#
# WHY THIS IS CHEAP
# -----------------
# A team rating is a sequential pass over the match list plus a one-coefficient
# scale fit. No GAM, no XGBoost, no rolling retrain — the whole comparison below
# runs in seconds, which is why it goes first: if none of these estimators beats
# production's Elo as a standalone margin predictor, the "our Elo is the weak
# component" hypothesis (§0a) is dead before any expensive work starts.
#
# LEAK SAFETY (three separate things, all enforced)
#  1. Ratings: `*_pre` values are a function only of strictly-earlier matches,
#     by construction. Building over all seasons at once does not leak.
#  2. Scale: the points-per-rating-unit coefficient is refit per test round on
#     matches strictly before it (same as elo_baseline_preds()).
#  3. Hyperparameters: k / carryover are tuned per test SEASON on strictly
#     prior seasons only (plan G6). Nothing is tuned on data it is scored on.
#
# VARIANTS
#   V0   production Elo                 — the incumbent (torp:::build_team_elo)
#   W1a  points-based power rating      — update on margin error, not win/loss
#   W1b  xScore power rating            — update on EXPECTED-score margin error.
#                                         torp-only: no competitor has xScore.
#   W1c  dual time constant             — fast + slow points ratings, both used
#                                         ("form" and "class" as separate terms)
#   W1d  offence/defence split          — separate scoring and conceding ratings
#
# Run: powershell.exe -Command 'Rscript "<this file>"'

suppressMessages({
  library(dplyr)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RES <- file.path(EXP, "results")
source(file.path(EXP, "rolling_lib.R"))
source(file.path(EXP, "signal_gate.R"))

TEST_SEASONS <- 2023:2026
MEAN_REVERT  <- 0  # ratings are centred at 0 in points space

# ---- match table -------------------------------------------------------------

.match_table <- function(team_mdl_df) {
  h <- team_mdl_df[team_mdl_df$team_type == "home" & !is.na(team_mdl_df$win), ]
  out <- data.frame(
    match_id    = h$match_id,
    date        = as.Date(h$utc_start_time),
    season      = h$season.x,
    round       = h$round_number.x,
    home        = as.character(h$team_name.x),
    away        = as.character(h$team_name.y),
    margin      = h$score_diff,
    xmargin     = h$xscore_diff,
    home_score  = h$home_score,
    away_score  = h$away_score,
    stringsAsFactors = FALSE
  )
  out <- out[order(out$season, out$round, out$date, out$match_id), ]
  # xScore is the whole point of W1b; fall back to actual margin only where the
  # xG join failed, and say how often that happens rather than hiding it.
  n_na <- sum(is.na(out$xmargin))
  if (n_na > 0) {
    cli::cli_alert_warning("{n_na} of {nrow(out)} matches have no xmargin - falling back to actual margin for those")
    out$xmargin[is.na(out$xmargin)] <- out$margin[is.na(out$xmargin)]
  }
  out
}

# ---- rating engines ----------------------------------------------------------

#' Generic points-space power rating.
#'
#' r_home - r_away + hga is the predicted margin directly (points units, no
#' logistic mapping), and the update is proportional to the prediction error:
#'   r_home += k * (observed - predicted);  r_away -= same
#' `target` selects what "observed" means: "margin" (W1a) or "xmargin" (W1b).
#' This is the structural difference from production's Elo, which updates on a
#' binary win/loss with a log-margin multiplier and therefore has to be mapped
#' from rating space to points afterwards.
.points_rating <- function(m, k = 0.10, hga = 8, carryover = 0.75,
                           target = c("margin", "xmargin")) {
  target <- match.arg(target)
  teams <- sort(unique(c(m$home, m$away)))
  r <- setNames(rep(0, length(teams)), teams)
  last_season <- setNames(rep(NA_integer_, length(teams)), teams)
  pre_h <- pre_a <- numeric(nrow(m))

  for (i in seq_len(nrow(m))) {
    h <- m$home[i]; a <- m$away[i]; s <- m$season[i]
    if (!is.na(last_season[[h]]) && last_season[[h]] < s) r[[h]] <- carryover * r[[h]]
    if (!is.na(last_season[[a]]) && last_season[[a]] < s) r[[a]] <- carryover * r[[a]]

    pre_h[i] <- r[[h]]; pre_a[i] <- r[[a]]
    pred <- (r[[h]] - r[[a]]) + hga
    obs  <- m[[target]][i]
    err  <- obs - pred
    r[[h]] <- r[[h]] + k * err
    r[[a]] <- r[[a]] - k * err
    last_season[[h]] <- s; last_season[[a]] <- s
  }
  data.frame(match_id = m$match_id, rating_diff = pre_h - pre_a)
}

#' Offence / defence split (W1d).
#'
#' Each team carries an attack rating (points scored above average) and a
#' defence rating (points conceded below average). Predicted home score is
#' base + att_home + def_away + hga/2. Gives a total-score signal as well as a
#' margin one — model 1 (total_xpoints) currently has no dynamic team term at
#' all, so this is the only variant here that could help it.
.offdef_rating <- function(m, k = 0.06, hga = 8, carryover = 0.75) {
  teams <- sort(unique(c(m$home, m$away)))
  att <- setNames(rep(0, length(teams)), teams)
  def <- setNames(rep(0, length(teams)), teams)
  last_season <- setNames(rep(NA_integer_, length(teams)), teams)
  base <- mean(c(m$home_score, m$away_score), na.rm = TRUE)
  d_margin <- d_total <- numeric(nrow(m))

  for (i in seq_len(nrow(m))) {
    h <- m$home[i]; a <- m$away[i]; s <- m$season[i]
    for (tm in c(h, a)) {
      if (!is.na(last_season[[tm]]) && last_season[[tm]] < s) {
        att[[tm]] <- carryover * att[[tm]]; def[[tm]] <- carryover * def[[tm]]
      }
    }
    ph <- base + att[[h]] + def[[a]] + hga / 2
    pa <- base + att[[a]] + def[[h]] - hga / 2
    d_margin[i] <- ph - pa
    d_total[i]  <- ph + pa

    eh <- m$home_score[i] - ph
    ea <- m$away_score[i] - pa
    att[[h]] <- att[[h]] + k * eh; def[[a]] <- def[[a]] + k * eh
    att[[a]] <- att[[a]] + k * ea; def[[h]] <- def[[h]] + k * ea
    last_season[[h]] <- s; last_season[[a]] <- s
  }
  data.frame(match_id = m$match_id, rating_diff = d_margin, total_pred = d_total)
}

#' Production Elo, expressed as a rating_diff in Elo units.
.prod_elo_rating <- function(m, k = 20, hga = 45, carryover = 0.75) {
  mm <- data.frame(match_id = m$match_id, date = m$date, season = m$season,
                   home_team = m$home, away_team = m$away, home_margin = m$margin)
  e <- torp:::build_team_elo(mm, k = k, hga = hga, carryover = carryover)$by_match
  key <- setNames(e$elo_pre, paste(e$match_id, e$team_name))
  data.frame(
    match_id = m$match_id,
    rating_diff = unname(key[paste(m$match_id, m$home)] - key[paste(m$match_id, m$away)])
  )
}

# ---- scoring -----------------------------------------------------------------

#' Score a rating as a standalone margin predictor, per-round refit scale.
#'
#' `rating_cols` may name more than one column (W1c uses fast + slow together),
#' in which case the per-round fit is a multiple regression on all of them.
.score_rating <- function(m, ratings, test_seasons, rating_cols = "rating_diff",
                          min_train = 100L) {
  d <- merge(m, ratings, by = "match_id")
  d <- d[order(d$season, d$round, d$date, d$match_id), ]
  rounds <- unique(d[d$season %in% test_seasons, c("season", "round")])
  rounds <- rounds[order(rounds$season, rounds$round), ]

  f <- stats::as.formula(paste("margin ~", paste(rating_cols, collapse = " + ")))
  acc <- vector("list", nrow(rounds))
  for (i in seq_len(nrow(rounds))) {
    s <- rounds$season[i]; r <- rounds$round[i]
    prior <- d$season < s | (d$season == s & d$round < r)
    this  <- d$season == s & d$round == r
    if (sum(prior) < min_train || !any(this)) next
    fit <- stats::lm(f, data = d[prior, ])
    te <- d[this, ]
    acc[[i]] <- data.frame(
      season = s, round = r, match_id = te$match_id,
      pred_margin = unname(stats::predict(fit, newdata = te)),
      margin = te$margin
    )
  }
  out <- dplyr::bind_rows(acc)
  out$pred_win <- 1 / (1 + exp(-out$pred_margin / 21))  # logistic map, scale ~ sd/2
  out$home_win <- ifelse(out$margin > 0, 1, ifelse(out$margin == 0, 0.5, 0))
  out
}

.metrics <- function(p, label) {
  data.frame(
    variant = label, n = nrow(p),
    MAE = mean(abs(p$pred_margin - p$margin)),
    RMSE = sqrt(mean((p$pred_margin - p$margin)^2)),
    cor = stats::cor(p$pred_margin, p$margin),
    slope = unname(stats::coef(stats::lm(margin ~ pred_margin, data = p))[2]),
    sd_pred = stats::sd(p$pred_margin)
  )
}

#' Tune a scalar hyperparameter per test season on strictly prior seasons (G6).
#'
#' Returns a rating table assembled season-by-season, each season's rows produced
#' by the k that was best on everything before it. Ratings themselves are always
#' built over the full history (past-only by construction); only the CHOICE of k
#' is restricted.
.tuned_rating <- function(m, builder, k_grid, test_seasons, label) {
  parts <- list()
  chosen <- c()
  for (s in test_seasons) {
    prior <- m[m$season < s, ]
    if (nrow(prior) < 150) next
    maes <- vapply(k_grid, function(k) {
      rt <- builder(prior, k = k)
      dd <- merge(prior, rt, by = "match_id")
      mean(abs(stats::predict(stats::lm(margin ~ rating_diff, data = dd)) - dd$margin))
    }, numeric(1))
    k_best <- k_grid[which.min(maes)]
    chosen <- c(chosen, k_best)
    rt_full <- builder(m, k = k_best)
    parts[[as.character(s)]] <- rt_full[m$season == s, ]
  }
  cli::cli_alert_info("{label}: k per season = {paste(test_seasons[seq_along(chosen)], chosen, sep='->', collapse=', ')}")
  dplyr::bind_rows(parts)
}

# ---- run ---------------------------------------------------------------------

team_mdl_df <- readRDS(file.path(RES, "team_mdl_df_cache_with_elo.rds"))
m <- .match_table(team_mdl_df)
cli::cli_alert_info("Matches: {nrow(m)} ({min(m$season)}-{max(m$season)}); test = {TEST_SEASONS}")

rt_v0  <- .prod_elo_rating(m)
rt_w1a <- .tuned_rating(m, function(mm, k) .points_rating(mm, k = k, target = "margin"),
                        c(0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24), TEST_SEASONS, "W1a points")
rt_w1b <- .tuned_rating(m, function(mm, k) .points_rating(mm, k = k, target = "xmargin"),
                        c(0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24), TEST_SEASONS, "W1b xScore")
rt_w1d <- .tuned_rating(m, function(mm, k) .offdef_rating(mm, k = k),
                        c(0.02, 0.04, 0.06, 0.08, 0.12), TEST_SEASONS, "W1d off/def")

# W1c: fast + slow, both fed to the per-round fit as separate terms.
rt_slow <- .points_rating(m, k = 0.06, target = "margin")
rt_fast <- .points_rating(m, k = 0.22, target = "margin")
rt_w1c  <- data.frame(match_id = rt_slow$match_id,
                      slow_diff = rt_slow$rating_diff,
                      fast_diff = rt_fast$rating_diff)

# W1bc: the two ideas combined - xScore slow + xScore fast.
rt_xs <- .points_rating(m, k = 0.06, target = "xmargin")
rt_xf <- .points_rating(m, k = 0.22, target = "xmargin")
rt_w1bc <- data.frame(match_id = rt_xs$match_id,
                      slow_diff = rt_xs$rating_diff, fast_diff = rt_xf$rating_diff)

preds <- list(
  `V0  production Elo`      = .score_rating(m, rt_v0,  TEST_SEASONS),
  `W1a points rating`       = .score_rating(m, rt_w1a, TEST_SEASONS),
  `W1b xScore rating`       = .score_rating(m, rt_w1b, TEST_SEASONS),
  `W1c fast+slow (margin)`  = .score_rating(m, rt_w1c, TEST_SEASONS, c("slow_diff", "fast_diff")),
  `W1bc fast+slow (xScore)` = .score_rating(m, rt_w1bc, TEST_SEASONS, c("slow_diff", "fast_diff")),
  `W1d off/def split`       = .score_rating(m, rt_w1d, TEST_SEASONS)
)

cli::cli_h1("Standalone margin prediction, rolling OOS, {min(TEST_SEASONS)}-{max(TEST_SEASONS)}")
print(do.call(rbind, Map(.metrics, preds, names(preds))), row.names = FALSE, digits = 4)

cli::cli_h1("2026 only")
p26 <- lapply(preds, function(p) p[p$season == 2026, ])
print(do.call(rbind, Map(.metrics, p26, names(p26))), row.names = FALSE, digits = 4)

cli::cli_h1("Incremental signal of each variant OVER production Elo")
v0 <- preds[["V0  production Elo"]]
for (nm in setdiff(names(preds), "V0  production Elo")) {
  s <- incremental_signal(preds[[nm]], v0)
  cli::cli_text("{nm}: delta r2 {round(s$delta_r2, 4)} | beta(V0) {round(s$beta_baseline, 3)} | beta(new) {round(s$beta_candidate, 3)} | p {signif(s$p_candidate, 3)} | cor {round(s$cor_preds, 3)}")
}

saveRDS(preds, file.path(RES, "ws1_elo_variants.rds"))
cli::cli_alert_success("Saved results/ws1_elo_variants.rds")
