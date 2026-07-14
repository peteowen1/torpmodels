# ws1_margin_recal.R — WS1: Post-hoc margin recalibration layer
# =================================================================
# FABLE-MATCH-MAE-PLAN.md WS1. Leak-safe expanding-window shrinkage of the
# G4 champion's (Input Blend) pred_margin, fit only on OOS predictions from
# rounds strictly before the round being scored.
#
# Champion (G4): Input Blend. Confirmed structurally, not just by heuristic —
# torp/R/match_model.R:685-694 (production's run_predictions_pipeline blend
# block) is byte-for-byte the same 50/50 GAM+XGB input blend, with pred_win
# re-derived from the GAM win model on the blended margin, that
# rolling_lib.R's run_rolling_eval() returns as `input_blend_preds`. So no
# separate "which variant matches production" search is needed here.
#
# Run structure: ONE run_rolling_eval(test_seasons = 2025:2026) call (not a
# 2025-only + 2026-only split). XGBoost nrounds are pre-optimised on
# season.x < min(test_seasons) = season.x < 2025 either way (G6), so a split
# run would silently use *different* nrounds for the 2026 leg than the
# pooled/confirm window uses -- the combined run keeps one nrounds regime and
# gives V1d's 2025 warm-start data for free out of the same call.
#
# Screen = subset the pooled predictions to season == 2026 (matches G2 "fast
# screen" apples-to-apples with the diagnosis doc's 2026-only numbers).
# Confirm = the full pooled 2025:2026 set (also G3's ship-gate window) --
# no second harness run needed.

# Setup ----
library(tidyverse)
library(xgboost)
library(mgcv)
library(MLmetrics)
library(geosphere)
library(cli)

devtools::load_all("C:/dev/torpverse/torp")

rolling_lib_candidates <- c(
  "rolling_lib.R",
  "experiments/rolling_lib.R",
  "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/rolling_lib.R"
)
rolling_lib_hits <- rolling_lib_candidates[file.exists(rolling_lib_candidates)]
if (length(rolling_lib_hits) == 0) stop("Cannot find experiments/rolling_lib.R")
source(rolling_lib_hits[1])

SCREEN_SEASON <- 2026
TEST_SEASONS  <- 2025:2026

# Build data ----
cli::cli_h1("WS1: Building Match Prediction Training Data")
tictoc::tic("total")

team_mdl_df <- build_team_mdl_df()
cli::cli_inform("team_mdl_df: {nrow(team_mdl_df)} rows, seasons {paste(sort(unique(team_mdl_df$season.x)), collapse = ', ')}")

# One combined rolling run over 2025:2026 ----
cli::cli_h1("WS1: Rolling OOS run (test_seasons = 2025:2026, default trainers)")
roll <- run_rolling_eval(team_mdl_df, TEST_SEASONS)

champion_preds <- roll$input_blend_preds |>
  mutate(season = as.integer(season), round = as.integer(round))

cli::cli_inform("Champion (Input Blend) OOS predictions: {nrow(champion_preds)} matches, seasons {paste(sort(unique(champion_preds$season)), collapse = ', ')}")

# Baseline card (fills in the WS0 baseline this plan step needed) ----
champ_2026  <- champion_preds |> filter(season == SCREEN_SEASON)
champ_pool  <- champion_preds

base_2026 <- .compute_metrics(champ_2026)
base_pool <- .compute_metrics(champ_pool)

.print_metrics <- function(m, label) {
  cat(sprintf(
    "%-28s MAE=%.3f RMSE=%.3f Brier=%.4f Slope=%.3f Cor=%.3f SDRatio=%.3f CloseMAE(n=%d)=%.3f\n",
    label, m$mae, m$rmse, m$brier, m$slope, m$cor, m$sd_ratio, m$close_n, m$close_mae
  ))
}

cat("\n=== WS1 Baseline Card: Champion (Input Blend) ===\n")
.print_metrics(base_2026, sprintf("2026 screen (n=%d)", nrow(champ_2026)))
.print_metrics(base_pool, sprintf("2025:2026 pooled (n=%d)", nrow(champ_pool)))

# Other three variants, 2026 screen only, for context (not used further) ----
cat("\n=== Other harness variants, 2026 screen (context only) ===\n")
for (nm in c("gam_preds", "xgb_preds", "blend_preds")) {
  p <- roll[[nm]] |> mutate(season = as.integer(season)) |> filter(season == SCREEN_SEASON)
  .print_metrics(.compute_metrics(p), nm)
}

# Bonus sanity check: correlation to submitted 2026 tips (non-blocking) ----
tryCatch({
  sub_tips <- fitzRoy::fetch_squiggle_data("tips", year = 2026) |>
    filter(source == "In The Game") |>
    mutate(round = as.integer(round), hteam_norm = torp_replace_teams(hteam),
           hmargin = as.numeric(hmargin))
  if (nrow(sub_tips) >= 10) {
    joined <- champ_2026 |>
      mutate(hteam_norm = torp_replace_teams(as.character(home_team))) |>
      inner_join(sub_tips |> select(round, hteam_norm, sub_pred_margin = hmargin),
                 by = c("round", "hteam_norm"))
    if (nrow(joined) >= 3) {
      cat(sprintf("\nSanity: Input Blend vs submitted 'In The Game' tips margin correlation = %.3f (n=%d)\n",
                   cor(joined$pred_margin, joined$sub_pred_margin), nrow(joined)))
    }
  }
}, error = function(e) cli::cli_warn("Squiggle sanity check skipped: {e$message}"))

# Recalibration engine ----

#' Leak-safe expanding-window margin recalibration
#'
#' For each (season, round) in `score_idx` (processed in chronological
#' order), fits the recalibration on rows in `history_pool_idx` whose
#' (season, round) is strictly before the round being scored, then applies
#' it to that round's original pred_margin. Identity fallback (b=1, a=0)
#' below `min_n` history rows (cold start).
#'
#' @param preds_all Full predictions data frame (pred_margin, margin, season, round)
#' @param score_idx Row indices (into preds_all) to compute recalibrated predictions for
#' @param history_pool_idx Row indices eligible to be used as recalibration history
#'   (V1a/b/c: same as score_idx, i.e. within-season only; V1d/confirm: all rows)
#' @param mode "slope_only" (V1a/d), "slope_intercept" (V1b), "nonlinear" (V1c)
#' @param min_n Minimum history rows required before recalibrating (else identity)
#' @return list(idx = score_idx sorted chronologically, pred_margin_recal, b_trace)
recal_expanding <- function(preds_all, score_idx, history_pool_idx,
                             mode = c("slope_only", "slope_intercept", "nonlinear"),
                             min_n = 30) {
  mode <- match.arg(mode)
  key <- preds_all$season * 1000L + preds_all$round

  score_idx <- score_idx[order(key[score_idx])]
  hist_key  <- key[history_pool_idx]

  out <- numeric(length(score_idx))
  b_trace <- vector("list", length(score_idx))

  for (k in seq_along(score_idx)) {
    i <- score_idx[k]
    cur_key <- key[i]
    hist_idx <- history_pool_idx[hist_key < cur_key]
    n_hist <- length(hist_idx)

    if (n_hist < min_n) {
      out[k] <- preds_all$pred_margin[i]
      b_trace[[k]] <- data.frame(season = preds_all$season[i], round = preds_all$round[i],
                                  b = 1, a = 0, n_hist = n_hist, applied = "identity")
      next
    }

    hist_df <- preds_all[hist_idx, c("pred_margin", "margin")]

    if (mode == "slope_only") {
      b <- unname(stats::coef(stats::lm(margin ~ pred_margin + 0, data = hist_df))[1])
      out[k] <- b * preds_all$pred_margin[i]
      b_trace[[k]] <- data.frame(season = preds_all$season[i], round = preds_all$round[i],
                                  b = b, a = 0, n_hist = n_hist, applied = mode)
    } else if (mode == "slope_intercept") {
      cf <- stats::coef(stats::lm(margin ~ pred_margin, data = hist_df))
      out[k] <- unname(cf[["(Intercept)"]] + cf[["pred_margin"]] * preds_all$pred_margin[i])
      b_trace[[k]] <- data.frame(season = preds_all$season[i], round = preds_all$round[i],
                                  b = unname(cf[["pred_margin"]]), a = unname(cf[["(Intercept)"]]),
                                  n_hist = n_hist, applied = mode)
    } else { # nonlinear
      fit <- tryCatch(
        mgcv::gam(margin ~ s(pred_margin, k = 4), data = hist_df),
        error = function(e) NULL
      )
      if (is.null(fit)) {
        b <- unname(stats::coef(stats::lm(margin ~ pred_margin + 0, data = hist_df))[1])
        out[k] <- b * preds_all$pred_margin[i]
        b_trace[[k]] <- data.frame(season = preds_all$season[i], round = preds_all$round[i],
                                    b = b, a = 0, n_hist = n_hist, applied = "slope_fallback")
      } else {
        center <- as.numeric(predict(fit, newdata = data.frame(pred_margin = 0)))
        val <- as.numeric(predict(fit, newdata = data.frame(pred_margin = preds_all$pred_margin[i]))) - center
        out[k] <- val
        b_trace[[k]] <- data.frame(season = preds_all$season[i], round = preds_all$round[i],
                                    b = NA_real_, a = NA_real_, n_hist = n_hist, applied = mode)
      }
    }
  }

  list(idx = score_idx, pred_margin_recal = out, b_trace = dplyr::bind_rows(b_trace))
}

#' Build a recalibrated predictions data frame for scoring
.apply_recal <- function(preds_all, res) {
  out <- preds_all[res$idx, ]
  out$pred_margin <- res$pred_margin_recal
  out
}

# 2026 screen: V1a-V1d ----
cli::cli_h1("WS1: 2026 screen — V1a-V1d")

idx_2026 <- which(champion_preds$season == SCREEN_SEASON)
idx_all  <- seq_len(nrow(champion_preds))

res_v1a <- recal_expanding(champion_preds, idx_2026, idx_2026, mode = "slope_only")
res_v1b <- recal_expanding(champion_preds, idx_2026, idx_2026, mode = "slope_intercept")
res_v1c <- recal_expanding(champion_preds, idx_2026, idx_2026, mode = "nonlinear")
res_v1d <- recal_expanding(champion_preds, idx_2026, idx_all,  mode = "slope_only")

preds_v1a <- .apply_recal(champion_preds, res_v1a)
preds_v1b <- .apply_recal(champion_preds, res_v1b)
preds_v1c <- .apply_recal(champion_preds, res_v1c)
preds_v1d <- .apply_recal(champion_preds, res_v1d)

m_v1a <- .compute_metrics(preds_v1a)
m_v1b <- .compute_metrics(preds_v1b)
m_v1c <- .compute_metrics(preds_v1c)
m_v1d <- .compute_metrics(preds_v1d)

# Champion's own 2026 rows in the same chronological order as the score sets,
# for a like-for-like boot_mae_diff comparison.
champ_2026_ordered <- champion_preds[res_v1a$idx, ]

boot_v1a <- boot_mae_diff(preds_v1a, champ_2026_ordered)
boot_v1b <- boot_mae_diff(preds_v1b, champ_2026_ordered)
boot_v1c <- boot_mae_diff(preds_v1c, champ_2026_ordered)
boot_v1d <- boot_mae_diff(preds_v1d, champ_2026_ordered)

cat("\n=== WS1 2026 Screen: V1a-V1d vs Champion (Input Blend) ===\n")
.print_metrics(base_2026, "Champion (no recal)")
.print_metrics(m_v1a, "V1a (global slope b)")
.print_metrics(m_v1b, "V1b (slope + intercept)")
.print_metrics(m_v1c, "V1c (nonlinear s(k=4))")
.print_metrics(m_v1d, "V1d (V1a + 2025 warm start)")

cat("\n--- Delta-MAE vs champion, match-level bootstrap 95% CI (negative = improvement) ---\n")
.print_boot <- function(b, label) {
  cat(sprintf("%-28s deltaMAE=%+.3f  95%%CI[%+.3f, %+.3f]  n=%d\n",
              label, b$mae_diff, b$mae_ci[1], b$mae_ci[2], b$n_matches))
}
.print_boot(boot_v1a, "V1a")
.print_boot(boot_v1b, "V1b")
.print_boot(boot_v1c, "V1c")
.print_boot(boot_v1d, "V1d")

# Pick screen winner: lowest 2026 MAE ----
screen_summary <- data.frame(
  variant = c("V1a", "V1b", "V1c", "V1d"),
  mae     = c(m_v1a$mae, m_v1b$mae, m_v1c$mae, m_v1d$mae),
  slope   = c(m_v1a$slope, m_v1b$slope, m_v1c$slope, m_v1d$slope),
  close_mae = c(m_v1a$close_mae, m_v1b$close_mae, m_v1c$close_mae, m_v1d$close_mae),
  delta_mae_ci_hi = c(boot_v1a$mae_ci[2], boot_v1b$mae_ci[2], boot_v1c$mae_ci[2], boot_v1d$mae_ci[2])
)
cat("\n=== Screen summary ===\n")
print(screen_summary, row.names = FALSE)

winner <- screen_summary$variant[which.min(screen_summary$mae)]
winner_mode <- switch(winner, V1a = "slope_only", V1b = "slope_intercept", V1c = "nonlinear", V1d = "slope_only")
winner_history <- if (winner == "V1d") "all" else "within_2026"
cli::cli_alert_info("WS1 screen winner: {winner} (mode={winner_mode}, history={winner_history})")

# Confirm on 2025:2026 pooled ----
cli::cli_h1("WS1: Confirm winner ({winner}) on 2025:2026 pooled")

# Pooled confirm = continuous expanding window across the whole window (no
# season-boundary reset) -- this is V1d's mechanism applied pool-wide, and
# is mathematically identical to V1a run pool-wide too (both reduce to "use
# every strictly-prior round's history"), so one call covers whichever mode
# won the screen.
res_confirm <- recal_expanding(champion_preds, idx_all, idx_all, mode = winner_mode)
preds_confirm <- .apply_recal(champion_preds, res_confirm)
m_confirm <- .compute_metrics(preds_confirm)

champ_pool_ordered <- champion_preds[res_confirm$idx, ]
boot_confirm_pool <- boot_mae_diff(preds_confirm, champ_pool_ordered)

# Also report the confirm-window's 2026-only slice (same population as the
# screen number, but now recalibrated with the continuous/warm-started
# window rather than the within-2026-only window used for V1a/b/c on screen)
preds_confirm_2026 <- preds_confirm |> filter(season == SCREEN_SEASON)
champ_2026_for_confirm <- champ_pool_ordered |> filter(season == SCREEN_SEASON)
m_confirm_2026 <- .compute_metrics(preds_confirm_2026)
boot_confirm_2026 <- boot_mae_diff(preds_confirm_2026, champ_2026_for_confirm)

cat("\n=== WS1 Confirm: winner", winner, "on pooled 2025:2026 ===\n")
.print_metrics(base_pool, "Champion (no recal), pooled")
.print_metrics(m_confirm, sprintf("%s recal, pooled", winner))
.print_boot(boot_confirm_pool, sprintf("%s pooled 2025:2026", winner))

cat("\n=== WS1 Confirm: winner", winner, "2026-only slice of the pooled/warm-started run ===\n")
.print_metrics(base_2026, "Champion (no recal), 2026")
.print_metrics(m_confirm_2026, sprintf("%s recal (warm-started), 2026", winner))
.print_boot(boot_confirm_2026, sprintf("%s 2026-only (warm-started)", winner))

# Success criteria check (plan WS1) ----
cli::cli_h1("WS1 Success Criteria Check")

target_mae_delta <- -0.3
slope_lo <- 0.92
slope_hi <- 1.05

mae_delta_pool  <- m_confirm$mae - base_pool$mae
ci_excludes_zero_pool <- boot_confirm_pool$mae_ci[2] < 0  # both bounds negative -> improvement, excludes 0
slope_ok        <- m_confirm$slope >= slope_lo & m_confirm$slope <= slope_hi
close_mae_delta_pool <- m_confirm$close_mae - base_pool$close_mae
brier_delta_pool <- m_confirm$brier - base_pool$brier   # pred_win untouched -> should be ~0

cat(sprintf("Pooled 2025:2026 MAE delta vs champion: %+.3f (target <= %.1f)\n", mae_delta_pool, target_mae_delta))
cat(sprintf("Pooled 95%% CI on delta-MAE: [%+.3f, %+.3f] (excludes 0, improvement: %s)\n",
            boot_confirm_pool$mae_ci[1], boot_confirm_pool$mae_ci[2], ci_excludes_zero_pool))
cat(sprintf("Pooled recalibrated slope: %.3f (target [%.2f, %.2f]): %s\n",
            m_confirm$slope, slope_lo, slope_hi, slope_ok))
cat(sprintf("Pooled close-bucket MAE delta: %+.3f (negative = improved)\n", close_mae_delta_pool))
cat(sprintf("Pooled Brier delta: %+.5f (guard: <= 0.002 worsening)\n", brier_delta_pool))

meets_mae_target <- mae_delta_pool <= target_mae_delta
ship_recommended <- meets_mae_target && ci_excludes_zero_pool && slope_ok && (brier_delta_pool <= 0.002)

cat(sprintf("\nMeets -0.3 MAE target: %s\n", meets_mae_target))
cat(sprintf("Ship-recommended (G3 gate: MAE target + CI excl 0 + slope band + Brier guard): %s\n", ship_recommended))

tictoc::toc()

cat("\n=== WS1 Final Summary ===\n")
cat("Champion (Input Blend) baseline: 2026 MAE=", round(base_2026$mae, 3),
    " slope=", round(base_2026$slope, 3),
    " | pooled 2025:2026 MAE=", round(base_pool$mae, 3),
    " slope=", round(base_pool$slope, 3), "\n", sep = "")
cat("Screen winner:", winner, "(mode =", winner_mode, ")\n")
cat("Confirm (pooled) MAE=", round(m_confirm$mae, 3),
    " slope=", round(m_confirm$slope, 3),
    " deltaMAE=", round(mae_delta_pool, 3),
    " 95%CI=[", round(boot_confirm_pool$mae_ci[1], 3), ", ", round(boot_confirm_pool$mae_ci[2], 3), "]\n", sep = "")
cat("Ship recommended:", ship_recommended, "\n")
