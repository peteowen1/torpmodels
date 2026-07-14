# Train Match Prediction Models — Rolling Evaluation Script
# =========================================================
# Rolling week-by-week out-of-sample evaluation: for each round in the
# test seasons, train GAMs + XGBoost on all prior completed matches,
# predict that round, then aggregate metrics and compare to Squiggle.
#
# The rolling loop itself lives in experiments/rolling_lib.R
# (run_rolling_eval() + .compute_metrics() + the margin-bucket report
# tables + boot_mae_diff()) — FABLE-MATCH-MAE-PLAN.md WS0. This script is
# now a thin caller: build data, run the harness with default trainers
# (byte-compatible with the pre-refactor inline loop), then report.
#
# XGBoost nrounds are pre-optimised via CV on all pre-test-season data
# (minimal leakage) inside run_rolling_eval(), then reused with fixed
# nrounds per rolling step (no per-week CV).
#
# Uses shared functions from torp/R/match_model.R and match_train.R.

# Parameters ----
# exists()-guarded so a caller can inject TEST_SEASONS before sourcing this
# script (e.g. eval_squiggle_rank.R) without it being clobbered.
if (!exists("TEST_SEASONS", inherits = FALSE)) TEST_SEASONS <- 2025:2026  # Seasons to evaluate (rolling week-by-week)

# Setup ----
library(tidyverse)
library(xgboost)
library(mgcv)
library(MLmetrics)
library(geosphere)
library(cli)

# Load torp — prefer devtools::load_all() for latest code, fall back to installed
torp_paths <- c("../torp", "../../torp", "../../../torp")
torp_loaded <- FALSE
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p)
    torp_loaded <- TRUE
    break
  }
}
if (!torp_loaded) {
  if (!requireNamespace("torp", quietly = TRUE)) {
    stop("Cannot find torp package. Install it or run from torpverse workspace.")
  }
  library(torp)
}

# Load the rolling harness (WS0 extraction — run_rolling_eval, .compute_metrics,
# margin_calibration_by_pred_bucket, mae_by_actual_bucket, boot_mae_diff) ----
rolling_lib_candidates <- c(
  "experiments/rolling_lib.R",
  "04-match-model/experiments/rolling_lib.R",
  "data-raw/04-match-model/experiments/rolling_lib.R",
  "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/rolling_lib.R"
)
rolling_lib_hits <- rolling_lib_candidates[file.exists(rolling_lib_candidates)]
if (length(rolling_lib_hits) == 0) stop("Cannot find experiments/rolling_lib.R")
source(rolling_lib_hits[1])

# Build Data ----
cli::cli_h1("Building Match Prediction Training Data")
tictoc::tic("total")

team_mdl_df <- build_team_mdl_df()

cli::cli_inform("Seasons: {paste(sort(unique(team_mdl_df$season.x)), collapse = ', ')}")

# Rolling evaluation ----
# Default trainers (.train_match_gams from torp, .train_xgb_fixed from
# rolling_lib.R) reproduce the pre-refactor inline loop exactly.
roll <- run_rolling_eval(team_mdl_df, TEST_SEASONS)

gam_preds         <- roll$gam_preds
xgb_preds         <- roll$xgb_preds
blend_preds       <- roll$blend_preds
input_blend_preds <- roll$input_blend_preds
xgb_nrounds       <- roll$xgb_nrounds
n_test_rounds     <- nrow(roll$test_rounds)

n_oos_matches <- nrow(gam_preds)
cli::cli_inform("Total out-of-sample predictions: {n_oos_matches} matches")

# Overall metrics ----
cli::cli_h1("Rolling Out-of-Sample Results")

gam_m   <- .compute_metrics(gam_preds)
xgb_m   <- .compute_metrics(xgb_preds)
blend_m <- .compute_metrics(blend_preds)
ib_m    <- .compute_metrics(input_blend_preds)

comparison <- data.frame(
  Model = c("GAM", "XGBoost", "Output Blend", "Input Blend"),
  N = n_oos_matches,
  LogLoss  = round(c(gam_m$logloss, xgb_m$logloss, blend_m$logloss, ib_m$logloss), 4),
  Accuracy = round(c(gam_m$accuracy, xgb_m$accuracy, blend_m$accuracy, ib_m$accuracy), 1),
  Brier    = round(c(gam_m$brier, xgb_m$brier, blend_m$brier, ib_m$brier), 4),
  MAE      = round(c(gam_m$mae, xgb_m$mae, blend_m$mae, ib_m$mae), 1),
  RMSE     = round(c(gam_m$rmse, xgb_m$rmse, blend_m$rmse, ib_m$rmse), 1),
  Slope    = round(c(gam_m$slope, xgb_m$slope, blend_m$slope, ib_m$slope), 3),
  SDRatio  = round(c(gam_m$sd_ratio, xgb_m$sd_ratio, blend_m$sd_ratio, ib_m$sd_ratio), 3),
  Cor      = round(c(gam_m$cor, xgb_m$cor, blend_m$cor, ib_m$cor), 3),
  CloseMAE = round(c(gam_m$close_mae, xgb_m$close_mae, blend_m$close_mae, ib_m$close_mae), 1),
  stringsAsFactors = FALSE
)

cat("\n=== TORP Rolling OOS Comparison ===\n")
print(comparison, row.names = FALSE)

# Per-season breakdown ----
.season_metrics <- function(preds, label) {
  preds |>
    group_by(season) |>
    summarise(
      Model = label,
      N = n(),
      LogLoss  = round(MLmetrics::LogLoss(pred_win, home_win), 4),
      Accuracy = round(mean(round(pred_win) == home_win) * 100, 1),
      Brier    = round(mean((pred_win - home_win)^2), 4),
      MAE      = round(mean(abs(pred_margin - margin)), 1),
      RMSE     = round(sqrt(mean((pred_margin - margin)^2)), 1),
      .groups  = "drop"
    )
}

season_breakdown <- bind_rows(
  .season_metrics(gam_preds, "GAM"),
  .season_metrics(xgb_preds, "XGBoost"),
  .season_metrics(blend_preds, "Output Blend"),
  .season_metrics(input_blend_preds, "Input Blend")
) |>
  arrange(season, Model)

cat("\n=== Per-Season Breakdown ===\n")
print(as.data.frame(season_breakdown), row.names = FALSE)

# Calibration bins (win probability) ----
cal_breaks <- c(0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1)
cal_labels <- c("<30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-85%", ">85%")

cal_table <- gam_preds |>
  mutate(
    bin = cut(pred_win, breaks = cal_breaks, labels = cal_labels, include.lowest = TRUE),
    xgb_prob = xgb_preds$pred_win,
    out_blend_prob = blend_preds$pred_win,
    in_blend_prob = input_blend_preds$pred_win,
    xgb_margin = xgb_preds$pred_margin,
    out_blend_margin = blend_preds$pred_margin,
    in_blend_margin = input_blend_preds$pred_margin
  ) |>
  group_by(bin) |>
  summarise(
    N = n(),
    Actual = round(mean(home_win) * 100, 1),
    GAM = round(mean(pred_win) * 100, 1),
    XGBoost = round(mean(xgb_prob) * 100, 1),
    OutBlend = round(mean(out_blend_prob) * 100, 1),
    InBlend = round(mean(in_blend_prob) * 100, 1),
    GAM_RMSE = round(sqrt(mean((pred_margin - margin)^2)), 1),
    XGB_RMSE = round(sqrt(mean((xgb_margin - margin)^2)), 1),
    OutB_RMSE = round(sqrt(mean((out_blend_margin - margin)^2)), 1),
    InB_RMSE = round(sqrt(mean((in_blend_margin - margin)^2)), 1),
    .groups = "drop"
  )

cat("\n=== Calibration Bins (mean pred % vs actual win %) ===\n")
print(as.data.frame(cal_table), row.names = FALSE)

# WS0: Margin calibration by predicted-margin bucket (actionable, pre-game view) ----
margin_bucket_report <- bind_rows(
  margin_calibration_by_pred_bucket(gam_preds)         |> mutate(Model = "GAM", .before = 1),
  margin_calibration_by_pred_bucket(xgb_preds)         |> mutate(Model = "XGBoost", .before = 1),
  margin_calibration_by_pred_bucket(blend_preds)       |> mutate(Model = "Output Blend", .before = 1),
  margin_calibration_by_pred_bucket(input_blend_preds) |> mutate(Model = "Input Blend", .before = 1)
)

cat("\n=== Margin Calibration by Predicted-Margin Bucket ===\n")
print(as.data.frame(margin_bucket_report), row.names = FALSE)

# WS0: MAE by actual-margin bucket (diagnostic only, not pre-game observable) ----
actual_bucket_report <- bind_rows(
  mae_by_actual_bucket(gam_preds)         |> mutate(Model = "GAM", .before = 1),
  mae_by_actual_bucket(xgb_preds)         |> mutate(Model = "XGBoost", .before = 1),
  mae_by_actual_bucket(blend_preds)       |> mutate(Model = "Output Blend", .before = 1),
  mae_by_actual_bucket(input_blend_preds) |> mutate(Model = "Input Blend", .before = 1)
)

cat("\n=== MAE by Actual-Margin Bucket (diagnostic) ===\n")
print(as.data.frame(actual_bucket_report), row.names = FALSE)

# Squiggle Comparison ----
cli::cli_h2("Squiggle Model Comparison")

squiggle_tips <- tryCatch(
  purrr::map_dfr(TEST_SEASONS, ~fitzRoy::fetch_squiggle_data("tips", year = .x)),
  error = function(e) { cli::cli_warn("Failed to fetch Squiggle data: {e$message}"); NULL }
)

if (!is.null(squiggle_tips) && nrow(squiggle_tips) > 0) {

  squiggle_tips <- squiggle_tips |>
    mutate(
      hteam_norm = torp_replace_teams(hteam),
      hconfidence = as.numeric(hconfidence),
      hmargin = as.numeric(hmargin),
      round = as.integer(round)
    )

  # Join squiggle tips to GAM rolling predictions
  sq_joined <- squiggle_tips |>
    inner_join(gam_preds, by = c("year" = "season", "round" = "round",
                                  "hteam_norm" = "home_team_chr"))

  n_matched <- n_distinct(sq_joined$gameid)
  cli::cli_inform("Squiggle tips matched: {n_matched} of {n_oos_matches} test matches")

  if (n_matched < n_oos_matches) {
    matched_keys <- sq_joined |> distinct(year, round, hteam_norm)
    unmatched <- gam_preds |>
      anti_join(matched_keys, by = c("season" = "year", "round" = "round",
                                      "home_team_chr" = "hteam_norm"))
    if (nrow(unmatched) > 0) {
      cli::cli_inform("Unmatched: {nrow(unmatched)} games (likely 2026 or incomplete rounds)")
    }
  }

  sq_joined <- sq_joined |>
    mutate(actual_margin = margin.y,
           home_win = ifelse(actual_margin > 0, 1,
                             ifelse(actual_margin == 0, 0.5, 0)))

  # Also join XGB/blend for matched games
  sq_match_keys <- sq_joined |> distinct(year, round, hteam_norm)
  sq_join_by <- c("season" = "year", "round" = "round", "home_team_chr" = "hteam_norm")

  xgb_matched <- xgb_preds |> semi_join(sq_match_keys, by = sq_join_by)
  blend_matched <- blend_preds |> semi_join(sq_match_keys, by = sq_join_by)
  ib_matched <- input_blend_preds |> semi_join(sq_match_keys, by = sq_join_by)

  # TORP metrics on matched games only
  gam_matched_m   <- .compute_metrics(gam_preds |> semi_join(sq_match_keys, by = sq_join_by))
  xgb_matched_m   <- .compute_metrics(xgb_matched)
  blend_matched_m <- .compute_metrics(blend_matched)
  ib_matched_m    <- .compute_metrics(ib_matched)

  # Per-source squiggle metrics
  sq_metrics <- sq_joined |>
    group_by(sourceid, source) |>
    summarise(
      n = n(),
      logloss  = MLmetrics::LogLoss(hconfidence / 100, home_win),
      accuracy = mean(round(hconfidence / 100) == home_win) * 100,
      brier    = mean((hconfidence / 100 - home_win)^2),
      mae      = mean(abs(hmargin - actual_margin)),
      rmse     = sqrt(mean((hmargin - actual_margin)^2)),
      .groups  = "drop"
    ) |>
    arrange(brier)

  full_comparison <- data.frame(
    Source   = c("TORP GAM", "TORP XGBoost", "TORP Output Blend", "TORP Input Blend"),
    N        = n_matched,
    Accuracy = round(c(gam_matched_m$accuracy, xgb_matched_m$accuracy, blend_matched_m$accuracy, ib_matched_m$accuracy), 1),
    LogLoss  = round(c(gam_matched_m$logloss, xgb_matched_m$logloss, blend_matched_m$logloss, ib_matched_m$logloss), 4),
    Brier    = round(c(gam_matched_m$brier, xgb_matched_m$brier, blend_matched_m$brier, ib_matched_m$brier), 4),
    MAE      = round(c(gam_matched_m$mae, xgb_matched_m$mae, blend_matched_m$mae, ib_matched_m$mae), 1),
    RMSE     = round(c(gam_matched_m$rmse, xgb_matched_m$rmse, blend_matched_m$rmse, ib_matched_m$rmse), 1),
    stringsAsFactors = FALSE
  )

  # Add squiggle sources (require 80%+ coverage)
  min_tips <- floor(n_matched * 0.8)
  sq_display <- sq_metrics |> filter(n >= min_tips)

  for (i in seq_len(nrow(sq_display))) {
    full_comparison <- rbind(full_comparison, data.frame(
      Source   = sq_display$source[i],
      N        = sq_display$n[i],
      Accuracy = round(sq_display$accuracy[i], 2),
      LogLoss  = round(sq_display$logloss[i], 4),
      Brier    = round(sq_display$brier[i], 4),
      MAE      = round(sq_display$mae[i], 2),
      RMSE     = round(sq_display$rmse[i], 2),
      stringsAsFactors = FALSE
    ))
  }

  # Rank by average z-score across LogLoss, Brier, MAE, RMSE (lower = better)
  rank_cols <- c("LogLoss", "Brier", "MAE", "RMSE")
  z_scores <- as.data.frame(lapply(full_comparison[rank_cols], function(x) {
    (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
  }))
  full_comparison$AvgZ <- round(rowMeans(z_scores, na.rm = TRUE), 3)
  full_comparison$Rank <- rank(full_comparison$AvgZ)
  full_comparison <- full_comparison[order(full_comparison$AvgZ), ]

  cat("\n=== Full Model Comparison (ranked by avg z-score: LogLoss/Brier/MAE/RMSE) ===\n")
  cat("Test matches:", n_matched, "| Seasons:", paste(TEST_SEASONS, collapse = "-"), "\n\n")
  print(full_comparison, row.names = FALSE)
}

tictoc::toc()

# WS0: Submitted tips vs harness variants — champion identification + stop condition ----
# FABLE-MATCH-MAE-PLAN.md WS0 deliverables 5-6: compute the same stats directly
# on torp's actually-submitted 2026 Squiggle tips (source "In The Game"), then
# identify which harness variant best matches them (G4 champion) and check the
# stop condition (no variant reproducing the submitted slope ~0.80 +/- 0.05
# would mean the defect lives in the serve path, not the model).
cli::cli_h1("WS0: Submitted Tips vs Harness Variants")

submitted_stats <- NULL
champion_variant <- NA_character_
stop_condition_triggered <- NA

sub_games_2026 <- tryCatch(
  fitzRoy::fetch_squiggle_data("games", year = 2026),
  error = function(e) { cli::cli_warn("Failed to fetch Squiggle games: {e$message}"); NULL }
)
sub_tips_2026 <- tryCatch(
  fitzRoy::fetch_squiggle_data("tips", year = 2026),
  error = function(e) { cli::cli_warn("Failed to fetch Squiggle tips: {e$message}"); NULL }
)

if (!is.null(sub_games_2026) && !is.null(sub_tips_2026)) {

  sub_games <- sub_games_2026 |>
    filter(complete == 100) |>
    transmute(
      gameid = as.integer(id),
      round = as.integer(round),
      hteam_norm = torp_replace_teams(hteam),
      actual_margin = as.numeric(hscore) - as.numeric(ascore)
    )

  submitted_raw <- sub_tips_2026 |>
    filter(source == "In The Game") |>
    mutate(
      gameid = as.integer(gameid),
      round = as.integer(round),
      hmargin = as.numeric(hmargin),
      hconfidence = as.numeric(hconfidence),
      hteam_norm = torp_replace_teams(hteam)
    ) |>
    inner_join(sub_games |> select(gameid, actual_margin), by = "gameid") |>
    filter(!is.na(hmargin), !is.na(hconfidence))

  n_submitted <- nrow(submitted_raw)
  cli::cli_inform("Submitted 'In The Game' Squiggle tips found: {n_submitted}")

  if (n_submitted >= 10) {
    submitted_preds <- submitted_raw |>
      transmute(
        round, hteam_norm,
        pred_margin = hmargin,
        pred_win = hconfidence / 100,
        margin = actual_margin,
        home_win = ifelse(actual_margin > 0, 1, ifelse(actual_margin == 0, 0.5, 0))
      )

    submitted_stats <- .compute_metrics(submitted_preds)

    cat("\n=== Submitted 2026 Squiggle Tips ('In The Game') — Direct Stats ===\n")
    cat(sprintf(
      "N=%d  MAE=%.2f  RMSE=%.2f  Brier=%.4f  Slope=%.3f  Cor=%.3f  SDRatio=%.3f  CloseMAE(n=%d)=%.2f\n",
      n_submitted, submitted_stats$mae, submitted_stats$rmse, submitted_stats$brier,
      submitted_stats$slope, submitted_stats$cor, submitted_stats$sd_ratio,
      submitted_stats$close_n, submitted_stats$close_mae
    ))

    # 2026-only slice of each harness variant, for apples-to-apples comparison
    # (submitted tips only exist for 2026 — diagnosis Finding 6)
    variant_list_2026 <- list(
      GAM            = gam_preds |> filter(season == 2026),
      XGBoost        = xgb_preds |> filter(season == 2026),
      `Output Blend` = blend_preds |> filter(season == 2026),
      `Input Blend`  = input_blend_preds |> filter(season == 2026)
    )

    variant_2026_metrics <- lapply(variant_list_2026, .compute_metrics)

    # Correlation of each variant's predicted margin vs submitted-tips predicted
    # margin, on the SAME matched games — this identifies the G4 champion.
    corr_table <- purrr::imap_dfr(variant_list_2026, function(preds, label) {
      joined <- preds |>
        mutate(hteam_norm = torp_replace_teams(as.character(home_team))) |>
        inner_join(
          submitted_preds |> select(round, hteam_norm,
                                     sub_pred_margin = pred_margin, sub_pred_win = pred_win),
          by = c("round", "hteam_norm")
        )
      m <- variant_2026_metrics[[label]]
      data.frame(
        Model = label,
        N_matched = nrow(joined),
        CorMarginVsSubmitted = if (nrow(joined) >= 3) round(cor(joined$pred_margin, joined$sub_pred_margin), 3) else NA_real_,
        CorWinVsSubmitted = if (nrow(joined) >= 3) round(cor(joined$pred_win, joined$sub_pred_win), 3) else NA_real_,
        Slope2026 = round(m$slope, 3),
        MAE2026 = round(m$mae, 2),
        CloseMAE2026 = round(m$close_mae, 2),
        stringsAsFactors = FALSE
      )
    })

    cat("\n=== WS0 Champion Identification: Correlation to Submitted Tips (2026) ===\n")
    print(corr_table, row.names = FALSE)

    champion_variant <- corr_table$Model[which.max(corr_table$CorMarginVsSubmitted)]
    cli::cli_alert_info("Champion variant (highest correlation to submitted tips): {champion_variant}")

    # WS0 stop condition: does ANY variant reproduce the submitted slope ~0.80 (+/-0.05)?
    slope_target <- 0.80
    slope_tol <- 0.05
    slopes_2026 <- vapply(variant_2026_metrics, function(m) m$slope, numeric(1))
    reproduces <- abs(slopes_2026 - slope_target) <= slope_tol
    stop_condition_triggered <- !any(reproduces)

    cat("\n=== WS0 Stop Condition Check ===\n")
    cat(sprintf("Submitted-tips slope: %.3f (target band [%.2f, %.2f])\n",
                submitted_stats$slope, slope_target - slope_tol, slope_target + slope_tol))
    for (nm in names(slopes_2026)) {
      cat(sprintf("  %-14s slope=%.3f  %s\n", nm, slopes_2026[[nm]],
                  if (reproduces[[nm]]) "REPRODUCES" else "does not reproduce"))
    }
    if (stop_condition_triggered) {
      cli::cli_alert_danger("WS0 STOP CONDITION TRIGGERED: no harness variant reproduces the submitted slope ~0.80 (+/-0.05). Escalate to a serve-path audit (blend block, forecast weather, submission staleness) before running WS1-WS5.")
    } else {
      cli::cli_alert_success("WS0 stop condition NOT triggered: at least one variant reproduces the submitted slope.")
    }

    # Demonstration + sanity check of boot_mae_diff() (E5 helper): Input Blend
    # vs GAM-only, pooled TEST_SEASONS window.
    boot_demo <- boot_mae_diff(input_blend_preds, gam_preds, B = 2000)
    cat("\n=== boot_mae_diff() demo: Input Blend vs GAM (pooled", paste(TEST_SEASONS, collapse = "-"), ") ===\n")
    cat(sprintf("N matches=%d  MAE diff (InputBlend-GAM)=%.3f  95%% CI [%.3f, %.3f]\n",
                boot_demo$n_matches, boot_demo$mae_diff, boot_demo$mae_ci[1], boot_demo$mae_ci[2]))
  } else {
    cli::cli_warn("Fewer than 10 submitted 'In The Game' tips found for 2026 — skipping champion identification.")
  }
} else {
  cli::cli_warn("Could not fetch Squiggle games/tips for 2026 — skipping WS0 champion identification.")
}

# Summary ----
# Evaluation only -- production match GAMs have exactly one publisher,
# torp::run_predictions_pipeline() (daily CI + rebuild Phase 9). This script
# never saves/uploads a production match_gams.rds.
cat("\n=== Final Summary ===\n")
cat("Rolling OOS evaluation:", n_oos_matches, "matches across", n_test_rounds, "rounds\n")
cat("GAM        Brier:", round(gam_m$brier, 4), "| MAE:", round(gam_m$mae, 1), "| RMSE:", round(gam_m$rmse, 1), "| Slope:", round(gam_m$slope, 3), "\n")
cat("XGB        Brier:", round(xgb_m$brier, 4), "| MAE:", round(xgb_m$mae, 1), "| RMSE:", round(xgb_m$rmse, 1), "| Slope:", round(xgb_m$slope, 3), "\n")
cat("Out Blend  Brier:", round(blend_m$brier, 4), "| MAE:", round(blend_m$mae, 1), "| RMSE:", round(blend_m$rmse, 1), "| Slope:", round(blend_m$slope, 3), "\n")
cat("In Blend   Brier:", round(ib_m$brier, 4), "| MAE:", round(ib_m$mae, 1), "| RMSE:", round(ib_m$rmse, 1), "| Slope:", round(ib_m$slope, 3), "\n")
if (!is.null(submitted_stats)) {
  cat("Submitted  Brier:", round(submitted_stats$brier, 4), "| MAE:", round(submitted_stats$mae, 1), "| RMSE:", round(submitted_stats$rmse, 1), "| Slope:", round(submitted_stats$slope, 3), "\n")
}
cat("WS0 champion variant (G4):", champion_variant, "\n")
cat("WS0 stop condition triggered:", isTRUE(stop_condition_triggered), "\n")
