# Train Match Prediction Models (GAM pipeline) — Evaluation Script
# =================================================================
# This script is for holdout evaluation and XGBoost comparison only.
# Production match predictions retrain daily via build_match_predictions.R
# in torp/data-raw/02-models/ (no pre-trained models needed).
#
# Uses shared functions from torp/R/match_model.R for data preparation
# and GAM training, then adds XGBoost comparison pipeline on top.
#
# Data pipeline:
#   1. build_team_mdl_df() — shared convenience wrapper loads all data
#   2. .train_match_gams() — shared GAM training (with holdout filter)
#   3. XGBoost comparison pipeline (evaluation only, this script)
#   4. Temporal holdout comparison (GAM vs XGBoost)
#   5. Save & optionally upload models

# Parameters ----
# Tweak these before running in RStudio
HOLDOUT_SEASON <- 2025   # Inf = production (train on all data). Set to e.g. 2025 for evaluation/comparison.
UPLOAD_TO_GITHUB <- FALSE # FALSE = save locally only, don't upload to GitHub releases

# Setup ----
library(tidyverse)
library(xgboost)
library(mgcv)
# library(fitzRoy)  # only used for Squiggle comparison (optional)
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

# Build Data ----
cli::cli_h1("Building Match Prediction Training Data")
tictoc::tic("total")

# Weather path: try relative paths from torpmodels
weather_path <- file.path("..", "torp", "data-raw", "weather_data.parquet")
if (!file.exists(weather_path)) weather_path <- file.path("..", "..", "torp", "data-raw", "weather_data.parquet")

# PSR coefficient path
psr_coef_path <- file.path("..", "torp", "data-raw", "cache-skills", "psr_v2_coefficients.csv")
if (!file.exists(psr_coef_path)) psr_coef_path <- file.path("..", "..", "torp", "data-raw", "cache-skills", "psr_v2_coefficients.csv")

team_mdl_df <- build_team_mdl_df(weather_path = weather_path, psr_coef_path = psr_coef_path)

cli::cli_inform("Seasons: {paste(sort(unique(team_mdl_df$season.x)), collapse = ', ')}")

# Train Models ----
cli::cli_h2("Training GAM pipeline (5 sequential models)")

train_filter <- if (is.finite(HOLDOUT_SEASON)) {
  team_mdl_df$season.x < HOLDOUT_SEASON
} else {
  rep(TRUE, nrow(team_mdl_df))
}

gam_result <- .train_match_gams(team_mdl_df, train_filter = train_filter)
team_mdl_df <- gam_result$data
match_gams <- gam_result$models

gam_df <- team_mdl_df |> filter(!is.na(win), season.x < HOLDOUT_SEASON)
cli::cli_inform("Train: {nrow(gam_df) / 2} matches ({paste(sort(unique(gam_df$season.x)), collapse=', ')})")
if (is.finite(HOLDOUT_SEASON)) {
  test_df <- team_mdl_df |> filter(!is.na(win), season.x >= HOLDOUT_SEASON)
  cli::cli_inform("Test:  {nrow(test_df) / 2} matches (season >= {HOLDOUT_SEASON})")
} else {
  cli::cli_inform("Production mode: training on all data (no holdout)")
}

# XGBoost Pipeline (evaluation mode only) ----
if (is.finite(HOLDOUT_SEASON)) {
cli::cli_h2("Training XGBoost pipeline (5 sequential models)")

xgb_result <- .train_match_xgb(team_mdl_df, train_filter = train_filter)
team_mdl_df <- xgb_result$data

## Temporal Holdout Evaluation ----
cli::cli_h1("Temporal Holdout: Train < {HOLDOUT_SEASON}, Test >= {HOLDOUT_SEASON}")

# Match-level mask for holdout test set
xgb_test_mask <- !is.na(team_mdl_df$win) & !is.na(team_mdl_df$total_xpoints_adj) &
                 !is.na(team_mdl_df$xscore_diff) & !is.na(team_mdl_df$shot_conv_diff) &
                 !is.na(team_mdl_df$score_diff) & team_mdl_df$season.x >= HOLDOUT_SEASON

# Average home/away predictions per match using .format_match_preds()
gam_holdout <- .format_match_preds(team_mdl_df |> filter(xgb_test_mask))

# XGBoost: map xgb predictions into the GAM column names, then format
xgb_holdout <- team_mdl_df |>
  filter(xgb_test_mask) |>
  mutate(pred_score_diff = xgb_pred_score_diff, pred_win = xgb_pred_win) |>
  .format_match_preds()

### GAM holdout metrics ----
# .format_match_preds() outputs: pred_margin, pred_win, margin (home perspective)
gam_win <- ifelse(gam_holdout$margin > 0, 1, ifelse(gam_holdout$margin == 0, 0.5, 0))
gam_logloss  <- MLmetrics::LogLoss(gam_holdout$pred_win, gam_win)
gam_accuracy <- mean(round(gam_holdout$pred_win) == gam_win)
gam_brier    <- mean((gam_holdout$pred_win - gam_win)^2)
gam_mae      <- mean(abs(gam_holdout$pred_margin - gam_holdout$margin))
gam_rmse     <- sqrt(mean((gam_holdout$pred_margin - gam_holdout$margin)^2))

### XGBoost holdout metrics ----
xgb_win <- ifelse(xgb_holdout$margin > 0, 1, ifelse(xgb_holdout$margin == 0, 0.5, 0))
xgb_logloss  <- MLmetrics::LogLoss(xgb_holdout$pred_win, xgb_win)
xgb_accuracy <- mean(round(xgb_holdout$pred_win) == xgb_win)
xgb_brier    <- mean((xgb_holdout$pred_win - xgb_win)^2)
xgb_mae      <- mean(abs(xgb_holdout$pred_margin - xgb_holdout$margin))
xgb_rmse     <- sqrt(mean((xgb_holdout$pred_margin - xgb_holdout$margin)^2))

### 50/50 Blend holdout metrics ----
blend_pred_win    <- 0.5 * gam_holdout$pred_win + 0.5 * xgb_holdout$pred_win
blend_pred_margin <- 0.5 * gam_holdout$pred_margin + 0.5 * xgb_holdout$pred_margin
blend_logloss  <- MLmetrics::LogLoss(blend_pred_win, gam_win)
blend_accuracy <- mean(round(blend_pred_win) == gam_win)
blend_brier    <- mean((blend_pred_win - gam_win)^2)
blend_mae      <- mean(abs(blend_pred_margin - gam_holdout$margin))
blend_rmse     <- sqrt(mean((blend_pred_margin - gam_holdout$margin)^2))

### GAM vs XGBoost vs Blend comparison ----
comparison <- data.frame(
  Metric = c("Win LogLoss", "Win Accuracy (%)", "Win Brier", "Score Diff MAE", "Score Diff RMSE"),
  GAM = c(round(gam_logloss, 4), round(gam_accuracy * 100, 1), round(gam_brier, 4), round(gam_mae, 1), round(gam_rmse, 1)),
  XGBoost = c(round(xgb_logloss, 4), round(xgb_accuracy * 100, 1), round(xgb_brier, 4), round(xgb_mae, 1), round(xgb_rmse, 1)),
  Blend = c(round(blend_logloss, 4), round(blend_accuracy * 100, 1), round(blend_brier, 4), round(blend_mae, 1), round(blend_rmse, 1)),
  stringsAsFactors = FALSE
)

n_holdout_matches <- nrow(gam_holdout)
cat("\n=== Holdout Test Set Comparison ===\n")
n_train_matches <- nrow(team_mdl_df |> filter(!is.na(win), season.x < HOLDOUT_SEASON)) / 2
cat("Train:", n_train_matches, "matches | Test:", n_holdout_matches, "matches\n\n")
print(comparison, row.names = FALSE)

### Calibration bins ----
cal_breaks <- c(0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1)
cal_labels <- c("<30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-85%", ">85%")

cal_df <- data.frame(
  bin = cut(gam_holdout$pred_win, breaks = cal_breaks, labels = cal_labels, include.lowest = TRUE),
  actual = gam_win,
  margin = gam_holdout$margin,
  gam_prob = gam_holdout$pred_win,
  xgb_prob = xgb_holdout$pred_win,
  blend_prob = blend_pred_win,
  gam_margin = gam_holdout$pred_margin,
  xgb_margin = xgb_holdout$pred_margin,
  blend_margin = blend_pred_margin
)

cal_table <- cal_df |>
  dplyr::group_by(bin) |>
  dplyr::summarise(
    N = dplyr::n(),
    Actual = round(mean(actual) * 100, 1),
    GAM = round(mean(gam_prob) * 100, 1),
    XGBoost = round(mean(xgb_prob) * 100, 1),
    Blend = round(mean(blend_prob) * 100, 1),
    GAM_RMSE = round(sqrt(mean((gam_margin - margin)^2)), 1),
    XGB_RMSE = round(sqrt(mean((xgb_margin - margin)^2)), 1),
    Blend_RMSE = round(sqrt(mean((blend_margin - margin)^2)), 1),
    .groups = "drop"
  )

cat("\n=== Calibration Bins (mean pred % vs actual win %) ===\n")
print(as.data.frame(cal_table), row.names = FALSE)

### XGBoost CV scores ----
cat("\n=== XGBoost CV Scores (on train set, per step) ===\n")
cat(sprintf("  Step 1 (total_xpoints) RMSE:    %.4f  nrounds: %d\n", xgb_result$steps$total_xpoints$cv_score, xgb_result$steps$total_xpoints$best_n))
cat(sprintf("  Step 2 (xscore_diff)   RMSE:    %.4f  nrounds: %d\n", xgb_result$steps$xscore_diff$cv_score, xgb_result$steps$xscore_diff$best_n))
cat(sprintf("  Step 3 (conv_diff)     RMSE:    %.4f  nrounds: %d\n", xgb_result$steps$conv_diff$cv_score, xgb_result$steps$conv_diff$best_n))
cat(sprintf("  Step 4 (score_diff)    RMSE:    %.4f  nrounds: %d\n", xgb_result$steps$score_diff$cv_score, xgb_result$steps$score_diff$best_n))
cat(sprintf("  Step 5 (win)           LogLoss: %.4f  nrounds: %d\n", xgb_result$steps$win$cv_score, xgb_result$steps$win$best_n))

### Feature importance ----
importance <- xgb.importance(model = xgb_result$models$win)
cat("\nTop 10 XGBoost Features (win step):\n")
print(head(importance, 10))

## Squiggle Model Comparison ----
cli::cli_h2("Squiggle Model Comparison")

holdout_seasons <- sort(unique(
  team_mdl_df$season.x[team_mdl_df$season.x >= HOLDOUT_SEASON & !is.na(team_mdl_df$win)]
))
squiggle_tips <- tryCatch(
  purrr::map_dfr(holdout_seasons, ~fitzRoy::fetch_squiggle_data("tips", year = .x)),
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

  # Match-level test set from .format_match_preds() — normalize team names
  home_test <- gam_holdout |>
    mutate(home_team = torp_replace_teams(as.character(home_team)))

  # Join squiggle tips to home test matches
  sq_joined <- squiggle_tips |>
    inner_join(home_test, by = c("year" = "season", "round" = "round",
                                  "hteam_norm" = "home_team"))

  n_test_matches <- nrow(home_test)
  n_matched <- n_distinct(sq_joined$gameid)
  cli::cli_inform("Squiggle tips matched: {n_matched} of {n_test_matches} test matches")

  # Diagnose unmatched games
  if (n_matched < n_test_matches) {
    matched_keys <- sq_joined |> distinct(year, round, hteam_norm)
    unmatched <- home_test |>
      anti_join(matched_keys, by = c("season" = "year", "round" = "round",
                                      "home_team" = "hteam_norm"))
    cli::cli_inform("Unmatched games ({nrow(unmatched)}):")
    print(unmatched |> select(season, round, home_team), n = 50)
  }

  # Per-source metrics (only sources that tipped all matched games)
  # Join creates margin.x (squiggle) and margin.y (torp actual); use .y
  sq_joined <- sq_joined |>
    mutate(actual_margin = margin.y,
           home_win = ifelse(actual_margin > 0, 1,
                             ifelse(actual_margin == 0, 0.5, 0)))

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

  # Reuse match-level predictions from holdout section (gam_holdout, xgb_holdout)
  full_comparison <- data.frame(
    Source = c("TORP GAM", "TORP XGBoost", "TORP Blend"),
    N = n_test_matches,
    LogLoss  = c(round(gam_logloss, 4), round(xgb_logloss, 4), round(blend_logloss, 4)),
    Accuracy = c(round(gam_accuracy * 100, 1), round(xgb_accuracy * 100, 1), round(blend_accuracy * 100, 1)),
    Brier    = c(round(gam_brier, 4), round(xgb_brier, 4), round(blend_brier, 4)),
    MAE      = c(round(gam_mae, 1), round(xgb_mae, 2), round(blend_mae, 2)),
    RMSE     = c(round(gam_rmse, 1), round(xgb_rmse, 2), round(blend_rmse, 2)),
    stringsAsFactors = FALSE
  )

  # Add squiggle sources (only those with enough tips)
  min_tips <- floor(n_test_matches * 0.8)  # require 80%+ coverage
  sq_display <- sq_metrics |> filter(n >= min_tips)

  for (i in seq_len(nrow(sq_display))) {
    full_comparison <- rbind(full_comparison, data.frame(
      Source   = sq_display$source[i],
      N        = sq_display$n[i],
      LogLoss  = round(sq_display$logloss[i], 4),
      Accuracy = round(sq_display$accuracy[i], 1),
      Brier    = round(sq_display$brier[i], 4),
      MAE      = round(sq_display$mae[i], 2),
      RMSE     = round(sq_display$rmse[i], 2),
      stringsAsFactors = FALSE
    ))
  }

  full_comparison <- full_comparison[order(full_comparison$RMSE), ]

  cat("\n=== Full Model Comparison (sorted by Brier) ===\n")
  cat("Test matches:", n_test_matches, "| Holdout season:", HOLDOUT_SEASON, "\n\n")
  print(full_comparison, row.names = FALSE)
}

} else {
  cli::cli_inform("Skipping XGBoost pipeline & evaluation (production mode, HOLDOUT_SEASON = Inf)")
}

# Save Models ----
cli::cli_h2("Saving models")

output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

## Save GAM pipeline ----
gam_path <- file.path(output_dir, "match_gams.rds")
saveRDS(match_gams, gam_path)
cli::cli_inform("Saved GAM pipeline to {gam_path}")

## Save XGBoost pipeline (evaluation mode only) ----
if (is.finite(HOLDOUT_SEASON)) {
  xgb_path <- file.path(output_dir, "match_xgb_pipeline.rds")
  saveRDS(xgb_result$models, xgb_path)
  cli::cli_inform("Saved XGBoost pipeline to {xgb_path}")
}

## Upload to GitHub releases ----
if (!UPLOAD_TO_GITHUB) {
  cli::cli_inform("Skipping GitHub upload (UPLOAD_TO_GITHUB = FALSE). Models saved locally at {gam_path}")
} else if (requireNamespace("piggyback", quietly = TRUE)) {
  cli::cli_inform("Uploading GAM models to GitHub release...")
  tryCatch({
    piggyback::pb_upload(gam_path, repo = "peteowen1/torpmodels", tag = "core-models")
    cli::cli_alert_success("Upload complete!")
  }, error = function(e) {
    cli::cli_warn(c(
      "Upload to GitHub failed: {e$message}",
      "i" = "Models saved locally at {gam_path}",
      "i" = "Upload manually with: piggyback::pb_upload('{gam_path}', repo='peteowen1/torpmodels', tag='core-models')"
    ))
  })
} else {
  cli::cli_warn(c(
    "piggyback package not installed -- skipping upload to GitHub releases.",
    "i" = "Models saved locally at {gam_path}"
  ))
}

tictoc::toc()

# Summary ----
cat("\n=== Final Summary ===\n")
cat("GAM trained on:", nrow(gam_df) / 2, "matches\n")
if (is.finite(HOLDOUT_SEASON)) {
  cat("Holdout test:", n_holdout_matches, "matches (season >=", HOLDOUT_SEASON, ")\n")
  cat("GAM holdout LogLoss:", round(gam_logloss, 4), "\n")
  cat("XGBoost holdout LogLoss:", round(xgb_logloss, 4), "\n")
  cat("Blend holdout LogLoss:", round(blend_logloss, 4), "\n")
  cat("Saved: match_gams.rds + match_xgb_pipeline.rds\n")
} else {
  cat("Production mode: all data used for training\n")
  cat("Saved & uploaded: match_gams.rds\n")
}
