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
library(fitzRoy)
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

team_mdl_df <- build_team_mdl_df(weather_path = weather_path)

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
cli::cli_inform("Train: {nrow(gam_df)} rows ({paste(sort(unique(gam_df$season.x)), collapse=', ')})")
if (is.finite(HOLDOUT_SEASON)) {
  test_df <- team_mdl_df |> filter(!is.na(win), season.x >= HOLDOUT_SEASON)
  cli::cli_inform("Test:  {nrow(test_df)} rows (season >= {HOLDOUT_SEASON})")
} else {
  cli::cli_inform("Production mode: training on all data (no holdout)")
}

# XGBoost Pipeline (evaluation mode only) ----
if (is.finite(HOLDOUT_SEASON)) {
cli::cli_h2("Training XGBoost pipeline (5 sequential models)")

xgb_complete <- team_mdl_df |>
  filter(!is.na(win), !is.na(total_xpoints_adj), !is.na(xscore_diff),
         !is.na(shot_conv_diff), !is.na(score_diff))
xgb_df   <- xgb_complete |> filter(season.x < HOLDOUT_SEASON)
xgb_test <- xgb_complete |> filter(season.x >= HOLDOUT_SEASON)
cli::cli_inform("XGBoost train: {nrow(xgb_df)} rows, test: {nrow(xgb_test)} rows")

## Base feature columns ----
xgb_base_cols <- c(
  "team_type_fac",
  "game_year_decimal.x", "game_prop_through_year.x",
  "game_prop_through_month.x", "game_prop_through_day.x",
  "torp_diff", "torp_recv_diff", "torp_disp_diff",
  "torp_spoil_diff", "torp_hitout_diff", "torp.x", "torp.y",
  "log_dist.x", "log_dist.y", "log_dist_diff",
  "familiarity.x", "familiarity.y", "familiarity_diff",
  "days_rest_diff_fac"
)

## Shared hyperparameters ----
xgb_reg_params <- list(
  objective = "reg:squarederror", eval_metric = "rmse",
  tree_method = "hist", eta = 0.05, subsample = 0.7,
  colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15
)
xgb_cls_params <- list(
  objective = "binary:logistic", eval_metric = "logloss",
  tree_method = "hist", eta = 0.05, subsample = 0.7,
  colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15
)

## Season-grouped CV folds ----
train_seasons <- sort(unique(xgb_df$season.x))
cli::cli_inform("Season folds: {paste(train_seasons, collapse=', ')} ({length(train_seasons)} folds)")
xgb_season_folds <- xgb_df$season.x
xgb_folds <- lapply(train_seasons, function(s) which(xgb_season_folds == s))

## Helper: train one XGBoost step with CV ----
train_xgb_step <- function(df, label, weights, feature_cols, params, folds, step_name) {
  fmat <- model.matrix(~ . - 1, data = df[, feature_cols, drop = FALSE])
  dtrain <- xgb.DMatrix(data = fmat, label = label, weight = weights)

  cli::cli_progress_step("CV for {step_name}")
  set.seed(1234)
  cv <- xgb.cv(params = params, data = dtrain, nrounds = 1000, folds = folds,
                early_stopping_rounds = 30, print_every_n = 50, verbose = 1)

  metric_col <- paste0("test_", params$eval_metric, "_mean")
  best_n <- which.min(cv$evaluation_log[[metric_col]])
  best_score <- min(cv$evaluation_log[[metric_col]])
  cli::cli_inform("{step_name}: nrounds={best_n}, CV {params$eval_metric}={round(best_score, 4)}")

  set.seed(1234)
  model <- xgb.train(params = params, data = dtrain, nrounds = best_n, print_every_n = 50)
  preds <- predict(model, dtrain)

  list(model = model, preds = preds, best_n = best_n, cv_score = best_score)
}

## Helper: predict on new data ----
predict_xgb_new <- function(model, df, feature_cols) {
  mat <- model.matrix(~ . - 1, data = df[, feature_cols, drop = FALSE])
  predict(model, xgb.DMatrix(data = mat))
}

## Step 1-5: Sequential XGBoost pipeline ----
step1 <- train_xgb_step(xgb_df, xgb_df$total_xpoints_adj, xgb_df$weightz,
                          xgb_base_cols, xgb_reg_params, xgb_folds, "Step 1: total_xpoints")
xgb_df$xgb_pred_tot_xscore   <- step1$preds
xgb_test$xgb_pred_tot_xscore <- predict_xgb_new(step1$model, xgb_test, xgb_base_cols)

step2_cols <- c(xgb_base_cols, "xgb_pred_tot_xscore")
step2 <- train_xgb_step(xgb_df, xgb_df$xscore_diff, xgb_df$weightz,
                          step2_cols, xgb_reg_params, xgb_folds, "Step 2: xscore_diff")
xgb_df$xgb_pred_xscore_diff   <- step2$preds
xgb_test$xgb_pred_xscore_diff <- predict_xgb_new(step2$model, xgb_test, step2_cols)

step3_cols <- c(xgb_base_cols, "xgb_pred_tot_xscore", "xgb_pred_xscore_diff")
step3 <- train_xgb_step(xgb_df, xgb_df$shot_conv_diff, xgb_df$shot_weightz,
                          step3_cols, xgb_reg_params, xgb_folds, "Step 3: conv_diff")
xgb_df$xgb_pred_conv_diff   <- step3$preds
xgb_test$xgb_pred_conv_diff <- predict_xgb_new(step3$model, xgb_test, step3_cols)

step4_cols <- c(xgb_base_cols, "xgb_pred_xscore_diff", "xgb_pred_conv_diff", "xgb_pred_tot_xscore")
step4 <- train_xgb_step(xgb_df, xgb_df$score_diff, xgb_df$weightz,
                          step4_cols, xgb_reg_params, xgb_folds, "Step 4: score_diff")
xgb_df$xgb_pred_score_diff   <- step4$preds
xgb_test$xgb_pred_score_diff <- predict_xgb_new(step4$model, xgb_test, step4_cols)

step5_cols <- c(xgb_base_cols, "xgb_pred_tot_xscore", "xgb_pred_score_diff")
step5 <- train_xgb_step(xgb_df, as.numeric(xgb_df$win), xgb_df$weightz,
                          step5_cols, xgb_cls_params, xgb_folds, "Step 5: win")
xgb_df$xgb_pred_win   <- step5$preds
xgb_test$xgb_pred_win <- predict_xgb_new(step5$model, xgb_test, step5_cols)

cli::cli_alert_success("XGBoost pipeline complete (5 sequential models)")

## Temporal Holdout Evaluation ----
cli::cli_h1("Temporal Holdout: Train < {HOLDOUT_SEASON}, Test >= {HOLDOUT_SEASON}")

# GAM test predictions (from team_mdl_df, aligned to xgb_test rows)
xgb_test_mask <- !is.na(team_mdl_df$win) & !is.na(team_mdl_df$total_xpoints_adj) &
                 !is.na(team_mdl_df$xscore_diff) & !is.na(team_mdl_df$shot_conv_diff) &
                 !is.na(team_mdl_df$score_diff) & team_mdl_df$season.x >= HOLDOUT_SEASON

gam_test_win   <- team_mdl_df$pred_win[xgb_test_mask]
gam_test_score <- team_mdl_df$pred_score_diff[xgb_test_mask]
test_labels     <- as.numeric(xgb_test$win)
test_score_diff <- xgb_test$score_diff

### GAM holdout metrics ----
gam_logloss  <- MLmetrics::LogLoss(gam_test_win, test_labels)
gam_accuracy <- mean(round(gam_test_win) == test_labels)
gam_brier    <- mean((gam_test_win - test_labels)^2)
gam_mae      <- mean(abs(gam_test_score - test_score_diff))
gam_rmse     <- sqrt(mean((gam_test_score - test_score_diff)^2))

### XGBoost holdout metrics ----
xgb_logloss  <- MLmetrics::LogLoss(xgb_test$xgb_pred_win, test_labels)
xgb_accuracy <- mean(round(xgb_test$xgb_pred_win) == test_labels)
xgb_brier    <- mean((xgb_test$xgb_pred_win - test_labels)^2)
xgb_mae      <- mean(abs(xgb_test$xgb_pred_score_diff - test_score_diff))
xgb_rmse     <- sqrt(mean((xgb_test$xgb_pred_score_diff - test_score_diff)^2))

### GAM vs XGBoost comparison ----
comparison <- data.frame(
  Metric = c("Win LogLoss", "Win Accuracy (%)", "Win Brier", "Score Diff MAE", "Score Diff RMSE"),
  GAM = c(round(gam_logloss, 4), round(gam_accuracy * 100, 1), round(gam_brier, 4), round(gam_mae, 1), round(gam_rmse, 1)),
  XGBoost = c(round(xgb_logloss, 4), round(xgb_accuracy * 100, 1), round(xgb_brier, 4), round(xgb_mae, 1), round(xgb_rmse, 1)),
  stringsAsFactors = FALSE
)

cat("\n=== Holdout Test Set Comparison ===\n")
cat("Train:", nrow(xgb_df), "rows | Test:", nrow(xgb_test), "rows\n\n")
print(comparison, row.names = FALSE)

### XGBoost CV scores ----
cat("\n=== XGBoost CV Scores (on train set, per step) ===\n")
cat(sprintf("  Step 1 (total_xpoints) RMSE:    %.4f  nrounds: %d\n", step1$cv_score, step1$best_n))
cat(sprintf("  Step 2 (xscore_diff)   RMSE:    %.4f  nrounds: %d\n", step2$cv_score, step2$best_n))
cat(sprintf("  Step 3 (conv_diff)     RMSE:    %.4f  nrounds: %d\n", step3$cv_score, step3$best_n))
cat(sprintf("  Step 4 (score_diff)    RMSE:    %.4f  nrounds: %d\n", step4$cv_score, step4$best_n))
cat(sprintf("  Step 5 (win)           LogLoss: %.4f  nrounds: %d\n", step5$cv_score, step5$best_n))

### Feature importance ----
importance <- xgb.importance(model = step5$model)
cat("\nTop 10 XGBoost Features (win step):\n")
print(head(importance, 10))

## Squiggle Model Comparison ----
cli::cli_h2("Squiggle Model Comparison")

squiggle_tips <- tryCatch(
  fitzRoy::fetch_squiggle_data("tips", year = HOLDOUT_SEASON),
  error = function(e) { cli::cli_warn("Failed to fetch Squiggle data: {e$message}"); NULL }
)

if (!is.null(squiggle_tips) && nrow(squiggle_tips) > 0) {

  squiggle_tips <- squiggle_tips |>
    mutate(
      hteam_norm = fitzRoy::replace_teams(hteam),
      hconfidence = as.numeric(hconfidence),
      hmargin = as.numeric(hmargin),
      round = as.integer(round)
    )

  # Home-only test set (metrics are identical to full long-form)
  home_test <- team_mdl_df |>
    filter(xgb_test_mask, team_type == "home") |>
    select(providerId, season = season.x, round = round.roundNumber.x,
           home_team = team_name.x, win, score_diff)

  # Join squiggle tips to home test matches
  sq_joined <- squiggle_tips |>
    inner_join(home_test, by = c("year" = "season", "round" = "round",
                                  "hteam_norm" = "home_team"))

  n_test_matches <- nrow(home_test)
  cli::cli_inform("Squiggle tips matched: {n_distinct(sq_joined$gameid)} of {n_test_matches} test matches")

  # Per-source metrics (only sources that tipped all matched games)
  sq_metrics <- sq_joined |>
    group_by(sourceid, source) |>
    summarise(
      n = n(),
      logloss  = MLmetrics::LogLoss(hconfidence / 100, as.numeric(win)),
      accuracy = mean(round(hconfidence / 100) == as.numeric(win)) * 100,
      brier    = mean((hconfidence / 100 - as.numeric(win))^2),
      mae      = mean(abs(hmargin - score_diff)),
      rmse     = sqrt(mean((hmargin - score_diff)^2)),
      .groups  = "drop"
    ) |>
    arrange(brier)

  # Recompute GAM/XGB on same home-only rows for fair comparison
  gam_home_win   <- team_mdl_df$pred_win[xgb_test_mask & team_mdl_df$team_type == "home"]
  gam_home_score <- team_mdl_df$pred_score_diff[xgb_test_mask & team_mdl_df$team_type == "home"]
  xgb_home       <- xgb_test |> filter(team_type == "home")
  home_labels    <- as.numeric(home_test$win)
  home_margins   <- home_test$score_diff

  # Build unified comparison table
  full_comparison <- data.frame(
    Source = c("TORP GAM", "TORP XGBoost"),
    N = n_test_matches,
    LogLoss  = c(round(MLmetrics::LogLoss(gam_home_win, home_labels), 4),
                 round(MLmetrics::LogLoss(xgb_home$xgb_pred_win, home_labels), 4)),
    Accuracy = c(round(mean(round(gam_home_win) == home_labels) * 100, 1),
                 round(mean(round(xgb_home$xgb_pred_win) == home_labels) * 100, 1)),
    Brier    = c(round(mean((gam_home_win - home_labels)^2), 4),
                 round(mean((xgb_home$xgb_pred_win - home_labels)^2), 4)),
    MAE      = c(round(mean(abs(gam_home_score - home_margins)), 1),
                 round(mean(abs(xgb_home$xgb_pred_score_diff - home_margins)), 1)),
    RMSE     = c(round(sqrt(mean((gam_home_score - home_margins)^2)), 1),
                 round(sqrt(mean((xgb_home$xgb_pred_score_diff - home_margins)^2)), 1)),
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
      MAE      = round(sq_display$mae[i], 1),
      RMSE     = round(sq_display$rmse[i], 1),
      stringsAsFactors = FALSE
    ))
  }

  full_comparison <- full_comparison[order(full_comparison$LogLoss), ]

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
  match_xgb <- list(
    total_xpoints = step1$model,
    xscore_diff   = step2$model,
    conv_diff      = step3$model,
    score_diff     = step4$model,
    win            = step5$model
  )
  xgb_path <- file.path(output_dir, "match_xgb_pipeline.rds")
  saveRDS(match_xgb, xgb_path)
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
cat("GAM trained on:", nrow(gam_df), "rows\n")
if (is.finite(HOLDOUT_SEASON)) {
  cat("Holdout test:", nrow(xgb_test), "rows (season >=", HOLDOUT_SEASON, ")\n")
  cat("GAM holdout LogLoss:", round(gam_logloss, 4), "\n")
  cat("XGBoost holdout LogLoss:", round(xgb_logloss, 4), "\n")
  cat("Saved: match_gams.rds + match_xgb_pipeline.rds\n")
} else {
  cat("Production mode: all data used for training\n")
  cat("Saved & uploaded: match_gams.rds\n")
}
