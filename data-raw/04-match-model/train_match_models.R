# Train Match Prediction Models — Rolling Evaluation Script
# =========================================================
# Rolling week-by-week out-of-sample evaluation: for each round in the
# test seasons, train GAMs + XGBoost on all prior completed matches,
# predict that round, then aggregate metrics and compare to Squiggle.
#
# XGBoost nrounds are pre-optimised via CV on all data (minimal leakage),
# then reused with fixed nrounds per rolling step (no per-week CV).
#
# Uses shared functions from torp/R/match_model.R and match_train.R.

# Parameters ----
TEST_SEASONS <- 2025:2026  # Seasons to evaluate (rolling week-by-week)
UPLOAD_TO_GITHUB <- FALSE

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

# Build Data ----
cli::cli_h1("Building Match Prediction Training Data")
tictoc::tic("total")

team_mdl_df <- build_team_mdl_df()

cli::cli_inform("Seasons: {paste(sort(unique(team_mdl_df$season.x)), collapse = ', ')}")

# Pre-optimise XGBoost nrounds via CV on all data ----
cli::cli_h2("Pre-optimising XGBoost nrounds (CV on all data)")
xgb_cv_result <- .train_match_xgb(team_mdl_df)
xgb_nrounds <- vapply(xgb_cv_result$steps, function(s) s$best_n, integer(1))
cli::cli_inform("XGBoost nrounds: {paste(names(xgb_nrounds), xgb_nrounds, sep='=', collapse=', ')}")

# XGBoost helper: train with fixed nrounds (no CV per week) ----
.train_xgb_fixed <- function(team_mdl_df, train_filter, nrounds_vec) {
  loadNamespace("xgboost")

  train_mask <- train_filter & !is.na(team_mdl_df$win) &
    !is.na(team_mdl_df$total_xpoints_adj) & !is.na(team_mdl_df$xscore_diff) &
    !is.na(team_mdl_df$shot_conv_diff) & !is.na(team_mdl_df$score_diff)

  xgb_df <- team_mdl_df[train_mask, ]
  if (nrow(xgb_df) == 0) return(team_mdl_df)

  osr_dsr_cols <- character(0)
  if (all(c("osr_diff", "dsr_diff") %in% names(team_mdl_df)) &&
      !all(is.na(team_mdl_df$osr_diff))) {
    osr_dsr_cols <- c("osr_diff", "dsr_diff")
  }

  base_cols <- c(
    "team_type_fac",
    "game_year_decimal.x", "game_prop_through_year.x",
    "game_prop_through_month.x", "game_prop_through_day.x",
    "epr_diff", "epr_recv_diff", "epr_disp_diff",
    "epr_spoil_diff", "epr_hitout_diff",
    "torp_diff", "psr_diff", osr_dsr_cols,
    "log_dist_diff", "familiarity_diff", "days_rest_diff_fac"
  )

  reg_params <- list(
    objective = "reg:squarederror", eval_metric = "rmse",
    tree_method = "hist", eta = 0.05, subsample = 0.7,
    colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15
  )
  cls_params <- list(
    objective = "binary:logistic", eval_metric = "logloss",
    tree_method = "hist", eta = 0.05, subsample = 0.7,
    colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15
  )

  train_fixed <- function(df, label, weights, feature_cols, params, nr) {
    fmat <- stats::model.matrix(~ . - 1, data = df[, feature_cols, drop = FALSE])
    dtrain <- xgboost::xgb.DMatrix(data = fmat, label = label, weight = weights)
    set.seed(1234)
    model <- xgboost::xgb.train(
      params = params, data = dtrain, nrounds = nr,
      print_every_n = 0, verbose = 0
    )
    model
  }

  predict_all <- function(model, df, feature_cols) {
    mat <- stats::model.matrix(~ . - 1, data = df[, feature_cols, drop = FALSE])
    predict(model, xgboost::xgb.DMatrix(data = mat))
  }

  # Step 1: total xPoints
  m1 <- train_fixed(xgb_df, xgb_df$total_xpoints_adj, xgb_df$weightz,
                     base_cols, reg_params, nrounds_vec["total_xpoints"])
  xgb_df$xgb_pred_tot_xscore <- predict_all(m1, xgb_df, base_cols)
  team_mdl_df$xgb_pred_tot_xscore <- predict_all(m1, team_mdl_df, base_cols)

  # Step 2: xScore diff
  s2_cols <- c(base_cols, "xgb_pred_tot_xscore")
  m2 <- train_fixed(xgb_df, xgb_df$xscore_diff, xgb_df$weightz,
                     s2_cols, reg_params, nrounds_vec["xscore_diff"])
  xgb_df$xgb_pred_xscore_diff <- predict_all(m2, xgb_df, s2_cols)
  team_mdl_df$xgb_pred_xscore_diff <- predict_all(m2, team_mdl_df, s2_cols)

  # Step 3: conv diff
  s3_cols <- c(base_cols, "xgb_pred_tot_xscore", "xgb_pred_xscore_diff")
  m3 <- train_fixed(xgb_df, xgb_df$shot_conv_diff, xgb_df$shot_weightz,
                     s3_cols, reg_params, nrounds_vec["conv_diff"])
  xgb_df$xgb_pred_conv_diff <- predict_all(m3, xgb_df, s3_cols)
  team_mdl_df$xgb_pred_conv_diff <- predict_all(m3, team_mdl_df, s3_cols)

  # Step 4: score diff
  s4_cols <- c(base_cols, "xgb_pred_xscore_diff", "xgb_pred_conv_diff", "xgb_pred_tot_xscore")
  m4 <- train_fixed(xgb_df, xgb_df$score_diff, xgb_df$weightz,
                     s4_cols, reg_params, nrounds_vec["score_diff"])
  xgb_df$xgb_pred_score_diff <- predict_all(m4, xgb_df, s4_cols)
  team_mdl_df$xgb_pred_score_diff <- predict_all(m4, team_mdl_df, s4_cols)

  # Step 5: win probability
  s5_cols <- c("team_type_fac", "xgb_pred_tot_xscore", "xgb_pred_score_diff",
               "log_dist_diff", "familiarity_diff", "days_rest_diff_fac")
  m5 <- train_fixed(xgb_df, as.numeric(xgb_df$win), xgb_df$weightz,
                     s5_cols, cls_params, nrounds_vec["win"])
  team_mdl_df$xgb_pred_win <- predict_all(m5, team_mdl_df, s5_cols)

  team_mdl_df
}

# Identify test rounds ----
test_rounds <- team_mdl_df |>
  filter(!is.na(win), season.x %in% TEST_SEASONS) |>
  distinct(season.x, round_number.x) |>
  arrange(season.x, round_number.x) |>
  rename(season = season.x, round = round_number.x)

n_test_rounds <- nrow(test_rounds)
n_test_matches <- sum(!is.na(team_mdl_df$win) & team_mdl_df$season.x %in% TEST_SEASONS) / 2
cli::cli_h1("Rolling Evaluation: {n_test_rounds} rounds, ~{n_test_matches} matches ({paste(TEST_SEASONS, collapse='-')})")

# Rolling evaluation loop ----
all_gam_preds <- list()
all_xgb_preds <- list()

for (i in seq_len(n_test_rounds)) {
  s <- test_rounds$season[i]
  r <- test_rounds$round[i]

  # Train on everything strictly before this round
  train_filter <- (team_mdl_df$season.x < s) |
    (team_mdl_df$season.x == s & team_mdl_df$round_number.x < r)

  # Test mask: this specific round, completed matches only
  test_mask <- !is.na(team_mdl_df$win) &
    team_mdl_df$season.x == s & team_mdl_df$round_number.x == r

  n_train <- sum(train_filter & !is.na(team_mdl_df$win)) / 2
  n_test <- sum(test_mask) / 2

  if (n_test == 0) next

  cli::cli_progress_step("{s} R{r}: train={n_train}, test={n_test}")

  # Train GAMs
  gam_result <- suppressMessages(
    .train_match_gams(team_mdl_df, train_filter = train_filter, nthreads = 4L)
  )
  gam_data <- gam_result$data

  # Train XGBoost (fixed nrounds, no CV)
  xgb_data <- suppressMessages(
    .train_xgb_fixed(gam_data, train_filter, xgb_nrounds)
  )

  # Extract GAM predictions for test round
  all_gam_preds[[i]] <- .format_match_preds(gam_data[test_mask, ])

  # Extract XGBoost predictions for test round
  all_xgb_preds[[i]] <- xgb_data[test_mask, ] |>
    mutate(pred_score_diff = xgb_pred_score_diff, pred_win = xgb_pred_win) |>
    .format_match_preds()
}

cli::cli_alert_success("Rolling evaluation complete")

# Combine all out-of-sample predictions ----
gam_preds <- bind_rows(all_gam_preds) |>
  mutate(
    home_win = ifelse(margin > 0, 1, ifelse(margin == 0, 0.5, 0)),
    home_team_chr = torp_replace_teams(as.character(home_team))
  )

xgb_preds <- bind_rows(all_xgb_preds) |>
  mutate(
    home_win = ifelse(margin > 0, 1, ifelse(margin == 0, 0.5, 0)),
    home_team_chr = torp_replace_teams(as.character(home_team))
  )

# 50/50 blend
blend_preds <- gam_preds |>
  mutate(
    pred_win = 0.5 * gam_preds$pred_win + 0.5 * xgb_preds$pred_win,
    pred_margin = 0.5 * gam_preds$pred_margin + 0.5 * xgb_preds$pred_margin
  )

n_oos_matches <- nrow(gam_preds)
cli::cli_inform("Total out-of-sample predictions: {n_oos_matches} matches")

# Compute metrics helper ----
.compute_metrics <- function(preds) {
  list(
    logloss  = MLmetrics::LogLoss(preds$pred_win, preds$home_win),
    accuracy = mean(round(preds$pred_win) == preds$home_win) * 100,
    brier    = mean((preds$pred_win - preds$home_win)^2),
    mae      = mean(abs(preds$pred_margin - preds$margin)),
    rmse     = sqrt(mean((preds$pred_margin - preds$margin)^2))
  )
}

# Overall metrics ----
cli::cli_h1("Rolling Out-of-Sample Results")

gam_m <- .compute_metrics(gam_preds)
xgb_m <- .compute_metrics(xgb_preds)
blend_m <- .compute_metrics(blend_preds)

comparison <- data.frame(
  Model = c("GAM", "XGBoost", "Blend"),
  N = n_oos_matches,
  LogLoss  = round(c(gam_m$logloss, xgb_m$logloss, blend_m$logloss), 4),
  Accuracy = round(c(gam_m$accuracy, xgb_m$accuracy, blend_m$accuracy), 1),
  Brier    = round(c(gam_m$brier, xgb_m$brier, blend_m$brier), 4),
  MAE      = round(c(gam_m$mae, xgb_m$mae, blend_m$mae), 1),
  RMSE     = round(c(gam_m$rmse, xgb_m$rmse, blend_m$rmse), 1),
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
  .season_metrics(blend_preds, "Blend")
) |>
  arrange(season, Model)

cat("\n=== Per-Season Breakdown ===\n")
print(as.data.frame(season_breakdown), row.names = FALSE)

# Calibration bins ----
cal_breaks <- c(0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1)
cal_labels <- c("<30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-85%", ">85%")

cal_table <- gam_preds |>
  mutate(
    bin = cut(pred_win, breaks = cal_breaks, labels = cal_labels, include.lowest = TRUE),
    xgb_prob = xgb_preds$pred_win,
    blend_prob = blend_preds$pred_win,
    xgb_margin = xgb_preds$pred_margin,
    blend_margin = blend_preds$pred_margin
  ) |>
  group_by(bin) |>
  summarise(
    N = n(),
    Actual = round(mean(home_win) * 100, 1),
    GAM = round(mean(pred_win) * 100, 1),
    XGBoost = round(mean(xgb_prob) * 100, 1),
    Blend = round(mean(blend_prob) * 100, 1),
    GAM_RMSE = round(sqrt(mean((pred_margin - margin)^2)), 1),
    XGB_RMSE = round(sqrt(mean((xgb_margin - margin)^2)), 1),
    Blend_RMSE = round(sqrt(mean((blend_margin - margin)^2)), 1),
    .groups = "drop"
  )

cat("\n=== Calibration Bins (mean pred % vs actual win %) ===\n")
print(as.data.frame(cal_table), row.names = FALSE)

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
  xgb_matched <- xgb_preds |>
    semi_join(sq_joined |> distinct(year, round, hteam_norm),
              by = c("season" = "year", "round" = "round", "home_team_chr" = "hteam_norm"))
  blend_matched <- blend_preds |>
    semi_join(sq_joined |> distinct(year, round, hteam_norm),
              by = c("season" = "year", "round" = "round", "home_team_chr" = "hteam_norm"))

  # TORP metrics on matched games only
  gam_matched_m   <- .compute_metrics(gam_preds |>
    semi_join(sq_joined |> distinct(year, round, hteam_norm),
              by = c("season" = "year", "round" = "round", "home_team_chr" = "hteam_norm")))
  xgb_matched_m   <- .compute_metrics(xgb_matched)
  blend_matched_m <- .compute_metrics(blend_matched)

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
    Source   = c("TORP GAM", "TORP XGBoost", "TORP Blend"),
    N        = n_matched,
    LogLoss  = round(c(gam_matched_m$logloss, xgb_matched_m$logloss, blend_matched_m$logloss), 4),
    Accuracy = round(c(gam_matched_m$accuracy, xgb_matched_m$accuracy, blend_matched_m$accuracy), 1),
    Brier    = round(c(gam_matched_m$brier, xgb_matched_m$brier, blend_matched_m$brier), 4),
    MAE      = round(c(gam_matched_m$mae, xgb_matched_m$mae, blend_matched_m$mae), 1),
    RMSE     = round(c(gam_matched_m$rmse, xgb_matched_m$rmse, blend_matched_m$rmse), 1),
    stringsAsFactors = FALSE
  )

  # Add squiggle sources (require 80%+ coverage)
  min_tips <- floor(n_matched * 0.8)
  sq_display <- sq_metrics |> filter(n >= min_tips)

  for (i in seq_len(nrow(sq_display))) {
    full_comparison <- rbind(full_comparison, data.frame(
      Source   = sq_display$source[i],
      N        = sq_display$n[i],
      LogLoss  = round(sq_display$logloss[i], 4),
      Accuracy = round(sq_display$accuracy[i], 2),
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

# Save production models (trained on all data) ----
cli::cli_h2("Training production models (all data)")
gam_result_prod <- .train_match_gams(team_mdl_df)
match_gams <- gam_result_prod$models

output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

gam_path <- file.path(output_dir, "match_gams.rds")
saveRDS(match_gams, gam_path)
cli::cli_inform("Saved GAM pipeline to {gam_path}")

if (UPLOAD_TO_GITHUB && requireNamespace("piggyback", quietly = TRUE)) {
  tryCatch({
    piggyback::pb_upload(gam_path, repo = "peteowen1/torpmodels", tag = "core-models")
    cli::cli_alert_success("Upload complete!")
  }, error = function(e) {
    cli::cli_warn("Upload failed: {e$message}. Models saved locally at {gam_path}")
  })
}

tictoc::toc()

# Summary ----
cat("\n=== Final Summary ===\n")
cat("Rolling OOS evaluation:", n_oos_matches, "matches across", n_test_rounds, "rounds\n")
cat("GAM   Brier:", round(gam_m$brier, 4), "| MAE:", round(gam_m$mae, 1), "| RMSE:", round(gam_m$rmse, 1), "\n")
cat("XGB   Brier:", round(xgb_m$brier, 4), "| MAE:", round(xgb_m$mae, 1), "| RMSE:", round(xgb_m$rmse, 1), "\n")
cat("Blend Brier:", round(blend_m$brier, 4), "| MAE:", round(blend_m$mae, 1), "| RMSE:", round(blend_m$rmse, 1), "\n")
cat("Production GAMs saved to:", gam_path, "\n")
