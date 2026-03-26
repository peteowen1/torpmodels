# Train Win Probability (WP) Model with Cross-Validated EP Predictions ----
# ========================================================================
# Fixes the in-sample EP overfitting issue: instead of using the full EP model
# to generate training features for WP, we use 5-fold cross-validated OOS EP
# predictions. This gives WP more honest EP inputs during training, improving
# calibration at inference time.
#
# Production inference is unchanged -- WP still uses the full EP model via
# add_epv_vars(). Only the training pipeline changes.
#
# See: train_wp_model.R (original, kept as reference)

library(devtools)
library(tidyverse)
library(zoo)
library(xgboost)

# Load torp (prefer devtools::load_all for access to internal functions) ----
torp_paths <- c("../torp", "../../torp", "../../../torp")
loaded <- FALSE
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p)
    loaded <- TRUE
    break
  }
}
if (!loaded) {
  if (!require(torp)) stop("Cannot find torp package. Install it or run from torpverse workspace.")
}

# Load and prepare data (same as EP training) ----
cli::cli_inform("Loading chains data...")
chains <- torp::load_chains(TRUE, TRUE)

cli::cli_inform("Cleaning play-by-play data...")
pbp <- torp::clean_pbp(chains)

cli::cli_inform("Preparing EPV model data...")
model_data_epv <- clean_model_data_epv(pbp)

# Prepare EP training inputs ----
epv_vars <- model_data_epv |> select_epv_model_vars()
X_all <- stats::model.matrix(~ . + 0, data = epv_vars)
y_all <- model_data_epv$label_ep

full_dmat <- xgboost::xgb.DMatrix(data = X_all, label = y_all)

ep_params <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  tree_method = "hist",
  num_class = 5,
  eta = 0.1,
  gamma = 0,
  subsample = 0.85,
  colsample_bytree = 0.85,
  max_depth = 6,
  min_child_weight = 25
)

# Create match-grouped CV folds (same seed/method as train_ep_model.R) ----
cli::cli_inform("Creating match-grouped CV folds...")
match_ids <- unique(model_data_epv$torp_match_id)
set.seed(1234)
match_folds <- sample(rep(1:5, length.out = length(match_ids)))
names(match_folds) <- match_ids
row_folds <- match_folds[model_data_epv$torp_match_id]
folds_idx <- lapply(1:5, function(k) which(row_folds == k))

# Find optimal nrounds via xgb.cv (reuses same folds) ----
cli::cli_inform("Running EP xgb.cv to find optimal nrounds...")
set.seed(1234)
cv_result <- xgboost::xgb.cv(
  params = ep_params,
  data = full_dmat,
  nrounds = 500,
  folds = folds_idx,
  early_stopping_rounds = 20,
  print_every_n = 20,
  verbose = 1
)

optimal_nrounds <- which.min(cv_result$evaluation_log$test_mlogloss_mean)
cli::cli_inform("EP optimal nrounds: {optimal_nrounds}")
cli::cli_inform("EP best CV mlogloss: {round(min(cv_result$evaluation_log$test_mlogloss_mean), 6)}")

# CV loop: generate out-of-sample EP predictions ----
cli::cli_inform("Generating OOS EP predictions via 5-fold CV...")
oos_ep_preds <- matrix(NA_real_, nrow = nrow(model_data_epv), ncol = 5)
colnames(oos_ep_preds) <- c("opp_goal", "opp_behind", "behind", "goal", "no_score")

for (k in 1:5) {
  cli::cli_inform("Fold {k}/5: training EP on {length(which(row_folds != k))} rows, predicting {length(folds_idx[[k]])} rows...")

  test_idx <- folds_idx[[k]]
  train_idx <- which(row_folds != k)

  dtrain <- xgboost::xgb.DMatrix(data = X_all[train_idx, ], label = y_all[train_idx])
  dtest <- xgboost::xgb.DMatrix(data = X_all[test_idx, ])

  set.seed(1234)
  fold_model <- xgboost::xgb.train(
    params = ep_params,
    data = dtrain,
    nrounds = optimal_nrounds,
    verbose = 0
  )

  preds_raw <- predict(fold_model, dtest)

  # Handle XGBoost 3.x matrix returns vs older flat vector
  if (is.matrix(preds_raw)) {
    oos_ep_preds[test_idx, ] <- preds_raw
  } else {
    oos_ep_preds[test_idx, ] <- matrix(preds_raw, ncol = 5, byrow = TRUE)
  }

  rm(fold_model, dtrain, dtest)
}

# Verify no NAs remain
stopifnot(!anyNA(oos_ep_preds))
cli::cli_inform("OOS EP predictions complete. Mean max-class prob: {round(mean(apply(oos_ep_preds, 1, max)), 4)}")

# Inject OOS EP predictions into model_data_epv ----
# Column names must match what clean_model_data_wp() expects:
# opp_goal, opp_behind, no_score, behind, goal, exp_pts
# Formula synced with add_epv_vars() line 32-34 of add_variables.R
oos_df <- as.data.frame(oos_ep_preds)
model_data_epv$opp_goal <- oos_df$opp_goal
model_data_epv$opp_behind <- oos_df$opp_behind
model_data_epv$behind <- oos_df$behind
model_data_epv$goal <- oos_df$goal
model_data_epv$no_score <- oos_df$no_score
model_data_epv$exp_pts <- round(-6 * oos_df$opp_goal - oos_df$opp_behind + oos_df$behind + 6 * oos_df$goal, 5)

# Build WP features using clean_model_data_wp() ----
cli::cli_inform("Preparing WP training data with OOS EP features...")
model_data_wp <- model_data_epv |>
  clean_model_data_wp()

cli::cli_inform("WP training data: {nrow(model_data_wp)} rows")

# Train WP model (identical to train_wp_model.R from here) ----
wp_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  tree_method = "hist",
  eta = 0.025,
  gamma = 0,
  monotone_constraints = "(0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0)",
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

wp_vars <- model_data_wp |> select_wp_model_vars()
X_train <- stats::model.matrix(~ . + 0, data = wp_vars)
y_train <- model_data_wp$label_wp

full_train <- xgboost::xgb.DMatrix(data = X_train, label = y_train)

# WP CV folds (match-grouped, same approach)
wp_match_ids <- unique(model_data_wp$torp_match_id)
set.seed(1234)
wp_match_folds <- sample(rep(1:5, length.out = length(wp_match_ids)))
names(wp_match_folds) <- wp_match_ids
wp_row_folds <- wp_match_folds[model_data_wp$torp_match_id]
wp_folds <- lapply(1:5, function(k) which(wp_row_folds == k))

cli::cli_inform("Running WP 5-fold CV...")
set.seed(1234)
wp_cv_result <- xgboost::xgb.cv(
  params = wp_params,
  data = full_train,
  nrounds = 500,
  folds = wp_folds,
  early_stopping_rounds = 20,
  print_every_n = 20,
  verbose = 1
)

wp_optimal_nrounds <- which.min(wp_cv_result$evaluation_log$test_logloss_mean)
cli::cli_inform("WP optimal nrounds: {wp_optimal_nrounds}")
cli::cli_inform("WP best CV logloss (with CV EP): {round(min(wp_cv_result$evaluation_log$test_logloss_mean), 6)}")

# Train final WP model
cli::cli_inform("Training final WP model...")
set.seed(1234)
wp_model <- xgboost::xgb.train(
  params = wp_params,
  data = full_train,
  nrounds = wp_optimal_nrounds,
  print_every_n = 10
)

# Save the model ----
output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

output_path <- file.path(output_dir, "wp_model.rds")
saveRDS(wp_model, output_path)
cli::cli_inform("Saved WP model to {output_path}")

# Upload to GitHub release ----
if (requireNamespace("piggyback", quietly = TRUE)) {
  cli::cli_inform("Uploading to GitHub release...")
  piggyback::pb_upload(
    output_path,
    repo = "peteowen1/torpmodels",
    tag = "core-models"
  )
  cli::cli_inform("Upload complete!")
}
