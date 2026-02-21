# Train Win Probability (WP) Model
# =================================
# This script trains the Win Probability model using XGBoost.
# Requires torp package and ep_model to be available.

library(devtools)
library(tidyverse)
library(zoo)
library(xgboost)

# Load torp for data loading functions
if (!require(torp)) {
  # Try common relative paths depending on working directory
  torp_paths <- c("../torp", "../../torp", "../../../torp")
  loaded <- FALSE
  for (p in torp_paths) {
    if (file.exists(file.path(p, "DESCRIPTION"))) {
      devtools::load_all(p)
      loaded <- TRUE
      break
    }
  }
  if (!loaded) stop("Cannot find torp package. Install it or run from torpverse workspace.")
}

# Load and prepare training data
# Note: This assumes clean_model_data_epv data exists or load it
cli::cli_inform("Loading chains data...")
chains <- torp::load_chains(TRUE, TRUE)

cli::cli_inform("Cleaning play-by-play data...")
pbp <- torp::clean_pbp(chains)

cli::cli_inform("Preparing EPV model data...")
model_data_epv <- torp::clean_model_data_epv(pbp)

# NOTE: EP predictions here are IN-SAMPLE — the EP model was trained on all data,
# so these predictions are not truly out-of-sample for WP training. This makes WP's
# CV metrics ~1-2% optimistic. For true OOS evaluation, use cross-validated EP preds.
cli::cli_inform("Adding EPV variables and preparing WP data...")
model_data_wp <- model_data_epv %>%
  torp::add_epv_vars() %>%
  torp::clean_model_data_wp()

# XGBoost parameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  tree_method = "hist",
  eta = 0.1,
  gamma = 0,
  monotone_constraints = "(0,0,0,1,1,1,0,1,0,0,0,0,0,0,0)",
  max_depth = 6,
  min_child_weight = 1
)

# Create training matrix
wp_vars <- model_data_wp %>% torp::select_wp_model_vars()
X_train <- stats::model.matrix(~ . + 0, data = wp_vars)
y_train <- model_data_wp$label_wp

full_train <- xgboost::xgb.DMatrix(data = X_train, label = y_train)

# Create match-grouped CV folds to prevent data leakage
cli::cli_inform("Creating match-grouped CV folds...")
match_ids <- unique(model_data_wp$torp_match_id)
set.seed(1234)
match_folds <- sample(rep(1:5, length.out = length(match_ids)))
names(match_folds) <- match_ids
row_folds <- match_folds[model_data_wp$torp_match_id]
folds <- lapply(1:5, function(k) which(row_folds == k))

# Cross-validation to find optimal nrounds
cli::cli_inform("Running 5-fold CV with match-grouped folds...")
set.seed(1234)
cv_result <- xgboost::xgb.cv(
  params = params,
  data = full_train,
  nrounds = 500,
  folds = folds,
  early_stopping_rounds = 20,
  print_every_n = 20,
  verbose = 1
)

# Get optimal nrounds
optimal_nrounds <- cv_result$best_iteration
if (is.null(optimal_nrounds) || length(optimal_nrounds) == 0) {
  optimal_nrounds <- which.min(cv_result$evaluation_log$test_logloss_mean)
}
cli::cli_inform("Optimal nrounds: {optimal_nrounds}")
cli::cli_inform("Best CV logloss: {min(cv_result$evaluation_log$test_logloss_mean)}")

# Train final model
cli::cli_inform("Training final WP model...")
set.seed(1234)
wp_model <- xgboost::xgb.train(
  params = params,
  data = full_train,
  nrounds = optimal_nrounds,
  print_every_n = 10
)

# Save the model
output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

output_path <- file.path(output_dir, "wp_model.rds")
saveRDS(wp_model, output_path)
cli::cli_inform("Saved WP model to {output_path}")

# Upload to GitHub release (if piggyback is available)
if (requireNamespace("piggyback", quietly = TRUE)) {
  cli::cli_inform("Uploading to GitHub release...")
  piggyback::pb_upload(
    output_path,
    repo = "peteowen1/torpmodels",
    tag = "core-models"
  )
  cli::cli_inform("Upload complete!")
}
