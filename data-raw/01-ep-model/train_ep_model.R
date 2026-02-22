# Train Expected Points (EP) Model
# =================================
# This script trains the Expected Points model using XGBoost.
# Requires torp package to be available for data loading and preprocessing.

library(devtools)
library(tidyverse)
library(zoo)
library(janitor)
library(lubridate)
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

# Load training data
cli::cli_inform("Loading chains data...")
chains <- torp::load_chains(TRUE, TRUE)

cli::cli_inform("Cleaning play-by-play data...")
pbp <- torp::clean_pbp(chains)

cli::cli_inform("Preparing EPV model data...")
model_data_epv <- torp::clean_model_data_epv(pbp)

# XGBoost parameters
params <- list(
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

# Create training matrix
epv_vars <- model_data_epv %>% torp::select_epv_model_vars()
X_train <- stats::model.matrix(~ . + 0, data = epv_vars)
y_train <- model_data_epv$label_ep

full_train <- xgboost::xgb.DMatrix(data = X_train, label = y_train)

# Create match-grouped CV folds to prevent data leakage
cli::cli_inform("Creating match-grouped CV folds...")
match_ids <- unique(model_data_epv$torp_match_id)
set.seed(1234)
match_folds <- sample(rep(1:5, length.out = length(match_ids)))
names(match_folds) <- match_ids
row_folds <- match_folds[model_data_epv$torp_match_id]
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
  optimal_nrounds <- which.min(cv_result$evaluation_log$test_mlogloss_mean)
}
cli::cli_inform("Optimal nrounds: {optimal_nrounds}")
cli::cli_inform("Best CV mlogloss: {min(cv_result$evaluation_log$test_mlogloss_mean)}")

# Train final model
cli::cli_inform("Training final EP model...")
set.seed(1234)
ep_model <- xgboost::xgb.train(
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

output_path <- file.path(output_dir, "ep_model.rds")
saveRDS(ep_model, output_path)
cli::cli_inform("Saved EP model to {output_path}")

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
