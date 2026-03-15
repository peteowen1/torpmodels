# Wrapper: train WP model using local dev torp
# =============================================

library(tidyverse)
library(zoo)
library(xgboost)

# Load dev torp first
devtools::load_all("C:/Users/peteo/OneDrive/Documents/torpverse/torp")

setwd("C:/Users/peteo/OneDrive/Documents/torpverse/torpmodels")

# --- Training steps (inlined from train_wp_model.R) ---

cli::cli_inform("Loading chains data...")
chains <- load_chains(2021:2025)

cli::cli_inform("Cleaning play-by-play data...")
pbp <- clean_pbp(chains)

cli::cli_inform("Preparing EPV model data...")
model_data_epv <- clean_model_data_epv(pbp)

cli::cli_inform("Adding EPV variables and preparing WP data...")
model_data_wp <- model_data_epv |>
  add_epv_vars() |>
  clean_model_data_wp()

params <- list(
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
cat("WP feature names:\n")
print(names(wp_vars))
cat("WP feature count:", ncol(wp_vars), "\n")

X_train <- stats::model.matrix(~ . + 0, data = wp_vars)
y_train <- model_data_wp$label_wp
full_train <- xgboost::xgb.DMatrix(data = X_train, label = y_train)

cli::cli_inform("Creating match-grouped CV folds...")
match_ids <- unique(model_data_wp$torp_match_id)
set.seed(1234)
match_folds <- sample(rep(1:5, length.out = length(match_ids)))
names(match_folds) <- match_ids
row_folds <- match_folds[model_data_wp$torp_match_id]
folds <- lapply(1:5, function(k) which(row_folds == k))

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

optimal_nrounds <- which.min(cv_result$evaluation_log$test_logloss_mean)
cli::cli_inform("Optimal nrounds: {optimal_nrounds}")
cli::cli_inform("Best CV logloss: {min(cv_result$evaluation_log$test_logloss_mean)}")

cli::cli_inform("Training final WP model...")
set.seed(1234)
wp_model <- xgboost::xgb.train(
  params = params,
  data = full_train,
  nrounds = optimal_nrounds,
  print_every_n = 10
)

output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
output_path <- file.path(output_dir, "wp_model.rds")
saveRDS(wp_model, output_path)
cli::cli_inform("Saved WP model to {output_path}")

if (requireNamespace("piggyback", quietly = TRUE)) {
  cli::cli_inform("Uploading to GitHub release...")
  piggyback::pb_upload(output_path, repo = "peteowen1/torpmodels", tag = "core-models")
  cli::cli_inform("Upload complete!")
}
