# Wrapper: train EP model using local dev torp
# =============================================
# Loads dev torp FIRST so all torp functions use the dev version.

library(tidyverse)
library(zoo)
library(janitor)
library(lubridate)
library(xgboost)

# Load dev torp (must be before any torp:: calls)
devtools::load_all("C:/Users/peteo/OneDrive/Documents/torpverse/torp")

setwd("C:/Users/peteo/OneDrive/Documents/torpverse/torpmodels")

# --- Training steps (inlined from train_ep_model.R) ---

cli::cli_inform("Loading chains data...")
chains <- load_chains(2021:2025)

cli::cli_inform("Cleaning play-by-play data...")
pbp <- clean_pbp(chains)

cli::cli_inform("Preparing EPV model data...")
model_data_epv <- clean_model_data_epv(pbp)

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

epv_vars <- model_data_epv |> select_epv_model_vars()
cat("EP feature names:\n")
print(names(epv_vars))
cat("EP feature count:", ncol(epv_vars), "\n")

X_train <- stats::model.matrix(~ . + 0, data = epv_vars)
y_train <- model_data_epv$label_ep
full_train <- xgboost::xgb.DMatrix(data = X_train, label = y_train)

cli::cli_inform("Creating match-grouped CV folds...")
match_ids <- unique(model_data_epv$torp_match_id)
set.seed(1234)
match_folds <- sample(rep(1:5, length.out = length(match_ids)))
names(match_folds) <- match_ids
row_folds <- match_folds[model_data_epv$torp_match_id]
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

optimal_nrounds <- which.min(cv_result$evaluation_log$test_mlogloss_mean)
cli::cli_inform("Optimal nrounds: {optimal_nrounds}")
cli::cli_inform("Best CV mlogloss: {min(cv_result$evaluation_log$test_mlogloss_mean)}")

cli::cli_inform("Training final EP model...")
set.seed(1234)
ep_model <- xgboost::xgb.train(
  params = params,
  data = full_train,
  nrounds = optimal_nrounds,
  print_every_n = 10
)

output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
output_path <- file.path(output_dir, "ep_model.rds")
saveRDS(ep_model, output_path)
cli::cli_inform("Saved EP model to {output_path}")

if (requireNamespace("piggyback", quietly = TRUE)) {
  cli::cli_inform("Uploading to GitHub release...")
  piggyback::pb_upload(output_path, repo = "peteowen1/torpmodels", tag = "core-models")
  cli::cli_inform("Upload complete!")
}
