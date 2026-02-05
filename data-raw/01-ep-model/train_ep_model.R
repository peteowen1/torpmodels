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
  devtools::load_all("../../torp")  # Load from sibling directory
}

# Load training data
cli::cli_inform("Loading chains data...")
chains <- torp::load_chains(TRUE, TRUE)

cli::cli_inform("Cleaning play-by-play data...")
pbp <- torp::clean_pbp(chains)

cli::cli_inform("Preparing EPV model data...")
model_data_epv <- torp::clean_model_data_epv(pbp)

# XGBoost parameters
nrounds <- 87
params <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  eval_metric = c("mlogloss"),
  tree_method = "hist",
  num_class = 5,
  eta = 0.15,
  gamma = 0,
  subsample = 0.85,
  colsample_bytree = 0.85,
  max_depth = 6,
  min_child_weight = 25
)

# Create training matrix
full_train <- xgboost::xgb.DMatrix(
  stats::model.matrix(~ . + 0,
    data = model_data_epv %>% torp::select_epv_model_vars()
  ),
  label = model_data_epv$label_ep
)

# Train the model
cli::cli_inform("Training EP model...")
set.seed(1234)
ep_model <- xgboost::xgboost(
  params = params,
  data = full_train,
  nrounds = nrounds,
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
