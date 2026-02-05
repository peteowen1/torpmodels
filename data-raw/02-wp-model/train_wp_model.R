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
  devtools::load_all("../../torp")  # Load from sibling directory
}

# Load and prepare training data
# Note: This assumes clean_model_data_epv data exists or load it
cli::cli_inform("Loading chains data...")
chains <- torp::load_chains(TRUE, TRUE)

cli::cli_inform("Cleaning play-by-play data...")
pbp <- torp::clean_pbp(chains)

cli::cli_inform("Preparing EPV model data...")
model_data_epv <- torp::clean_model_data_epv(pbp)

cli::cli_inform("Adding EPV variables and preparing WP data...")
model_data_wp <- model_data_epv %>%
  torp::add_epv_vars() %>%
  torp::clean_model_data_wp()

# XGBoost parameters
nrounds <- 100
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = c("logloss"),
  tree_method = "hist",
  eta = 0.1,
  gamma = 0,
  monotone_constraints = "(0,0,0,1,1,1,0,1,0,0,0,0,0,0,0)",
  max_depth = 6,
  min_child_weight = 1
)

# Create training matrix
full_train <- xgboost::xgb.DMatrix(
  stats::model.matrix(~ . + 0,
    data = model_data_wp %>% torp::select_wp_model_vars()
  ),
  label = model_data_wp$label_wp
)

# Train the model
cli::cli_inform("Training WP model...")
set.seed(1234)
wp_model <- xgboost::xgboost(
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
