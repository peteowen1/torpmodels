# Train XGBoost Win Prediction Model
# ===================================
# This script trains the XGBoost model for predicting match winners.
# Requires team_mdl_df from torp's build_match_predictions.R workflow.

library(tidyverse)
library(caret)
library(xgboost)
library(MLmetrics)

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

# Load team_mdl_df: check memory first, then tempdir, then abort with instructions
if (!exists("team_mdl_df")) {
  team_mdl_path <- file.path(tempdir(), "team_mdl_df.rds")
  if (file.exists(team_mdl_path)) {
    cli::cli_inform("Loading team_mdl_df from {team_mdl_path}")
    team_mdl_df <- readRDS(team_mdl_path)
  } else {
    stop(
      "`team_mdl_df` not found in memory or at ", team_mdl_path, ".\n",
      "Run build_match_predictions.R first (it saves team_mdl_df to tempdir)."
    )
  }
}

# Prepare clean modeling dataset
model_df <- team_mdl_df %>%
  dplyr::filter(!is.na(win)) %>%
  dplyr::mutate(win = as.numeric(win)) %>%
  torp::select_afl_model_vars() %>%
  tidyr::drop_na()

# Create train/test split
set.seed(1234)
train_idx <- which(model_df$season.x < torp::get_afl_season())
train_df <- model_df[train_idx, ]
test_df <- model_df[-train_idx, ]

# Create model matrices
feature_cols <- setdiff(names(train_df), c("win", "providerId", "team_type"))

train_matrix <- model.matrix(~ . - 1, data = train_df[, feature_cols])
train_label <- train_df$win

test_matrix <- model.matrix(~ . - 1, data = test_df[, feature_cols])
test_label <- test_df$win

cat("Train matrix rows:", nrow(train_matrix), "Train label length:", length(train_label), "\n")
cat("Test matrix rows:", nrow(test_matrix), "Test label length:", length(test_label), "\n")

# Create XGBoost matrices
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

# Hyperparameter tuning grid
gr <- expand.grid(
  eta = c(0.05, 0.1, 0.2),
  max_depth = c(4, 6, 8),
  subsample = c(0.8, 1),
  colsample_bytree = c(0.8, 1),
  min_child_weight = c(1, 5),
  gamma = c(0, 1)
)

# Create match-grouped CV folds to prevent data leakage
# Each match has 2 rows (home/away), ensure both are in same fold
cli::cli_inform("Creating match-grouped CV folds...")
match_ids <- unique(train_df$providerId)
set.seed(1234)
match_folds <- sample(rep(1:5, length.out = length(match_ids)))
names(match_folds) <- match_ids
row_folds <- match_folds[train_df$providerId]
folds <- lapply(1:5, function(k) which(row_folds == k))

# Cross-validation for hyperparameter tuning with match-grouped folds
cli::cli_inform("Running hyperparameter tuning with match-grouped folds...")
cv_results <- purrr::pmap(gr, function(eta, max_depth, subsample,
                                       colsample_bytree, min_child_weight, gamma) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = eta,
    max_depth = max_depth,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma
  )

  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 1000,
    folds = folds,  # use match-grouped folds
    verbose = 0,
    early_stopping_rounds = 50
  )

  best_iter <- cv$best_iteration
  if (is.null(best_iter) || length(best_iter) == 0) {
    best_iter <- which.min(cv$evaluation_log$test_logloss_mean)
  }

  tibble(
    eta = eta,
    max_depth = max_depth,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    best_logloss = min(cv$evaluation_log$test_logloss_mean),
    best_nrounds = best_iter
  )
}, .progress = TRUE) %>%
  bind_rows()

# Select best parameters
best <- cv_results %>%
  dplyr::arrange(best_logloss) %>%
  dplyr::slice(1)

cat("Best CV LogLoss:", round(best$best_logloss, 4), "\n")

best_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = best$eta,
  max_depth = best$max_depth,
  subsample = best$subsample,
  colsample_bytree = best$colsample_bytree,
  min_child_weight = best$min_child_weight,
  gamma = best$gamma
)

# Train final model
cli::cli_inform("Training final model...")
xgb_win_model <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best$best_nrounds,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 0
)

# Evaluate on test data
test_pred <- predict(xgb_win_model, dtest)
logloss <- MLmetrics::LogLoss(test_pred, test_label)
print(glue::glue("Test LogLoss: {round(logloss, 4)}"))

# Feature importance
importance <- xgb.importance(model = xgb_win_model)
print("Top 10 Most Important Features:")
print(head(importance, 10))

# Save the model
output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

output_path <- file.path(output_dir, "xgb_win_model.rds")
saveRDS(xgb_win_model, output_path)
cli::cli_inform("Saved XGBoost win model to {output_path}")

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

# Print summary
cat("\nModel Summary:\n")
cat("Training samples:", nrow(train_df), "\n")
cat("Test samples:", nrow(test_df), "\n")
cat("Features:", ncol(train_matrix), "\n")
cat("Test LogLoss:", round(logloss, 4), "\n")
