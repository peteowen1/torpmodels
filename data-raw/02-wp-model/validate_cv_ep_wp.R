# Validate CV EP vs In-Sample EP for WP Model ----
# =================================================
# Compares WP models trained with in-sample vs cross-validated EP predictions.
# Run after training both models to quantify the improvement.

library(devtools)
library(tidyverse)
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

# Load and prepare shared data ----
cli::cli_inform("Loading chains data...")
chains <- torp::load_chains(TRUE, TRUE)

cli::cli_inform("Cleaning play-by-play data...")
pbp <- torp::clean_pbp(chains)

cli::cli_inform("Preparing EPV model data...")
model_data_epv <- clean_model_data_epv(pbp)

# Shared EP setup ----
epv_vars <- model_data_epv |> select_epv_model_vars()
X_all <- stats::model.matrix(~ . + 0, data = epv_vars)
y_all <- model_data_epv$label_ep
full_dmat <- xgboost::xgb.DMatrix(data = X_all, label = y_all)

ep_params <- list(
  booster = "gbtree", objective = "multi:softprob", eval_metric = "mlogloss",
  tree_method = "hist", num_class = 5, eta = 0.1, gamma = 0,
  subsample = 0.85, colsample_bytree = 0.85, max_depth = 6, min_child_weight = 25
)

# Match-grouped folds ----
match_ids <- unique(model_data_epv$torp_match_id)
set.seed(1234)
match_folds <- sample(rep(1:5, length.out = length(match_ids)))
names(match_folds) <- match_ids
row_folds <- match_folds[model_data_epv$torp_match_id]
folds_idx <- lapply(1:5, function(k) which(row_folds == k))

# Optimal EP nrounds ----
set.seed(1234)
cv_result <- xgboost::xgb.cv(
  params = ep_params, data = full_dmat, nrounds = 500,
  folds = folds_idx, early_stopping_rounds = 20, print_every_n = 50, verbose = 1
)
optimal_nrounds <- which.min(cv_result$evaluation_log$test_mlogloss_mean)

# Generate in-sample EP predictions (full model) ----
cli::cli_inform("Training full EP model for in-sample predictions...")
set.seed(1234)
full_ep_model <- xgboost::xgb.train(
  params = ep_params, data = full_dmat, nrounds = optimal_nrounds, verbose = 0
)
insample_preds_raw <- predict(full_ep_model, full_dmat)
if (is.matrix(insample_preds_raw)) {
  insample_ep <- insample_preds_raw
} else {
  insample_ep <- matrix(insample_preds_raw, ncol = 5, byrow = TRUE)
}
colnames(insample_ep) <- c("opp_goal", "opp_behind", "behind", "goal", "no_score")

# Generate OOS EP predictions (5-fold CV) ----
cli::cli_inform("Generating OOS EP predictions via 5-fold CV...")
oos_ep <- matrix(NA_real_, nrow = nrow(model_data_epv), ncol = 5)
colnames(oos_ep) <- c("opp_goal", "opp_behind", "behind", "goal", "no_score")

for (k in 1:5) {
  cli::cli_inform("EP fold {k}/5...")
  test_idx <- folds_idx[[k]]
  train_idx <- which(row_folds != k)
  dtrain <- xgboost::xgb.DMatrix(data = X_all[train_idx, ], label = y_all[train_idx])
  dtest <- xgboost::xgb.DMatrix(data = X_all[test_idx, ])
  set.seed(1234)
  fold_model <- xgboost::xgb.train(params = ep_params, data = dtrain, nrounds = optimal_nrounds, verbose = 0)
  preds_raw <- predict(fold_model, dtest)
  if (is.matrix(preds_raw)) {
    oos_ep[test_idx, ] <- preds_raw
  } else {
    oos_ep[test_idx, ] <- matrix(preds_raw, ncol = 5, byrow = TRUE)
  }
  rm(fold_model, dtrain, dtest)
}

# Compare EP prediction sharpness ----
insample_sharpness <- mean(apply(insample_ep, 1, max))
oos_sharpness <- mean(apply(oos_ep, 1, max))

cli::cli_h1("EP Prediction Comparison")
cli::cli_inform("In-sample mean max-class prob: {round(insample_sharpness, 4)}")
cli::cli_inform("OOS mean max-class prob:       {round(oos_sharpness, 4)}")
cli::cli_inform("Difference:                    {round(insample_sharpness - oos_sharpness, 4)} (in-sample is sharper)")

# Helper: build WP data from EP predictions ----
build_wp_data <- function(ep_preds, base_data) {
  df <- base_data
  ep_df <- as.data.frame(ep_preds)
  colnames(ep_df) <- c("opp_goal", "opp_behind", "behind", "goal", "no_score")
  df$opp_goal <- ep_df$opp_goal
  df$opp_behind <- ep_df$opp_behind
  df$behind <- ep_df$behind
  df$goal <- ep_df$goal
  df$no_score <- ep_df$no_score
  df$exp_pts <- round(-6 * ep_df$opp_goal - ep_df$opp_behind + ep_df$behind + 6 * ep_df$goal, 5)
  df |> clean_model_data_wp()
}

# Helper: run WP CV and return logloss ----
run_wp_cv <- function(wp_data, label = "WP") {
  wp_params <- list(
    booster = "gbtree", objective = "binary:logistic", eval_metric = "logloss",
    tree_method = "hist", eta = 0.025, gamma = 0,
    monotone_constraints = "(0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0)",
    max_depth = 6, min_child_weight = 1, subsample = 0.8, colsample_bytree = 0.8
  )

  wp_vars <- wp_data |> select_wp_model_vars()
  X <- stats::model.matrix(~ . + 0, data = wp_vars)
  y <- wp_data$label_wp
  dmat <- xgboost::xgb.DMatrix(data = X, label = y)

  wp_match_ids <- unique(wp_data$torp_match_id)
  set.seed(1234)
  wp_match_folds <- sample(rep(1:5, length.out = length(wp_match_ids)))
  names(wp_match_folds) <- wp_match_ids
  wp_row_folds <- wp_match_folds[wp_data$torp_match_id]
  wp_folds <- lapply(1:5, function(k) which(wp_row_folds == k))

  set.seed(1234)
  cv <- xgboost::xgb.cv(
    params = wp_params, data = dmat, nrounds = 500,
    folds = wp_folds, early_stopping_rounds = 20, print_every_n = 50, verbose = 1
  )

  best_round <- which.min(cv$evaluation_log$test_logloss_mean)
  best_logloss <- min(cv$evaluation_log$test_logloss_mean)
  list(best_round = best_round, best_logloss = best_logloss, cv = cv)
}

# Run WP CV with in-sample EP ----
cli::cli_h1("WP Model with In-Sample EP")
wp_data_insample <- build_wp_data(insample_ep, model_data_epv)
result_insample <- run_wp_cv(wp_data_insample, "In-sample EP")
cli::cli_inform("Best round: {result_insample$best_round}")
cli::cli_inform("Best CV logloss: {round(result_insample$best_logloss, 6)}")

# Run WP CV with OOS EP ----
cli::cli_h1("WP Model with CV (OOS) EP")
wp_data_oos <- build_wp_data(oos_ep, model_data_epv)
result_oos <- run_wp_cv(wp_data_oos, "OOS EP")
cli::cli_inform("Best round: {result_oos$best_round}")
cli::cli_inform("Best CV logloss: {round(result_oos$best_logloss, 6)}")

# Summary comparison ----
cli::cli_h1("Summary")
cli::cli_inform("")
cat(sprintf("%-30s %-15s %-15s\n", "", "In-Sample EP", "CV (OOS) EP"))
cat(sprintf("%-30s %-15s %-15s\n", "---", "---", "---"))
cat(sprintf("%-30s %-15d %-15d\n", "WP optimal nrounds", result_insample$best_round, result_oos$best_round))
cat(sprintf("%-30s %-15.6f %-15.6f\n", "WP CV logloss", result_insample$best_logloss, result_oos$best_logloss))
cat(sprintf("%-30s %-15.4f %-15.4f\n", "EP mean max-class prob", insample_sharpness, oos_sharpness))
cli::cli_inform("")

logloss_diff <- result_insample$best_logloss - result_oos$best_logloss
if (logloss_diff < 0) {
  cli::cli_inform("In-sample EP WP has {round(abs(logloss_diff) / result_oos$best_logloss * 100, 2)}% lower logloss (appears better but is optimistic)")
} else {
  cli::cli_inform("CV EP WP has {round(abs(logloss_diff) / result_insample$best_logloss * 100, 2)}% lower logloss (genuine improvement)")
}
cli::cli_inform("Note: In-sample EP logloss is optimistic due to data leakage. CV EP gives honest metrics.")
