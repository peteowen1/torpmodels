# Validate CV EP vs In-Sample EP for WP Model ----
# =================================================
# Compares WP models trained with in-sample vs cross-validated EP predictions.
# Run to quantify the improvement CV-EP gives over the legacy in-sample
# approach. Sources lib/train_lib.R for the shared data/fold/CV plumbing --
# the canonical trainer (TRAINING-CONSOLIDATION-PLAN.md Step 3).

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

devtools::load_all(".")
source("data-raw/lib/train_lib.R")

# Load and prepare shared data ----
cli::cli_inform("Loading chains data...")
model_data_epv <- load_training_pbp(default_training_seasons())

# Shared EP setup (same machinery train_models.R uses) ----
epv_vars <- model_data_epv |> torp:::select_epv_model_vars()
X_all <- stats::model.matrix(~ . + 0, data = epv_vars, na.action = na.pass)
y_all <- model_data_epv$label_ep

row_folds <- make_match_folds(model_data_epv$torp_match_id)
folds_idx <- lapply(1:5, function(k) which(row_folds == k))

ep_p <- ep_params()

# Optimal EP nrounds ----
set.seed(1234)
cv_result <- xgboost::xgb.cv(
  params = ep_p, data = xgboost::xgb.DMatrix(data = X_all, label = y_all), nrounds = 500,
  folds = folds_idx, early_stopping_rounds = 20, print_every_n = 50, verbose = 1
)
optimal_nrounds <- which.min(cv_result$evaluation_log$test_mlogloss_mean)

# Generate in-sample EP predictions (full model) ----
cli::cli_inform("Training full EP model for in-sample predictions...")
set.seed(1234)
full_ep_model <- xgboost::xgb.train(
  params = ep_p, data = xgboost::xgb.DMatrix(data = X_all, label = y_all),
  nrounds = optimal_nrounds, verbose = 0
)
insample_preds_raw <- predict(full_ep_model, xgboost::xgb.DMatrix(data = X_all))
if (is.matrix(insample_preds_raw)) {
  insample_ep <- insample_preds_raw
} else {
  insample_ep <- matrix(insample_preds_raw, ncol = 5, byrow = TRUE)
}
colnames(insample_ep) <- c("opp_goal", "opp_behind", "behind", "goal", "no_score")

# Generate OOS EP predictions (5-fold CV) via the shared lib ----
cli::cli_inform("Generating OOS EP predictions via 5-fold CV...")
oos_ep <- cv_ep_oos_preds(X_all, y_all, folds_idx, row_folds, ep_p, optimal_nrounds)

# Compare EP prediction sharpness ----
insample_sharpness <- mean(apply(insample_ep, 1, max))
oos_sharpness <- mean(apply(oos_ep, 1, max))

cli::cli_h1("EP Prediction Comparison")
cli::cli_inform("In-sample mean max-class prob: {round(insample_sharpness, 4)}")
cli::cli_inform("OOS mean max-class prob:       {round(oos_sharpness, 4)}")
cli::cli_inform("Difference:                    {round(insample_sharpness - oos_sharpness, 4)} (in-sample is sharper)")

# Run WP CV with in-sample EP ----
cli::cli_h1("WP Model with In-Sample EP")
wp_data_insample <- build_wp_data(model_data_epv, insample_ep)
result_insample <- fit_wp(wp_data_insample)
cli::cli_inform("Best round: {result_insample$optimal_nrounds}")
cli::cli_inform("Best CV logloss: {round(result_insample$cv_logloss, 6)}")

# Run WP CV with OOS EP ----
cli::cli_h1("WP Model with CV (OOS) EP")
wp_data_oos <- build_wp_data(model_data_epv, oos_ep)
result_oos <- fit_wp(wp_data_oos)
cli::cli_inform("Best round: {result_oos$optimal_nrounds}")
cli::cli_inform("Best CV logloss: {round(result_oos$cv_logloss, 6)}")

# Summary comparison ----
cli::cli_h1("Summary")
cli::cli_inform("")
cat(sprintf("%-30s %-15s %-15s\n", "", "In-Sample EP", "CV (OOS) EP"))
cat(sprintf("%-30s %-15s %-15s\n", "---", "---", "---"))
cat(sprintf("%-30s %-15d %-15d\n", "WP optimal nrounds", result_insample$optimal_nrounds, result_oos$optimal_nrounds))
cat(sprintf("%-30s %-15.6f %-15.6f\n", "WP CV logloss", result_insample$cv_logloss, result_oos$cv_logloss))
cat(sprintf("%-30s %-15.4f %-15.4f\n", "EP mean max-class prob", insample_sharpness, oos_sharpness))
cli::cli_inform("")

logloss_diff <- result_insample$cv_logloss - result_oos$cv_logloss
if (logloss_diff < 0) {
  cli::cli_inform("In-sample EP WP has {round(abs(logloss_diff) / result_oos$cv_logloss * 100, 2)}% lower logloss (appears better but is optimistic)")
} else {
  cli::cli_inform("CV EP WP has {round(abs(logloss_diff) / result_insample$cv_logloss * 100, 2)}% lower logloss (genuine improvement)")
}
cli::cli_inform("Note: In-sample EP logloss is optimistic due to data leakage. CV EP gives honest metrics.")
