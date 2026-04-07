# Train Live EP Model (8 features)
# =================================
# Lean XGBoost for Cloudflare Worker inference on live chain data.
# Uses only the top 8 features by variable importance (97% of gain).
# Exports tree structure as JSON for JS tree walker.

library(devtools)
library(xgboost)
library(jsonlite)

# Load torp (need load_all for internal functions like clean_model_data_epv)
torp_paths <- c("../../torp", "../torp", "../../../torp")
loaded <- FALSE
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p)
    loaded <- TRUE
    break
  }
}
if (!loaded) stop("Cannot find torp package. Run from torpmodels or torpverse workspace.")

# ── 1. Load and prepare training data ──────────────────────────
cli::cli_inform("Loading chains data...")
chains <- load_chains(TRUE, TRUE)

cli::cli_inform("Cleaning play-by-play data...")
pbp <- clean_pbp(chains)

cli::cli_inform("Preparing EPV model data...")
model_data <- clean_model_data_epv(pbp)

# ── 2. Select live features (top 8 by variable importance) ─────
live_vars <- c(
  "est_qtr_remaining",        # 56.2% gain
  "goal_x",                   # 16.1% gain
  "period_seconds",           # 11.6% gain
  "y",                        #  3.2% gain
  "shot_row",                 #  2.7% gain
  "lag_goal_x",               #  2.4% gain
  "est_match_remaining",      #  2.0% gain
  "phase_of_play_set_shot"    #  1.9% gain
)

cli::cli_inform("Selected {length(live_vars)} features: {paste(live_vars, collapse = ', ')}")

epv_live <- as.data.frame(model_data)[, live_vars, drop = FALSE]
X_train <- stats::model.matrix(~ . + 0, data = epv_live)
y_train <- model_data$label_ep
feature_names <- colnames(X_train)

cli::cli_inform("Training matrix: {nrow(X_train)} rows x {ncol(X_train)} features")
cli::cli_inform("Features: {paste(feature_names, collapse = ', ')}")

full_train <- xgboost::xgb.DMatrix(data = X_train, label = y_train)

# ── 3. Match-grouped CV ───────────────────────────────────────
cli::cli_inform("Creating match-grouped CV folds...")
match_ids <- unique(model_data$torp_match_id)
set.seed(1234)
match_folds <- sample(rep(1:5, length.out = length(match_ids)))
names(match_folds) <- match_ids
row_folds <- match_folds[model_data$torp_match_id]
folds <- lapply(1:5, function(k) which(row_folds == k))

# XGBoost parameters (same as full model)
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

cli::cli_inform("Running 5-fold CV...")
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
best_loss <- min(cv_result$evaluation_log$test_mlogloss_mean)
cli::cli_inform("Optimal nrounds: {optimal_nrounds}")
cli::cli_inform("Best CV mlogloss: {round(best_loss, 6)}")

# ── 4. Train final model ──────────────────────────────────────
cli::cli_inform("Training final live EP model...")
set.seed(1234)
ep_model_live <- xgboost::xgb.train(
  params = params,
  data = full_train,
  nrounds = optimal_nrounds,
  print_every_n = 10
)

# ── 5. Validate against full 19-feature model ─────────────────
cli::cli_inform("\n── Validation ──────────────────────────────────────")

# Live model predictions
live_preds_raw <- predict(ep_model_live, X_train)
live_matrix <- matrix(live_preds_raw, ncol = 5, byrow = TRUE)
live_ep <- -6 * live_matrix[, 1] - live_matrix[, 2] + live_matrix[, 3] + 6 * live_matrix[, 4]

cli::cli_inform("Live model exp_pts range: [{round(min(live_ep), 3)}, {round(max(live_ep), 3)}]")
cli::cli_inform("Live model exp_pts SD: {round(sd(live_ep), 4)}")

# Full model predictions (for comparison)
tryCatch({
  ep_model_full <- load_model_with_fallback("ep")
  full_vars <- model_data |> select_epv_model_vars()
  X_full <- stats::model.matrix(~ . + 0, data = full_vars)
  full_preds_raw <- predict(ep_model_full, X_full)
  full_matrix <- matrix(full_preds_raw, ncol = 5, byrow = TRUE)
  full_ep <- -6 * full_matrix[, 1] - full_matrix[, 2] + full_matrix[, 3] + 6 * full_matrix[, 4]

  corr <- cor(live_ep, full_ep)
  rmse <- sqrt(mean((live_ep - full_ep)^2))
  mae <- mean(abs(live_ep - full_ep))
  cli::cli_inform("Correlation with full model: {round(corr, 4)}")
  cli::cli_inform("RMSE vs full model: {round(rmse, 4)}")
  cli::cli_inform("MAE vs full model: {round(mae, 4)}")
}, error = function(e) {
  cli::cli_warn("Could not load full model for comparison: {e$message}")
})

# Variable importance
imp <- xgb.importance(feature_names = feature_names, model = ep_model_live)
cli::cli_inform("\nVariable importance (live model):")
for (i in seq_len(nrow(imp))) {
  cli::cli_inform("  {i}. {imp$Feature[i]}: Gain={round(imp$Gain[i], 4)}")
}

# ── 6. Save RDS model ���────────────────────────────────────────
output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

rds_path <- file.path(output_dir, "ep_model_live.rds")
saveRDS(ep_model_live, rds_path)
cli::cli_inform("\nSaved RDS: {rds_path} ({round(file.size(rds_path) / 1024, 1)} KB)")

# ── 7. Export tree JSON for Cloudflare Worker ──────────────────
cli::cli_inform("\n── Exporting tree JSON ─────────────────────────────")

json_trees <- xgb.dump(ep_model_live, dump_format = "json")
json_parsed <- fromJSON(json_trees, simplifyVector = FALSE)

n_trees <- length(json_parsed)
cli::cli_inform("Total trees: {n_trees} ({n_trees / 5} rounds x 5 classes)")

# Save with feature name metadata
export <- list(
  trees = json_parsed,
  feature_names = feature_names,
  num_class = 5,
  class_labels = c("opp_goal", "opp_behind", "behind", "goal", "no_score"),
  num_rounds = optimal_nrounds,
  cv_mlogloss = best_loss
)

json_path <- file.path(getwd(), "inst", "models", "core", "ep_model_live.json")
write_json(export, json_path, auto_unbox = TRUE, pretty = FALSE)
json_size <- file.size(json_path)
cli::cli_inform("Saved JSON: {json_path}")
cli::cli_inform("JSON size: {round(json_size / 1024, 1)} KB")

# Also save a copy for the blog repo
blog_path <- normalizePath(file.path(getwd(), "..", "..", "inthegame-blog", "afl", "ep-model-live.json"), mustWork = FALSE)
if (dir.exists(dirname(blog_path))) {
  file.copy(json_path, blog_path, overwrite = TRUE)
  cli::cli_inform("Copied to: {blog_path}")
}

cli::cli_inform("\n── Done ────────────────────────────────────────────")
cli::cli_inform("Next steps:")
cli::cli_inform("  1. Upload ep-model-live.json to R2: afl/ep-model-live.json")
cli::cli_inform("  2. Build JS tree walker in worker/src/ep-model.js")
cli::cli_inform("  3. Wire into handleAflLiveChains in worker/src/index.js")
