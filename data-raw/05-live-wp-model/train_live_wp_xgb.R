# Train Live WP Model — XGBoost (replaces GAM lookup table)
# =========================================================
# Binary:logistic XGBoost trained on symmetrized data (no home bias).
# Features match what Squiggle API provides at runtime:
#   margin, period, period_seconds, game_seconds
#
# Phase 1: Compare GAM vs XGBoost on key Q4 scenarios
# Phase 2: Export tree JSON for browser inference (same pattern as EP model)

library(devtools)
library(data.table)
library(xgboost)
library(jsonlite)

# Load torp
torp_paths <- c("../../torp", "../torp", "../../../torp",
                "C:/dev/torpverse/torp")
loaded <- FALSE
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p)
    loaded <- TRUE
    cli::cli_alert_success("Loaded torp from {normalizePath(p)}")
    break
  }
}
if (!loaded) stop("Cannot find torp package.")

# ── 1. Load and prepare training data ────────────────────────────────────────

cli::cli_h1("Training Live Win Probability Model (XGBoost)")
cli::cli_alert_info("Loading pre-cleaned PBP data...")
pbp <- torp::load_pbp(TRUE)

dt <- data.table::as.data.table(pbp)
cli::cli_alert_info("PBP rows: {nrow(dt)}, columns: {ncol(dt)}")
cli::cli_alert_info("Seasons: {paste(sort(unique(dt$season)), collapse = ', ')}")

# ── 2. Build features from HOME team perspective ─────────────────────────────
# We want: margin = home_points - away_points (always from home team POV)
# label = did the home team win? (1/0/0.5 for draw)

required_cols <- c("period", "period_seconds", "home_points", "away_points",
                   "home_score", "away_score", "torp_match_id", "season")
missing <- setdiff(required_cols, names(dt))
if (length(missing) > 0) cli::cli_abort("Missing columns: {paste(missing, collapse = ', ')}")

# Feature: margin from home team's perspective
dt[, margin := home_points - away_points]

# Feature: game_seconds (same formula as existing GAM)
dt[, game_seconds := (period - 1L) * 2000 + period_seconds]

# Label: did home team win?
dt[, label := data.table::fcase(
  home_score > away_score, 1,
  home_score == away_score, 0.5,
  home_score < away_score, 0,
  default = NA_real_
)]

# Keep only what we need (home=1 since all rows are from home team perspective)
wp <- dt[!is.na(label) & !is.na(period_seconds) & !is.na(margin),
         .(margin, period, period_seconds, game_seconds, home = 1L, label,
           torp_match_id, season)]

n_matches <- uniqueN(wp$torp_match_id)
n_draws <- uniqueN(wp[label == 0.5]$torp_match_id)
cli::cli_alert_info("Training rows: {nrow(wp)} across {n_matches} matches ({min(wp$season)}-{max(wp$season)}), {n_draws} draws")

# ── 3. Symmetrize data ──────────────────────────────────────────────────────
# For each row (margin=M, home=1, label=L), add (margin=-M, home=0, label=1-L)
# Original = home team perspective, flipped = away team perspective
# Model learns home advantage from the home feature, not from data imbalance

wp_flip <- copy(wp)
wp_flip[, `:=`(margin = -margin, home = 0L, label = 1 - label)]
# Give flipped rows distinct match IDs for CV grouping (same match, different fold OK)
wp_flip[, torp_match_id := paste0(torp_match_id, "_flip")]

wp_sym <- rbindlist(list(wp, wp_flip))
cli::cli_alert_info("After symmetrization: {nrow(wp_sym)} rows ({nrow(wp)} original + {nrow(wp_flip)} flipped)")

# ── 4. Train XGBoost ────────────────────────────────────────────────────────

feature_cols <- c("margin", "period", "period_seconds", "game_seconds", "home")
X_train <- as.matrix(wp_sym[, ..feature_cols])
y_train <- wp_sym$label

dtrain <- xgb.DMatrix(data = X_train, label = y_train)

# Match-grouped CV (keep both original + flipped rows of same match in same fold)
base_match_ids <- gsub("_flip$", "", wp_sym$torp_match_id)
unique_base <- unique(base_match_ids)
set.seed(42)
match_folds <- sample(rep(1:5, length.out = length(unique_base)))
names(match_folds) <- unique_base
row_folds <- match_folds[base_match_ids]
folds <- lapply(1:5, function(k) which(row_folds == k))

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  tree_method = "hist",
  eta = 0.1,
  max_depth = 4,
  min_child_weight = 50,
  subsample = 0.85,
  colsample_bytree = 0.85,
  gamma = 0
)

cli::cli_alert_info("Running 5-fold match-grouped CV...")
set.seed(42)
cv_result <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 500,
  folds = folds,
  early_stopping_rounds = 20,
  print_every_n = 20,
  verbose = 1
)

optimal_nrounds <- which.min(cv_result$evaluation_log$test_logloss_mean)
best_loss <- min(cv_result$evaluation_log$test_logloss_mean)
cli::cli_alert_info("Optimal nrounds: {optimal_nrounds}, best CV logloss: {round(best_loss, 6)}")

# Train final model
set.seed(42)
wp_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = optimal_nrounds,
  print_every_n = 10
)

# Variable importance
imp <- xgb.importance(feature_names = feature_cols, model = wp_model)
cli::cli_h2("Variable importance")
for (i in seq_len(nrow(imp))) {
  cli::cli_alert_info("  {i}. {imp$Feature[i]}: Gain={round(imp$Gain[i] * 100, 1)}%")
}

# ── 5. Load existing GAM for comparison ──────────────────────────────────────

cli::cli_h1("GAM vs XGBoost Comparison")

# Load the existing GAM model
gam_rds <- file.path(getwd(), "inst", "models", "core", "live_wp_model.rds")
if (!file.exists(gam_rds)) {
  # Try alternative paths
  gam_rds <- "C:/dev/torpverse/torpmodels/inst/models/core/live_wp_model.rds"
}

gam_available <- file.exists(gam_rds)
if (gam_available) {
  gam_fit <- readRDS(gam_rds)
  cli::cli_alert_success("Loaded existing GAM from {gam_rds}")
} else {
  cli::cli_warn("GAM model not found at {gam_rds} — will show XGBoost only")
}

# ── 6. Compare on key scenarios ──────────────────────────────────────────────

test <- data.table(
  scenario = c(
    # Q4 scenarios (the problematic ones)
    "Q4 start, tied",
    "Q4 6min, down 1",
    "Q4 6min, up 1",
    "Q4 15min, tied",
    "Q4 15min, down 1",
    "Q4 15min, up 1",
    "Q4 25min, tied",
    "Q4 25min, down 1",
    "Q4 25min, up 1",
    "Q4 25min, down 6",
    "Q4 25min, up 6",
    "Q4 30min, tied",
    "Q4 30min, down 1",
    "Q4 30min, up 1",
    # Earlier quarter scenarios
    "Q1 start, tied",
    "Q2 end, up 18",
    "Q2 end, down 18",
    "HT tied",
    "Q3 end, up 30",
    "Q3 end, down 30",
    # Symmetry checks
    "Q4 15min, down 10",
    "Q4 15min, up 10"
  ),
  margin = c(
    0, -1, 1, 0, -1, 1, 0, -1, 1, -6, 6, 0, -1, 1,
    0, 18, -18, 0, 30, -30,
    -10, 10
  ),
  period = c(
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    1, 2, 2, 2, 3, 3,
    4, 4
  ),
  period_seconds = c(
    0, 360, 360, 1200, 1200, 1200, 1500, 1500, 1500, 1500, 1500, 1800, 1800, 1800,
    0, 1800, 1800, 1800, 1800, 1800,
    1200, 1200
  )
)

test[, game_seconds := (period - 1L) * 2000 + period_seconds]

# XGBoost predictions (home=1, same as GAM comparison)
X_test_home <- as.matrix(test[, .(margin, period, period_seconds, game_seconds, home = 1L)])
test[, xgb_home := round(predict(wp_model, X_test_home) * 100, 1)]

# Also predict with home=0 to show the away perspective
X_test_away <- as.matrix(test[, .(margin, period, period_seconds, game_seconds, home = 0L)])
test[, xgb_away := round(predict(wp_model, X_test_away) * 100, 1)]

# GAM predictions (with home=1, as the lookup was generated)
if (gam_available) {
  gam_input <- data.table(
    total_seconds = test$game_seconds,
    points_diff = test$margin,
    home = 1L
  )
  test[, gam_wp := round(predict(gam_fit, newdata = gam_input, type = "response") * 100, 1)]
}

# Print comparison
cli::cli_h2("Q4 Scenario Comparison: Home team WP (%)")
if (gam_available) {
  print(test[, .(scenario, margin, period, period_seconds, GAM = gam_wp, XGB_home = xgb_home, XGB_away = xgb_away)])
} else {
  print(test[, .(scenario, margin, period, period_seconds, XGB_home = xgb_home, XGB_away = xgb_away)])
}

# Home advantage check: home=1 vs home=0 at margin=0 (difference = home advantage)
cli::cli_h2("Home advantage (margin=0, home=1 vs home=0)")
for (ps in c(0, 600, 1200, 1800)) {
  gs <- 3 * 2000 + ps  # Q4
  h <- predict(wp_model, matrix(c(0, 4, ps, gs, 1), nrow = 1))
  a <- predict(wp_model, matrix(c(0, 4, ps, gs, 0), nrow = 1))
  cli::cli_alert_info("Q4 {ps}s: home={round(h*100,1)}% away={round(a*100,1)}% advantage={round((h-a)*100,1)}pp")
}

# Symmetry check: WP(margin=+M, home=1) should equal 1 - WP(margin=-M, home=0)
cli::cli_h2("Symmetry check: WP(+M,home) + WP(-M,away) should = ~100%")
for (m in c(1, 6, 10, 18, 30)) {
  xgb_home_pos <- predict(wp_model, matrix(c(m, 4, 1200, 7200, 1), nrow = 1))
  xgb_away_neg <- predict(wp_model, matrix(c(-m, 4, 1200, 7200, 0), nrow = 1))
  cli::cli_alert_info("home +{m}: {round(xgb_home_pos*100,1)}% | away -{m}: {round(xgb_away_neg*100,1)}% | sum: {round((xgb_home_pos+xgb_away_neg)*100,1)}%")
}

# ── 7. Calibration ──────────────────────────────────────────────────────────

cli::cli_h2("Calibration: predicted vs actual (original data only, not flipped)")
wp_orig <- wp_sym[!grepl("_flip$", torp_match_id)]
X_orig <- as.matrix(wp_orig[, ..feature_cols])
wp_orig[, predicted := predict(wp_model, X_orig)]
wp_orig[, pred_bucket := cut(predicted, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)]

calib <- wp_orig[, .(
  actual = round(mean(label) * 100, 1),
  predicted = round(mean(predicted) * 100, 1),
  n = .N
), by = pred_bucket][order(pred_bucket)]
print(calib)

# ── 8. Export tree JSON ─────────────────────────────────────────────────────

cli::cli_h1("Exporting tree JSON")

json_trees <- xgb.dump(wp_model, dump_format = "json")
json_parsed <- fromJSON(json_trees, simplifyVector = FALSE)

n_trees <- length(json_parsed)
cli::cli_alert_info("Total trees: {n_trees}")

export <- list(
  description = "AFL Live Win Probability XGBoost (symmetrized, no home bias)",
  trained_on = paste0(min(wp$season), "-", max(wp$season)),
  n_matches = n_matches,
  n_plays = nrow(wp),
  n_draws = n_draws,
  num_rounds = optimal_nrounds,
  cv_logloss = best_loss,
  feature_names = feature_cols,
  num_class = 1,
  trees = json_parsed
)

# Save to torpmodels
output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
json_path <- file.path(output_dir, "wp_model_live.json")
write_json(export, json_path, auto_unbox = TRUE, pretty = FALSE)
json_size <- file.size(json_path)
cli::cli_alert_success("Saved JSON: {json_path} ({round(json_size / 1024, 1)} KB)")

# Copy to blog repo
blog_path <- normalizePath(
  file.path(getwd(), "..", "..", "inthegame-blog", "afl", "wp-model.json"),
  mustWork = FALSE
)
if (dir.exists(dirname(blog_path))) {
  file.copy(json_path, blog_path, overwrite = TRUE)
  cli::cli_alert_success("Copied to: {blog_path}")
}

# Also save RDS
rds_path <- file.path(output_dir, "wp_model_live.rds")
saveRDS(wp_model, rds_path)
cli::cli_alert_success("Saved RDS: {rds_path}")

cli::cli_h1("Done!")
cli::cli_alert_info("Next: review the comparison table above.")
cli::cli_alert_info("If XGBoost looks good, upload wp-model.json to R2: afl/wp-model.json")
