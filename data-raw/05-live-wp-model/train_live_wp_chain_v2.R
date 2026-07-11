# Train Chain-Aware Live WP Model v2 — minimal delta from train_live_wp_xgb.R
# ============================================================================
# v1 (train_live_wp_chain.R) trained on 15 features but calibration was poor
# at margin extremes (tied Q2 = 50% ✓, but Q3 down 20 = 50% ❌, Q4 up 30 = 51% ❌).
# With max_depth=4 and 15 competing features, many splits went to play_type /
# phase_of_play / chain_action_num instead of using the tree depth to express
# strong margin × game_seconds conditions.
#
# v2 copies train_live_wp_xgb.R (base 5-feature model) exactly, adds ONE new
# feature: exp_pts (EPV state value). Same hyperparams, same symmetrization,
# same train harness. Expectation: model retains base's correct margin
# sensitivity and gains exp_pts-driven chain variance.
#
# Output: wp-model-chain.json (overwrites v1).

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

# ── 1. Load data & compute exp_pts via EPV pipeline ──────────────────────────
# We need exp_pts which isn't in load_pbp()'s output. Use the canonical chain
# pipeline (analyze_match.R) to get it, then treat the resulting rows exactly
# as the base script treats load_pbp() rows.
cli::cli_h1("Training Chain-Aware Live WP v2 (XGBoost, base + exp_pts)")

cli::cli_alert_info("Loading chains + running EPV pipeline for exp_pts...")
chains <- torp::load_chains(TRUE, TRUE)
md <- chains |>
  torp::clean_pbp() |>
  torp:::clean_model_data_epv() |>
  torp:::clean_shots_data() |>
  torp::add_shot_vars() |>
  torp::add_epv_vars()
dt <- data.table::as.data.table(md)

# ── 2. Build features from HOME team perspective (exact base script logic) ───
required_cols <- c("period", "period_seconds", "home_points", "away_points",
                   "home_score", "away_score", "torp_match_id", "season", "exp_pts")
missing <- setdiff(required_cols, names(dt))
if (length(missing) > 0) cli::cli_abort("Missing columns: {paste(missing, collapse = ', ')}")

dt[, margin := home_points - away_points]
dt[, game_seconds := (period - 1L) * 2000 + period_seconds]
dt[, label := data.table::fcase(
  home_score > away_score, 1,
  home_score == away_score, 0.5,
  home_score < away_score, 0,
  default = NA_real_
)]

wp <- dt[!is.na(label) & !is.na(period_seconds) & !is.na(margin) & !is.na(exp_pts),
         .(margin, period, period_seconds, game_seconds, home = 1L, exp_pts,
           label, torp_match_id, season)]

n_matches <- uniqueN(wp$torp_match_id)
n_draws <- uniqueN(wp[label == 0.5]$torp_match_id)
cli::cli_alert_info("Training rows: {nrow(wp)} across {n_matches} matches ({min(wp$season)}-{max(wp$season)}), {n_draws} draws")
cli::cli_alert_info("exp_pts range: [{round(min(wp$exp_pts),2)}, {round(max(wp$exp_pts),2)}]  mean={round(mean(wp$exp_pts),3)}")

# ── 3. Symmetrize (identical to base, with exp_pts flip) ─────────────────────
wp_flip <- copy(wp)
wp_flip[, `:=`(margin = -margin, exp_pts = -exp_pts, home = 0L, label = 1 - label)]
wp_flip[, torp_match_id := paste0(torp_match_id, "_flip")]
wp_sym <- rbindlist(list(wp, wp_flip))
cli::cli_alert_info("After symmetrization: {nrow(wp_sym)} rows")

# ── 4. Train (identical hyperparams to train_live_wp_xgb.R) ──────────────────
feature_cols <- c("margin", "period", "period_seconds", "game_seconds", "home", "exp_pts")
X_train <- as.matrix(wp_sym[, ..feature_cols])
y_train <- wp_sym$label
dtrain <- xgb.DMatrix(data = X_train, label = y_train)

# Match-grouped CV (same as base)
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
  params = params, data = dtrain, nrounds = 500, folds = folds,
  early_stopping_rounds = 20, print_every_n = 20, verbose = 1
)
optimal_nrounds <- which.min(cv_result$evaluation_log$test_logloss_mean)
best_loss <- min(cv_result$evaluation_log$test_logloss_mean)
cli::cli_alert_info("Optimal nrounds: {optimal_nrounds}, best CV logloss: {round(best_loss, 6)}")

set.seed(42)
model <- xgb.train(params = params, data = dtrain, nrounds = optimal_nrounds, verbose = 1)

# Variable importance
imp <- xgb.importance(feature_names = feature_cols, model = model)
cli::cli_h2("Variable importance")
for (i in seq_len(nrow(imp))) {
  cli::cli_alert_info("  {i}. {imp$Feature[i]}: Gain={round(imp$Gain[i] * 100, 1)}%")
}

# ── 5. Sample-state sanity check (same scenarios as v1) ──────────────────────
cli::cli_h1("Sample-state comparison: v2 chain WP vs base 5-feat WP")

baseline_rds <- "C:/dev/torpverse/torpmodels/inst/models/core/wp_model_live.rds"
baseline <- if (file.exists(baseline_rds)) readRDS(baseline_rds) else NULL

samples <- data.table(
  scenario = c("Q2 end tied", "Q3 end down 20", "Q4 28min up 30",
               "Q4 30min tied, shot EPV=3", "Q4 30min tied, EPV=-0.2",
               "Q1 start tied", "Q4 start tied", "HT down 10", "HT up 10"),
  margin         = c(0, -20, 30, 0, 0, 0, 0, -10, 10),
  period         = c(2L, 3L, 4L, 4L, 4L, 1L, 4L, 2L, 2L),
  period_seconds = c(1800, 1800, 1680, 1800, 1800, 0, 0, 1800, 1800),
  exp_pts        = c(0, 0, 0, 3.0, -0.2, 0, 0, 0, 0)
)
samples[, game_seconds := (period - 1L) * 2000 + period_seconds]
samples[, home := 1L]

Xs <- as.matrix(samples[, ..feature_cols])
samples[, wp_v2 := round(predict(model, Xs) * 100, 1)]

if (!is.null(baseline)) {
  base_cols <- c("margin", "period", "period_seconds", "game_seconds", "home")
  Xs_base <- as.matrix(samples[, ..base_cols])
  samples[, wp_base := round(predict(baseline, Xs_base) * 100, 1)]
  print(samples[, .(scenario, margin, period, period_seconds, exp_pts, wp_base, wp_v2)])
} else {
  print(samples[, .(scenario, margin, period, period_seconds, exp_pts, wp_v2)])
}

# ── 6. Richness check ────────────────────────────────────────────────────────
cli::cli_h1("Richness check")
test_mid <- dt[season == max(season) & round_number == max(round_number, na.rm = TRUE),
               unique(match_id)][1]
if (is.na(test_mid)) test_mid <- dt$match_id[1]
cli::cli_alert_info("Test match: {test_mid}")

mdt <- dt[match_id == test_mid][order(period, period_seconds)]
mdt[, `:=`(margin = home_points - away_points, game_seconds = (period - 1L) * 2000 + period_seconds, home = 1L)]
mdt_f <- mdt[!is.na(exp_pts) & !is.na(margin)]
mdt_X <- as.matrix(mdt_f[, ..feature_cols])
mdt_f[, wp := predict(model, mdt_X)]
non_scoring <- mdt_f[!description %in% c("Goal", "Behind", "Rushed")]
non_scoring[, wp_delta := abs(wp - shift(wp))]
shift_thresh <- 0.001
shifted <- sum(non_scoring$wp_delta > shift_thresh, na.rm = TRUE)
total <- sum(!is.na(non_scoring$wp_delta))
cli::cli_alert_info("Non-scoring consecutive rows: {total}. With |wp_delta|>{shift_thresh}: {shifted} ({round(100*shifted/total, 2)}%)")

# ── 7. Export JSON ───────────────────────────────────────────────────────────
cli::cli_h1("Exporting v2 JSON")

raw_model_json <- xgb.dump(model, with_stats = FALSE, dump_format = "json")
trees_parsed <- fromJSON(paste(raw_model_json, collapse = ""), simplifyVector = FALSE)

# xgb.dump returns a list of per-tree JSON strings when dump_format="json"
# so we need to pull out the trees array properly.
if (length(raw_model_json) == 1) {
  trees_parsed <- fromJSON(raw_model_json, simplifyVector = FALSE)
} else {
  trees_parsed <- lapply(raw_model_json, function(tj) fromJSON(tj, simplifyVector = FALSE))
}

# Pull base_score from booster config
cfg_str <- tryCatch(xgb.config(model), error = function(e) NULL)
base_score <- 0.5
if (!is.null(cfg_str)) {
  cfg <- fromJSON(cfg_str, simplifyVector = FALSE)
  bs <- cfg$learner$learner_model_param$base_score
  if (!is.null(bs)) base_score <- as.numeric(bs)
}
cli::cli_alert_info("Booster base_score = {round(base_score, 5)}")

envelope <- list(
  model_type = "wp_afl_chain_v2",
  objective = "binary:logistic",
  num_class = 1L,
  feature_names = feature_cols,
  nrounds = optimal_nrounds,
  base_score = base_score,
  n_matches = n_matches,
  trained_on = paste0(min(wp$season), "-", max(wp$season)),
  trees = trees_parsed,
  exported_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
)

out_json <- "C:/dev/torpverse/torpmodels/data-raw/05-live-wp-model/wp-model-chain.json"
write(toJSON(envelope, auto_unbox = TRUE, digits = NA), out_json)
cli::cli_alert_success("Saved: {out_json} ({round(file.size(out_json)/1024, 1)} KB)")
