# Train Chain-Aware Live WP Model — XGBoost
# ==========================================
# Extends the 5-feature live WP XGBoost (margin, period, period_seconds,
# game_seconds, home) with chain/EPV state features so every chain row
# produces a meaningfully different prediction — mirroring ep_model_live_v2
# (14-feature EPV) for WPA.
#
# New features (on top of baseline 5):
#   exp_pts           (EPV state value from add_epv_vars)
#   shot_row          (1 if shot_at_goal TRUE else 0)
#   chain_action_num  (row index within match/chain)
#   play_type_handball, play_type_kick, play_type_reception
#   phase_of_play_handball_received, phase_of_play_hard_ball,
#   phase_of_play_loose_ball, phase_of_play_set_shot
#
# Training data: clean_model_data_epv() output (chain rows, dummified) +
# exp_pts/chain_action_num joined in. Label = did HOME win (1/0/0.5 draw).
# Symmetrized (home=1 + flipped home=0) same as existing baseline.
#
# Output: wp-model-chain.json locally. Does NOT overwrite wp-model.json.

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

# ── 1. Build chain-level training data ───────────────────────────────────────
cli::cli_h1("Chain-aware Live WP (XGBoost)")

cli::cli_alert_info("Loading chains + running canonical EPV pipeline...")
# Canonical pipeline from torp/R/analyze_match.R:62-69. Uses two internal
# (unexported) helpers: clean_model_data_epv and clean_shots_data. Access
# via triple-colon since we need them before add_epv_vars (which depends
# on the dummies + lag features that clean_model_data_epv adds).
chains <- torp::load_chains(TRUE, TRUE)
md <- chains |>
  torp::clean_pbp() |>
  torp:::clean_model_data_epv() |>
  torp:::clean_shots_data() |>
  torp::add_shot_vars() |>
  torp::add_epv_vars()

dt <- data.table::as.data.table(md)

# chain_action_num: row index within each (match, chain_number)
dt[, chain_action_num := seq_len(.N), by = .(match_id, chain_number)]

# shot_row: shot-at-goal flag as 0/1 numeric
if (!"shot_row" %in% names(dt)) {
  if ("shot_at_goal" %in% names(dt)) {
    dt[, shot_row := as.integer(as.logical(shot_at_goal))]
    dt[is.na(shot_row), shot_row := 0L]
  } else {
    dt[, shot_row := 0L]
  }
}

# Compute margin (home − away points), game_seconds on model's 0-8000 axis
stopifnot(all(c("home_points", "away_points", "period", "period_seconds") %in% names(dt)))
dt[, margin := home_points - away_points]
dt[, game_seconds := (period - 1L) * 2000 + period_seconds]

# Guard against missing dummy columns (edge case: some factor levels absent in a subset)
needed_dummies <- c(
  "play_type_handball", "play_type_kick", "play_type_reception",
  "phase_of_play_handball_received", "phase_of_play_hard_ball",
  "phase_of_play_loose_ball", "phase_of_play_set_shot"
)
for (col in needed_dummies) {
  if (!col %in% names(dt)) {
    cli::cli_alert_warning("Column {col} missing — filling 0")
    dt[, (col) := 0L]
  }
}

# Label: did HOME team win? (1 / 0 / 0.5 draw)
required <- c("home_score", "away_score", "torp_match_id", "season", "label_wp")
miss <- setdiff(required, names(dt))
if (length(miss) > 0) cli::cli_abort("Missing cols: {paste(miss, collapse=', ')}")

# label_wp already lives on the pbp row (from HOME POV) — use it directly
wp <- dt[!is.na(label_wp) & !is.na(margin) & !is.na(period_seconds) & !is.na(exp_pts),
         .(margin, period, period_seconds, game_seconds, home = 1L,
           exp_pts, shot_row, chain_action_num,
           play_type_handball, play_type_kick, play_type_reception,
           phase_of_play_handball_received, phase_of_play_hard_ball,
           phase_of_play_loose_ball, phase_of_play_set_shot,
           label = label_wp, torp_match_id, season)]

n_matches <- uniqueN(wp$torp_match_id)
n_draws   <- uniqueN(wp[label == 0.5]$torp_match_id)
cli::cli_alert_info("Training rows: {nrow(wp)} across {n_matches} matches ({min(wp$season)}-{max(wp$season)}), {n_draws} draws")
cli::cli_alert_info("exp_pts range: [{round(min(wp$exp_pts),2)}, {round(max(wp$exp_pts),2)}]  mean={round(mean(wp$exp_pts),3)}")

# ── 2. Symmetrize (home=1 rows + flipped home=0 rows) ────────────────────────
# Flipping home POV: margin → -margin, exp_pts → -exp_pts (EPV is home-relative
# via home flag in EP model → for WP training we're modelling P(home team POV
# wins), so flip sign of margin and exp_pts, flip label).
wp_flip <- copy(wp)
wp_flip[, `:=`(margin = -margin, exp_pts = -exp_pts, home = 0L, label = 1 - label)]
wp_flip[, torp_match_id := paste0(torp_match_id, "_flip")]

wp_sym <- rbindlist(list(wp, wp_flip))
cli::cli_alert_info("After symmetrization: {nrow(wp_sym)} rows")

# ── 3. Train XGBoost ─────────────────────────────────────────────────────────
feature_cols <- c("margin", "period", "period_seconds", "game_seconds", "home",
                  "exp_pts", "shot_row", "chain_action_num",
                  "play_type_handball", "play_type_kick", "play_type_reception",
                  "phase_of_play_handball_received", "phase_of_play_hard_ball",
                  "phase_of_play_loose_ball", "phase_of_play_set_shot")

X <- as.matrix(wp_sym[, ..feature_cols])
y <- wp_sym$label
dtrain <- xgb.DMatrix(data = X, label = y)

# Match-grouped CV folds (original + flipped stay in same fold)
base_ids <- gsub("_flip$", "", wp_sym$torp_match_id)
uniq <- unique(base_ids)
set.seed(42)
fold_map <- sample(rep(1:5, length.out = length(uniq)))
names(fold_map) <- uniq
row_fold <- fold_map[base_ids]
folds <- lapply(1:5, function(k) which(row_fold == k))

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
cv <- xgb.cv(
  params = params, data = dtrain, nrounds = 500, folds = folds,
  early_stopping_rounds = 20, print_every_n = 25, verbose = 1
)
best_nr <- which.min(cv$evaluation_log$test_logloss_mean)
best_loss <- min(cv$evaluation_log$test_logloss_mean)
cli::cli_alert_info("Optimal nrounds: {best_nr}, CV logloss: {round(best_loss, 6)}")

# Final model
set.seed(42)
model <- xgb.train(params = params, data = dtrain, nrounds = best_nr, print_every_n = 25)

# Variable importance
imp <- xgb.importance(feature_names = feature_cols, model = model)
cli::cli_h2("Variable importance")
for (i in seq_len(nrow(imp))) {
  cli::cli_alert_info("  {i}. {imp$Feature[i]}: Gain={round(imp$Gain[i] * 100, 1)}%")
}

# ── 4. Metrics — logloss, Brier, calibration on HELD-OUT (fold 1) ────────────
cli::cli_h1("Validation metrics (held-out fold 1)")

oof_idx <- folds[[1]]
p_oof <- predict(model, X[oof_idx, , drop = FALSE])
y_oof <- y[oof_idx]
clip <- function(p) pmin(pmax(p, 1e-6), 1 - 1e-6)
logloss_oof <- -mean(y_oof * log(clip(p_oof)) + (1 - y_oof) * log(1 - clip(p_oof)))
brier_oof   <- mean((p_oof - y_oof)^2)
cli::cli_alert_info("Fold-1 logloss: {round(logloss_oof, 6)}")
cli::cli_alert_info("Fold-1 Brier:   {round(brier_oof, 6)}")

# Calibration by decile on fold 1
cal_dt <- data.table(pred = p_oof, actual = y_oof)
cal_dt[, decile := cut(pred, breaks = seq(0, 1, 0.1), include.lowest = TRUE)]
cal <- cal_dt[, .(
  n = .N,
  mean_pred = round(mean(pred) * 100, 1),
  actual_rate = round(mean(actual) * 100, 1)
), by = decile][order(decile)]
cli::cli_h2("Calibration (fold 1)")
print(cal)

# ── 5. Sample-state comparison vs baseline 5-feature WP ──────────────────────
cli::cli_h1("Sample-state comparison: chain WP vs existing 5-feat WP")

# Load existing 5-feat model (optional — for reference)
baseline_json <- "C:/dev/torpverse/torpmodels/inst/models/core/wp_model_live.json"
baseline_rds  <- sub("\\.json$", ".rds", baseline_json)
baseline <- if (file.exists(baseline_rds)) readRDS(baseline_rds) else NULL

# Sample game states — we need a reasonable exp_pts for each. For EPV at a
# neutral field-position (centre bounce-ish) we'll use exp_pts=0 for "tied"
# rows, and for illustration vary chain features to show richness.
samples <- data.table(
  scenario = c(
    "Q2 end tied, neutral chain",
    "Q3 end down 20, neutral",
    "Q4 28min up 30, neutral",
    "Q4 30min tied, shot imminent",
    "Q4 30min tied, early chain"
  ),
  margin            = c(0, -20, 30, 0, 0),
  period            = c(2L, 3L, 4L, 4L, 4L),
  period_seconds    = c(1800, 1800, 1680, 1800, 1800),
  exp_pts           = c(0, 0, 0, 3.0, -0.2),
  shot_row          = c(0L, 0L, 0L, 1L, 0L),
  chain_action_num  = c(1L, 1L, 1L, 8L, 1L),
  play_type_handball            = c(0L, 0L, 0L, 0L, 0L),
  play_type_kick                = c(0L, 0L, 0L, 1L, 0L),
  play_type_reception           = c(0L, 0L, 0L, 0L, 1L),
  phase_of_play_handball_received = c(0L, 0L, 0L, 0L, 1L),
  phase_of_play_hard_ball         = c(0L, 0L, 0L, 0L, 0L),
  phase_of_play_loose_ball        = c(0L, 0L, 0L, 0L, 0L),
  phase_of_play_set_shot          = c(0L, 0L, 0L, 1L, 0L)
)
samples[, game_seconds := (period - 1L) * 2000 + period_seconds]
samples[, home := 1L]

Xs <- as.matrix(samples[, ..feature_cols])
samples[, wp_chain := round(predict(model, Xs) * 100, 1)]

if (!is.null(baseline)) {
  Xs_base <- as.matrix(samples[, .(margin, period, period_seconds, game_seconds, home = 1L)])
  samples[, wp_base := round(predict(baseline, Xs_base) * 100, 1)]
  cli::cli_h2("Sample states")
  print(samples[, .(scenario, margin, period, period_seconds,
                    exp_pts, shot_row, wp_base, wp_chain)])
} else {
  cli::cli_alert_warning("Baseline 5-feat model not found — showing chain WP only")
  print(samples[, .(scenario, margin, period, period_seconds,
                    exp_pts, shot_row, wp_chain)])
}

# ── 6. Richness check: fraction of consecutive non-scoring rows that shift WP ─
cli::cli_h1("Richness check (consecutive non-scoring chain rows)")

# Pick a recent match from data
mid_candidates <- dt[season == max(season) & round_number == max(round_number, na.rm = TRUE),
                     unique(match_id)]
test_mid <- if (length(mid_candidates) > 0) mid_candidates[1] else dt$match_id[1]
cli::cli_alert_info("Test match: {test_mid}")

mdt <- dt[match_id == test_mid][order(period, period_seconds)]
if (nrow(mdt) > 10) {
  mdt[, margin_chain := home_points - away_points]
  mdt[, game_seconds := (period - 1L) * 2000 + period_seconds]
  mdt[, home := 1L]

  # Build feature matrix (defaulting any missing dummy to 0)
  for (col in feature_cols) {
    if (!col %in% names(mdt)) mdt[, (col) := 0]
  }
  # margin in mdt has come from home_points - away_points already above
  mdt[, margin := margin_chain]

  Xm <- as.matrix(mdt[, ..feature_cols])
  mdt[, wp := predict(model, Xm)]
  # Non-scoring: no shot_at_goal between rows → margin unchanged
  mdt[, margin_next := shift(margin, -1L)]
  mdt[, wp_next := shift(wp, -1L)]
  non_scoring <- mdt[!is.na(wp_next) & margin == margin_next]
  shifted <- sum(abs(non_scoring$wp_next - non_scoring$wp) > 1e-3)
  total   <- nrow(non_scoring)
  frac    <- round(100 * shifted / max(total, 1), 2)
  cli::cli_alert_info(
    "Non-scoring consecutive rows: {total}. With |wp_delta|>0.001: {shifted} ({frac}%)"
  )
} else {
  cli::cli_alert_warning("Not enough rows in test match for richness check")
}

# ── 7. Export JSON ───────────────────────────────────────────────────────────
cli::cli_h1("Exporting chain-aware JSON")

json_trees <- xgb.dump(model, dump_format = "json")
json_parsed <- fromJSON(json_trees, simplifyVector = FALSE)

export <- list(
  description  = "AFL Chain-Aware Live Win Probability XGBoost (symmetrized)",
  trained_on   = paste0(min(wp$season), "-", max(wp$season)),
  n_matches    = n_matches,
  n_plays      = nrow(wp),
  n_draws      = n_draws,
  num_rounds   = best_nr,
  cv_logloss   = best_loss,
  fold1_logloss = logloss_oof,
  fold1_brier   = brier_oof,
  feature_names = feature_cols,
  num_class    = 1,
  trees        = json_parsed
)

out_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# Keep local to torpmodels. Do NOT overwrite wp_model_live.json.
local_out <- file.path("C:/dev/torpverse/torpmodels/data-raw/05-live-wp-model",
                      "wp-model-chain.json")
write_json(export, local_out, auto_unbox = TRUE, pretty = FALSE)
cli::cli_alert_success("Saved JSON: {local_out} ({round(file.size(local_out)/1024, 1)} KB)")

rds_out <- sub("\\.json$", ".rds", local_out)
saveRDS(model, rds_out)
cli::cli_alert_success("Saved RDS: {rds_out}")

cli::cli_h1("Done — DID NOT upload to R2 or overwrite existing wp_model_live.json")
