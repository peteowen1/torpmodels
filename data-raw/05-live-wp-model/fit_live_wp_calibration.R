# Fit the live/worker WP model's OWN calibration (leverage-interaction form)
# =============================================================================
# torpverse/docs/plans/LIVE-WP-CALIBRATION-PLAN.md. The worker's live WP model
# (train_live_wp_chain_v4.R -- possession-POV chain-aware xgboost, feature_cols
# c("xmargin","period","period_seconds","game_seconds","is_home")) is a
# DIFFERENT model from torp's canonical WP model (WP_MODEL_FEATURES). Its own
# calibration must be fitted on ITS OWN predictions -- torp's published
# wp_calibration (a, b, c) corrects the canonical model's underconfidence and
# must NOT be reused here (plan's central warning). The live model is
# suspected OVERconfident (visibly hotter than torp's WP on the same Q4
# sequence per AFL_CHAIN_PARQUET_PLAN), so expect the opposite sign regime
# (b possibly < 1) -- that is a finding to report, not a bug to "fix" by
# tweaking the fit.
#
# Protocol (mirrors FABLE-RECAL-PLAN.md §7 Path B + §8 gate protocol v2,
# reusing train_lib.R's gate-v2 machinery directly rather than re-deriving it):
#   1. Train a TEMPORAL twin of the chain-v4 model on seasons < gate_season
#      (2021:2024), using honest cross-fitted OOS EP predictions for xmargin
#      on the training window (mirrors fit_wp_temporal_variant()'s EP step).
#   2. Score the held-out gate_season (2025) with EP features from the
#      TEMPORAL EP model (never saw gate_season) -- honest recent-season OOS.
#   3. Print RAW (uncalibrated) pooled slopes first -- the direction of
#      miscalibration is itself the finding this plan exists to measure.
#   4. Cross-fit calibrate (wp_crossfit_calibrate(), leverage_interaction_v1
#      form) and gate the pooled OOF slopes: all within +/-0.10 of 1,
#      q4close within +/-0.25 (validate_wp_temporal_slope() -- aborts loudly
#      and publishes nothing on a breach).
#   5. On pass: full-gate-season refit (both folds, deduped convention, the
#      shipped artifact) -> publish live_wp_calibration.json to torpmodels'
#      core-models release (same repo/tag as wp_calibration.rds's sibling;
#      the worker's OWN model JSONs -- wp-model-chain.json etc. -- are
#      instead uploaded straight to Cloudflare R2 via a manual `wrangler r2
#      object put`, never committed/released on GitHub -- see the session
#      report for why this script still targets the GitHub release).
#
# Run from torpmodels root: Rscript data-raw/05-live-wp-model/fit_live_wp_calibration.R
# Local-only (no upload): Rscript data-raw/05-live-wp-model/fit_live_wp_calibration.R --no-upload

library(devtools)
library(data.table)
library(xgboost)
library(jsonlite)
library(cli)

args <- commandArgs(trailingOnly = TRUE)
upload <- !("--no-upload" %in% args)

# --- Locate + load dev torp, dev torpmodels, the training library ----------

torp_paths <- c("../torp", "../../torp", "../../../torp", "C:/dev/torpverse/torp")
torp_path <- NA_character_
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    torp_path <- p
    break
  }
}
if (is.na(torp_path)) stop("Cannot find torp package (tried ../torp, ../../torp, ../../../torp, C:/dev/torpverse/torp).")
cli::cli_inform("Loading dev torp from {torp_path}...")
devtools::load_all(torp_path)

cli::cli_inform("Loading dev torpmodels from '.'...")
devtools::load_all(".")

source("data-raw/lib/train_lib.R")

torp_sha <- tryCatch(
  system2("git", c("-C", torp_path, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE),
  error = function(e) NA_character_
)
if (length(torp_sha) != 1 || !nzchar(torp_sha)) torp_sha <- NA_character_
torpmodels_sha <- tryCatch(
  system2("git", c("rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE),
  error = function(e) NA_character_
)
if (length(torpmodels_sha) != 1 || !nzchar(torpmodels_sha)) torpmodels_sha <- NA_character_

# --- 1. Load + prep data ------------------------------------------------------

cli::cli_h1("Fitting live-WP-chain calibration")

seasons <- default_training_seasons()          # 2021:(current_season - 1), e.g. 2021:2025
gate_season <- max(seasons)                     # 2025 -- last completed season
cli::cli_inform("Training window: {min(seasons)}-{max(seasons)} | gate season: {gate_season}")

model_data_epv <- load_training_pbp(seasons)
# clean_model_data_epv() already restricts to EPV_RELEVANT_DESCRIPTIONS, which
# is the IDENTICAL 27-description whitelist train_live_wp_chain_v4.R's own
# `relevant_desc` re-filters to -- so model_data_epv IS the chain-v4 "dt_rel"
# already; no further description filter needed here.
if (!"season" %in% names(model_data_epv)) {
  model_data_epv[, season := as.numeric(substr(match_id, 5, 8))]
}

train_data <- model_data_epv[season < gate_season]
gate_data  <- model_data_epv[season == gate_season]
if (nrow(train_data) == 0) cli::cli_abort("No rows with season < {gate_season}")
if (nrow(gate_data) == 0) cli::cli_abort("No rows for gate_season {gate_season}")
cli::cli_inform("Train rows (season < {gate_season}): {nrow(train_data)} across {uniqueN(train_data$torp_match_id)} matches")
cli::cli_inform("Gate rows (season == {gate_season}): {nrow(gate_data)} across {uniqueN(gate_data$torp_match_id)} matches")

# --- 2. Temporal EP variant: train on seasons < gate_season only -------------
# Mirrors fit_wp_temporal_variant()'s EP step exactly (same params, same fold
# construction) so xmargin on both windows is honestly out-of-sample.

cli::cli_h2("Temporal EP variant (train < {gate_season})")
ep_fit <- fit_ep(train_data, params = ep_params())

row_folds_train <- make_match_folds(train_data$torp_match_id)
ep_oos_train <- cv_ep_oos_preds(ep_fit$X, ep_fit$y, ep_fit$folds, row_folds_train,
                                ep_params(), ep_fit$optimal_nrounds)
exp_pts_train <- round(-6 * ep_oos_train[, "opp_goal"] - ep_oos_train[, "opp_behind"] +
                         ep_oos_train[, "behind"] + 6 * ep_oos_train[, "goal"], 5)

# Gate season: scored by the TEMPORAL EP model directly (never saw gate_season
# rows in training) -- honest recent-season OOS, same pattern score_wp_rows()
# uses for the canonical model's gate scoring.
epv_vars_gate <- gate_data |> torp:::select_epv_model_vars()
abort_on_feature_na(epv_vars_gate, "EP (gate season score)")
X_ep_gate <- stats::model.matrix(~ . + 0, data = epv_vars_gate, na.action = na.pass)
stopifnot(nrow(X_ep_gate) == nrow(epv_vars_gate))
ep_preds_gate <- predict(ep_fit$model, X_ep_gate)
if (!is.matrix(ep_preds_gate)) ep_preds_gate <- matrix(ep_preds_gate, ncol = 5, byrow = TRUE)
colnames(ep_preds_gate) <- c("opp_goal", "opp_behind", "behind", "goal", "no_score")
exp_pts_gate <- round(-6 * ep_preds_gate[, "opp_goal"] - ep_preds_gate[, "opp_behind"] +
                        ep_preds_gate[, "behind"] + 6 * ep_preds_gate[, "goal"], 5)

train_data[, exp_pts := exp_pts_train]
gate_data[, exp_pts := exp_pts_gate]

# --- 3. Build chain-v4 possession-POV features --------------------------------
# `home`/`points_diff` are ALREADY possession-POV (team_id_mdl vs home_team_id)
# as computed by clean_model_data_epv()'s add_epv_team_vars_dt() -- identical
# semantics to train_live_wp_chain_v4.R's own is_home/margin_poss, so no need
# to recompute them from home_points/away_points by hand. game_seconds uses
# v4's own (period-1)*2000+period_seconds convention (NOT torp's
# est_match_elapsed -- that's a different, stoppage-adjusted clock reserved
# for the leverage weight below), matching the actually-deployed model.
build_chain_features <- function(dt) {
  dt[, is_home := home]
  dt[, margin_poss := points_diff]
  dt[, xmargin := margin_poss + exp_pts]
  dt[, game_seconds := (period - 1L) * 2000 + period_seconds]
  dt
}
train_data <- build_chain_features(train_data)
gate_data  <- build_chain_features(gate_data)

wp_train <- train_data[!is.na(label_wp) & !is.na(period_seconds) & !is.na(xmargin)]
wp_gate  <- gate_data[!is.na(label_wp) & !is.na(period_seconds) & !is.na(xmargin)]
cli::cli_inform("wp_train: {nrow(wp_train)} rows | wp_gate: {nrow(wp_gate)} rows")

# --- 4. Train the temporal chain-v4 twin --------------------------------------
# Faithful port of train_live_wp_chain_v4.R's own architecture (same feature
# order, monotone constraint, hyperparameters, seed) -- only the data window
# changes (season < gate_season instead of all history). Draws (label_wp ==
# 0.5) are NOT dropped from training here, matching the real deployed
# script's convention (binary:logistic with a 0.5 soft label) -- the S5
# draw-exclusion finding is specific to the canonical WP model and wasn't
# re-validated for this architecture, so the twin should mirror what's
# actually live, not silently adopt a newer convention.
feature_cols <- c("xmargin", "period", "period_seconds", "game_seconds", "is_home")
X_train <- as.matrix(wp_train[, ..feature_cols])
y_train <- wp_train$label_wp
dtrain <- xgboost::xgb.DMatrix(data = X_train, label = y_train)

set.seed(42)
unique_matches <- unique(wp_train$torp_match_id)
match_fold <- sample(rep(1:5, length.out = length(unique_matches)))
names(match_fold) <- unique_matches
row_fold <- match_fold[wp_train$torp_match_id]
cv_folds <- lapply(1:5, function(k) which(row_fold == k))

mono_vec <- c(1L, 0L, 0L, 0L, 0L)
names(mono_vec) <- feature_cols

chain_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  tree_method = "hist",
  eta = 0.1,
  max_depth = 4,
  min_child_weight = 50,
  subsample = 0.85,
  colsample_bytree = 0.85,
  gamma = 0,
  monotone_constraints = paste0("(", paste(mono_vec, collapse = ","), ")")
)

cli::cli_h2("Training temporal chain-v4 twin ({min(train_data$season)}-{max(train_data$season)})")
set.seed(42)
cv <- xgboost::xgb.cv(
  params = chain_params, data = dtrain, nrounds = 500, folds = cv_folds,
  early_stopping_rounds = 20, print_every_n = 25, verbose = 1
)
best_nr <- which.min(cv$evaluation_log$test_logloss_mean)
cli::cli_inform("Temporal twin optimal nrounds: {best_nr}, CV logloss: {round(min(cv$evaluation_log$test_logloss_mean), 6)}")

set.seed(42)
twin_model <- xgboost::xgb.train(params = chain_params, data = dtrain, nrounds = best_nr, verbose = 1)

# --- 5. Score the gate season with the temporal twin --------------------------

X_gate <- as.matrix(wp_gate[, ..feature_cols])
gate_preds <- predict(twin_model, X_gate)
gate_labels <- wp_gate$label_wp
gate_meta <- wp_gate[, .(period, points_diff = margin_poss, est_match_elapsed,
                         est_match_remaining, match_id)]
gate_meta <- as.data.frame(gate_meta)

# --- 6. RAW slopes FIRST -- the direction of miscalibration is the finding ----

cli::cli_h1("RAW (uncalibrated) live-model gate-season slopes")
slope_all_raw <- wp_gate_slope(gate_preds, gate_labels, gate_meta, cell = "all")
q4c_raw <- wp_gate_slope(gate_preds, gate_labels, gate_meta, cell = "q4close", detail = TRUE)
cli::cli_alert_info("slope_all (raw)     = {round(slope_all_raw, 4)}")
cli::cli_alert_info("slope_q4close (raw) = {round(q4c_raw$slope, 4)} (n = {q4c_raw$n} deduped obs, SE = {round(q4c_raw$se, 4)})")
wp_gate_cell_diagnostics(gate_preds, gate_labels, gate_meta, label = "raw, gate season")

# --- 7. Cross-fit calibrate + gate --------------------------------------------

temporal <- list(preds = gate_preds, labels = gate_labels, meta_cols = gate_meta)
half <- make_match_folds(gate_meta$match_id, k = 2L)
n_foldA_matches <- length(unique(gate_meta$match_id[half == 1L]))
n_foldB_matches <- length(unique(gate_meta$match_id[half == 2L]))
cli::cli_inform("Cross-fit split within gate season {gate_season}: fold A = {n_foldA_matches} matches ({sum(half == 1L)} rows), fold B = {n_foldB_matches} matches ({sum(half == 2L)} rows)")

cf <- wp_crossfit_calibrate(temporal, form = "leverage_interaction_v1")
cli::cli_inform(paste0(
  "Per-fold leverage-interaction fits: fold A a = {round(cf$calib_A$a, 4)}, b = {round(cf$calib_A$b, 4)}, c = {round(cf$calib_A$c, 4)} | ",
  "fold B a = {round(cf$calib_B$a, 4)}, b = {round(cf$calib_B$b, 4)}, c = {round(cf$calib_B$c, 4)}"
))

cli::cli_h1("Pooled out-of-fold calibrated slopes (gate check)")
gated <- validate_wp_temporal_slope(cf$oof_preds, cf$oof_labels, cf$oof_meta,
                                    threshold = 0.10, threshold_q4close = 0.25)
wp_gate_cell_diagnostics(cf$oof_preds, cf$oof_labels, cf$oof_meta, label = "pooled OOF (gate)")

# --- 8. Pass: full gate-season refit (shipped artifact) -----------------------

calib <- fit_wp_calibration_deduped(gate_preds, gate_labels, gate_meta, form = "leverage_interaction_v1")
cli::cli_inform("Full gate-season refit (shipped): a = {round(calib$a, 4)}, b = {round(calib$b, 4)}, c = {round(calib$c, 4)}")

w_shipped <- wp_leverage_weight(gate_meta$est_match_remaining / 60, gate_meta$points_diff)
calibrated_full <- apply_wp_calibration(gate_preds, calib, w = w_shipped)
wp_gate_cell_diagnostics(calibrated_full, gate_labels, gate_meta, label = "full refit (shipped)")

out_dir <- file.path("inst", "models", "core")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_path <- file.path(out_dir, "live_wp_calibration.json")

live_wp_calibration <- list(
  model = "wp_afl_chain_v4_poss",
  form = calib$form,
  a = calib$a, b = calib$b, c = calib$c,
  ramp_mins = 20, margin_cap = 18,
  cell = list(period = 4, margin_abs_max = 12),
  formula = "sigmoid(a + (b + c*w) * logit(p)), w = max(0, 1 - mins_remaining/ramp_mins) * max(0, 1 - abs(margin)/margin_cap)",
  fitted_on = "temporal-oos-full-gate-season-refit",
  gate_validated_on = "temporal-oos-cross-fit-pooled-oof",
  gate_season = gate_season,
  n_fit = nrow(gate_meta),
  n_foldA_matches = n_foldA_matches,
  n_foldB_matches = n_foldB_matches,
  slope_all_raw = slope_all_raw,
  slope_q4close_raw = q4c_raw$slope,
  slope_all_calibrated_oof = gated$slope_all,
  slope_q4close_calibrated_oof = gated$slope_q4close,
  trained_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  torp_sha = torp_sha,
  torpmodels_sha = torpmodels_sha,
  script = "fit_live_wp_calibration.R"
)
jsonlite::write_json(live_wp_calibration, out_path, auto_unbox = TRUE, digits = NA, pretty = TRUE)
cli::cli_alert_success("Saved: {out_path}")

reloaded <- jsonlite::fromJSON(out_path, simplifyVector = TRUE)
stopifnot(identical(reloaded$form, calib$form), is.finite(reloaded$a), is.finite(reloaded$b))
cli::cli_alert_success("Reload verification passed.")

if (upload) {
  cli::cli_h2("Publishing live_wp_calibration.json to torpmodels core-models")
  piggyback::pb_upload(out_path, repo = "peteowen1/torpmodels", tag = "core-models")
  cli::cli_alert_success("Published live_wp_calibration.json to peteowen1/torpmodels@core-models")
  cli::cli_alert_info("This is the versioned source of truth. The worker reads model JSON exclusively from Cloudflare R2 (env.R2 binding), not GitHub releases -- a follow-up manual step must mirror this file to R2, e.g.:")
  cli::cli_inform("  wrangler r2 object put inthegame-data/afl/live-wp-calibration.json --file={out_path} --remote")
} else {
  cli::cli_alert_warning("--no-upload: local artifact only, nothing published.")
}

cli::cli_h1("Done")
