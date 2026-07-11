# Train Chain-Aware Live WP Model v3 — xmargin feature + relevant-row filter
# ============================================================================
# v2 gave correct margin calibration but exp_pts was only 0.1% feature gain
# (model basically ignored it). Per Pete's suggestion: collapse margin +
# exp_pts into a single composite feature xmargin = margin + signed_exp_pts
# (signed relative to home team — positive when home's in possession with a
# good state, negative when away's in possession with a good state). This
# gives the model one clean predictor of "home's effective margin including
# current chain state" instead of asking it to discover the interaction.
#
# Also filter to EPV_RELEVANT_DESCRIPTIONS (possession-level chain actions)
# like the EPV model does — excludes Goal/Behind/Rushed rows and other admin
# events. Mirrors how the EP model is trained.
#
# Features (5 total, matching base model's feature count for regularisation):
#   xmargin, period, period_seconds, game_seconds, home

library(devtools)
library(data.table)
library(xgboost)
library(jsonlite)

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

# ── 1. Load data + EPV pipeline ──────────────────────────────────────────────
cli::cli_h1("Training Chain-Aware Live WP v3 (xmargin + relevant filter)")

cli::cli_alert_info("Loading chains + running EPV pipeline...")
chains <- torp::load_chains(TRUE, TRUE)
md <- chains |>
  torp::clean_pbp() |>
  torp:::clean_model_data_epv() |>
  torp:::clean_shots_data() |>
  torp::add_shot_vars() |>
  torp::add_epv_vars()
dt <- data.table::as.data.table(md)

cli::cli_alert_info("Raw rows: {nrow(dt)}")

# ── 2. Filter to EPV-relevant chain actions only ─────────────────────────────
# Mirrors torp's internal filter_relevant_descriptions — keeps the ~27
# possession-level action types (Kick, Handball, Mark, etc.), drops the
# admin/outcome rows (Goal, Behind, Rushed, Start/End quarter, etc.) that
# don't represent a player-actionable possession state.
relevant_desc <- c(
  "Ball Up Call", "Bounce", "Centre Bounce", "Contested Knock On", "Contested Mark",
  "Free Advantage", "Free For", "Free For: Before the Bounce", "Free For: In Possession",
  "Free For: Off The Ball", "Gather", "Gather From Hitout", "Gather from Opposition",
  "Ground Kick", "Handball", "Handball Received", "Hard Ball Get", "Hard Ball Get Crumb",
  "Kick", "Knock On", "Loose Ball Get", "Loose Ball Get Crumb", "Mark On Lead",
  "Out of Bounds", "Out On Full After Kick", "Ruck Hard Ball Get", "Uncontested Mark"
)
dt_rel <- dt[description %in% relevant_desc]
cli::cli_alert_info("Relevant rows: {nrow(dt_rel)} ({round(100*nrow(dt_rel)/nrow(dt), 1)}%)")

# ── 3. Build features ────────────────────────────────────────────────────────
required_cols <- c("period", "period_seconds", "home_points", "away_points",
                   "home_score", "away_score", "torp_match_id", "season",
                   "exp_pts", "team_id_mdl", "home_team_id")
miss <- setdiff(required_cols, names(dt_rel))
if (length(miss) > 0) cli::cli_abort("Missing: {paste(miss, collapse=', ')}")

dt_rel[, margin := home_points - away_points]
dt_rel[, game_seconds := (period - 1L) * 2000 + period_seconds]

# Sign exp_pts to home's perspective: if home team has possession, exp_pts
# is already from home's perspective (positive = home about to score). If
# away team has possession, flip the sign (positive exp_pts means away is
# about to score, which is bad for home).
dt_rel[, home_possession := team_id_mdl == home_team_id]
dt_rel[, home_exp_pts := ifelse(home_possession, exp_pts, -exp_pts)]
dt_rel[, xmargin := margin + home_exp_pts]

dt_rel[, label := data.table::fcase(
  home_score > away_score, 1,
  home_score == away_score, 0.5,
  home_score < away_score, 0,
  default = NA_real_
)]

wp <- dt_rel[!is.na(label) & !is.na(period_seconds) & !is.na(xmargin),
             .(xmargin, period, period_seconds, game_seconds, home = 1L,
               label, torp_match_id, season)]

n_matches <- uniqueN(wp$torp_match_id)
n_draws <- uniqueN(wp[label == 0.5]$torp_match_id)
cli::cli_alert_info("Training rows: {nrow(wp)} across {n_matches} matches ({min(wp$season)}-{max(wp$season)}), {n_draws} draws")
cli::cli_alert_info("xmargin range: [{round(min(wp$xmargin),1)}, {round(max(wp$xmargin),1)}]  mean={round(mean(wp$xmargin),2)}")

# ── 4. Symmetrize ────────────────────────────────────────────────────────────
wp_flip <- copy(wp)
wp_flip[, `:=`(xmargin = -xmargin, home = 0L, label = 1 - label)]
wp_flip[, torp_match_id := paste0(torp_match_id, "_flip")]
wp_sym <- rbindlist(list(wp, wp_flip))
cli::cli_alert_info("After symmetrization: {nrow(wp_sym)} rows")

# ── 5. Train (identical hyperparams to base model) ───────────────────────────
feature_cols <- c("xmargin", "period", "period_seconds", "game_seconds", "home")
X_train <- as.matrix(wp_sym[, ..feature_cols])
y_train <- wp_sym$label
dtrain <- xgb.DMatrix(data = X_train, label = y_train)

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

imp <- xgb.importance(feature_names = feature_cols, model = model)
cli::cli_h2("Variable importance")
for (i in seq_len(nrow(imp))) {
  cli::cli_alert_info("  {i}. {imp$Feature[i]}: Gain={round(imp$Gain[i] * 100, 1)}%")
}

# ── 6. Sample-state sanity check ─────────────────────────────────────────────
cli::cli_h1("Sample-state comparison: v3 vs base")

baseline_rds <- "C:/dev/torpverse/torpmodels/inst/models/core/wp_model_live.rds"
baseline <- if (file.exists(baseline_rds)) readRDS(baseline_rds) else NULL

samples <- data.table(
  scenario = c(
    "Q2 end tied, neutral",
    "Q2 end tied, home attacking EPV=3",
    "Q2 end tied, away attacking EPV=3",
    "Q3 end down 20, neutral",
    "Q4 28min up 30, neutral",
    "Q4 30min tied, home about to score (EPV=3)",
    "Q4 30min tied, away about to score (EPV=3)",
    "HT down 10",
    "HT up 10"
  ),
  margin         = c(0, 0, 0, -20, 30, 0, 0, -10, 10),
  period         = c(2L, 2L, 2L, 3L, 4L, 4L, 4L, 2L, 2L),
  period_seconds = c(1800, 1800, 1800, 1800, 1680, 1800, 1800, 1800, 1800),
  home_exp_pts   = c(0, 3, -3, 0, 0, 3, -3, 0, 0)  # signed for home
)
samples[, xmargin := margin + home_exp_pts]
samples[, game_seconds := (period - 1L) * 2000 + period_seconds]
samples[, home := 1L]

Xs <- as.matrix(samples[, ..feature_cols])
samples[, wp_v3 := round(predict(model, Xs) * 100, 1)]

if (!is.null(baseline)) {
  # Base model uses margin only — for comparison feed it straight margin (no EPV)
  base_cols <- c("margin", "period", "period_seconds", "game_seconds", "home")
  Xs_base <- as.matrix(samples[, ..base_cols])
  samples[, wp_base := round(predict(baseline, Xs_base) * 100, 1)]
  print(samples[, .(scenario, margin, home_exp_pts, xmargin, wp_base, wp_v3)])
} else {
  print(samples[, .(scenario, margin, home_exp_pts, xmargin, wp_v3)])
}

# ── 7. Richness check ────────────────────────────────────────────────────────
cli::cli_h1("Richness check")
test_mid <- dt_rel[season == max(season) & round_number == max(round_number, na.rm = TRUE),
                   unique(match_id)][1]
if (is.na(test_mid)) test_mid <- dt_rel$match_id[1]
cli::cli_alert_info("Test match: {test_mid}")

mdt <- dt_rel[match_id == test_mid][order(period, period_seconds)]
mdt[, `:=`(margin = home_points - away_points, game_seconds = (period - 1L) * 2000 + period_seconds, home = 1L)]
mdt[, home_possession := team_id_mdl == home_team_id]
mdt[, home_exp_pts := ifelse(home_possession, exp_pts, -exp_pts)]
mdt[, xmargin := margin + home_exp_pts]
mdt_f <- mdt[!is.na(xmargin)]
mdt_X <- as.matrix(mdt_f[, ..feature_cols])
mdt_f[, wp := predict(model, mdt_X)]
mdt_f[, wp_delta := abs(wp - shift(wp))]
shifted <- sum(mdt_f$wp_delta > 0.001, na.rm = TRUE)
total <- sum(!is.na(mdt_f$wp_delta))
cli::cli_alert_info("Consecutive relevant rows: {total}. With |wp_delta|>0.001: {shifted} ({round(100*shifted/total, 2)}%)")

# ── 8. Export JSON (using xgb.dump tempfile pattern like panna does) ────────
cli::cli_h1("Exporting v3 JSON")

tmp_raw_json <- tempfile(fileext = ".json")
xgboost::xgb.dump(model, fname = tmp_raw_json, dump_format = "json")
trees_nested <- jsonlite::fromJSON(tmp_raw_json, simplifyDataFrame = FALSE, simplifyVector = FALSE)
cli::cli_alert_info("xgb.dump produced {length(trees_nested)} trees")

# Pull base_score from booster config
base_score <- tryCatch({
  cfg_raw <- xgboost::xgb.config(model)
  if (is.character(cfg_raw)) {
    cfg <- jsonlite::fromJSON(cfg_raw, simplifyDataFrame = FALSE, simplifyVector = FALSE)
    as.numeric(cfg$learner$learner_model_param$base_score %||% 0.5)
  } else 0.5
}, error = function(e) 0.5)
cli::cli_alert_info("base_score: {round(base_score, 5)}")

envelope <- list(
  model_type = "wp_afl_chain_v3",
  objective = "binary:logistic",
  num_class = 1L,
  feature_names = feature_cols,
  nrounds = optimal_nrounds,
  base_score = base_score,
  n_matches = n_matches,
  trained_on = paste0(min(wp$season), "-", max(wp$season)),
  trees = trees_nested,
  exported_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
)

out_json <- "C:/dev/torpverse/torpmodels/data-raw/05-live-wp-model/wp-model-chain.json"
jsonlite::write_json(envelope, out_json, auto_unbox = TRUE, digits = NA, pretty = FALSE)
cli::cli_alert_success("Saved: {out_json} ({round(file.size(out_json)/1024, 1)} KB)")

# Verify reload
reloaded <- jsonlite::fromJSON(out_json, simplifyDataFrame = FALSE, simplifyVector = FALSE)
stopifnot(length(reloaded$trees) == length(trees_nested))
cli::cli_alert_success("Reload verification passed.")
