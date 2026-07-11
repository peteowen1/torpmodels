# Train Chain-Aware Live WP Model v4 — Possession-POV unified convention
# =======================================================================
# v4 unifies AFL with the possession-POV methodology used by football
# (pannaverse/panna/R/wp_model.R). The model predicts:
#
#   P(possession_team wins match | current state)
#
# not P(home wins). Features per row are ALL from possession team's POV:
#
#   margin_poss = possession_team_points - opponent_points  (sign by poss)
#   exp_pts     (already possession-POV from torp's EPV model)
#   xmargin     = margin_poss + exp_pts
#   is_home     = 1 if possession team is home team (home-advantage feature)
#   wp_label    = 1 if possession team won match, 0.5 draw, 0 if lost
#
# Benefits:
#   - EPV and WP share the same reference frame (possession team's POV)
#   - No sym-flip needed — data naturally has both home-poss and away-poss rows
#   - Consistent with football panna v5+ methodology
#   - Monotonicity on xmargin still enforced (non-decreasing)
#
# Output: wp-model-chain.json (overwrites v3).

library(devtools)
library(data.table)
library(xgboost)
library(jsonlite)

torp_paths <- c("../../torp", "../torp", "../../../torp",
                "C:/dev/torpverse/torp")
loaded <- FALSE
torp_path <- NA_character_
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p)
    loaded <- TRUE
    torp_path <- p
    cli::cli_alert_success("Loaded torp from {normalizePath(p)}")
    break
  }
}
if (!loaded) stop("Cannot find torp package.")

torp_sha <- tryCatch(
  system2("git", c("-C", torp_path, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE),
  error = function(e) NA_character_
)
if (length(torp_sha) != 1 || !nzchar(torp_sha)) torp_sha <- NA_character_

# ── 1. Load data + EPV pipeline ──────────────────────────────────────────────
cli::cli_h1("Training Chain-Aware Live WP v4 (possession-POV)")

cli::cli_alert_info("Loading chains + running EPV pipeline...")
chains <- torp::load_chains(TRUE, TRUE)
md <- chains |>
  torp::clean_pbp() |>
  torp:::clean_model_data_epv() |>
  torp:::clean_shots_data() |>
  torp::add_shot_vars() |>
  torp::add_epv_vars()
dt <- data.table::as.data.table(md)

# ── 2. Filter to EPV-relevant chain actions only ─────────────────────────────
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

# ── 3. Build features (POSSESSION-POV) ───────────────────────────────────────
required_cols <- c("period", "period_seconds", "home_points", "away_points",
                   "home_score", "away_score", "torp_match_id", "season",
                   "exp_pts", "team_id_mdl", "home_team_id")
miss <- setdiff(required_cols, names(dt_rel))
if (length(miss) > 0) cli::cli_abort("Missing: {paste(miss, collapse=', ')}")

# is_home: is the possession team the home team?
dt_rel[, is_home := as.integer(team_id_mdl == home_team_id)]

# margin_poss: possession team's current points - opponent's current points
dt_rel[, margin_poss := data.table::fifelse(is_home == 1L,
                                            home_points - away_points,
                                            away_points - home_points)]

# xmargin: possession-POV margin + possession-POV EPV (already possession-POV
# in torp's output, no sign flipping needed)
dt_rel[, xmargin := margin_poss + exp_pts]

dt_rel[, game_seconds := (period - 1L) * 2000 + period_seconds]

# wp_label: did possession team win match?
dt_rel[, wp_label := data.table::fcase(
  home_score == away_score, 0.5,
  is_home == 1L & home_score > away_score, 1,
  is_home == 1L & home_score < away_score, 0,
  is_home == 0L & away_score > home_score, 1,
  is_home == 0L & away_score < home_score, 0
)]

wp <- dt_rel[!is.na(wp_label) & !is.na(period_seconds) & !is.na(xmargin),
             .(xmargin, period, period_seconds, game_seconds, is_home,
               label = wp_label, torp_match_id, season)]

n_matches <- uniqueN(wp$torp_match_id)
cli::cli_alert_info("Training rows: {nrow(wp)} across {n_matches} matches ({min(wp$season)}-{max(wp$season)})")
cli::cli_alert_info("xmargin range: [{round(min(wp$xmargin),1)}, {round(max(wp$xmargin),1)}]  mean={round(mean(wp$xmargin),2)}")
cli::cli_alert_info("is_home split: {round(100*mean(wp$is_home), 1)}% home possession")
cli::cli_alert_info("Label balance: win={round(100*mean(wp$label == 1), 1)}%, draw={round(100*mean(wp$label == 0.5), 1)}%, loss={round(100*mean(wp$label == 0), 1)}%")

# ── 4. Train (match AFL v3 hyperparams + monotonicity) ──────────────────────
feature_cols <- c("xmargin", "period", "period_seconds", "game_seconds", "is_home")
X_train <- as.matrix(wp[, ..feature_cols])
y_train <- wp$label
dtrain <- xgb.DMatrix(data = X_train, label = y_train)

# Match-grouped CV folds
set.seed(42)
unique_matches <- unique(wp$torp_match_id)
match_fold <- sample(rep(1:5, length.out = length(unique_matches)))
names(match_fold) <- unique_matches
row_fold <- match_fold[wp$torp_match_id]
folds <- lapply(1:5, function(k) which(row_fold == k))

# Monotonicity: xmargin non-decreasing (higher → higher WP)
mono_vec <- c(1L, 0L, 0L, 0L, 0L)  # matches feature_cols order
names(mono_vec) <- feature_cols

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
  gamma = 0,
  monotone_constraints = paste0("(", paste(mono_vec, collapse = ","), ")")
)

cli::cli_alert_info("Running 5-fold match-grouped CV (possession-POV, monotonic xmargin)...")
set.seed(42)
cv <- xgb.cv(
  params = params, data = dtrain, nrounds = 500, folds = folds,
  early_stopping_rounds = 20, print_every_n = 25, verbose = 1
)
best_nr <- which.min(cv$evaluation_log$test_logloss_mean)
best_loss <- min(cv$evaluation_log$test_logloss_mean)
cli::cli_alert_info("Optimal nrounds: {best_nr}, CV logloss: {round(best_loss, 6)}")

set.seed(42)
model <- xgb.train(params = params, data = dtrain, nrounds = best_nr, verbose = 1)

imp <- xgb.importance(feature_names = feature_cols, model = model)
cli::cli_h2("Variable importance")
for (i in seq_len(nrow(imp))) {
  cli::cli_alert_info("  {i}. {imp$Feature[i]}: Gain={round(imp$Gain[i] * 100, 1)}%")
}

# ── 5. Sample-state sanity check ─────────────────────────────────────────────
cli::cli_h1("Sample-state checks")

# Scenarios from possession-POV. is_home=1 means "home team has the ball", 0 means away.
samples <- data.table(
  scenario = c(
    "Q2 end, home poss, tied (xm=0)",
    "Q2 end, home poss, home leading by 10 (xm=10)",
    "Q2 end, away poss, tied (xm=0)",
    "Q2 end, away poss, away leading by 10 (xm=10)",
    "Q4 30min, home poss, tied, about to score (xm=+3)",
    "Q4 30min, home poss, tied (xm=0)",
    "Q4 30min, home poss, down 20 (xm=-20)",
    "Q4 30min, away poss, down 20 (away=home in POV, xm=-20)",
    "Q4 30min, away poss, up 20 (xm=+20)",
    "Q4 30min, home poss, up 30 (xm=30)"
  ),
  margin_poss    = c(0, 10, 0, 10, 0, 0, -20, -20, 20, 30),
  exp_pts        = c(0,  0, 0,  0, 3, 0,   0,   0,  0,  0),
  period         = c(2L, 2L, 2L, 2L, 4L, 4L, 4L, 4L, 4L, 4L),
  period_seconds = c(1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800),
  is_home        = c(1L, 1L, 0L, 0L, 1L, 1L, 1L, 0L, 0L, 1L)
)
samples[, xmargin := margin_poss + exp_pts]
samples[, game_seconds := (period - 1L) * 2000 + period_seconds]

Xs <- as.matrix(samples[, ..feature_cols])
samples[, wp_poss := round(predict(model, Xs) * 100, 1)]
print(samples[, .(scenario, xmargin, is_home, wp_poss)])

# ── 6. Richness check ────────────────────────────────────────────────────────
cli::cli_h1("Richness check")
test_mid <- dt_rel[season == max(season) & round_number == max(round_number, na.rm = TRUE),
                   unique(match_id)][1]
if (is.na(test_mid)) test_mid <- dt_rel$match_id[1]
cli::cli_alert_info("Test match: {test_mid}")

mdt <- dt_rel[match_id == test_mid][order(period, period_seconds)]
mdt[, `:=`(
  is_home = as.integer(team_id_mdl == home_team_id),
  game_seconds = (period - 1L) * 2000 + period_seconds
)]
mdt[, margin_poss := data.table::fifelse(is_home == 1L,
                                         home_points - away_points,
                                         away_points - home_points)]
mdt[, xmargin := margin_poss + exp_pts]
mdt_f <- mdt[!is.na(xmargin)]
mdt_X <- as.matrix(mdt_f[, ..feature_cols])
mdt_f[, wp := predict(model, mdt_X)]
mdt_f[, wp_delta := abs(wp - shift(wp))]
shifted <- sum(mdt_f$wp_delta > 0.001, na.rm = TRUE)
total <- sum(!is.na(mdt_f$wp_delta))
cli::cli_alert_info("Consecutive rows: {total}. With |wp_delta|>0.001: {shifted} ({round(100*shifted/total, 2)}%)")

# ── 7. Export JSON ───────────────────────────────────────────────────────────
cli::cli_h1("Exporting v4 JSON")

tmp_raw_json <- tempfile(fileext = ".json")
xgboost::xgb.dump(model, fname = tmp_raw_json, dump_format = "json")
trees_nested <- jsonlite::fromJSON(tmp_raw_json, simplifyDataFrame = FALSE, simplifyVector = FALSE)
cli::cli_alert_info("{length(trees_nested)} trees")

base_score <- tryCatch({
  cfg <- xgboost::xgb.config(model)
  if (is.character(cfg)) cfg <- jsonlite::fromJSON(cfg, simplifyDataFrame = FALSE, simplifyVector = FALSE)
  bs_raw <- cfg$learner$learner_model_param$base_score
  as.numeric(gsub("[\\[\\]]", "", bs_raw, perl = TRUE))
}, error = function(e) 0.5)
cli::cli_alert_info("base_score: {round(base_score, 5)}")

envelope <- list(
  model_type = "wp_afl_chain_v4_poss",
  objective = "binary:logistic",
  num_class = 1L,
  feature_names = feature_cols,
  nrounds = best_nr,
  base_score = base_score,
  n_matches = n_matches,
  trained_on = paste0(min(wp$season), "-", max(wp$season)),
  trees = trees_nested,
  exported_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  torp_sha = torp_sha,
  script = "train_live_wp_chain_v4.R"
)

out_json <- "C:/dev/torpverse/torpmodels/data-raw/05-live-wp-model/wp-model-chain.json"
jsonlite::write_json(envelope, out_json, auto_unbox = TRUE, digits = NA, pretty = FALSE)
cli::cli_alert_success("Saved: {out_json} ({round(file.size(out_json)/1024, 1)} KB)")

reloaded <- jsonlite::fromJSON(out_json, simplifyDataFrame = FALSE, simplifyVector = FALSE)
stopifnot(length(reloaded$trees) == length(trees_nested))
cli::cli_alert_success("Reload verification passed.")
