# Train Live EP Model v2 (13 features)
# ====================================
# Compared to v1 (8 features):
#   KEEP:   goal_x, y, period_seconds, est_qtr_remaining, est_match_remaining, shot_row
#   DROP:   lag_goal_x (conflates transition with state value)
#           phase_of_play_set_shot (replaced by full phase_of_play)
#   ADD:    play_type_handball, play_type_kick, play_type_reception (from full model)
#           phase_of_play_handball_received, phase_of_play_hard_ball,
#           phase_of_play_loose_ball, phase_of_play_set_shot (full categorical)
#           chain_action_num (position within chain — new feature)
#
# Total: 13 features (6 numeric + 3 play_type + 4 phase_of_play)

library(xgboost)
library(data.table)

# Load torp
torp_paths <- c("../../torp", "../torp", "../../../torp")
loaded <- FALSE
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p)
    loaded <- TRUE
    break
  }
}
if (!loaded) stop("Cannot find torp package.")

# ── 1. Load and prepare training data ────────────────────────
cli::cli_inform("Loading chains data...")
chains <- load_chains(TRUE, TRUE)

cli::cli_inform("Running clean_pbp...")
pbp <- clean_pbp(chains)

cli::cli_inform("Preparing EPV model data...")
model_data <- clean_model_data_epv(pbp)

# Add chain_action_num: row number within each (match, chain_number)
# Must be computed AFTER clean_model_data_epv filtering (some rows removed)
dt <- as.data.table(model_data)
dt[, chain_action_num := seq_len(.N), by = .(match_id, chain_number)]
cli::cli_inform("chain_action_num range: [{min(dt$chain_action_num)}, {max(dt$chain_action_num)}]")
cli::cli_inform("chain_action_num median: {median(dt$chain_action_num)}")

# ── 2. Define v1 and v2 feature sets ────────────────────────
v1_vars <- c(
  "est_qtr_remaining", "goal_x", "period_seconds", "y",
  "shot_row", "lag_goal_x", "est_match_remaining", "phase_of_play_set_shot"
)

v2_vars <- c(
  # Numeric (state-based, no lag features)
  "goal_x", "y", "period_seconds",
  "est_qtr_remaining", "est_match_remaining",
  "shot_row", "chain_action_num",
  # Play type (what is the player doing?)
  "play_type_handball", "play_type_kick", "play_type_reception",
  # Phase of play (how did they get the ball?)
  "phase_of_play_handball_received", "phase_of_play_hard_ball",
  "phase_of_play_loose_ball", "phase_of_play_set_shot"
)

cli::cli_inform("v1 features ({length(v1_vars)}): {paste(v1_vars, collapse = ', ')}")
cli::cli_inform("v2 features ({length(v2_vars)}): {paste(v2_vars, collapse = ', ')}")

# Build training matrices
X_v1 <- model.matrix(~ . + 0, data = as.data.frame(dt)[, v1_vars])
X_v2 <- model.matrix(~ . + 0, data = as.data.frame(dt)[, v2_vars])
y_train <- dt$label_ep

cli::cli_inform("Training rows: {nrow(X_v1)}")

# ── 3. Match-grouped CV (same folds for both) ───────────────
cli::cli_inform("Creating match-grouped CV folds...")
match_ids <- unique(dt$torp_match_id)
set.seed(1234)
match_folds <- sample(rep(1:5, length.out = length(match_ids)))
names(match_folds) <- match_ids
row_folds <- match_folds[dt$torp_match_id]
folds <- lapply(1:5, function(k) which(row_folds == k))

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

# ── 4. Train v1 (baseline) ──────────────────────────────────
cli::cli_inform("\n── Training v1 (8 features) ────────────────────────")
dm_v1 <- xgb.DMatrix(data = X_v1, label = y_train)
set.seed(1234)
cv_v1 <- xgb.cv(params = params, data = dm_v1, nrounds = 500, folds = folds,
                 early_stopping_rounds = 20, print_every_n = 50, verbose = 1)
best_v1 <- which.min(cv_v1$evaluation_log$test_mlogloss_mean)
loss_v1 <- min(cv_v1$evaluation_log$test_mlogloss_mean)
cli::cli_inform("v1 optimal nrounds: {best_v1}, CV mlogloss: {round(loss_v1, 6)}")

set.seed(1234)
model_v1 <- xgb.train(params = params, data = dm_v1, nrounds = best_v1, print_every_n = 50)

# ── 5. Train v2 (new features) ──────────────────────────────
cli::cli_inform("\n── Training v2 (14 features) ────────────────────────")
dm_v2 <- xgb.DMatrix(data = X_v2, label = y_train)
set.seed(1234)
cv_v2 <- xgb.cv(params = params, data = dm_v2, nrounds = 500, folds = folds,
                 early_stopping_rounds = 20, print_every_n = 50, verbose = 1)
best_v2 <- which.min(cv_v2$evaluation_log$test_mlogloss_mean)
loss_v2 <- min(cv_v2$evaluation_log$test_mlogloss_mean)
cli::cli_inform("v2 optimal nrounds: {best_v2}, CV mlogloss: {round(loss_v2, 6)}")

set.seed(1234)
model_v2 <- xgb.train(params = params, data = dm_v2, nrounds = best_v2, print_every_n = 50)

# ── 6. Compare on CV metrics ────────────────────────────────
cli::cli_inform("\n══ CV COMPARISON ══════════════════════════════════")
cli::cli_inform("v1 (8 feat):  mlogloss = {round(loss_v1, 6)}, nrounds = {best_v1}")
cli::cli_inform("v2 (14 feat): mlogloss = {round(loss_v2, 6)}, nrounds = {best_v2}")
improvement <- (loss_v1 - loss_v2) / loss_v1 * 100
cli::cli_inform("Improvement: {round(improvement, 2)}%")

# Variable importance for v2
imp <- xgb.importance(feature_names = colnames(X_v2), model = model_v2)
cli::cli_inform("\nv2 variable importance:")
for (i in seq_len(nrow(imp))) {
  cli::cli_inform("  {i}. {imp$Feature[i]}: Gain={round(imp$Gain[i] * 100, 1)}%")
}

# ── 7. Compare on Daicos R6 rows ────────────────────────────
cli::cli_inform("\n══ DAICOS R6 COMPARISON ═══════════════════════════")
mid <- dt[season == 2026 & round_number == 6 & grepl("Carl", home_team_name)][1, match_id]
match_dt <- dt[match_id == mid]
cat("Match:", mid, "  Rows:", nrow(match_dt), "\n")

# Score match rows with both models
score_ep <- function(model, X) {
  p <- predict(model, X)
  m <- matrix(p, ncol = 5, byrow = FALSE)
  -6 * m[, 1] - m[, 2] + m[, 3] + 6 * m[, 4]
}

mX_v1 <- model.matrix(~ . + 0, data = as.data.frame(match_dt)[, v1_vars])
mX_v2 <- model.matrix(~ . + 0, data = as.data.frame(match_dt)[, v2_vars])

match_dt[, ep_v1 := round(score_ep(model_v1, mX_v1), 3)]
match_dt[, ep_v2 := round(score_ep(model_v2, mX_v2), 3)]

# Also score with full 19-feature model for reference
full_model <- readRDS(file.path(
  normalizePath("~/OneDrive/Documents/torpverse/torpmodels/inst/models/core"),
  "ep_model.rds"))
full_vars <- select_epv_model_vars(match_dt)
mX_full <- model.matrix(~ . + 0, data = full_vars)
match_dt[, ep_full := round(score_ep(full_model, mX_full), 3)]

# Q4 15:00-15:15
nick_id <- "CD_I1023261"
fmtT <- function(s) paste0(s %/% 60, ":", formatC(s %% 60, width = 2, flag = "0"))

cat("\n=== Q4 15:00-15:15 ===\n")
cat(sprintf("%-3s Q %-5s %-22s %4s %4s %3s %7s %7s %7s %7s %-6s %-6s\n",
            "", "", "Time", "Desc", "gx", "lgx", "chn",
            "EP_full", "EP_v1", "EP_v2", "v2-v1", "ptype", "phase"))
q4 <- match_dt[period == 4 & period_seconds >= 895 & period_seconds <= 920]
for (i in seq_len(nrow(q4))) {
  r <- q4[i]
  mk <- ifelse(r$player_id == nick_id, ">>>", "   ")
  cat(sprintf("%s %d %-5s %-22s %4.0f %4.0f %3d %+7.3f %+7.3f %+7.3f %+7.3f %-6s %-6s\n",
              mk, r$period, fmtT(r$period_seconds),
              substr(r$description, 1, 22),
              r$goal_x, r$lag_goal_x, r$chain_action_num,
              r$ep_full, r$ep_v1, r$ep_v2, r$ep_v2 - r$ep_v1,
              substr(as.character(r$play_type), 1, 6),
              substr(as.character(r$phase_of_play), 1, 6)))
}

# Daicos totals
nick <- match_dt[player_id == nick_id]
cat(sprintf("\n=== DAICOS TOTALS (%d rows) ===\n", nrow(nick)))
cat(sprintf("  EP_full (19 feat): cor=%.4f  mean|diff from v2|=%.3f\n",
            cor(nick$ep_full, nick$ep_v2), mean(abs(nick$ep_full - nick$ep_v2))))
cat(sprintf("  EP_v1   (8 feat):  cor=%.4f  mean|diff from v2|=%.3f\n",
            cor(nick$ep_v1, nick$ep_v2), mean(abs(nick$ep_v1 - nick$ep_v2))))

# Top differences v1 vs v2
nick[, v2_minus_v1 := ep_v2 - ep_v1]
top <- nick[order(-abs(v2_minus_v1))][1:15]
cat("\n=== TOP 15 v2-vs-v1 DIFFERENCES ===\n")
cat(sprintf("%-1s %-5s %-22s %7s %7s %7s %7s %-6s %-6s %4s %4s\n",
            "Q", "Time", "Description", "EP_full", "EP_v1", "EP_v2", "v2-v1", "ptype", "phase", "gx", "lgx"))
for (i in seq_len(nrow(top))) {
  r <- top[i]
  cat(sprintf("%-1d %-5s %-22s %+7.3f %+7.3f %+7.3f %+7.3f %-6s %-6s %4.0f %4.0f\n",
              r$period, fmtT(r$period_seconds),
              substr(r$description, 1, 22),
              r$ep_full, r$ep_v1, r$ep_v2, r$v2_minus_v1,
              substr(as.character(r$play_type), 1, 6),
              substr(as.character(r$phase_of_play), 1, 6),
              r$goal_x, r$lag_goal_x))
}

# ── 8. Save v2 model ────────────────────────────────────────
output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

rds_path <- file.path(output_dir, "ep_model_live_v2.rds")
saveRDS(model_v2, rds_path)
cli::cli_inform("\nSaved: {rds_path}")

# Export JSON for Worker
json_trees <- xgb.dump(model_v2, dump_format = "json")
json_parsed <- jsonlite::fromJSON(json_trees, simplifyVector = FALSE)

export <- list(
  trees = json_parsed,
  feature_names = colnames(X_v2),
  num_class = 5,
  class_labels = c("opp_goal", "opp_behind", "behind", "goal", "no_score"),
  num_rounds = best_v2,
  cv_mlogloss = loss_v2
)

json_path <- file.path(output_dir, "ep_model_live_v2.json")
jsonlite::write_json(export, json_path, auto_unbox = TRUE, pretty = FALSE)
cli::cli_inform("Saved JSON: {json_path} ({round(file.size(json_path)/1024, 1)} KB)")
