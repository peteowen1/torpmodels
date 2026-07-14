# h2o_automl_compare.R
# =====================================================================
# Standalone curiosity check (NOT part of FABLE-MATCH-MAE-PLAN.md): how
# does a literal h2o::h2o.automl() stack up against the production
# GAM/XGB/blend match-prediction stack, on identical features and
# leak-safe temporal splits?
#
# Usage:
#   Rscript h2o_automl_compare.R [runtime_secs] [splits]
#     runtime_secs : per-h2o.automl() time budget in seconds (default 600)
#     splits       : comma-separated subset of "A,B" (default "A,B")
#
# Reuses .compute_metrics() from ../rolling_lib.R verbatim so the numbers
# here are computed identically to the champion numbers already measured
# on the same 2026 rolling-OOS screen.
# =====================================================================

args <- commandArgs(trailingOnly = TRUE)
runtime_secs  <- if (length(args) >= 1) as.integer(args[1]) else 600L
splits_to_run <- if (length(args) >= 2) strsplit(args[2], ",")[[1]] else c("A", "B")

cat(sprintf("[%s] Starting h2o AutoML comparison: runtime_secs=%d splits=%s\n",
            format(Sys.time()), runtime_secs, paste(splits_to_run, collapse = ",")))

suppressPackageStartupMessages({
  library(h2o)
  library(dplyr)
  library(MLmetrics)
})

# h2o masks several base/stats functions on attach (ifelse, %in%, is.numeric,
# is.character, round, cor, sd, var, apply, colnames, log*, trunc, signif).
# This script operates on plain R data frames/vectors throughout (H2OFrame
# ops always go through explicit h2o:: calls) — restore base semantics for
# the handful we use so a masked ifelse()/%in%() doesn't silently misbehave
# on ordinary vectors.
ifelse     <- base::ifelse
`%in%`     <- base::`%in%`
is.numeric <- base::is.numeric
is.character <- base::is.character
round      <- base::round

exp_dir <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
out_dir <- file.path(exp_dir, "h2o_automl")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# .compute_metrics() only (MAE/RMSE/Brier/logloss/accuracy/slope/sd_ratio/
# close_mae) — reused verbatim so these numbers are computed exactly like
# the champion's.
source(file.path(exp_dir, "rolling_lib.R"))

# --- Load cached feature dataset (built once this morning by a parallel
# effort; do NOT rebuild — rebuilding hits live AFL-API/weather calls) ---
cache_path <- file.path(exp_dir, "results", "team_mdl_df_cache.rds")
stopifnot(file.exists(cache_path))
team_mdl_df <- as.data.frame(readRDS(cache_path))
cat(sprintf("Loaded team_mdl_df_cache.rds: %d rows, %d cols\n", nrow(team_mdl_df), ncol(team_mdl_df)))

MATCH_MIN_DATA_SEASON <- 2021L  # torp::constants_match.R; cache already respects the
                                 # MATCH_MIN_DATA_ROUND=14 floor for 2021 upstream

# Exact production feature set (torp/R/match_train.R .train_xgb_fixed() base_cols) —
# same columns for h2o, no new features (that would confound the comparison).
feature_cols <- c(
  "team_type_fac", "game_year_decimal.x", "game_prop_through_year.x",
  "game_prop_through_month.x", "game_prop_through_day.x",
  "epr_diff", "epr_recv_diff", "epr_disp_diff", "epr_spoil_diff", "epr_hitout_diff",
  "torp_diff", "psr_diff", "osr_diff", "dsr_diff",
  "log_dist_diff", "familiarity_diff", "days_rest_diff_fac"
)
stopifnot(all(feature_cols %in% names(team_mdl_df)))

# Strip stray attributes (e.g. game_prop_through_month.x carries a leftover
# `names` attribute from an upstream vapply) before handing columns to h2o.
for (col in feature_cols) {
  if (is.numeric(team_mdl_df[[col]])) team_mdl_df[[col]] <- unname(team_mdl_df[[col]])
}

# Completed-match universe: non-NA win, season >= 2021.
completed_mask <- !is.na(team_mdl_df$win) & team_mdl_df$season.x >= MATCH_MIN_DATA_SEASON
cat(sprintf("Completed matches (season >= %d): %d rows (%d matches)\n",
            MATCH_MIN_DATA_SEASON, sum(completed_mask), sum(completed_mask) / 2))

# --- fuse_preds(): replicate torp:::.format_match_preds()'s home/away
# averaging (match_model.R) so h2o predictions are combined into one
# row/match exactly the way the production pipeline turns its 2-rows-per-
# match long format into 1-row/match predictions: away-perspective raw
# predictions are sign/complement-flipped to the home perspective, then
# averaged with the home-perspective row. Skipping this step would silently
# score h2o on 2x too many "matches" (double-counting each game) and would
# not match how margin/win-prob are actually blended in production. ---
fuse_preds <- function(df, pred_margin_col, pred_win_col) {
  d <- df[, c("match_id", "team_type_fac", "score_diff", pred_margin_col, pred_win_col)]
  names(d)[4:5] <- c("pred_margin", "pred_win")
  home <- d[d$team_type_fac == "home", c("match_id", "score_diff", "pred_margin", "pred_win")]
  away <- d[d$team_type_fac == "away", c("match_id", "score_diff", "pred_margin", "pred_win")]
  away$score_diff  <- -away$score_diff
  away$pred_margin <- -away$pred_margin
  away$pred_win    <- 1 - away$pred_win
  names(away)[2:4] <- paste0(names(away)[2:4], "_away")
  merged <- merge(home, away, by = "match_id")
  stopifnot(nrow(merged) == nrow(home), nrow(merged) == nrow(away))
  data.frame(
    match_id    = merged$match_id,
    margin      = merged$score_diff,
    pred_margin = (merged$pred_margin + merged$pred_margin_away) / 2,
    pred_win    = (merged$pred_win + merged$pred_win_away) / 2,
    home_win    = ifelse(merged$score_diff > 0, 1, ifelse(merged$score_diff == 0, 0.5, 0))
  )
}

# First non-StackedEnsemble leaderboard row (or NULL if none exists — guard
# for a short runtime budget or a Windows h2o build without every backend).
best_non_stacked <- function(aml) {
  lb <- as.data.frame(h2o::h2o.get_leaderboard(aml, extra_columns = "algo"))
  lb$algo <- as.character(lb$algo)
  non_stacked <- lb[lb$algo != "StackedEnsemble", , drop = FALSE]
  if (nrow(non_stacked) == 0) return(NULL)
  h2o::h2o.getModel(as.character(non_stacked$model_id[1]))
}

safe_varimp <- function(model) {
  tryCatch(h2o::h2o.varimp(model), error = function(e) {
    cat(sprintf("  (varimp unavailable for %s: %s)\n", model@model_id, conditionMessage(e)))
    NULL
  })
}

h2o::h2o.init(nthreads = 4, max_mem_size = "4G")
h2o::h2o.no_progress()

all_results <- list()
results_path <- file.path(out_dir, "h2o_results.rds")

run_split <- function(split_name, train_mask, test_mask) {
  n_train_matches <- sum(train_mask & completed_mask) / 2
  n_test_matches  <- sum(test_mask) / 2
  cat(sprintf("\n==== Split %s: train=%d matches, test=%d matches ====\n",
              split_name, n_train_matches, n_test_matches))

  train_df <- team_mdl_df[train_mask & completed_mask, ]
  test_df  <- team_mdl_df[test_mask, ]

  train_h2o <- h2o::as.h2o(train_df[, c(feature_cols, "score_diff", "win", "weightz", "match_id")])
  # Prediction-only frame: feature_cols alone. Deliberately excludes score_diff/
  # win/weightz — h2o's Model.adaptTestForTrain() hard-errors if a same-named
  # column differs in type between train and test frames (e.g. numeric "win"
  # here vs the factor "win" the win-prob model trained on), even though
  # h2o.predict() never needs the response/weight columns to produce
  # predictions. Fusion/metrics are computed in R from test_df directly, so
  # nothing is lost by leaving them out of test_h2o.
  test_h2o  <- h2o::as.h2o(test_df[, feature_cols])

  n_draws <- sum(train_df$win == 0.5)
  cat(sprintf("Dropping %d draw rows (win==0.5) from win-prob TRAINING only ", n_draws))
  cat("(evaluation keeps draws, home_win=0.5, same as .compute_metrics())\n")
  cls_train_df  <- train_df[train_df$win %in% c(0, 1), ]
  cls_train_h2o <- h2o::as.h2o(cls_train_df[, c(feature_cols, "win", "weightz", "match_id")])
  cls_train_h2o$win <- h2o::as.factor(cls_train_h2o$win)

  # --- Margin regression AutoML ---
  cat(sprintf("[%s] Margin AutoML (score_diff) starting, budget=%ds\n", format(Sys.time()), runtime_secs))
  t0 <- Sys.time()
  aml_margin <- h2o::h2o.automl(
    x = feature_cols, y = "score_diff", training_frame = train_h2o,
    weights_column = "weightz", max_runtime_secs = runtime_secs, max_models = NULL,
    seed = 1234, exclude_algos = c("DeepLearning"),
    project_name = paste0("torp_margin_", split_name, "_", runtime_secs)
  )
  cat(sprintf("[%s] Margin AutoML done (%.1fs elapsed). Leader: %s\n",
              format(Sys.time()), as.numeric(difftime(Sys.time(), t0, units = "secs")),
              aml_margin@leader@model_id))

  # --- Win-prob classification AutoML ---
  cat(sprintf("[%s] Win-prob AutoML (win) starting, budget=%ds\n", format(Sys.time()), runtime_secs))
  t0 <- Sys.time()
  aml_win <- h2o::h2o.automl(
    x = feature_cols, y = "win", training_frame = cls_train_h2o,
    weights_column = "weightz", max_runtime_secs = runtime_secs, max_models = NULL,
    seed = 1234, exclude_algos = c("DeepLearning"),
    project_name = paste0("torp_win_", split_name, "_", runtime_secs)
  )
  cat(sprintf("[%s] Win-prob AutoML done (%.1fs elapsed). Leader: %s\n",
              format(Sys.time()), as.numeric(difftime(Sys.time(), t0, units = "secs")),
              aml_win@leader@model_id))

  margin_leader <- aml_margin@leader
  win_leader    <- aml_win@leader
  margin_best   <- best_non_stacked(aml_margin)
  win_best      <- best_non_stacked(aml_win)
  if (is.null(margin_best)) margin_best <- margin_leader  # leader already non-stacked (or no other model)
  if (is.null(win_best))    win_best    <- win_leader

  predict_col <- function(model, newdata, col) as.data.frame(h2o::h2o.predict(model, newdata))[[col]]

  test_df$pred_margin_leader <- predict_col(margin_leader, test_h2o, "predict")
  test_df$pred_margin_best   <- predict_col(margin_best,   test_h2o, "predict")
  test_df$pred_win_leader    <- predict_col(win_leader,    test_h2o, "p1")
  test_df$pred_win_best      <- predict_col(win_best,      test_h2o, "p1")

  preds_leader <- fuse_preds(test_df, "pred_margin_leader", "pred_win_leader")
  preds_best   <- fuse_preds(test_df, "pred_margin_best",   "pred_win_best")

  metrics_leader <- .compute_metrics(preds_leader)
  metrics_best   <- .compute_metrics(preds_best)

  result <- list(
    split = split_name,
    n_train = n_train_matches,
    n_test = n_test_matches,
    margin_leader_id = margin_leader@model_id,
    margin_best_id   = margin_best@model_id,
    win_leader_id    = win_leader@model_id,
    win_best_id      = win_best@model_id,
    metrics_leader = metrics_leader,
    metrics_best   = metrics_best,
    varimp_margin_leader = safe_varimp(margin_leader),
    varimp_margin_best   = safe_varimp(margin_best),
    varimp_win_leader    = safe_varimp(win_leader),
    varimp_win_best      = safe_varimp(win_best),
    leaderboard_margin = as.data.frame(h2o::h2o.get_leaderboard(aml_margin, extra_columns = "algo")),
    leaderboard_win    = as.data.frame(h2o::h2o.get_leaderboard(aml_win, extra_columns = "algo"))
  )

  cat(sprintf("\n-- %s LEADER combo: margin=%s win=%s\n", split_name, result$margin_leader_id, result$win_leader_id))
  cat(sprintf("   MAE=%.2f RMSE=%.2f slope=%.3f sd_ratio=%.3f close_mae=%.2f (n_close=%d)\n",
              metrics_leader$mae, metrics_leader$rmse, metrics_leader$slope,
              metrics_leader$sd_ratio, metrics_leader$close_mae, metrics_leader$close_n))
  cat(sprintf("   Brier=%.4f logloss=%.4f acc=%.1f\n",
              metrics_leader$brier, metrics_leader$logloss, metrics_leader$accuracy))
  cat(sprintf("-- %s BEST-SINGLE combo: margin=%s win=%s\n", split_name, result$margin_best_id, result$win_best_id))
  cat(sprintf("   MAE=%.2f RMSE=%.2f slope=%.3f sd_ratio=%.3f close_mae=%.2f (n_close=%d)\n",
              metrics_best$mae, metrics_best$rmse, metrics_best$slope,
              metrics_best$sd_ratio, metrics_best$close_mae, metrics_best$close_n))
  cat(sprintf("   Brier=%.4f logloss=%.4f acc=%.1f\n\n",
              metrics_best$brier, metrics_best$logloss, metrics_best$accuracy))

  result
}

if ("A" %in% splits_to_run) {
  train_mask_A <- team_mdl_df$season.x < 2026
  test_mask_A  <- team_mdl_df$season.x == 2026 & !is.na(team_mdl_df$win)
  all_results$A <- run_split("A_static", train_mask_A, test_mask_A)
  saveRDS(all_results, results_path)
}

if ("B" %in% splits_to_run) {
  train_mask_B <- team_mdl_df$season.x < 2026 |
    (team_mdl_df$season.x == 2026 & team_mdl_df$round_number.x <= 9)
  test_mask_B <- team_mdl_df$season.x == 2026 & team_mdl_df$round_number.x >= 10 &
    team_mdl_df$round_number.x <= 19 & !is.na(team_mdl_df$win)
  all_results$B <- run_split("B_recent_warm", train_mask_B, test_mask_B)
  saveRDS(all_results, results_path)
}

cat(sprintf("[%s] All done. Results saved to %s\n", format(Sys.time()), results_path))

h2o::h2o.shutdown(prompt = FALSE)
