# rolling_lib.R — Reusable rolling week-by-week OOS evaluation harness
# =====================================================================
# Extracted from train_match_models.R (FABLE-MATCH-MAE-PLAN.md WS0). The
# only leak-safe evaluator in the codebase (plan G1): train strictly on
# rounds before the test round, predict the test round, roll forward.
#
# Swappable gam_trainer / xgb_trainer let experiments/*.R (WS1-WS5) test
# structural variants without touching torp/R/*.R (plan G5). The default
# call — gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
# extra_feature_cols = NULL — reproduces train_match_models.R's original
# rolling loop exactly (byte-compatible default path).
#
# Assumes the caller has already devtools::load_all()'d torp (for
# .train_match_gams(), .train_match_xgb(), .format_match_preds(),
# torp_replace_teams()) and library()'d dplyr/xgboost/mgcv/cli/MLmetrics,
# exactly as train_match_models.R does before sourcing this file.

# .train_xgb_fixed ----

#' Train the 5-step XGBoost pipeline with fixed nrounds (no per-round CV)
#'
#' Mirrors torp:::.train_match_xgb()'s structure but skips per-round CV,
#' reusing nrounds pre-optimised once on pre-test-season data
#' (run_rolling_eval() does this via torp:::.train_match_xgb()).
#' `extra_feature_cols` (e.g. "elo_diff") are appended to the shared
#' base_cols used by steps 1-4, for feature-variant experiments (WS2).
#'
#' @keywords internal
.train_xgb_fixed <- function(team_mdl_df, train_filter, nrounds_vec, extra_feature_cols = NULL,
                              xgb_nthread = NULL) {
  loadNamespace("xgboost")

  train_mask <- train_filter & !is.na(team_mdl_df$win) &
    !is.na(team_mdl_df$total_xpoints_adj) & !is.na(team_mdl_df$xscore_diff) &
    !is.na(team_mdl_df$shot_conv_diff) & !is.na(team_mdl_df$score_diff)

  xgb_df <- team_mdl_df[train_mask, ]
  if (nrow(xgb_df) == 0) return(team_mdl_df)

  osr_dsr_cols <- character(0)
  if (all(c("osr_diff", "dsr_diff") %in% names(team_mdl_df)) &&
      !all(is.na(team_mdl_df$osr_diff))) {
    osr_dsr_cols <- c("osr_diff", "dsr_diff")
  }

  base_cols <- c(
    "team_type_fac",
    "game_year_decimal.x", "game_prop_through_year.x",
    "game_prop_through_month.x", "game_prop_through_day.x",
    "epr_diff", "epr_recv_diff", "epr_disp_diff",
    "epr_spoil_diff", "epr_hitout_diff",
    "torp_diff", "psr_diff", osr_dsr_cols,
    "log_dist_diff", "familiarity_diff", "days_rest_diff_fac",
    extra_feature_cols
  )

  # xgb_nthread: XGBoost's default (unset) grabs ALL logical cores per
  # xgb.train() call. Fine for the sequential rolling loop (one round at a
  # time) but MUST be capped when run_rolling_eval_parallel() runs several
  # rounds concurrently, or every round's xgb.train() calls oversubscribe the
  # same cores and everything gets slower, not faster. NULL preserves today's
  # exact (uncapped) behaviour for the sequential path.
  reg_params <- list(
    objective = "reg:squarederror", eval_metric = "rmse",
    tree_method = "hist", eta = 0.05, subsample = 0.7,
    colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15
  )
  cls_params <- list(
    objective = "binary:logistic", eval_metric = "logloss",
    tree_method = "hist", eta = 0.05, subsample = 0.7,
    colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15
  )
  if (!is.null(xgb_nthread)) {
    reg_params$nthread <- xgb_nthread
    cls_params$nthread <- xgb_nthread
  }

  train_fixed <- function(df, label, weights, feature_cols, params, nr) {
    fmat <- stats::model.matrix(~ . - 1, data = df[, feature_cols, drop = FALSE])
    dtrain <- xgboost::xgb.DMatrix(data = fmat, label = label, weight = weights)
    set.seed(1234)
    xgboost::xgb.train(
      params = params, data = dtrain, nrounds = nr,
      print_every_n = 0, verbose = 0
    )
  }

  # Mirrors torp:::.predict_all_rows() (torp R/match_train.R), and must keep
  # mirroring it. model.matrix()'s default na.action is na.omit, so ANY row with
  # an NA feature is dropped from the matrix and the returned vector comes back
  # shorter than the frame it is assigned onto. When the lengths happen to divide
  # evenly R recycles instead of erroring, and predictions land on the wrong
  # matches with nothing logged -- the first symptom is bad numbers, not a
  # failure. torp fixed this on its own side (PR #155); this file has its own
  # copy and did not get the fix (torpmodels#34). Live trigger: unplayed finals
  # placeholder fixtures, which carry NA rating features.
  predict_all <- function(model, df, feature_cols) {
    fdf <- df[, feature_cols, drop = FALSE]
    mf <- stats::model.frame(~ . - 1, data = fdf, na.action = stats::na.pass)
    mat <- stats::model.matrix(~ . - 1, data = mf)
    if (nrow(mat) != nrow(df)) {
      stop(sprintf(paste0("predict_all(): design matrix has %d row(s) for a %d-row frame -- ",
                          "rows were dropped, so predictions cannot be aligned to matches. ",
                          "Expected na.pass to preserve every row; check for a non-numeric feature column."),
                   nrow(mat), nrow(df)))
    }
    preds <- predict(model, xgboost::xgb.DMatrix(data = mat))
    if (length(preds) != nrow(df)) {
      stop(sprintf(paste0("predict_all(): predicted %d value(s) for %d row(s) -- ",
                          "refusing to return a vector that would recycle onto the wrong matches."),
                   length(preds), nrow(df)))
    }
    preds
  }

  # Season-grouped out-of-fold assignment (same reasoning/shape as
  # torp:::.train_match_xgb()'s fix, R/match_train.R) -- used only to de-leak
  # the stacked cascade features below (steps 1, 2 and 3's predictions feed
  # later steps as inputs), NOT a substitute for `train_filter`.
  stopifnot(!anyNA(xgb_df$season.x))
  xgb_seasons <- sort(unique(xgb_df$season.x))
  folds <- lapply(xgb_seasons, function(s) which(xgb_df$season.x == s))

  # Out-of-fold prediction for one cascade stage at this function's ALREADY-
  # FIXED nrounds `nr` (this function's whole design point is a pre-chosen
  # nrounds -- see roxygen above -- so no per-fold xgb.cv, just refit at `nr`
  # on each fold's complement and predict on the held-out fold).
  oof_predict_fixed <- function(df, label, weights, feature_cols, params, nr) {
    oof <- rep(NA_real_, nrow(df))
    for (f in folds) {
      fmat_tr <- stats::model.matrix(~ . - 1, data = df[-f, feature_cols, drop = FALSE])
      dtr <- xgboost::xgb.DMatrix(data = fmat_tr, label = label[-f], weight = weights[-f])
      set.seed(1234)
      fit <- xgboost::xgb.train(params = params, data = dtr, nrounds = nr, print_every_n = 0, verbose = 0)
      fmat_te <- stats::model.matrix(~ . - 1, data = df[f, feature_cols, drop = FALSE])
      oof[f] <- predict(fit, xgboost::xgb.DMatrix(data = fmat_te))
    }
    oof
  }

  # Step 1: total xPoints
  m1 <- train_fixed(xgb_df, xgb_df$total_xpoints_adj, xgb_df$weightz,
                     base_cols, reg_params, nrounds_vec["total_xpoints"])
  # Out-of-fold correction for training rows: steps 2-4 all consume
  # xgb_pred_tot_xscore as an input feature. team_mdl_df's non-training rows
  # (upcoming/future rounds) keep the full-model prediction -- already
  # legitimately out-of-sample for them.
  xgb_df$xgb_pred_tot_xscore <- oof_predict_fixed(xgb_df, xgb_df$total_xpoints_adj,
                                                   xgb_df$weightz, base_cols, reg_params,
                                                   nrounds_vec["total_xpoints"])
  team_mdl_df$xgb_pred_tot_xscore <- predict_all(m1, team_mdl_df, base_cols)
  team_mdl_df$xgb_pred_tot_xscore[train_mask] <- xgb_df$xgb_pred_tot_xscore

  # Step 2: xScore diff
  s2_cols <- c(base_cols, "xgb_pred_tot_xscore")
  m2 <- train_fixed(xgb_df, xgb_df$xscore_diff, xgb_df$weightz,
                     s2_cols, reg_params, nrounds_vec["xscore_diff"])
  # Out-of-fold correction: steps 3 and 4 both consume xgb_pred_xscore_diff.
  xgb_df$xgb_pred_xscore_diff <- oof_predict_fixed(xgb_df, xgb_df$xscore_diff,
                                                    xgb_df$weightz, s2_cols, reg_params,
                                                    nrounds_vec["xscore_diff"])
  team_mdl_df$xgb_pred_xscore_diff <- predict_all(m2, team_mdl_df, s2_cols)
  team_mdl_df$xgb_pred_xscore_diff[train_mask] <- xgb_df$xgb_pred_xscore_diff

  # Step 3: conv diff
  s3_cols <- c(base_cols, "xgb_pred_tot_xscore", "xgb_pred_xscore_diff")
  m3 <- train_fixed(xgb_df, xgb_df$shot_conv_diff, xgb_df$shot_weightz,
                     s3_cols, reg_params, nrounds_vec["conv_diff"])
  # Out-of-fold correction: step 4 consumes xgb_pred_conv_diff.
  xgb_df$xgb_pred_conv_diff <- oof_predict_fixed(xgb_df, xgb_df$shot_conv_diff,
                                                  xgb_df$shot_weightz, s3_cols, reg_params,
                                                  nrounds_vec["conv_diff"])
  team_mdl_df$xgb_pred_conv_diff <- predict_all(m3, team_mdl_df, s3_cols)
  team_mdl_df$xgb_pred_conv_diff[train_mask] <- xgb_df$xgb_pred_conv_diff

  # Step 4: score diff
  s4_cols <- c(base_cols, "xgb_pred_xscore_diff", "xgb_pred_conv_diff", "xgb_pred_tot_xscore")
  m4 <- train_fixed(xgb_df, xgb_df$score_diff, xgb_df$weightz,
                     s4_cols, reg_params, nrounds_vec["score_diff"])
  # Out-of-fold correction: step 5 consumes xgb_pred_score_diff, and it is
  # also served directly (production blend uses it for every row including
  # completed matches) -- so training-row honesty here matters beyond just
  # step 5's own fit.
  xgb_df$xgb_pred_score_diff <- oof_predict_fixed(xgb_df, xgb_df$score_diff,
                                                   xgb_df$weightz, s4_cols, reg_params,
                                                   nrounds_vec["score_diff"])
  team_mdl_df$xgb_pred_score_diff <- predict_all(m4, team_mdl_df, s4_cols)
  team_mdl_df$xgb_pred_score_diff[train_mask] <- xgb_df$xgb_pred_score_diff

  # Step 5: win probability
  s5_cols <- c("team_type_fac", "xgb_pred_tot_xscore", "xgb_pred_score_diff",
               "log_dist_diff", "familiarity_diff", "days_rest_diff_fac")
  m5 <- train_fixed(xgb_df, as.numeric(xgb_df$win), xgb_df$weightz,
                     s5_cols, cls_params, nrounds_vec["win"])
  team_mdl_df$xgb_pred_win <- predict_all(m5, team_mdl_df, s5_cols)

  team_mdl_df
}

# .train_match_xgb_ext ----

#' Copy of torp:::.train_match_xgb() (match_train.R) extended to accept
#' extra_feature_cols (e.g. "elo_diff"), appended to base_cols BEFORE the
#' per-step CV/early-stopping nrounds search runs.
#'
#' Round 2 fix (FABLE-MATCH-MAE-PLAN.md WS2 follow-up): round 1 diagnosed
#' elo_diff hurting the XGB side (MAE 27.41 vs champion XGB 26.92) as an
#' artifact of run_rolling_eval()'s nrounds pre-optimisation calling
#' torp:::.train_match_xgb() on the elo-FREE base_cols — nrounds got tuned
#' before elo_diff ever entered the feature set, so .train_xgb_fixed()'s
#' actual per-round training (which DOES include elo_diff via
#' extra_feature_cols) ran with a stopping point chosen for a different
#' model. This function gives elo_diff a fair shake at nrounds-tuning time.
#'
#' extra_feature_cols = NULL reduces to base_cols unchanged (c(base_cols,
#' NULL) is a no-op), so this is behaviourally identical to
#' torp:::.train_match_xgb() on the default (NULL) path. elo_diff is
#' deliberately NOT added to s5_cols (win-probability step) — matches
#' .train_xgb_fixed()'s own s5_cols, which never included extra_feature_cols
#' either, so the fixed-nrounds training this feeds stays apples-to-apples.
#' Weather cols are included in s1 (mirrors torp:::.train_match_xgb()); this
#' is a pre-existing quirk of the production function (weather is CV'd into
#' step 1's nrounds search but .train_xgb_fixed()'s actual step 1 training
#' never receives weather_cols) that round 2 intentionally leaves alone —
#' fixing it would be a second variable confounding the elo_diff comparison.
#'
#' @keywords internal
.train_match_xgb_ext <- function(team_mdl_df, train_filter = NULL, extra_feature_cols = NULL) {
  loadNamespace("xgboost")

  if (is.null(train_filter)) {
    train_mask <- !is.na(team_mdl_df$win) & !is.na(team_mdl_df$total_xpoints_adj) &
      !is.na(team_mdl_df$xscore_diff) & !is.na(team_mdl_df$shot_conv_diff) &
      !is.na(team_mdl_df$score_diff)
  } else {
    train_mask <- train_filter & !is.na(team_mdl_df$win) &
      !is.na(team_mdl_df$total_xpoints_adj) & !is.na(team_mdl_df$xscore_diff) &
      !is.na(team_mdl_df$shot_conv_diff) & !is.na(team_mdl_df$score_diff)
  }

  xgb_df <- team_mdl_df[train_mask, ]
  cli::cli_inform("XGBoost nrounds CV (extended feature set) training on {nrow(xgb_df)} rows")
  if (nrow(xgb_df) == 0) {
    cli::cli_abort("Cannot train XGBoost: 0 complete rows after filtering")
  }

  osr_dsr_cols <- character(0)
  if (all(c("osr_diff", "dsr_diff") %in% names(team_mdl_df)) &&
      !all(is.na(team_mdl_df$osr_diff))) {
    osr_dsr_cols <- c("osr_diff", "dsr_diff")
  }

  base_cols <- c(
    "team_type_fac",
    "game_year_decimal.x", "game_prop_through_year.x",
    "game_prop_through_month.x", "game_prop_through_day.x",
    "epr_diff", "epr_recv_diff", "epr_disp_diff",
    "epr_spoil_diff", "epr_hitout_diff",
    "torp_diff", "psr_diff", osr_dsr_cols,
    "log_dist_diff",
    "familiarity_diff",
    "days_rest_diff_fac",
    extra_feature_cols
  )

  weather_cols <- character(0)
  weather_candidates <- c("log_wind", "log_precip", "temp_avg", "humidity_avg", "is_roof")
  if (all(weather_candidates %in% names(team_mdl_df))) {
    weather_cols <- weather_candidates
  }
  s1_cols <- c(base_cols, weather_cols)

  reg_params <- list(
    objective = "reg:squarederror", eval_metric = "rmse",
    tree_method = "hist", eta = 0.05, subsample = 0.7,
    colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15
  )
  cls_params <- list(
    objective = "binary:logistic", eval_metric = "logloss",
    tree_method = "hist", eta = 0.05, subsample = 0.7,
    colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15
  )

  train_seasons <- sort(unique(xgb_df$season.x))
  folds <- lapply(train_seasons, function(s) which(xgb_df$season.x == s))

  train_step <- function(df, label, weights, feature_cols, params) {
    fmat <- stats::model.matrix(~ . - 1, data = df[, feature_cols, drop = FALSE])
    dtrain <- xgboost::xgb.DMatrix(data = fmat, label = label, weight = weights)

    withr::local_seed(1234)
    cv <- xgboost::xgb.cv(
      params = params, data = dtrain, nrounds = 1000, folds = folds,
      early_stopping_rounds = 30, print_every_n = 0, verbose = 0
    )
    metric_col <- paste0("test_", params$eval_metric, "_mean")
    best_n <- which.min(cv$evaluation_log[[metric_col]])
    cv_score <- min(cv$evaluation_log[[metric_col]])

    withr::local_seed(1234)
    model <- xgboost::xgb.train(
      params = params, data = dtrain, nrounds = best_n,
      print_every_n = 0, verbose = 0
    )
    list(model = model, preds = predict(model, dtrain),
         best_n = best_n, cv_score = cv_score)
  }

  # Mirrors torp:::.predict_all_rows() (torp R/match_train.R), and must keep
  # mirroring it. model.matrix()'s default na.action is na.omit, so ANY row with
  # an NA feature is dropped from the matrix and the returned vector comes back
  # shorter than the frame it is assigned onto. When the lengths happen to divide
  # evenly R recycles instead of erroring, and predictions land on the wrong
  # matches with nothing logged -- the first symptom is bad numbers, not a
  # failure. torp fixed this on its own side (PR #155); this file has its own
  # copy and did not get the fix (torpmodels#34). Live trigger: unplayed finals
  # placeholder fixtures, which carry NA rating features.
  predict_all <- function(model, df, feature_cols) {
    fdf <- df[, feature_cols, drop = FALSE]
    mf <- stats::model.frame(~ . - 1, data = fdf, na.action = stats::na.pass)
    mat <- stats::model.matrix(~ . - 1, data = mf)
    if (nrow(mat) != nrow(df)) {
      stop(sprintf(paste0("predict_all(): design matrix has %d row(s) for a %d-row frame -- ",
                          "rows were dropped, so predictions cannot be aligned to matches. ",
                          "Expected na.pass to preserve every row; check for a non-numeric feature column."),
                   nrow(mat), nrow(df)))
    }
    preds <- predict(model, xgboost::xgb.DMatrix(data = mat))
    if (length(preds) != nrow(df)) {
      stop(sprintf(paste0("predict_all(): predicted %d value(s) for %d row(s) -- ",
                          "refusing to return a vector that would recycle onto the wrong matches."),
                   length(preds), nrow(df)))
    }
    preds
  }

  s1 <- train_step(xgb_df, xgb_df$total_xpoints_adj, xgb_df$weightz, s1_cols, reg_params)
  xgb_df$xgb_pred_tot_xscore <- s1$preds
  team_mdl_df$xgb_pred_tot_xscore <- predict_all(s1$model, team_mdl_df, s1_cols)

  s2_cols <- c(base_cols, "xgb_pred_tot_xscore")
  s2 <- train_step(xgb_df, xgb_df$xscore_diff, xgb_df$weightz, s2_cols, reg_params)
  xgb_df$xgb_pred_xscore_diff <- s2$preds
  team_mdl_df$xgb_pred_xscore_diff <- predict_all(s2$model, team_mdl_df, s2_cols)

  s3_cols <- c(base_cols, "xgb_pred_tot_xscore", "xgb_pred_xscore_diff")
  s3 <- train_step(xgb_df, xgb_df$shot_conv_diff, xgb_df$shot_weightz, s3_cols, reg_params)
  xgb_df$xgb_pred_conv_diff <- s3$preds
  team_mdl_df$xgb_pred_conv_diff <- predict_all(s3$model, team_mdl_df, s3_cols)

  s4_cols <- c(base_cols, "xgb_pred_xscore_diff", "xgb_pred_conv_diff", "xgb_pred_tot_xscore")
  s4 <- train_step(xgb_df, xgb_df$score_diff, xgb_df$weightz, s4_cols, reg_params)
  xgb_df$xgb_pred_score_diff <- s4$preds
  team_mdl_df$xgb_pred_score_diff <- predict_all(s4$model, team_mdl_df, s4_cols)

  s5_cols <- c(
    "team_type_fac",
    "xgb_pred_tot_xscore", "xgb_pred_score_diff",
    "log_dist_diff", "familiarity_diff", "days_rest_diff_fac"
  )
  s5 <- train_step(xgb_df, as.numeric(xgb_df$win), xgb_df$weightz, s5_cols, cls_params)
  xgb_df$xgb_pred_win <- s5$preds
  team_mdl_df$xgb_pred_win <- predict_all(s5$model, team_mdl_df, s5_cols)

  cli::cli_alert_success("XGBoost nrounds CV, extended ({s1$best_n}/{s2$best_n}/{s3$best_n}/{s4$best_n}/{s5$best_n})")

  steps <- list(
    total_xpoints = list(best_n = s1$best_n, cv_score = s1$cv_score),
    xscore_diff   = list(best_n = s2$best_n, cv_score = s2$cv_score),
    conv_diff      = list(best_n = s3$best_n, cv_score = s3$cv_score),
    score_diff     = list(best_n = s4$best_n, cv_score = s4$cv_score),
    win            = list(best_n = s5$best_n, cv_score = s5$cv_score)
  )
  list(steps = steps)
}

# run_rolling_eval ----

#' Rolling week-by-week out-of-sample evaluation
#'
#' For each round in `test_seasons`, trains `gam_trainer` + `xgb_trainer` on
#' all strictly-prior completed matches, predicts that round, and
#' accumulates out-of-sample predictions for four variants: GAM-only,
#' XGBoost-only, Output Blend (50/50 average of final GAM/XGB outputs), and
#' Input Blend (50/50 average of intermediate outputs, WP re-derived from
#' the GAM win model on the blended margin — mirrors match_model.R's
#' production blend block, plan G4).
#'
#' XGBoost nrounds are pre-optimised once via CV on data strictly before
#' `min(test_seasons)` (plan G6 — no leakage into TEST_SEASONS), then reused
#' fixed for every rolling step (no per-round CV).
#'
#' @param team_mdl_df Complete model dataset from build_team_mdl_df()
#' @param test_seasons Integer vector of seasons to roll through
#' @param gam_trainer Function(team_mdl_df, train_filter, nthreads) -> list(models, data);
#'   default torp:::.train_match_gams. Swap for experiments/ws3_*, ws4_* variants.
#' @param xgb_trainer Function(team_mdl_df, train_filter, nrounds_vec, extra_feature_cols) ->
#'   team_mdl_df with xgb_pred_* columns; default .train_xgb_fixed above.
#' @param extra_feature_cols Character vector of extra column names to feed
#'   xgb_trainer's base_cols (e.g. "elo_diff" for WS2); NULL = today's feature set.
#' @param cv_extra_feature_cols Character vector of extra column names to
#'   include in the nrounds CV pre-optimisation step's base_cols (round 2
#'   fix, WS2 follow-up). NULL (default) = unchanged behaviour, CV runs via
#'   torp:::.train_match_xgb() on today's base_cols only, exactly as before
#'   — every existing caller (round 1 scripts, decay experiments, etc.) is
#'   byte-compatible. Pass e.g. "elo_diff" here (usually the same value as
#'   `extra_feature_cols`) to fix the round-1 gap where nrounds got tuned
#'   BEFORE elo_diff ever entered the XGB feature set, so .train_xgb_fixed()
#'   was training with a stopping point chosen for a different model. Kept
#'   as a separate parameter (not reusing `extra_feature_cols` for both
#'   purposes) so opting into the fix is explicit per-call.
#' @param nthreads Threads passed to gam_trainer (default 4L)
#' @param verbose Print progress via cli (default TRUE)
#' @return list(gam_preds, xgb_preds, blend_preds, input_blend_preds,
#'   xgb_nrounds, test_rounds)
#' @keywords internal
run_rolling_eval <- function(team_mdl_df,
                              test_seasons,
                              gam_trainer = .train_match_gams,
                              xgb_trainer = .train_xgb_fixed,
                              extra_feature_cols = NULL,
                              cv_extra_feature_cols = NULL,
                              cv_trainer = NULL,
                              nthreads = 4L,
                              verbose = TRUE) {
  stopifnot(is.function(gam_trainer), is.function(xgb_trainer))

  # Pre-optimise XGBoost nrounds via CV on pre-test-season data only (G6) ----
  # Using the full dataset (including test_seasons) would leak: best_n would
  # be tuned with knowledge of folds that include the rounds we're about to
  # predict in the rolling loop below.
  cv_train_mask <- team_mdl_df$season.x < min(test_seasons)
  cv_train_df   <- team_mdl_df[cv_train_mask, ]
  if (verbose) {
    cli::cli_h2("Pre-optimising XGBoost nrounds (CV on pre-test-season data)")
    cli::cli_inform("nrounds CV input: {sum(cv_train_mask)/2} matches (seasons < {min(test_seasons)})")
  }
  # cv_trainer (round 3, FABLE-MATCH-MAE-PLAN.md wide-window CI extension):
  # optional override for the nrounds-CV step, function(cv_train_df) ->
  # list(steps = ...). Needed because torp:::.train_match_xgb() is PRODUCTION
  # code that keeps evolving underneath this research harness (2026-07-14:
  # gained elo_diff in base_cols via the "Integrate C6" commit, then a
  # further uncommitted edit pinned xgb_nthread) -- callers that need to
  # reproduce an EARLIER cached champion definition byte-for-byte should pass
  # their own pinned copy here rather than depend on today's live production
  # behaviour. NULL (default) preserves every existing caller's behaviour
  # unchanged (falls through to cv_extra_feature_cols / .train_match_xgb()).
  xgb_cv_result <- if (!is.null(cv_trainer)) {
    cv_trainer(cv_train_df)
  } else if (!is.null(cv_extra_feature_cols)) {
    .train_match_xgb_ext(cv_train_df, extra_feature_cols = cv_extra_feature_cols)
  } else {
    .train_match_xgb(cv_train_df)
  }
  xgb_nrounds <- vapply(xgb_cv_result$steps, function(s) s$best_n, integer(1))
  if (verbose) {
    cli::cli_inform("XGBoost nrounds: {paste(names(xgb_nrounds), xgb_nrounds, sep='=', collapse=', ')}")
  }

  # Identify test rounds ----
  test_rounds <- team_mdl_df |>
    dplyr::filter(!is.na(win), season.x %in% test_seasons) |>
    dplyr::distinct(season.x, round_number.x) |>
    dplyr::arrange(season.x, round_number.x) |>
    dplyr::rename(season = season.x, round = round_number.x)

  n_test_rounds <- nrow(test_rounds)
  n_test_matches <- sum(!is.na(team_mdl_df$win) & team_mdl_df$season.x %in% test_seasons) / 2
  if (verbose) {
    cli::cli_h1("Rolling Evaluation: {n_test_rounds} rounds, ~{n_test_matches} matches ({paste(test_seasons, collapse='-')})")
  }

  # Rolling evaluation loop ----
  all_gam_preds <- list()
  all_xgb_preds <- list()
  all_input_blend_preds <- list()

  for (i in seq_len(n_test_rounds)) {
    s <- test_rounds$season[i]
    r <- test_rounds$round[i]

    # Train on everything strictly before this round
    train_filter <- (team_mdl_df$season.x < s) |
      (team_mdl_df$season.x == s & team_mdl_df$round_number.x < r)

    # Test mask: this specific round, completed matches only
    test_mask <- !is.na(team_mdl_df$win) &
      team_mdl_df$season.x == s & team_mdl_df$round_number.x == r

    n_train <- sum(train_filter & !is.na(team_mdl_df$win)) / 2
    n_test <- sum(test_mask) / 2

    if (n_test == 0) next

    if (verbose) cli::cli_progress_step("{s} R{r}: train={n_train}, test={n_test}")

    # Train GAMs
    gam_result <- suppressMessages(
      gam_trainer(team_mdl_df, train_filter = train_filter, nthreads = nthreads)
    )
    gam_data <- gam_result$data

    # Train XGBoost (fixed nrounds, no CV)
    xgb_data <- suppressMessages(
      xgb_trainer(gam_data, train_filter, xgb_nrounds, extra_feature_cols = extra_feature_cols)
    )

    # Extract GAM predictions for test round
    all_gam_preds[[i]] <- .format_match_preds(gam_data[test_mask, ])

    # Extract XGBoost predictions for test round
    all_xgb_preds[[i]] <- xgb_data[test_mask, ] |>
      dplyr::mutate(pred_score_diff = xgb_pred_score_diff, pred_win = xgb_pred_win) |>
      .format_match_preds()

    # Input blend: blend intermediate outputs, derive WP from GAM model
    ib_data <- xgb_data
    ib_data$pred_tot_xscore <- 0.5 * ib_data$pred_tot_xscore +
      0.5 * ib_data$xgb_pred_tot_xscore
    ib_data$pred_xscore_diff <- 0.5 * ib_data$pred_xscore_diff +
      0.5 * ib_data$xgb_pred_xscore_diff
    ib_data$pred_score_diff <- 0.5 * ib_data$pred_score_diff +
      0.5 * ib_data$xgb_pred_score_diff
    ib_data$pred_win <- predict(
      gam_result$models$win, newdata = ib_data, type = "response"
    )
    all_input_blend_preds[[i]] <- .format_match_preds(ib_data[test_mask, ])
  }

  if (verbose) cli::cli_alert_success("Rolling evaluation complete")

  # Combine all out-of-sample predictions ----
  gam_preds <- dplyr::bind_rows(all_gam_preds) |>
    dplyr::mutate(
      home_win = ifelse(margin > 0, 1, ifelse(margin == 0, 0.5, 0)),
      home_team_chr = torp_replace_teams(as.character(home_team))
    )

  xgb_preds <- dplyr::bind_rows(all_xgb_preds) |>
    dplyr::mutate(
      home_win = ifelse(margin > 0, 1, ifelse(margin == 0, 0.5, 0)),
      home_team_chr = torp_replace_teams(as.character(home_team))
    )

  # Output blend (legacy): 50/50 average of final probabilities
  blend_preds <- gam_preds |>
    dplyr::mutate(
      pred_win = 0.5 * gam_preds$pred_win + 0.5 * xgb_preds$pred_win,
      pred_margin = 0.5 * gam_preds$pred_margin + 0.5 * xgb_preds$pred_margin
    )

  # Input blend: blend intermediate model outputs, derive WP from GAM model
  input_blend_preds <- dplyr::bind_rows(all_input_blend_preds) |>
    dplyr::mutate(
      home_win = ifelse(margin > 0, 1, ifelse(margin == 0, 0.5, 0)),
      home_team_chr = torp_replace_teams(as.character(home_team))
    )

  list(
    gam_preds = gam_preds,
    xgb_preds = xgb_preds,
    blend_preds = blend_preds,
    input_blend_preds = input_blend_preds,
    xgb_nrounds = xgb_nrounds,
    test_rounds = test_rounds
  )
}

# .compute_metrics ----

#' Compute aggregate WP + margin metrics for one prediction set
#'
#' Extended (WS0, FABLE-MATCH-MAE-PLAN.md) beyond the original
#' logloss/accuracy/brier/mae/rmse with margin calibration slope +
#' intercept (`lm(margin ~ pred_margin)`), `cor(pred_margin, margin)`,
#' `sd(pred_margin)/sd(margin)`, and close-bucket (|actual margin| <= 12) MAE.
#'
#' Round 2 (FABLE decay-on-C6 retest) adds `bits` — mean per-game bits
#' score, Squiggle's own convention: `1 + log2(p_winner)` where p_winner is
#' pred_win if the home team won, `1 - pred_win` if the away team won, and
#' `0.5 * log2(p*(1-p))`-style (via the 1 + ... wrapper) for the rare draw
#' (`home_win == 0.5`). Matches the `team_mdl_df$bits` column already
#' computed inside `.train_match_gams_v4b_elo()` (ws5_grid.R c6 trainer).
#'
#' @param preds One-row-per-match predictions (as returned by
#'   .format_match_preds() + the home_win/home_team_chr mutate that
#'   run_rolling_eval() applies) — needs columns pred_margin, margin,
#'   pred_win, home_win.
#' @keywords internal
.compute_metrics <- function(preds) {
  close_mask <- abs(preds$margin) <= 12
  fit <- stats::lm(margin ~ pred_margin, data = preds)
  cf  <- stats::coef(fit)

  bits_vec <- dplyr::case_when(
    preds$home_win == 1 ~ 1 + log2(preds$pred_win),
    preds$home_win == 0 ~ 1 + log2(1 - preds$pred_win),
    TRUE ~ 1 + 0.5 * log2(preds$pred_win * (1 - preds$pred_win))
  )

  list(
    logloss   = MLmetrics::LogLoss(preds$pred_win, preds$home_win),
    accuracy  = mean(round(preds$pred_win) == preds$home_win) * 100,
    brier     = mean((preds$pred_win - preds$home_win)^2),
    bits      = mean(bits_vec),
    mae       = mean(abs(preds$pred_margin - preds$margin)),
    rmse      = sqrt(mean((preds$pred_margin - preds$margin)^2)),
    slope     = unname(cf[["pred_margin"]]),
    intercept = unname(cf[["(Intercept)"]]),
    cor       = stats::cor(preds$pred_margin, preds$margin),
    sd_pred   = stats::sd(preds$pred_margin),
    sd_actual = stats::sd(preds$margin),
    sd_ratio  = stats::sd(preds$pred_margin) / stats::sd(preds$margin),
    close_n   = sum(close_mask),
    close_mae = mean(abs(preds$pred_margin[close_mask] - preds$margin[close_mask]))
  )
}

# margin_calibration_by_pred_bucket ----

#' Margin calibration by predicted-margin bucket
#'
#' The actionable, pre-game-observable view (WS0 report table 1): buckets on
#' |pred_margin| (0-6, 6-12, 12-24, 24+), each with n, mean|pred|,
#' mean|actual|, MAE, and within-bucket OLS slope (actual ~ pred).
#'
#' @param preds One-row-per-match predictions (pred_margin, margin columns)
#' @keywords internal
margin_calibration_by_pred_bucket <- function(preds) {
  abs_pred <- abs(preds$pred_margin)
  bucket <- dplyr::case_when(
    abs_pred <= 6  ~ "0-6",
    abs_pred <= 12 ~ "6-12",
    abs_pred <= 24 ~ "12-24",
    TRUE           ~ "24+"
  )
  preds$bucket <- factor(bucket, levels = c("0-6", "6-12", "12-24", "24+"))

  preds |>
    dplyr::group_by(bucket, .drop = FALSE) |>
    dplyr::summarise(
      n               = dplyr::n(),
      mean_abs_pred   = round(mean(abs(pred_margin)), 2),
      mean_abs_actual = round(mean(abs(margin)), 2),
      mae             = round(mean(abs(pred_margin - margin)), 2),
      # OLS slope via cov/var (equivalent to lm(margin ~ pred_margin) coefficient,
      # but safe to compute inside a dplyr data-masked summarise without formula/env gotchas)
      slope           = if (dplyr::n() >= 5) round(stats::cov(pred_margin, margin) / stats::var(pred_margin), 3) else NA_real_,
      .groups = "drop"
    )
}

# mae_by_actual_bucket ----

#' MAE by actual-margin bucket
#'
#' Diagnostic-only (WS0 report table 2, not conditionally interpretable
#' pre-game): buckets on |actual margin| (<=12, 13-30, 31-60, 60+), matching
#' the shape of docs/reviews/2026-MATCH-MAE-DIAGNOSIS.md's Squiggle table,
#' with n, mean pred, mean actual, MAE per bucket.
#'
#' @param preds One-row-per-match predictions (pred_margin, margin columns)
#' @keywords internal
mae_by_actual_bucket <- function(preds) {
  abs_actual <- abs(preds$margin)
  bucket <- dplyr::case_when(
    abs_actual <= 12 ~ "<=12 (close)",
    abs_actual <= 30 ~ "13-30 (moderate)",
    abs_actual <= 60 ~ "31-60 (big)",
    TRUE             ~ "60+ (blowout)"
  )
  preds$bucket <- factor(bucket, levels = c("<=12 (close)", "13-30 (moderate)", "31-60 (big)", "60+ (blowout)"))

  preds |>
    dplyr::group_by(bucket, .drop = FALSE) |>
    dplyr::summarise(
      n           = dplyr::n(),
      mean_pred   = round(mean(pred_margin), 2),
      mean_actual = round(mean(margin), 2),
      mae         = round(mean(abs(pred_margin - margin)), 2),
      .groups = "drop"
    )
}

# boot_mae_diff ----

#' Match-level block bootstrap CI on delta-MAE / delta-Brier between two
#' prediction sets (FABLE-METHODOLOGY.md E5)
#'
#' Resamples unique `match_id`s with replacement (not rows-within-match,
#' since one row already = one match here), recomputing the mean per-match
#' MAE/Brier difference each draw. A 95% CI excluding 0 is the plan G3 ship
#' gate for any ΔMAE claim.
#'
#' @param preds_a,preds_b One-row-per-match prediction sets to compare
#'   (need match_id, pred_margin, margin, pred_win, home_win columns).
#'   Compared on their intersection of match_id (inner join).
#' @param B Number of bootstrap draws (default 2000)
#' @param seed RNG seed for reproducibility (default 1234)
#' @return list(n_matches, mae_diff, mae_ci, brier_diff, brier_ci) where
#'   diffs are mean(a - b) and *_ci are c(2.5%, 97.5%) bootstrap quantiles
#' @keywords internal
boot_mae_diff <- function(preds_a, preds_b, B = 2000, seed = 1234) {
  need_cols <- c("match_id", "pred_margin", "pred_win", "margin", "home_win")
  stopifnot(all(need_cols %in% names(preds_a)), all(need_cols %in% names(preds_b)))

  a <- as.data.frame(preds_a[, need_cols])
  b <- as.data.frame(preds_b[, need_cols])
  names(a)[2:5] <- paste0(names(a)[2:5], "_a")
  names(b)[2:5] <- paste0(names(b)[2:5], "_b")

  joined <- merge(a, b, by = "match_id")
  if (nrow(joined) == 0) {
    stop("boot_mae_diff: no overlapping match_id between preds_a and preds_b")
  }
  if (anyDuplicated(joined$match_id)) {
    cli::cli_warn("boot_mae_diff: duplicate match_id after join — preds_a/preds_b should be one row per match")
  }

  mae_a <- abs(joined$pred_margin_a - joined$margin_a)
  mae_b <- abs(joined$pred_margin_b - joined$margin_b)
  brier_a <- (joined$pred_win_a - joined$home_win_a)^2
  brier_b <- (joined$pred_win_b - joined$home_win_b)^2
  d_mae <- mae_a - mae_b
  d_brier <- brier_a - brier_b

  n_ids <- nrow(joined)

  withr::with_seed(seed, {
    boots <- replicate(B, {
      s <- sample.int(n_ids, n_ids, replace = TRUE)
      c(mae = mean(d_mae[s]), brier = mean(d_brier[s]))
    })
  })

  mae_ci <- stats::quantile(boots["mae", ], c(0.025, 0.975))
  brier_ci <- stats::quantile(boots["brier", ], c(0.025, 0.975))

  list(
    n_matches  = n_ids,
    mae_diff   = mean(d_mae),
    mae_ci     = mae_ci,
    brier_diff = mean(d_brier),
    brier_ci   = brier_ci
  )
}

# run_rolling_eval_parallel ----

#' Parallel-across-rounds variant of run_rolling_eval()
#'
#' Perf note (raised during the match-MAE investigation, not a plan
#' workstream): each rolling round's fit only depends on `team_mdl_df` + a
#' boolean training mask — NOT on any other round's fitted model — so the
#' sequential `for` loop in run_rolling_eval() is embarrassingly parallel.
#' One full 2025:2026 pooled run measured at ~527s (WS1); on a 24-core box
#' with the sequential loop using only nthreads=4 for mgcv and (by default)
#' ALL cores for xgboost sequentially, most cores sit idle most of the time.
#'
#' Two things make this non-trivial to just wrap in mclapply():
#' 1. Windows: `parallel::mclapply()` forks, which Windows doesn't support —
#'    it silently ignores `mc.cores` and runs sequentially with NO error or
#'    warning. `future.apply::future_lapply()` with a `multisession` plan
#'    (separate persistent worker *processes*, not forks) works cross-
#'    platform including Windows — that's why this uses it, not mclapply.
#' 2. XGBoost's `xgb.train()` has no thread cap set anywhere in this file's
#'    default params — by default it grabs ALL logical cores per call. Fine
#'    one round at a time; catastrophic (oversubscription, slower than
#'    sequential) if naively run N rounds at once. This function caps it via
#'    `.train_xgb_fixed()`'s new `xgb_nthread` param — every parallel worker
#'    MUST pass a capped value here, sized so `n_workers * (gam_nthreads +
#'    xgb_nthread) <=` roughly the machine's core count.
#'
#' `multisession` workers are separate R processes — they don't inherit the
#' calling session's `devtools::load_all()`'d torp package or attached
#' libraries. Each worker loads torp + required libraries ONCE on first use
#' (guarded by a sentinel in that worker's global env) and reuses it for
#' every subsequent round future_lapply schedules onto that same worker.
#'
#' VALIDATED 2026-07-14 at FULL SCALE (2025:2026 pooled, 48 rounds, 369
#' matches, default production gam_trainer/xgb_trainer, n_workers=5) — this
#' supersedes an earlier 5-round smoke test; the caveat below is CONFIRMED,
#' not resolved, at real scale:
#' Sequential run_rolling_eval() took 596.9s (9.95 min); parallel took 191.9s
#' (3.20 min) — a real 3.11x speedup (measured under some concurrent CPU load
#' from an unrelated script sharing the machine, so treat as approximate,
#' not a clean isolated benchmark; the 5-round test's ~1.07x was an artifact
#' of worker-startup overhead dominating a run that small — at full scale
#' that cost amortises and the speedup is real).
#' Correctness: GAM predictions are close in aggregate (seq vs par MAE
#' 25.619 vs 25.637, Brier 0.1718 vs 0.1720) but "~1e-4 float noise" from the
#' 5-round test UNDERSTATES the full-scale tail — mean |pred_margin diff|
#' 0.11 but max 4.55 in the worst single round, mean |pred_win diff| 0.0009
#' but max 0.045 (gam_nthreads 4 sequential vs 2 parallel evidently nudges
#' mgcv's REML smoothing-parameter search in a handful of rounds; small next
#' to the xgboost effect below, but not literally negligible).
#' XGBoost/Input-Blend/Output-Blend predictions did NOT converge with more
#' data — mean |pred_margin diff| 2.4-4.8 points, MAX 12.6-25.2 points; mean
#' |pred_win diff| 0.02-0.05, max 0.09-0.28. In aggregate: XGB-only MAE
#' 26.857 (seq) vs 26.622 (par), a 0.24-point delta; blend MAE 25.923 vs
#' 25.792, a 0.13-point delta — sized comparably to the real candidate-vs-
#' champion ΔMAE this repo ships/rejects on (FABLE-MATCH-MAE-PLAN gates are
#' often 0.1-0.5). Root cause unchanged: `tree_method = "hist"` is not
#' deterministic across different `nthread` values even with a fixed seed
#' (known xgboost behaviour, not a bug here); this path necessarily runs
#' xgboost at a capped `xgb_nthread` while run_rolling_eval()'s default is
#' uncapped (all cores), and that thread-count difference alone is enough to
#' diverge the trees, compounding over many boosting rounds — and this does
#' NOT wash out at 48 rounds vs 5; it's structural, not sampling noise.
#' CONSEQUENCE / VERDICT: do not compare a candidate scored via this
#' function against a baseline established via run_rolling_eval() (or vice
#' versa) for a real MAE/ship decision — the seq-vs-par delta alone is large
#' enough to be mistaken for a genuine model improvement or regression at
#' this project's effect sizes. Confirmed safe for: fast relative screening
#' across many candidates, provided every candidate in the comparison is run
#' through the SAME path at the SAME thread count (the systematic
#' divergence then affects all candidates alike and cancels in the
#' comparison) — a real 3.11x speedup makes this genuinely worth reaching
#' for at that stage (a ~10 min sequential run becomes ~3 min). Any
#' apparent winner from a parallel-screened comparison must be re-confirmed
#' through the sequential run_rolling_eval() before a real ship decision.
#'
#' @inheritParams run_rolling_eval
#' @param torp_path Path to the torp package for workers to devtools::load_all()
#'   (default tries the same relative paths train_match_models.R uses).
#' @param n_workers Number of persistent multisession workers (default: leaves
#'   ~2 cores free, i.e. max(1, availableCores() - 2), capped at 6 — beyond
#'   6 concurrent rounds most machines run out of RAM headroom before CPU,
#'   since each worker holds its own copy of team_mdl_df + fitted models).
#' @param gam_nthreads Threads per mgcv::bam() call inside each worker
#'   (default 2 — lower than run_rolling_eval()'s solo default of 4, since
#'   now n_workers copies run at once; tune down further if n_workers is large).
#' @param xgb_nthread Threads per xgb.train() call inside each worker
#'   (default 2, same reasoning — see point 2 above; NEVER leave this NULL
#'   here or xgboost oversubscribes every core on every worker at once).
#' @keywords internal
run_rolling_eval_parallel <- function(team_mdl_df,
                                       test_seasons,
                                       gam_trainer = .train_match_gams,
                                       xgb_trainer = .train_xgb_fixed,
                                       extra_feature_cols = NULL,
                                       cv_extra_feature_cols = NULL,
                                       torp_path = NULL,
                                       n_workers = NULL,
                                       gam_nthreads = 2L,
                                       xgb_nthread = 2L,
                                       verbose = TRUE) {
  stopifnot(is.function(gam_trainer), is.function(xgb_trainer))
  if (!requireNamespace("future.apply", quietly = TRUE)) {
    stop("run_rolling_eval_parallel() needs the future.apply package (install.packages('future.apply'))")
  }

  if (is.null(torp_path)) {
    candidates <- c("../../../torp", "../../torp", "../torp", "C:/dev/torpverse/torp")
    hit <- Filter(function(p) file.exists(file.path(p, "DESCRIPTION")), candidates)
    if (length(hit) == 0) stop("run_rolling_eval_parallel(): could not auto-detect torp_path; pass it explicitly")
    torp_path <- normalizePath(hit[[1]])
  }

  n_cores <- future::availableCores()
  if (is.null(n_workers)) n_workers <- min(6L, max(1L, n_cores - 2L))
  if (verbose) {
    cli::cli_inform("run_rolling_eval_parallel: {n_workers} workers x (gam_nthreads={gam_nthreads} + xgb_nthread={xgb_nthread}) on {n_cores} logical cores")
  }

  # Same nrounds CV pre-optimisation as run_rolling_eval() — single run, not
  # worth parallelising on its own (G6: strictly pre-test-season data only).
  cv_train_mask <- team_mdl_df$season.x < min(test_seasons)
  cv_train_df   <- team_mdl_df[cv_train_mask, ]
  xgb_cv_result <- if (!is.null(cv_extra_feature_cols)) {
    .train_match_xgb_ext(cv_train_df, extra_feature_cols = cv_extra_feature_cols)
  } else {
    .train_match_xgb(cv_train_df)
  }
  xgb_nrounds <- vapply(xgb_cv_result$steps, function(s) s$best_n, integer(1))

  test_rounds <- team_mdl_df |>
    dplyr::filter(!is.na(win), season.x %in% test_seasons) |>
    dplyr::distinct(season.x, round_number.x) |>
    dplyr::arrange(season.x, round_number.x) |>
    dplyr::rename(season = season.x, round = round_number.x)
  n_test_rounds <- nrow(test_rounds)
  if (verbose) cli::cli_h1("Rolling Evaluation (parallel): {n_test_rounds} rounds ({paste(test_seasons, collapse='-')})")

  old_plan <- future::plan()
  on.exit(future::plan(old_plan), add = TRUE)
  future::plan(future::multisession, workers = n_workers)

  one_round <- function(i, team_mdl_df, test_rounds, gam_trainer, xgb_trainer,
                         xgb_nrounds, extra_feature_cols, torp_path,
                         gam_nthreads, xgb_nthread) {
    if (!exists(".rre_parallel_worker_ready", envir = .GlobalEnv)) {
      suppressPackageStartupMessages({
        devtools::load_all(torp_path, quiet = TRUE)
        library(dplyr); library(xgboost); library(mgcv); library(cli); library(MLmetrics)
      })
      assign(".rre_parallel_worker_ready", TRUE, envir = .GlobalEnv)
    }

    s <- test_rounds$season[i]
    r <- test_rounds$round[i]
    train_filter <- (team_mdl_df$season.x < s) |
      (team_mdl_df$season.x == s & team_mdl_df$round_number.x < r)
    test_mask <- !is.na(team_mdl_df$win) &
      team_mdl_df$season.x == s & team_mdl_df$round_number.x == r
    if (sum(test_mask) == 0) return(NULL)

    gam_result <- suppressMessages(
      gam_trainer(team_mdl_df, train_filter = train_filter, nthreads = gam_nthreads)
    )
    gam_data <- gam_result$data

    # Guard: custom xgb_trainer copies (WS2-WS5 experiment variants) may not
    # declare xgb_nthread — only pass it if the function actually accepts it,
    # so this stays compatible with trainers written before this parameter
    # existed (at the cost of no thread cap for those, i.e. don't run THOSE
    # through the parallel path with more than 1-2 workers).
    xgb_call_args <- list(gam_data, train_filter, xgb_nrounds, extra_feature_cols = extra_feature_cols)
    if ("xgb_nthread" %in% names(formals(xgb_trainer))) {
      xgb_call_args$xgb_nthread <- xgb_nthread
    }
    xgb_data <- suppressMessages(do.call(xgb_trainer, xgb_call_args))

    ib_data <- xgb_data
    ib_data$pred_tot_xscore <- 0.5 * ib_data$pred_tot_xscore + 0.5 * ib_data$xgb_pred_tot_xscore
    ib_data$pred_xscore_diff <- 0.5 * ib_data$pred_xscore_diff + 0.5 * ib_data$xgb_pred_xscore_diff
    ib_data$pred_score_diff <- 0.5 * ib_data$pred_score_diff + 0.5 * ib_data$xgb_pred_score_diff
    ib_data$pred_win <- predict(gam_result$models$win, newdata = ib_data, type = "response")

    list(
      gam = .format_match_preds(gam_data[test_mask, ]),
      xgb = xgb_data[test_mask, ] |>
        dplyr::mutate(pred_score_diff = xgb_pred_score_diff, pred_win = xgb_pred_win) |>
        .format_match_preds(),
      ib  = .format_match_preds(ib_data[test_mask, ])
    )
  }

  results <- future.apply::future_lapply(
    seq_len(n_test_rounds), one_round,
    team_mdl_df = team_mdl_df, test_rounds = test_rounds,
    gam_trainer = gam_trainer, xgb_trainer = xgb_trainer,
    xgb_nrounds = xgb_nrounds, extra_feature_cols = extra_feature_cols,
    torp_path = torp_path, gam_nthreads = gam_nthreads, xgb_nthread = xgb_nthread,
    future.seed = TRUE
  )
  results <- Filter(Negate(is.null), results)
  if (verbose) cli::cli_alert_success("Rolling evaluation (parallel) complete")

  gam_preds <- dplyr::bind_rows(lapply(results, `[[`, "gam")) |>
    dplyr::mutate(home_win = ifelse(margin > 0, 1, ifelse(margin == 0, 0.5, 0)),
                  home_team_chr = torp_replace_teams(as.character(home_team)))
  xgb_preds <- dplyr::bind_rows(lapply(results, `[[`, "xgb")) |>
    dplyr::mutate(home_win = ifelse(margin > 0, 1, ifelse(margin == 0, 0.5, 0)),
                  home_team_chr = torp_replace_teams(as.character(home_team)))
  blend_preds <- gam_preds |>
    dplyr::mutate(pred_win = 0.5 * gam_preds$pred_win + 0.5 * xgb_preds$pred_win,
                  pred_margin = 0.5 * gam_preds$pred_margin + 0.5 * xgb_preds$pred_margin)
  input_blend_preds <- dplyr::bind_rows(lapply(results, `[[`, "ib")) |>
    dplyr::mutate(home_win = ifelse(margin > 0, 1, ifelse(margin == 0, 0.5, 0)),
                  home_team_chr = torp_replace_teams(as.character(home_team)))

  list(
    gam_preds = gam_preds, xgb_preds = xgb_preds,
    blend_preds = blend_preds, input_blend_preds = input_blend_preds,
    xgb_nrounds = xgb_nrounds, test_rounds = test_rounds
  )
}
