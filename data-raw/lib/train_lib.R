# Canonical Training Library
# ==========================
# Plain functions, no library() calls and no top-level execution -- callers
# (train_models.R, rebuild_everything.R Phase 4) must devtools::load_all()
# torp AND torpmodels before sourcing this file. Internal torp/torpmodels
# symbols are addressed via torp::: / torpmodels::: (both stay @keywords
# internal -- see model_meta.R); publish_model_group() is torpmodels's one
# exported entry point this file calls. All three resolve correctly whether
# torp/torpmodels were library()'d or load_all()'d.
#
# torpverse/docs/plans/TRAINING-CONSOLIDATION-PLAN.md Step 3 -- the single canonical trainer for
# EP, WP, and shot models. Fixes F1 (WP monotone constraints derived from
# WP_MODEL_FEATURES, never hand-inlined), F3 (shot + shot_player_df upload
# atomically), F4 (one training-window default), C2 (every artifact stamped
# with provenance before it's saved).

#' The canonical EP/WP/shot training window
#'
#' Completed seasons only -- matches rebuild's intent and keeps the
#' in-progress season available as an out-of-sample check.
#' @keywords internal
default_training_seasons <- function() 2021:(torp::get_afl_season() - 1L)

#' Canonical EP hyperparameters
#' Source: train_ep_model.R == rebuild Phase 4a == train_wp_model_cv_ep.R (all agree).
#' @keywords internal
ep_params <- function() {
  list(
    booster = "gbtree", objective = "multi:softprob", eval_metric = "mlogloss",
    tree_method = "hist", num_class = 5, eta = 0.1, gamma = 0,
    subsample = 0.85, colsample_bytree = 0.85, max_depth = 6, min_child_weight = 25
  )
}

#' Canonical WP hyperparameters
#' Source: train_wp_model.R == train_wp_model_cv_ep.R. The rebuild variant
#' (eta 0.1, subsample/colsample 0.85, 15-entry constraints) was the F1 bug
#' and is retired -- monotone_constraints is now DERIVED, never hand-inlined.
#' @keywords internal
wp_params <- function() {
  list(
    booster = "gbtree", objective = "binary:logistic", eval_metric = "logloss",
    tree_method = "hist", eta = 0.025, gamma = 0,
    monotone_constraints = torp:::wp_monotone_constraints(),
    max_depth = 6, min_child_weight = 1, subsample = 0.8, colsample_bytree = 0.8
  )
}

#' Load and clean PBP for EP/WP training, once, shared by both models
#' @keywords internal
load_training_pbp <- function(seasons) {
  cli::cli_inform("Loading chains for seasons {min(seasons)}-{max(seasons)}...")
  chains <- torp::load_chains(seasons = seasons, rounds = TRUE)
  cli::cli_inform("Chains: {nrow(chains)} rows")
  pbp <- torp::clean_pbp(chains)
  torp:::clean_model_data_epv(pbp)
}

#' Match-grouped 5-fold assignment, by row
#'
#' The identical fold construction previously copy-pasted across every
#' trainer: match-level assignment first (so no match's rows split across
#' folds), then broadcast to rows.
#'
#' @param match_ids Character/numeric vector, one match id per training row.
#' @param k Integer number of folds.
#' @param seed Integer seed (kept at 1234 to match every historical trainer).
#' @return Integer vector, same length/order as `match_ids`: the fold id per row.
#' @keywords internal
make_match_folds <- function(match_ids, k = 5L, seed = 1234L) {
  uniq <- unique(match_ids)
  set.seed(seed)
  match_fold_map <- sample(rep(seq_len(k), length.out = length(uniq)))
  names(match_fold_map) <- uniq
  match_fold_map[match_ids]
}

#' Fit the EP model with match-grouped CV to pick nrounds
#'
#' @param model_data_epv Output of [load_training_pbp()].
#' @param params EP xgboost params, default [ep_params()].
#' @param nrounds_max Integer cap for CV/final training rounds.
#' @return list(model, optimal_nrounds, cv_logloss, X, y, folds)
#' @keywords internal
fit_ep <- function(model_data_epv, params = ep_params(), nrounds_max = 500L) {
  epv_vars <- model_data_epv |> torp:::select_epv_model_vars()
  X <- stats::model.matrix(~ . + 0, data = epv_vars, na.action = na.pass)
  stopifnot(nrow(X) == nrow(epv_vars))
  stopifnot(identical(colnames(X), torp:::EPV_MODEL_FEATURES))
  y <- model_data_epv$label_ep

  row_folds <- make_match_folds(model_data_epv$torp_match_id)
  folds <- lapply(seq_len(max(row_folds)), function(k) which(row_folds == k))

  full_train <- xgboost::xgb.DMatrix(data = X, label = y)

  cli::cli_inform("Running EP 5-fold CV (match-grouped)...")
  set.seed(1234)
  cv_result <- xgboost::xgb.cv(
    params = params, data = full_train, nrounds = nrounds_max,
    folds = folds, early_stopping_rounds = 20, print_every_n = 20, verbose = 1
  )

  optimal_nrounds <- cv_result$best_iteration
  if (is.null(optimal_nrounds) || length(optimal_nrounds) == 0) {
    optimal_nrounds <- which.min(cv_result$evaluation_log$test_mlogloss_mean)
  }
  cv_logloss <- min(cv_result$evaluation_log$test_mlogloss_mean)
  cli::cli_inform("EP optimal nrounds: {optimal_nrounds} | CV mlogloss: {round(cv_logloss, 6)}")

  set.seed(1234)
  model <- xgboost::xgb.train(params = params, data = full_train,
                              nrounds = optimal_nrounds, print_every_n = 10)

  list(model = model, optimal_nrounds = optimal_nrounds, cv_logloss = cv_logloss,
       X = X, y = y, folds = folds)
}

#' Generate out-of-sample EP predictions via 5-fold CV
#'
#' Fixes the in-sample EP overfitting that made WP's shipped CV metrics
#' optimistic (train_wp_model_cv_ep.R:89-125).
#'
#' @param X EP feature matrix (from [fit_ep()]).
#' @param y EP labels (from [fit_ep()]).
#' @param folds List of row-index vectors per fold (from [fit_ep()]).
#' @param row_folds Integer vector, fold id per row (from [make_match_folds()]).
#' @param params EP params used to refit each fold.
#' @param nrounds Rounds to use for each fold's fit (the tuned optimal_nrounds).
#' @return Matrix of OOS class probabilities, columns
#'   `opp_goal, opp_behind, behind, goal, no_score`.
#' @keywords internal
cv_ep_oos_preds <- function(X, y, folds, row_folds, params, nrounds) {
  oos_preds <- matrix(NA_real_, nrow = nrow(X), ncol = 5)
  colnames(oos_preds) <- c("opp_goal", "opp_behind", "behind", "goal", "no_score")

  for (k in seq_along(folds)) {
    cli::cli_inform("EP OOS fold {k}/{length(folds)}...")
    test_idx <- folds[[k]]
    train_idx <- which(row_folds != k)

    dtrain <- xgboost::xgb.DMatrix(data = X[train_idx, ], label = y[train_idx])
    dtest <- xgboost::xgb.DMatrix(data = X[test_idx, ])

    set.seed(1234)
    fold_model <- xgboost::xgb.train(params = params, data = dtrain, nrounds = nrounds, verbose = 0)

    preds_raw <- predict(fold_model, dtest)
    if (is.matrix(preds_raw)) {
      oos_preds[test_idx, ] <- preds_raw
    } else {
      oos_preds[test_idx, ] <- matrix(preds_raw, ncol = 5, byrow = TRUE)
    }
    rm(fold_model, dtrain, dtest)
  }

  stopifnot(!anyNA(oos_preds))
  oos_preds
}

#' Inject OOS EP predictions into model_data_epv and build WP training data
#'
#' Formula synced with add_epv_vars() (add_variables.R): exp_pts = -6*opp_goal
#' - opp_behind + behind + 6*goal.
#'
#' @param model_data_epv Output of [load_training_pbp()].
#' @param oos_ep_preds Output of [cv_ep_oos_preds()].
#' @return WP training data (post [torp:::clean_model_data_wp()]).
#' @keywords internal
build_wp_data <- function(model_data_epv, oos_ep_preds) {
  df <- model_data_epv
  ep_df <- as.data.frame(oos_ep_preds)
  df$opp_goal <- ep_df$opp_goal
  df$opp_behind <- ep_df$opp_behind
  df$behind <- ep_df$behind
  df$goal <- ep_df$goal
  df$no_score <- ep_df$no_score
  df$exp_pts <- round(-6 * ep_df$opp_goal - ep_df$opp_behind + ep_df$behind + 6 * ep_df$goal, 5)
  df |> torp:::clean_model_data_wp()
}

#' Hard train-time guard against the F1 defect (constraint/feature misalignment)
#'
#' @param X WP feature matrix.
#' @param params WP params (with `monotone_constraints`).
#' @keywords internal
validate_wp_spec <- function(X, params) {
  n_constraints <- length(strsplit(gsub("[()]", "", params$monotone_constraints), ",")[[1]])
  stopifnot(n_constraints == ncol(X))
  stopifnot(identical(colnames(X), torp:::WP_MODEL_FEATURES))
}

#' Fit the WP model with match-grouped CV to pick nrounds
#'
#' @param model_data_wp Output of [build_wp_data()] (cv path) or
#'   `add_epv_vars() |> clean_model_data_wp()` (insample path).
#' @param params WP xgboost params, default [wp_params()].
#' @param nrounds_max Integer cap for CV/final training rounds.
#' @return list(model, optimal_nrounds, cv_logloss, X, y, folds)
#' @keywords internal
fit_wp <- function(model_data_wp, params = wp_params(), nrounds_max = 500L) {
  wp_vars <- model_data_wp |> torp:::select_wp_model_vars()
  X <- stats::model.matrix(~ . + 0, data = wp_vars, na.action = na.pass)
  stopifnot(nrow(X) == nrow(wp_vars))
  stopifnot(identical(colnames(X), torp:::WP_MODEL_FEATURES))
  validate_wp_spec(X, params)

  y <- model_data_wp$label_wp
  row_folds <- make_match_folds(model_data_wp$torp_match_id)
  folds <- lapply(seq_len(max(row_folds)), function(k) which(row_folds == k))

  full_train <- xgboost::xgb.DMatrix(data = X, label = y)

  cli::cli_inform("Running WP 5-fold CV (match-grouped)...")
  set.seed(1234)
  cv_result <- xgboost::xgb.cv(
    params = params, data = full_train, nrounds = nrounds_max,
    folds = folds, early_stopping_rounds = 20, print_every_n = 20, verbose = 1
  )

  optimal_nrounds <- cv_result$best_iteration
  if (is.null(optimal_nrounds) || length(optimal_nrounds) == 0) {
    optimal_nrounds <- which.min(cv_result$evaluation_log$test_logloss_mean)
  }
  cv_logloss <- min(cv_result$evaluation_log$test_logloss_mean)
  cli::cli_inform("WP optimal nrounds: {optimal_nrounds} | CV logloss: {round(cv_logloss, 6)}")

  set.seed(1234)
  model <- xgboost::xgb.train(params = params, data = full_train,
                              nrounds = optimal_nrounds, print_every_n = 10)

  list(model = model, optimal_nrounds = optimal_nrounds, cv_logloss = cv_logloss,
       X = X, y = y, folds = folds)
}

#' Generate out-of-sample WP predictions via k-fold CV (binary objective)
#'
#' Sibling of [cv_ep_oos_preds()] for the binary WP objective: same
#' per-fold-refit pattern, returns a full-length OOS probability vector.
#' Report-only diagnostic (random-CV slope and future comparisons) -- the
#' temporal path in [fit_wp_temporal_variant()] is the load-bearing one for
#' the release gate (torpverse/docs/plans/FABLE-RECAL-PLAN.md Step 1).
#'
#' @param X WP feature matrix.
#' @param y WP labels (0/0.5/1).
#' @param folds List of row-index vectors per fold (test indices).
#' @param params WP xgboost params used to refit each fold.
#' @param nrounds Rounds to use for each fold's fit (the tuned optimal_nrounds).
#' @return Numeric vector, OOS probability per row.
#' @keywords internal
cv_wp_oos_preds <- function(X, y, folds, params, nrounds) {
  oos_preds <- rep(NA_real_, nrow(X))

  for (k in seq_along(folds)) {
    cli::cli_inform("WP OOS fold {k}/{length(folds)}...")
    test_idx <- folds[[k]]
    train_idx <- setdiff(seq_len(nrow(X)), test_idx)

    dtrain <- xgboost::xgb.DMatrix(data = X[train_idx, ], label = y[train_idx])
    dtest <- xgboost::xgb.DMatrix(data = X[test_idx, ])

    set.seed(1234)
    fold_model <- xgboost::xgb.train(params = params, data = dtrain, nrounds = nrounds, verbose = 0)

    preds_raw <- predict(fold_model, dtest)
    if (is.matrix(preds_raw)) preds_raw <- as.vector(preds_raw)
    oos_preds[test_idx] <- preds_raw
    rm(fold_model, dtrain, dtest)
  }

  stopifnot(!anyNA(oos_preds))
  oos_preds
}

#' Score EPV rows through an already-fitted EP model, then an already-fitted
#' WP model
#'
#' Shared plumbing for [fit_wp_temporal_variant()]'s gate-season predictions
#' and the trainer's report-only in-progress-season check: builds EP
#' features, predicts with `ep_model`, builds WP features via
#' [build_wp_data()], predicts with `wp_model`. Never touches
#' torpmodels/the network -- both models are passed in already fitted, and
#' prediction goes through the plain-matrix `predict()` path (mirrors
#' torp's serving convention in `get_epv_preds()`/`get_wp_preds()`, not the
#' `xgb.DMatrix` CV-refit path).
#'
#' @param epv_rows Rows from [load_training_pbp()] (or a subset), i.e. the
#'   pre-`add_epv_vars()`/`build_wp_data()` model_data_epv shape.
#' @param ep_model A fitted EP `xgb.Booster`.
#' @param wp_model A fitted WP `xgb.Booster`.
#' @return `list(preds, labels, meta_cols)` -- `meta_cols` carries `period`,
#'   `points_diff`, `est_match_elapsed`, `match_id` for the gate cell.
#' @keywords internal
score_wp_rows <- function(epv_rows, ep_model, wp_model) {
  epv_vars <- epv_rows |> torp:::select_epv_model_vars()
  X_ep <- stats::model.matrix(~ . + 0, data = epv_vars, na.action = na.pass)
  ep_preds <- predict(ep_model, X_ep)
  if (!is.matrix(ep_preds)) ep_preds <- matrix(ep_preds, ncol = 5, byrow = TRUE)
  colnames(ep_preds) <- c("opp_goal", "opp_behind", "behind", "goal", "no_score")

  wp_data <- build_wp_data(epv_rows, ep_preds)
  wp_vars <- wp_data |> torp:::select_wp_model_vars()
  X_wp <- stats::model.matrix(~ . + 0, data = wp_vars, na.action = na.pass)
  preds <- predict(wp_model, X_wp)
  if (is.matrix(preds)) preds <- as.vector(preds)

  list(
    preds = preds,
    labels = wp_data$label_wp,
    meta_cols = wp_data |> dplyr::select("period", "points_diff",
                                         "est_match_elapsed", "match_id")
  )
}

#' Fit the WP temporal variant and score it on the held-out gate season
#'
#' Mirrors torpverse/docs/reviews/FABLE-WP-EXPERIMENTS.md \enc{§}{Section}5's protocol exactly:
#' EP is trained on seasons strictly before `gate_season` (CV nrounds inside
#' that window), 5-fold OOS EP predictions feed WP feature construction, WP
#' is trained on the same window, and the fitted pair is then used to score
#' `gate_season` -- honest recent-season OOS predictions, since neither
#' model saw a single `gate_season` row in training.
#'
#' @param model_data_epv Output of [load_training_pbp()] (full window).
#' @param gate_season Integer. The held-out season (`S_gate`, typically
#'   `max(seasons)`).
#' @param params_ep EP xgboost params, default [ep_params()].
#' @param params_wp WP xgboost params, default [wp_params()].
#' @return `list(preds, labels, meta_cols)` for `gate_season` rows -- see
#'   [score_wp_rows()].
#' @keywords internal
fit_wp_temporal_variant <- function(model_data_epv, gate_season,
                                    params_ep = ep_params(), params_wp = wp_params()) {
  data <- model_data_epv
  if (!"season" %in% names(data)) {
    data$season <- as.numeric(substr(data$match_id, 5, 8))
  }

  train_data <- data |> dplyr::filter(.data$season < gate_season)
  gate_data  <- data |> dplyr::filter(.data$season == gate_season)
  if (nrow(train_data) == 0) {
    cli::cli_abort("fit_wp_temporal_variant: no rows with season < {gate_season}")
  }
  if (nrow(gate_data) == 0) {
    cli::cli_abort("fit_wp_temporal_variant: no rows for gate_season {gate_season}")
  }

  train_seasons <- sort(unique(train_data$season))
  cli::cli_inform("Temporal variant: EP+WP trained on {min(train_seasons)}-{max(train_seasons)}, scored on gate season {gate_season}")

  # 1. EP on seasons < gate_season only (CV nrounds inside the window)
  ep_fit <- fit_ep(train_data, params = params_ep)

  # 2. 5-fold OOS EP preds within the training window
  row_folds <- make_match_folds(train_data$torp_match_id)
  ep_oos <- cv_ep_oos_preds(ep_fit$X, ep_fit$y, ep_fit$folds, row_folds, params_ep, ep_fit$optimal_nrounds)

  # 3. WP features built from the OOS EP preds, WP trained on the window
  model_data_wp_train <- build_wp_data(train_data, ep_oos)
  wp_fit <- fit_wp(model_data_wp_train, params = params_wp)

  # 4. Predict gate-season rows -- EP features come from the temporal EP
  #    model, which never saw gate_season. Honest recent-season OOS.
  score_wp_rows(gate_data, ep_fit$model, wp_fit$model)
}

#' Fit the two-parameter Platt-on-logit calibration (D1)
#'
#' `p' = plogis(a + b * qlogis(p))`. Draw rows (`label == 0.5`) are dropped
#' before fitting, matching torpverse/docs/reviews/FABLE-WP-EXPERIMENTS.md's convention ("kept for
#' logloss, excluded from calibration slopes").
#'
#' @param preds Numeric vector of raw (uncalibrated) WP predictions.
#' @param labels Numeric vector, same length, values in `{0, 0.5, 1}`.
#' @return `list(a, b)`.
#' @keywords internal
fit_wp_calibration <- function(preds, labels) {
  keep <- labels %in% c(0, 1)
  y <- labels[keep]
  p <- pmin(pmax(preds[keep], 1e-6), 1 - 1e-6)

  fit <- stats::glm(y ~ stats::qlogis(p), family = stats::binomial())
  co <- unname(stats::coef(fit))
  a <- co[1]
  b <- co[2]

  stopifnot(is.finite(a), is.finite(b), b > 0)
  list(a = a, b = b)
}

#' Calibration-slope gate statistic (D5)
#'
#' `cell = "all"`: row-level `glm(label ~ qlogis(pred), binomial)` slope
#' over every non-draw row. `cell = "q4close"`: the same GLM restricted to
#' `period == 4 & abs(points_diff) <= 12`, after the anti-pseudoreplication
#' dedup convention quoted in torpverse/docs/plans/FABLE-RECAL-PLAN.md D5 -- bucket rows into
#' 5-minute buckets (`pmin(4, pmax(0, est_match_elapsed %/% 300 - 11))`)
#' and keep only the last row per `(match_id, bucket)` before fitting.
#'
#' @param preds Numeric vector of (typically calibrated) WP predictions.
#' @param labels Numeric vector, same length, values in `{0, 0.5, 1}`.
#' @param meta_cols Data frame with `period`, `points_diff`,
#'   `est_match_elapsed`, `match_id` columns, same row order/length as
#'   `preds`/`labels` (see [score_wp_rows()]).
#' @param cell One of `"q4close"` (default) or `"all"`.
#' @return Numeric scalar slope, or `NA_real_` if the cell is degenerate
#'   (too few rows / a single class).
#' @keywords internal
wp_gate_slope <- function(preds, labels, meta_cols, cell = c("q4close", "all")) {
  cell <- match.arg(cell)

  d <- data.frame(
    pred = preds, label_wp = labels,
    period = meta_cols$period, points_diff = meta_cols$points_diff,
    est_match_elapsed = meta_cols$est_match_elapsed, match_id = meta_cols$match_id
  )
  d <- d[d$label_wp %in% c(0, 1), ]

  if (cell == "q4close") {
    d <- d[d$period == 4 & abs(d$points_diff) <= 12, ]
    d$bucket <- pmin(4, pmax(0, d$est_match_elapsed %/% 300 - 11))
    d <- d |>
      dplyr::group_by(.data$match_id, .data$bucket) |>
      dplyr::summarise(label_wp = dplyr::last(.data$label_wp),
                       pred = dplyr::last(.data$pred), .groups = "drop")
  }

  if (nrow(d) < 2 || length(unique(d$label_wp)) < 2) return(NA_real_)

  p <- pmin(pmax(d$pred, 1e-6), 1 - 1e-6)
  fit <- stats::glm(d$label_wp ~ stats::qlogis(p), family = stats::binomial())
  unname(stats::coef(fit)[2])
}

#' The temporal Q4/close release gate (D5) -- aborts before anything is
#' written or published
#'
#' Gates BOTH the all-rows and Q4/close-cell calibrated slopes at the same
#' threshold: `|slope - 1| <= threshold`. Fires as `cli::cli_abort()` in the
#' same idiom as [validate_wp_spec()]/`publish_model_group()` -- a loud
#' bulleted message with the measured slopes, before any `saveRDS`/upload.
#'
#' @param calibrated_preds Numeric vector of calibrated WP predictions.
#' @param labels Numeric vector, same length, values in `{0, 0.5, 1}`.
#' @param meta_cols Data frame, see [wp_gate_slope()].
#' @param threshold Numeric, default `0.10` (D5).
#' @return `list(slope_all, slope_q4close)`, invisibly reachable via return
#'   value only on a pass (the function aborts on a breach).
#' @keywords internal
validate_wp_temporal_slope <- function(calibrated_preds, labels, meta_cols, threshold = 0.10) {
  slope_all <- wp_gate_slope(calibrated_preds, labels, meta_cols, cell = "all")
  slope_q4close <- wp_gate_slope(calibrated_preds, labels, meta_cols, cell = "q4close")

  ok_all <- !is.na(slope_all) && abs(slope_all - 1) <= threshold
  ok_q4close <- !is.na(slope_q4close) && abs(slope_q4close - 1) <= threshold

  if (!ok_all || !ok_q4close) {
    cli::cli_abort(c(
      "WP temporal slope gate FAILED -- |slope - 1| must be <= {threshold}",
      "x" = "slope_all = {round(slope_all, 3)} ({if (ok_all) 'ok' else 'BREACH'})",
      "x" = "slope_q4close = {round(slope_q4close, 3)} ({if (ok_q4close) 'ok' else 'BREACH'})",
      "i" = "Nothing written or published. Emergency override: train_models.R --skip-slope-gate."
    ))
  }

  cli::cli_alert_success(
    "WP temporal slope gate passed: slope_all = {round(slope_all, 3)}, slope_q4close = {round(slope_q4close, 3)} (threshold {threshold})"
  )

  list(slope_all = slope_all, slope_q4close = slope_q4close)
}

#' Fit the shot outcome model (ordered categorical GAM)
#'
#' Body ported from train_shot_model.R. Window unification: the historical
#' script used `seasons = TRUE` (all seasons) and rebuild used
#' `training_seasons`; canonical is the `seasons` argument, default
#' [default_training_seasons()].
#'
#' @param seasons Integer vector of seasons to train on.
#' @return list(model, shot_player_df)
#' @keywords internal
fit_shot <- function(seasons = default_training_seasons()) {
  cli::cli_inform("Loading play-by-play data for shot model...")
  shots_prep <- torp::load_pbp(seasons = seasons, rounds = TRUE)

  shots <- shots_prep |>
    dplyr::filter(!is.na(shot_at_goal), x > 0, goal_x < 65, abs_y < 45) |>
    dplyr::mutate(
      scored_shot = ifelse(!is.na(points_shot), 1, 0),
      shot_cat = dplyr::case_when(
        is.na(points_shot) ~ 1,
        points_shot == 1 ~ 2,
        points_shot == 6 ~ 3
      )
    )

  shots$player_id_shot <- forcats::fct_lump_min(shots$player_id, 10, other_level = "Other")

  player_name_mapping <- shots |>
    dplyr::group_by(player_id_shot = player_id) |>
    dplyr::summarise(player_name_shot = dplyr::last(player_name))

  shot_player_df <- tibble::tibble(
    player_id_shot = levels(shots$player_id_shot)
  ) |>
    dplyr::left_join(player_name_mapping, by = "player_id_shot")

  cli::cli_inform("Training shot model ({nrow(shots)} shots)...")
  shot_ocat_mdl <- mgcv::bam(
    shot_cat ~
      ti(goal_x, abs_y, by = phase_of_play, bs = "ts")
      + ti(goal_x, abs_y, bs = "ts")
      + s(goal_x, bs = "ts")
      + s(abs_y, bs = "ts")
      + ti(lag_goal_x, lag_y)
      + s(lag_goal_x, bs = "ts")
      + s(lag_y, bs = "ts")
      + s(play_type, bs = "re")
      + s(phase_of_play, bs = "re")
      + s(player_position_fac, bs = "re")
      + s(player_id_shot, bs = "re"),
    data = shots,
    family = ocat(R = 3),
    nthreads = 4,
    select = TRUE,
    discrete = TRUE,
    drop.unused.levels = FALSE
  )

  list(model = shot_ocat_mdl, shot_player_df = shot_player_df)
}

#' Train (and optionally publish) EP/WP/shot models through one canonical path
#'
#' Orchestrator used by BOTH train_models.R and rebuild_everything.R Phase 4.
#' Trains in dependency order (EP before WP, since WP consumes EP
#' predictions), stamps provenance on every model, saves with the existing
#' filenames, and -- when `upload = TRUE` -- publishes each model group
#' atomically via [torpmodels::publish_model_group()].
#'
#' @param models Character vector, subset of `c("ep", "wp", "shot")`.
#' @param seasons Integer vector of training seasons, default
#'   [default_training_seasons()].
#' @param upload Logical. Publish to GitHub releases after training.
#' @param wp_ep_source `"cv"` (canonical: 5-fold out-of-sample EP predictions
#'   feed WP training) or `"insample"` (legacy `train_wp_model.R` semantics
#'   -- comparison/debug only; forces `upload = FALSE`).
#' @param slope_gate Logical, default `TRUE`. The temporal Q4/close release
#'   gate (torpverse/docs/plans/FABLE-RECAL-PLAN.md D5) -- `FALSE` disables it (emergencies
#'   only; the calibration still fits and slopes still print, just unfenced).
#' @param calibrate Logical, default `TRUE`. Fits the WP recalibration layer
#'   (D1-D3) and its release gate. `FALSE` (or `wp_ep_source = "insample"`)
#'   skips the temporal variant + calibration entirely and forces
#'   `upload = FALSE` -- an uncalibrated WP model can be built locally but
#'   never published through the front door.
#' @param output_dir Directory to save trained artifacts.
#' @return Named list of `torp_meta` objects, one per trained model.
#' @keywords internal
train_core_models <- function(models = c("ep", "wp", "shot"),
                              seasons = default_training_seasons(),
                              upload = TRUE,
                              wp_ep_source = c("cv", "insample"),
                              slope_gate = TRUE,
                              calibrate = TRUE,
                              output_dir = file.path("inst", "models", "core")) {
  models <- match.arg(models, c("ep", "wp", "shot"), several.ok = TRUE)
  wp_ep_source <- match.arg(wp_ep_source, c("cv", "insample"))

  if (wp_ep_source == "insample" && upload) {
    cli::cli_alert_warning("wp_ep_source = 'insample' forces upload = FALSE (comparison/debug only -- the legacy train_wp_model.R semantics)")
    upload <- FALSE
  }

  skip_calibration <- !calibrate || wp_ep_source == "insample"
  if (skip_calibration && upload) {
    cli::cli_alert_warning("calibrate = FALSE (or wp_ep_source = 'insample') forces upload = FALSE -- an uncalibrated WP model can never publish through the front door")
    upload <- FALSE
  }
  if (!isTRUE(slope_gate)) {
    cli::cli_alert_warning("!! slope_gate = FALSE -- the temporal Q4/close release gate is DISABLED. Emergency use only.")
  }

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  results <- list()
  model_data_epv <- NULL
  ep_fit <- NULL

  if (any(c("ep", "wp") %in% models) || (("wp" %in% models) && wp_ep_source == "cv")) {
    model_data_epv <- load_training_pbp(seasons)
  }

  if ("ep" %in% models) {
    cli::cli_h2("Training EP model")
    ep_fit <- fit_ep(model_data_epv)
    ep_meta <- torpmodels:::build_model_meta(
      "ep", seasons, ep_params(), torp:::EPV_MODEL_FEATURES,
      cv_metric = ep_fit$cv_logloss, n_rows = nrow(model_data_epv),
      n_matches = length(unique(model_data_epv$torp_match_id)),
      extra = list(script = "train_models.R")
    )
    saveRDS(torpmodels:::stamp_model_meta(ep_fit$model, ep_meta),
            file.path(output_dir, "ep_model.rds"))
    if (upload) torpmodels::publish_model_group("ep", output_dir)
    results$ep <- ep_meta
  }

  if ("wp" %in% models) {
    cli::cli_h2("Training WP model")

    if (wp_ep_source == "cv") {
      if (is.null(ep_fit)) ep_fit <- fit_ep(model_data_epv)
      row_folds <- make_match_folds(model_data_epv$torp_match_id)
      oos_preds <- cv_ep_oos_preds(ep_fit$X, ep_fit$y, ep_fit$folds, row_folds,
                                    ep_params(), ep_fit$optimal_nrounds)
      model_data_wp <- build_wp_data(model_data_epv, oos_preds)
      ep_source_label <- "cv"
    } else {
      model_data_wp <- model_data_epv |> torp::add_epv_vars() |> torp:::clean_model_data_wp()
      ep_source_label <- "insample"
    }

    wp_fit <- fit_wp(model_data_wp)

    if (!skip_calibration) {
      gate_season <- max(seasons)
      cli::cli_h3("WP recalibration + temporal slope gate (gate season {gate_season})")

      temporal <- fit_wp_temporal_variant(model_data_epv, gate_season,
                                          params_ep = ep_params(), params_wp = wp_params())

      slope_all_before <- wp_gate_slope(temporal$preds, temporal$labels, temporal$meta_cols, "all")
      slope_q4close_before <- wp_gate_slope(temporal$preds, temporal$labels, temporal$meta_cols, "q4close")
      cli::cli_inform("Uncalibrated temporal slopes: all = {round(slope_all_before, 3)}, q4close = {round(slope_q4close_before, 3)}")

      calib <- fit_wp_calibration(temporal$preds, temporal$labels)
      cli::cli_inform("Fitted calibration: a = {round(calib$a, 4)}, b = {round(calib$b, 4)}")

      calibrated_preds <- stats::plogis(calib$a + calib$b * stats::qlogis(temporal$preds))

      if (isTRUE(slope_gate)) {
        gated <- validate_wp_temporal_slope(calibrated_preds, temporal$labels, temporal$meta_cols, threshold = 0.10)
        slope_all_after <- gated$slope_all
        slope_q4close_after <- gated$slope_q4close
      } else {
        slope_all_after <- wp_gate_slope(calibrated_preds, temporal$labels, temporal$meta_cols, "all")
        slope_q4close_after <- wp_gate_slope(calibrated_preds, temporal$labels, temporal$meta_cols, "q4close")
        cli::cli_alert_warning("Slope gate skipped (slope_gate = FALSE): all = {round(slope_all_after, 3)}, q4close = {round(slope_q4close_after, 3)}")
      }

      wp_calibration_obj <- list(
        a = calib$a, b = calib$b, formula = "plogis(a + b*qlogis(p))",
        fitted_on = "temporal-oos", gate_season = gate_season,
        n_fit = length(temporal$labels),
        slope_before = slope_all_before, slope_after = slope_all_after,
        slope_q4close_before = slope_q4close_before, slope_q4close_after = slope_q4close_after
      )
      wp_calibration_meta <- torpmodels:::build_model_meta(
        "wp_calibration", seasons, list(formula = "plogis(a + b*qlogis(p))"), c("a", "b"),
        n_rows = length(temporal$labels),
        extra = list(
          script = "train_models.R", a = calib$a, b = calib$b,
          gate_season = gate_season, n_fit = length(temporal$labels),
          slope_before = slope_all_before, slope_after = slope_all_after,
          slope_q4close_before = slope_q4close_before, slope_q4close_after = slope_q4close_after
        )
      )
      saveRDS(torpmodels:::stamp_model_meta(wp_calibration_obj, wp_calibration_meta),
              file.path(output_dir, "wp_calibration.rds"))
      results$wp_calibration <- wp_calibration_meta

      # Report-only diagnostics -- printed, never gated (D2/D5).
      row_folds_wp <- make_match_folds(model_data_wp$torp_match_id)
      folds_wp <- lapply(seq_len(max(row_folds_wp)), function(k) which(row_folds_wp == k))
      cv_oos <- cv_wp_oos_preds(wp_fit$X, wp_fit$y, folds_wp, wp_params(), wp_fit$optimal_nrounds)
      cv_oos_calibrated <- stats::plogis(calib$a + calib$b * stats::qlogis(cv_oos))
      meta_full <- model_data_wp |> dplyr::select("period", "points_diff",
                                                   "est_match_elapsed", "match_id")
      cli::cli_inform(
        "[report-only, not gated] Random-CV calibrated slope: all = {round(wp_gate_slope(cv_oos_calibrated, model_data_wp$label_wp, meta_full, 'all'), 3)}, q4close = {round(wp_gate_slope(cv_oos_calibrated, model_data_wp$label_wp, meta_full, 'q4close'), 3)}"
      )

      in_progress_season <- gate_season + 1L
      ip_chains <- tryCatch(torp::load_chains(seasons = in_progress_season, rounds = TRUE), error = function(e) NULL)
      if (is.null(ip_chains) || nrow(ip_chains) == 0) {
        cli::cli_inform("[report-only, not gated] In-progress season {in_progress_season}: no rows available yet")
      } else {
        ip_epv <- torp::clean_pbp(ip_chains) |> torp:::clean_model_data_epv()
        ip_scored <- score_wp_rows(ip_epv, ep_fit$model, wp_fit$model)
        ip_preds_cal <- stats::plogis(calib$a + calib$b * stats::qlogis(ip_scored$preds))
        cli::cli_inform(
          "[report-only, not gated] In-progress season {in_progress_season}: calibrated slope all = {round(wp_gate_slope(ip_preds_cal, ip_scored$labels, ip_scored$meta_cols, 'all'), 3)}, q4close = {round(wp_gate_slope(ip_preds_cal, ip_scored$labels, ip_scored$meta_cols, 'q4close'), 3)} (n = {length(ip_scored$labels)})"
        )
      }
    } else {
      cli::cli_alert_warning("WP calibration skipped (calibrate = FALSE or wp_ep_source = 'insample') -- shipping uncalibrated WP model, local-only")
    }

    wp_meta <- torpmodels:::build_model_meta(
      "wp", seasons, wp_params(), torp:::WP_MODEL_FEATURES,
      cv_metric = wp_fit$cv_logloss, n_rows = nrow(model_data_wp),
      n_matches = length(unique(model_data_wp$torp_match_id)),
      extra = list(script = "train_models.R", ep_source = ep_source_label)
    )
    saveRDS(torpmodels:::stamp_model_meta(wp_fit$model, wp_meta),
            file.path(output_dir, "wp_model.rds"))
    if (upload) torpmodels::publish_model_group("wp", output_dir)
    results$wp <- wp_meta
  }

  if ("shot" %in% models) {
    cli::cli_h2("Training shot model")
    shot_fit <- fit_shot(seasons)
    shot_meta <- torpmodels:::build_model_meta(
      "shot", seasons, list(family = "ocat(R=3)"), NA_character_,
      n_rows = nrow(shot_fit$shot_player_df),
      extra = list(script = "train_models.R")
    )
    saveRDS(torpmodels:::stamp_model_meta(shot_fit$model, shot_meta),
            file.path(output_dir, "shot_ocat_mdl.rds"))
    saveRDS(shot_fit$shot_player_df, file.path(output_dir, "shot_player_df.rds"))
    if (upload) torpmodels::publish_model_group("shot", output_dir)
    results$shot <- shot_meta
  }

  results
}
