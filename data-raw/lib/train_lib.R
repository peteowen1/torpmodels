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

#' Abort loudly if EP/WP training features contain NAs
#'
#' `na.action = na.pass` in [fit_ep()]/[fit_wp()]/[score_wp_rows()] was
#' meant (torpverse/docs/plans/TRAINING-CONSOLIDATION-PLAN.md
#' \enc{§}{Section}1.5, "Model-matrix decision") to keep every row aligned
#' through `stats::model.matrix()`. It's actually inert: `na.action` is not
#' a formal argument of `stats:::model.matrix.default()` (it has only
#' `object, data, contrasts.arg, xlev, ...`), so it's silently swallowed by
#' `...` and `model.matrix()` falls back to `getOption("na.action")`
#' (`na.omit` by default) -- which DROPS the NA row from the returned
#' matrix, exactly like the deleted standalone trainers
#' (`train_ep_model.R`/`train_wp_model.R`, git history retains them) did
#' with their implicit default. Verified empirically: `model.matrix(~ . + 0,
#' data = df, na.action = na.pass)` on a 5-row frame with one NA returns 4
#' rows, identical to `na.action = na.omit`; only pre-building the
#' model.frame with `na.action = na.pass` and passing THAT to
#' `model.matrix()` actually preserves the row.
#'
#' Consequence: in [fit_ep()]/[fit_wp()] the row-drop is still caught --
#' `stopifnot(nrow(X) == nrow(vars))` fires immediately after
#' `model.matrix()` -- so those callers were never actually exposed (a
#' feature NA aborts training today, just via an opaque assertion message).
#' This check runs BEFORE `model.matrix()`, directly on the source
#' dataframe, so the abort names the offending columns instead of just
#' failing an assertion, and so it stays correct if `model.matrix()`'s
#' na-handling is ever "fixed" to actually honor `na.pass` (which would
#' make the post-hoc `nrow` stopifnot permanently true and blind). In
#' [score_wp_rows()], however, there is NO such stopifnot -- a feature NA
#' there silently drops a row from `X_ep`/`X_wp`, so `predict()` returns
#' fewer rows than `epv_rows`, and [build_wp_data()] recycles the short
#' vector against the full-length frame with no error. That path (which
#' feeds the gate-season score in [fit_wp_temporal_variant()], i.e. the WP
#' release gate) was the one actually exposed; this guard closes it.
#'
#' `clean_model_data_epv()`/`clean_model_data_wp()` are designed to leave
#' every `EPV_MODEL_FEATURES`/`WP_MODEL_FEATURES` column fully populated
#' (boundary lag NAs filled via locf/nocb/first/last, `speed1`/`speed5`
#' NaNs zeroed -- see `torp/R/clean_features.R`), and no feature column
#' here is documented as intentionally missing. So any NA reaching this
#' point means upstream cleaning regressed, not a deliberate xgboost
#' missing-value signal.
#'
#' @param vars Feature dataframe, pre-`model.matrix()` (e.g. the output of
#'   `select_epv_model_vars()`/`select_wp_model_vars()`).
#' @param model_label Character, prefixes the abort message (e.g. `"EP"`).
#' @keywords internal
abort_on_feature_na <- function(vars, model_label) {
  na_counts <- vapply(vars, function(col) sum(is.na(col)), integer(1))
  bad <- na_counts[na_counts > 0]
  if (length(bad) == 0) return(invisible(NULL))

  bullets <- paste0(names(bad), ": ", bad, " NA row", ifelse(bad == 1, "", "s"))
  names(bullets) <- rep("x", length(bullets))
  cli::cli_abort(c(
    "{model_label} features have NAs in {length(bad)} column{?s} -- clean_pbp()/clean_model_data_epv()/clean_model_data_wp() are designed to leave these columns fully populated, so this looks like a data-cleaning regression upstream, not intentional xgboost missingness",
    bullets,
    "i" = "If an NA here is genuinely intentional (a new xgboost-missing signal), update this guard rather than silently ignoring it."
  ))
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
  abort_on_feature_na(epv_vars, "EP")
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
  abort_on_feature_na(wp_vars, "WP")
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
#'   `points_diff`, `est_match_elapsed`, `est_match_remaining`, `match_id`
#'   for the gate cell (`est_match_remaining` feeds [wp_leverage_weight()]).
#' @keywords internal
score_wp_rows <- function(epv_rows, ep_model, wp_model) {
  epv_vars <- epv_rows |> torp:::select_epv_model_vars()
  abort_on_feature_na(epv_vars, "EP (score_wp_rows)")
  X_ep <- stats::model.matrix(~ . + 0, data = epv_vars, na.action = na.pass)
  ep_preds <- predict(ep_model, X_ep)
  if (!is.matrix(ep_preds)) ep_preds <- matrix(ep_preds, ncol = 5, byrow = TRUE)
  colnames(ep_preds) <- c("opp_goal", "opp_behind", "behind", "goal", "no_score")

  wp_data <- build_wp_data(epv_rows, ep_preds)
  wp_vars <- wp_data |> torp:::select_wp_model_vars()
  abort_on_feature_na(wp_vars, "WP (score_wp_rows)")
  X_wp <- stats::model.matrix(~ . + 0, data = wp_vars, na.action = na.pass)
  preds <- predict(wp_model, X_wp)
  if (is.matrix(preds)) preds <- as.vector(preds)

  list(
    preds = preds,
    labels = wp_data$label_wp,
    meta_cols = wp_data |> dplyr::select("period", "points_diff",
                                         "est_match_elapsed", "est_match_remaining",
                                         "match_id")
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

  # 3. WP features built from the OOS EP preds, WP trained on the window.
  #    S5_exclude_draws: mirrors the production WP branch's draw-drop
  #    (training only) so the temporal variant honestly reflects what
  #    production training does -- gate_data/score_wp_rows() below keeps
  #    draws for eval, unchanged.
  model_data_wp_train <- build_wp_data(train_data, ep_oos)
  model_data_wp_train <- model_data_wp_train[model_data_wp_train$label_wp %in% c(0, 1), ]
  wp_fit <- fit_wp(model_data_wp_train, params = params_wp)

  # 4. Predict gate-season rows -- EP features come from the temporal EP
  #    model, which never saw gate_season. Honest recent-season OOS.
  score_wp_rows(gate_data, ep_fit$model, wp_fit$model)
}

#' Path B continuous leverage weight (FABLE-RECAL-PLAN.md \enc{§}{Section}7)
#'
#' Pre-registered, NOT a fitting parameter: `w = pmax(0, 1 -
#' minutes_remaining/ramp_mins) * pmax(0, 1 - abs(points_diff)/margin_cap)`,
#' `ramp_mins = 20`, `margin_cap = 18` fixed by design. Continuous in both
#' match-time and margin -- no step at the Q3->Q4 boundary or a hard margin
#' threshold the way [wp_cell_flag()]'s binary Q4/close cell has (margin
#' still steps at real scoring events, which is genuine signal, not an
#' artifact); rationale is WPA consumes DIFFERENCES of consecutive
#' predictions, so a discontinuous calibration would spike/zero it right at
#' the boundary. `w` is ~1 deep in the Q4/close cell and 0 through the bulk
#' of the first three quarters.
#'
#' `minutes_remaining` maps to `est_match_remaining / 60` --
#' `WP_MODEL_FEATURES`' own match-time-remaining column
#' (`torp/R/clean_pbp.R`: `est_match_remaining = AFL_PLAY_GAME_SECONDS -
#' est_match_elapsed`, in seconds; `AFL_PLAY_GAME_SECONDS = 4800` = 4 x
#' `AFL_PLAY_QUARTER_SECONDS` (1200), so `ramp_mins = 20` is exactly one
#' quarter of playing time) -- the identical variable the WP model itself
#' trains on, not a new feature.
#'
#' @param minutes_remaining Numeric vector, match time remaining in minutes
#'   (`est_match_remaining / 60`).
#' @param points_diff Numeric vector, signed scoring margin.
#' @param ramp_mins Numeric, default `20` (fixed, not fitted).
#' @param margin_cap Numeric, default `18` (fixed, not fitted).
#' @return Numeric vector in `[0, 1]`, same length as the inputs.
#' @keywords internal
wp_leverage_weight <- function(minutes_remaining, points_diff, ramp_mins = 20, margin_cap = 18) {
  pmax(0, 1 - minutes_remaining / ramp_mins) * pmax(0, 1 - abs(points_diff) / margin_cap)
}

#' Fit the Platt-on-logit calibration: global, Q4/close-interaction, or
#' Path B leverage-interaction form
#'
#' Global (2 params): `p' = plogis(a + b * qlogis(p))`. Q4/close interaction
#' (D1's original pre-authorized escalation, 4 params, kept for backward
#' compatibility/debugging -- retired as of FABLE-RECAL-PLAN.md \enc{§}{Section}7 Path B, no
#' longer fitted by [train_core_models()]):
#' `glm(label ~ qlogis(pred) * is_q4close, binomial)`, i.e.
#' `p' = plogis((a + a_q4c*I) + (b + b_q4c*I) * qlogis(p))` with
#' `I = is_q4close`. Leverage interaction (\enc{§}{Section}7 Path B, the form
#' [train_core_models()] actually fits and ships -- a single global slope
#' was measured unable to satisfy both gate cells simultaneously, see \enc{§}{Section}7's
#' evidence): `glm(label ~ qlogis(pred) + I(w*qlogis(pred)), binomial)`, i.e.
#' `p' = plogis(a + (b + c*w) * qlogis(p))` with `w` from
#' [wp_leverage_weight()]. Per \enc{§}{Section}7: do NOT go further (no splines, no
#' per-quarter models -- that is retraining by another name), and do not
#' tune `ramp_mins`/`margin_cap` against the gate.
#'
#' Draw rows (`label == 0.5`) are dropped before fitting, matching
#' FABLE-WP-EXPERIMENTS.md's convention ("kept for logloss, excluded from
#' calibration slopes"). All forms return the same shape so the shipped
#' artifact schema is uniform: unused fields are 0.
#'
#' @param preds Numeric vector of raw (uncalibrated) WP predictions.
#' @param labels Numeric vector, same length, values in `{0, 0.5, 1}`.
#' @param is_q4close Logical vector, same length as `labels`: the plan's
#'   exact cell (`period == 4 & abs(points_diff) <= 12`, see
#'   [wp_cell_flag()]). Required by (and only used by) the
#'   `"q4close_interaction"` form; `NA` is treated as out-of-cell.
#' @param w Numeric vector, same length as `labels`: [wp_leverage_weight()].
#'   Required by (and only used by) the `"leverage_interaction_v1"` form.
#' @param form `"global"` (default), `"q4close_interaction"`, or
#'   `"leverage_interaction_v1"`.
#' @return `list(a, b, c, a_q4c, b_q4c, form)`.
#' @keywords internal
fit_wp_calibration <- function(preds, labels, is_q4close = NULL, w = NULL,
                               form = c("global", "q4close_interaction", "leverage_interaction_v1")) {
  form <- match.arg(form)
  keep <- labels %in% c(0, 1)
  y <- labels[keep]
  p <- pmin(pmax(preds[keep], 1e-6), 1 - 1e-6)
  x <- stats::qlogis(p)
  c_coef <- 0

  if (form == "global") {
    fit <- stats::glm(y ~ x, family = stats::binomial())
    co <- unname(stats::coef(fit))
    a <- co[1]
    b <- co[2]
    a_q4c <- 0
    b_q4c <- 0
    stopifnot(is.finite(a), is.finite(b), b > 0)
  } else if (form == "q4close_interaction") {
    if (is.null(is_q4close)) {
      cli::cli_abort("fit_wp_calibration(form = 'q4close_interaction') requires {.arg is_q4close}")
    }
    stopifnot(length(is_q4close) == length(labels))
    I <- as.numeric(is_q4close[keep])
    I[is.na(I)] <- 0
    fit <- stats::glm(y ~ x * I, family = stats::binomial())
    co <- stats::coef(fit)
    a <- unname(co[["(Intercept)"]])
    b <- unname(co[["x"]])
    a_q4c <- unname(co[["I"]])
    b_q4c <- unname(co[["x:I"]])
    # b > 0 keeps the out-of-cell map strictly monotone; b + b_q4c > 0 keeps
    # the in-cell map strictly monotone. WPA consumes DIFFERENCES of
    # consecutive predictions, so monotonicity is a hard requirement (D1).
    stopifnot(is.finite(a), is.finite(b), is.finite(a_q4c), is.finite(b_q4c),
              b > 0, b + b_q4c > 0)
  } else {
    if (is.null(w)) {
      cli::cli_abort("fit_wp_calibration(form = 'leverage_interaction_v1') requires {.arg w}")
    }
    stopifnot(length(w) == length(labels))
    w_k <- w[keep]
    fit <- stats::glm(y ~ x + I(w_k * x), family = stats::binomial())
    co <- stats::coef(fit)
    a <- unname(co[["(Intercept)"]])
    b <- unname(co[["x"]])
    c_coef <- unname(co[["I(w_k * x)"]])
    a_q4c <- 0
    b_q4c <- 0
    # b > 0 keeps the w = 0 map strictly monotone; b + c > 0 keeps the w = 1
    # map strictly monotone -- w is linear in [0, 1], so these two endpoints
    # bound the slope (b + c*w) over its whole range. Same WPA-monotonicity
    # rationale as the interaction form above.
    stopifnot(is.finite(a), is.finite(b), is.finite(c_coef),
              b > 0, b + c_coef > 0)
  }

  list(a = a, b = b, c = c_coef, a_q4c = a_q4c, b_q4c = b_q4c, form = form)
}

#' The plan's exact gate cell, as a per-row flag
#'
#' `period == 4 & abs(points_diff) <= 12`. `NA` period/points_diff rows are
#' out-of-cell (global calibration arm) -- mirrors torp's serve-time guard
#' in `get_wp_preds()`.
#'
#' @param meta_cols Data frame with `period` and `points_diff` columns.
#' @param period Integer, cell quarter (default 4).
#' @param margin_abs_max Numeric, cell half-width in points (default 12).
#' @return Logical vector, one flag per row, never `NA`.
#' @keywords internal
wp_cell_flag <- function(meta_cols, period = 4, margin_abs_max = 12) {
  flag <- meta_cols$period == period & abs(meta_cols$points_diff) <= margin_abs_max
  flag[is.na(flag)] <- FALSE
  flag
}

#' Shared dedup observation-set builder (gate protocol v2, FABLE-RECAL-PLAN.md
#' \enc{§}{Section}8 point 2)
#'
#' Draws-excluded, last-row-per-(match_id, 5-min bucket) dedup -- the single
#' anti-pseudoreplication convention now used by BOTH the calibration fit
#' ([fit_wp_calibration_deduped()]) and the gate statistic ([wp_gate_slope()]),
#' so fit and gate are provably built from the same observation set. \enc{§}{Section}8:
#' "aligns objective with metric and removes the pseudoreplication that
#' inflated b to ~1.4." Supersedes the pre-\enc{§}{Section}8 split where only the
#' `"q4close"` cell deduped and `"all"` fit/gated on raw undeduped event rows.
#'
#' Bucket = `est_match_elapsed \%/\% 300` (plain 5-minute windows since
#' kickoff) -- NOT the pre-\enc{§}{Section}8 `"q4close"`-only dedup's
#' `pmin(4, pmax(0, ... - 11))` offset/clamp, which assumed every input row
#' was already restricted to Q4 (an assumption `"all"` doesn't satisfy). A
#' plain floor works uniformly for both cells and produces the identical
#' grouping within Q4/close except for rare extended-time-on rows, which now
#' land in their own bucket instead of being merged into a shared tail
#' bucket.
#'
#' @param preds Numeric vector of WP predictions.
#' @param labels Numeric vector, same length, values in `{0, 0.5, 1}`.
#' @param meta_cols Data frame with `period`, `points_diff`,
#'   `est_match_elapsed`, `match_id` (and `est_match_remaining` if present --
#'   carried through so callers can derive [wp_leverage_weight()] on the
#'   deduped rows without re-joining).
#' @param cell One of `"all"` (default, no cell restriction before dedup) or
#'   `"q4close"` (restrict to [wp_cell_flag()] before bucketing/deduping).
#' @return Data frame, one row per kept `(match_id, bucket)`: `pred`,
#'   `label_wp`, `match_id`, `bucket`, `period`, `points_diff`,
#'   `est_match_elapsed`, `est_match_remaining` (`NA` if absent from
#'   `meta_cols`).
#' @keywords internal
wp_dedup_observations <- function(preds, labels, meta_cols, cell = c("all", "q4close")) {
  cell <- match.arg(cell)
  d <- data.frame(
    pred = preds, label_wp = labels,
    period = meta_cols$period, points_diff = meta_cols$points_diff,
    est_match_elapsed = meta_cols$est_match_elapsed,
    est_match_remaining = if ("est_match_remaining" %in% names(meta_cols)) {
      meta_cols$est_match_remaining
    } else {
      NA_real_
    },
    match_id = meta_cols$match_id
  )
  d <- d[d$label_wp %in% c(0, 1), ]
  if (cell == "q4close") d <- d[wp_cell_flag(d), ]

  d$bucket <- d$est_match_elapsed %/% 300
  d |>
    dplyr::group_by(.data$match_id, .data$bucket) |>
    dplyr::summarise(
      label_wp = dplyr::last(.data$label_wp),
      pred = dplyr::last(.data$pred),
      period = dplyr::last(.data$period),
      points_diff = dplyr::last(.data$points_diff),
      est_match_elapsed = dplyr::last(.data$est_match_elapsed),
      est_match_remaining = dplyr::last(.data$est_match_remaining),
      .groups = "drop"
    )
}

#' Fit a WP calibration on the shared dedup observation set (gate protocol
#' v2, \enc{§}{Section}8 point 2)
#'
#' Builds the deduped observation set ([wp_dedup_observations()], `cell =
#' "all"` -- the whole row set passed in, not restricted to Q4/close),
#' derives the leverage weight on those deduped rows, and fits
#' [fit_wp_calibration()] on them. [wp_gate_slope()] measures its slopes via
#' the SAME dedup helper, so fit and gate are provably built from the same
#' observation set.
#'
#' @param preds,labels,meta_cols As [wp_dedup_observations()].
#' @param form Passed to [fit_wp_calibration()], default
#'   `"leverage_interaction_v1"`.
#' @return See [fit_wp_calibration()].
#' @keywords internal
fit_wp_calibration_deduped <- function(preds, labels, meta_cols, form = "leverage_interaction_v1") {
  dd <- wp_dedup_observations(preds, labels, meta_cols, cell = "all")
  w <- wp_leverage_weight(dd$est_match_remaining / 60, dd$points_diff)
  fit_wp_calibration(dd$pred, dd$label_wp, w = w, form = form)
}

#' Apply a fitted WP calibration to raw predictions
#'
#' Dispatches on `calib$form`. `"leverage_interaction_v1"` (\enc{§}{Section}7 Path B):
#' `plogis(a + (b + c*w) * qlogis(p))`, `w` from [wp_leverage_weight()] --
#' the single formula this trainer-side evaluation implements here and
#' torp's `get_wp_preds()` implements independently at serve time. Every
#' other form (`"global"`, `"q4close_interaction"`, or missing/unrecognized
#' -- both retired D1 forms stay supported here for backward
#' compatibility/debugging): `plogis((a + a_q4c*I) + (b + b_q4c*I) *
#' qlogis(p))`, `I = is_q4close`. Tolerant of missing/non-finite
#' `a_q4c`/`b_q4c`/`c` (treated as 0), so a global-form or pre-escalation
#' 2-param artifact stays valid. `NA`s in `is_q4close` are out-of-cell.
#'
#' @param preds Numeric vector of raw WP predictions.
#' @param calib Calibration list with at least `a` and `b`.
#' @param is_q4close Logical vector, or `NULL` for all-out-of-cell (pure
#'   global application). Used by the `"global"`/`"q4close_interaction"`
#'   forms only.
#' @param w Numeric vector, [wp_leverage_weight()] per row. Required when
#'   `calib$form == "leverage_interaction_v1"`.
#' @return Numeric vector of calibrated predictions.
#' @keywords internal
apply_wp_calibration <- function(preds, calib, is_q4close = NULL, w = NULL) {
  if (identical(calib$form, "leverage_interaction_v1")) {
    if (is.null(w)) {
      cli::cli_abort("apply_wp_calibration(calib$form = 'leverage_interaction_v1') requires {.arg w}")
    }
    c_coef <- calib$c
    if (is.null(c_coef) || length(c_coef) != 1 || !is.finite(c_coef)) c_coef <- 0
    return(stats::plogis(calib$a + (calib$b + c_coef * w) * stats::qlogis(preds)))
  }

  a_q4c <- calib$a_q4c
  if (is.null(a_q4c) || length(a_q4c) != 1 || !is.finite(a_q4c)) a_q4c <- 0
  b_q4c <- calib$b_q4c
  if (is.null(b_q4c) || length(b_q4c) != 1 || !is.finite(b_q4c)) b_q4c <- 0

  I <- if (is.null(is_q4close)) rep(0, length(preds)) else as.numeric(is_q4close)
  I[is.na(I)] <- 0

  stats::plogis((calib$a + a_q4c * I) + (calib$b + b_q4c * I) * stats::qlogis(preds))
}

#' Cell-boundary discontinuity of a calibration (report-only design guard)
#'
#' A hard cell boundary can discontinuously jump WP when a goal takes
#' |margin| across 12 or play crosses into Q4 -- WPA turns that jump into
#' spurious credit spikes. For a grid of raw p, computes
#' `|calibrated_in_cell - calibrated_out_of_cell|`; the trainer prints the
#' max and the p = 0.5 value in WP points and stores `max_boundary_jump`
#' (probability scale) in the calibration meta. NOT gated -- Pete decides
#' if it needs smoothing later. Identically 0 for a global-form fit.
#'
#' @param calib Calibration list (see [fit_wp_calibration()]).
#' @param grid Numeric vector of raw probabilities to scan.
#' @return `list(max_jump, max_at_p, jump_at_p50)`, probability scale.
#' @keywords internal
wp_calibration_boundary_jump <- function(calib, grid = seq(0.05, 0.95, by = 0.01)) {
  in_cell <- apply_wp_calibration(grid, calib, rep(TRUE, length(grid)))
  out_cell <- apply_wp_calibration(grid, calib, rep(FALSE, length(grid)))
  jump <- abs(in_cell - out_cell)
  list(
    max_jump = max(jump),
    max_at_p = grid[which.max(jump)],
    jump_at_p50 = jump[which.min(abs(grid - 0.5))]
  )
}

#' Print gate-cell sample-size + reliability diagnostics
#'
#' Answers "is the measured Q4/close slope a stable estimate or thin-cell
#' noise": prints the cell's match count, raw row count, deduped
#' `(match_id, bucket)` observation count, and per-decile reliability
#' (mean predicted vs actual) on the deduped cell -- the same dedup the
#' gate statistic uses.
#'
#' @param preds Numeric vector of (typically calibrated) WP predictions.
#' @param labels Numeric vector, values in `{0, 0.5, 1}`.
#' @param meta_cols Data frame with `period`, `points_diff`,
#'   `est_match_elapsed`, `match_id` (see [score_wp_rows()]).
#' @param label Optional character tag prefixed to the printed lines (e.g.
#'   `"fit half"` / `"gate half"`).
#' @return The per-decile reliability data frame, invisibly (`NULL` if the
#'   cell is empty).
#' @keywords internal
wp_gate_cell_diagnostics <- function(preds, labels, meta_cols, label = NULL) {
  tag <- if (is.null(label)) "" else paste0("[", label, "] ")

  n_total_matches <- length(unique(meta_cols$match_id))
  raw <- data.frame(period = meta_cols$period, points_diff = meta_cols$points_diff,
                    label_wp = labels)
  raw <- raw[raw$label_wp %in% c(0, 1), ]
  n_raw_cell <- sum(wp_cell_flag(raw))

  dd <- wp_dedup_observations(preds, labels, meta_cols, cell = "q4close")

  if (nrow(dd) == 0) {
    cli::cli_warn("{tag}Q4/close gate cell is EMPTY -- no diagnostics possible")
    return(invisible(NULL))
  }

  cli::cli_inform(
    "{tag}{n_total_matches} matches total | Q4/close cell: {length(unique(dd$match_id))} matches, {n_raw_cell} raw rows, {nrow(dd)} deduped (match_id, bucket) obs"
  )

  dd$decile <- cut(dd$pred, seq(0, 1, 0.1), include.lowest = TRUE)
  rel <- dd |>
    dplyr::group_by(.data$decile) |>
    dplyr::summarise(
      n = dplyr::n(),
      mean_pred = round(mean(.data$pred), 3),
      actual = round(mean(.data$label_wp), 3),
      gap = round(mean(.data$label_wp) - mean(.data$pred), 3),
      .groups = "drop"
    )
  cli::cli_inform("{tag}Per-decile reliability (deduped Q4/close cell):")
  print(as.data.frame(rel))

  invisible(rel)
}

#' Calibration-slope gate statistic (D5, deduped convention per gate
#' protocol v2 \enc{§}{Section}8 point 2)
#'
#' `cell = "all"`: `glm(label ~ qlogis(pred), binomial)` slope over the
#' deduped observation set ([wp_dedup_observations()], `cell = "all"`).
#' `cell = "q4close"`: the same GLM restricted to `period == 4 &
#' abs(points_diff) <= 12` before deduping ([wp_dedup_observations()], `cell
#' = "q4close"`). Both cells now share the identical last-row-per-(match_id,
#' 5-min bucket) anti-pseudoreplication convention (\enc{§}{Section}8 point 2 -- prior to
#' gate protocol v2, only `"q4close"` deduped and `"all"` fit/gated on raw
#' undeduped event rows, which \enc{§}{Section}8's diagnosis identified as the source of
#' the fit/gate split's pseudoreplication).
#'
#' @param preds Numeric vector of (typically calibrated) WP predictions.
#' @param labels Numeric vector, same length, values in `{0, 0.5, 1}`.
#' @param meta_cols Data frame with `period`, `points_diff`,
#'   `est_match_elapsed`, `match_id` columns, same row order/length as
#'   `preds`/`labels` (see [score_wp_rows()]).
#' @param cell One of `"q4close"` (default) or `"all"`.
#' @param detail Logical, default `FALSE`. When `TRUE`, returns
#'   `list(slope, se, n)` (`se` = the GLM coefficient's standard error,
#'   `n` = rows the GLM saw after cell filter + dedup) instead of the bare
#'   slope -- used by the gate to judge breach-vs-noise on thin cells.
#' @return Numeric scalar slope (or `NA_real_` if the cell is degenerate:
#'   too few rows / a single class); with `detail = TRUE`, a list.
#' @keywords internal
wp_gate_slope <- function(preds, labels, meta_cols, cell = c("q4close", "all"),
                          detail = FALSE) {
  cell <- match.arg(cell)
  d <- wp_dedup_observations(preds, labels, meta_cols, cell = cell)

  if (nrow(d) < 2 || length(unique(d$label_wp)) < 2) {
    if (detail) return(list(slope = NA_real_, se = NA_real_, n = nrow(d)))
    return(NA_real_)
  }

  p <- pmin(pmax(d$pred, 1e-6), 1 - 1e-6)
  fit <- stats::glm(d$label_wp ~ stats::qlogis(p), family = stats::binomial())
  slope <- unname(stats::coef(fit)[2])

  if (!detail) return(slope)

  se <- tryCatch(unname(summary(fit)$coefficients[2, "Std. Error"]),
                 error = function(e) NA_real_)
  list(slope = slope, se = se, n = nrow(d))
}

#' The temporal Q4/close release gate (D5, precision-honest tolerances per
#' gate protocol v2 \enc{§}{Section}8 point 3) -- aborts before anything is written or
#' published
#'
#' Gates the all-rows calibrated slope at `|slope_all - 1| <= threshold`
#' (default `0.10`) and the Q4/close-cell slope at `|slope_q4close - 1| <=
#' threshold_q4close` (default `0.25` -- widened from the original single
#' `0.10` because the q4close cell's split sd (~0.6, n≈288, SE≈0.17) made the
#' tighter tolerance near-unpassable for a genuinely calibrated model; \enc{§}{Section}8
#' full evidence in `docs/reviews/2026-WP-GATE-SPLIT-DIAGNOSIS.md`). Fires as
#' `cli::cli_abort()` in the same idiom as
#' [validate_wp_spec()]/`publish_model_group()` -- a loud bulleted message
#' with the measured slopes, before any `saveRDS`/upload.
#'
#' @param calibrated_preds Numeric vector of calibrated WP predictions.
#' @param labels Numeric vector, same length, values in `{0, 0.5, 1}`.
#' @param meta_cols Data frame, see [wp_gate_slope()].
#' @param threshold Numeric, default `0.10` -- the all-rows tolerance.
#' @param threshold_q4close Numeric, default `0.25` -- the Q4/close-cell
#'   tolerance (\enc{§}{Section}8 point 3). Neither threshold widens further on a thin
#'   cell -- the gate additionally reports the slope's GLM standard error
#'   (deduped n < 150) so a breach can be judged against noise, but still
#'   gates at `threshold_q4close`.
#' @return `list(slope_all, slope_q4close, se_q4close, n_q4close)`,
#'   reachable via return value only on a pass (the function aborts on a
#'   breach).
#' @keywords internal
validate_wp_temporal_slope <- function(calibrated_preds, labels, meta_cols,
                                       threshold = 0.10, threshold_q4close = 0.25) {
  slope_all <- wp_gate_slope(calibrated_preds, labels, meta_cols, cell = "all")
  q4c <- wp_gate_slope(calibrated_preds, labels, meta_cols, cell = "q4close", detail = TRUE)
  slope_q4close <- q4c$slope

  ok_all <- !is.na(slope_all) && abs(slope_all - 1) <= threshold
  ok_q4close <- !is.na(slope_q4close) && abs(slope_q4close - 1) <= threshold_q4close

  thin_cell <- !is.na(q4c$n) && q4c$n < 150
  q4c_detail <- if (thin_cell) {
    " [THIN CELL: n = {q4c$n} deduped obs < 150 -- slope SE = {round(q4c$se, 3)}; judge breach vs noise, gate still {threshold_q4close}]"
  } else {
    " (n = {q4c$n} deduped obs)"
  }

  if (!ok_all || !ok_q4close) {
    cli::cli_abort(c(
      "WP temporal slope gate FAILED -- |slope_all - 1| must be <= {threshold}, |slope_q4close - 1| must be <= {threshold_q4close}",
      "x" = "slope_all = {round(slope_all, 3)} ({if (ok_all) 'ok' else 'BREACH'})",
      "x" = paste0("slope_q4close = {round(slope_q4close, 3)} ({if (ok_q4close) 'ok' else 'BREACH'})", q4c_detail),
      "i" = "Nothing written or published. Emergency override: train_models.R --skip-slope-gate."
    ))
  }

  cli::cli_alert_success(paste0(
    "WP temporal slope gate passed: slope_all = {round(slope_all, 3)} (threshold {threshold}), slope_q4close = {round(slope_q4close, 3)} (threshold {threshold_q4close})",
    q4c_detail
  ))

  list(slope_all = slope_all, slope_q4close = slope_q4close,
       se_q4close = q4c$se, n_q4close = q4c$n)
}

#' Cross-fitted leverage calibration over the gate season (gate protocol v2,
#' FABLE-RECAL-PLAN.md \enc{§}{Section}8 point 1)
#'
#' Splits the gate season's matches into 2 folds (match-grouped, seed-stable
#' [make_match_folds()] -- the identical split construction the pre-\enc{§}{Section}8
#' split-half protocol used), fits a calibration on each fold's deduped
#' observations ([fit_wp_calibration_deduped()]), and cross-applies: fold
#' A's rows are calibrated using fold B's fit and vice versa, so no row is
#' ever scored by a calibration fitted on its own fold ("out-of-fold").
#' Returns the pooled OOF calibrated predictions at raw event-row
#' granularity, same order/length as `temporal$preds` -- callers dedup at
#' measurement time via [wp_gate_slope()]/[wp_gate_cell_diagnostics()].
#'
#' @param temporal `list(preds, labels, meta_cols)`, e.g. from
#'   [fit_wp_temporal_variant()].
#' @param form Calibration form passed to [fit_wp_calibration_deduped()],
#'   default `"leverage_interaction_v1"`.
#' @param seed Fold-split seed, default `1234L` (matches
#'   [make_match_folds()]'s own default).
#' @return `list(fold, calib_A, calib_B, oof_preds, oof_labels, oof_meta)` --
#'   `fold` is the per-row fold id (1L/2L, same order as `temporal$preds`);
#'   `oof_preds[i]` is `temporal$preds[i]` calibrated by the OTHER fold's
#'   fit; `oof_labels`/`oof_meta` are `temporal$labels`/`temporal$meta_cols`
#'   unchanged (returned for caller convenience).
#' @keywords internal
wp_crossfit_calibrate <- function(temporal, form = "leverage_interaction_v1", seed = 1234L) {
  fold <- make_match_folds(temporal$meta_cols$match_id, k = 2L, seed = seed)
  idx_A <- which(fold == 1L)
  idx_B <- which(fold == 2L)
  meta_A <- temporal$meta_cols[idx_A, , drop = FALSE]
  meta_B <- temporal$meta_cols[idx_B, , drop = FALSE]

  calib_A <- fit_wp_calibration_deduped(temporal$preds[idx_A], temporal$labels[idx_A], meta_A, form = form)
  calib_B <- fit_wp_calibration_deduped(temporal$preds[idx_B], temporal$labels[idx_B], meta_B, form = form)

  w_A <- wp_leverage_weight(meta_A$est_match_remaining / 60, meta_A$points_diff)
  w_B <- wp_leverage_weight(meta_B$est_match_remaining / 60, meta_B$points_diff)

  oof_preds <- rep(NA_real_, length(temporal$preds))
  oof_preds[idx_A] <- apply_wp_calibration(temporal$preds[idx_A], calib_B, w = w_A)  # A scored by B's fit
  oof_preds[idx_B] <- apply_wp_calibration(temporal$preds[idx_B], calib_A, w = w_B)  # B scored by A's fit
  stopifnot(!anyNA(oof_preds))

  list(fold = fold, calib_A = calib_A, calib_B = calib_B,
       oof_preds = oof_preds, oof_labels = temporal$labels, oof_meta = temporal$meta_cols)
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

    # S5_exclude_draws (FABLE-RECAL-PLAN.md \enc{§}{Section}7 addendum, WP-SHARPNESS-RESULTS.md \enc{§}{Section}7 --
    # the one strictly-non-negative S1-S5 finding): drop draw rows
    # (label_wp == 0.5) from WP TRAINING only. Reassigning model_data_wp
    # here (rather than filtering only the frame passed to fit_wp()) keeps
    # every downstream use in this branch (folds_wp, meta_full, wp_fit's
    # own X/y/folds) consistently row-aligned -- gate/eval data and the
    # calibration fit's own draw-exclusion (fit_wp_calibration() already
    # drops label == 0.5) are untouched.
    if ("label_wp" %in% names(model_data_wp)) {
      n_before_drop <- nrow(model_data_wp)
      model_data_wp <- model_data_wp[model_data_wp$label_wp %in% c(0, 1), ]
      n_dropped <- n_before_drop - nrow(model_data_wp)
      if (n_dropped > 0) {
        cli::cli_inform("S5_exclude_draws: dropped {n_dropped} draw row{?s} (label_wp == 0.5) from WP training data")
      }
    }

    wp_fit <- fit_wp(model_data_wp)

    if (!skip_calibration) {
      gate_season <- max(seasons)
      cli::cli_h3("WP recalibration + temporal slope gate (gate season {gate_season})")

      temporal <- fit_wp_temporal_variant(model_data_epv, gate_season,
                                          params_ep = ep_params(), params_wp = wp_params())

      # ---- Cross-fitted gating within the holdout season (gate protocol v2,
      # FABLE-RECAL-PLAN.md \enc{§}{Section}8, 2026-07-22) ---------------------------------
      # The pre-\enc{§}{Section}8 split-half protocol (fit on half A, gate on held-out
      # half B) was diagnosed as pseudoreplicated and noisy across two
      # consecutive gate failures: 175K "rows" per half collapse to ~1,700
      # quasi-independent 5-min buckets, and the fit/gate slope gap swung
      # -0.26 to +0.47 across 7 reseedings of the SAME split construction
      # (docs/reviews/2026-WP-GATE-SPLIT-DIAGNOSIS.md). Cross-fitting pools
      # BOTH folds' out-of-fold calibrated predictions into one gate
      # statistic over the full gate season instead of judging one noisy
      # half against another.
      half <- make_match_folds(temporal$meta_cols$match_id, k = 2L)
      n_foldA_matches <- length(unique(temporal$meta_cols$match_id[half == 1L]))
      n_foldB_matches <- length(unique(temporal$meta_cols$match_id[half == 2L]))
      cli::cli_inform(
        "Cross-fit split within gate season {gate_season}: fold A = {n_foldA_matches} matches ({sum(half == 1L)} rows), fold B = {n_foldB_matches} matches ({sum(half == 2L)} rows)"
      )

      slope_all_before <- wp_gate_slope(temporal$preds, temporal$labels, temporal$meta_cols, "all")
      slope_q4close_before <- wp_gate_slope(temporal$preds, temporal$labels, temporal$meta_cols, "q4close")
      cli::cli_inform("Uncalibrated temporal slopes (full gate season {gate_season}): all = {round(slope_all_before, 3)}, q4close = {round(slope_q4close_before, 3)}")

      # Report-only counterfactual: what the retired D1 global 2-param fit
      # would have shipped, cross-fit pooled the same way as the shipped
      # form below -- kept purely for log continuity/monitoring, never
      # applied, never gated. Shows exactly the defect Path B exists to fix.
      cf_global <- wp_crossfit_calibrate(temporal, form = "global")
      slope_all_global <- wp_gate_slope(cf_global$oof_preds, cf_global$oof_labels, cf_global$oof_meta, "all")
      slope_q4close_global <- wp_gate_slope(cf_global$oof_preds, cf_global$oof_labels, cf_global$oof_meta, "q4close")
      cli::cli_inform(paste0(
        "[report-only, retired D1 form] Global-form cross-fit pooled OOF slopes: all = {round(slope_all_global, 3)}, q4close = {round(slope_q4close_global, 3)} ",
        "(fold A: a = {round(cf_global$calib_A$a, 4)}, b = {round(cf_global$calib_A$b, 4)} | fold B: a = {round(cf_global$calib_B$a, 4)}, b = {round(cf_global$calib_B$b, 4)})"
      ))

      # Path B (FABLE-RECAL-PLAN.md \enc{§}{Section}7, 2026-07-21): state-dependent
      # calibration form -- REPLACES D1's global fit (and its conditional
      # 4-param q4close_interaction escalation) as the form that is actually
      # fitted and shipped. The 2026-07-21 gated retrain proved a single
      # global slope cannot pass the gate on this data (b = 1.4198
      # overcorrected all-rows to 0.715 while q4close stayed 1.245 -- both
      # breaches); S1-S5 raw-model sharpening came back null
      # (WP-SHARPNESS-RESULTS.md). The leverage-weighted interaction fits a
      # continuous per-row correction instead of one global slope. Cross-fit
      # per point 1: fit on each fold's deduped rows, apply cross-fold.
      cf_leverage <- wp_crossfit_calibrate(temporal, form = "leverage_interaction_v1")
      cli::cli_inform(paste0(
        "Per-fold leverage-interaction fits: fold A a = {round(cf_leverage$calib_A$a, 4)}, b = {round(cf_leverage$calib_A$b, 4)}, c = {round(cf_leverage$calib_A$c, 4)} | ",
        "fold B a = {round(cf_leverage$calib_B$a, 4)}, b = {round(cf_leverage$calib_B$b, 4)}, c = {round(cf_leverage$calib_B$c, 4)}"
      ))

      # Pooled OOF calibrated predictions over the FULL gate season -- the
      # only rows the gate ever sees. Genuinely out-of-fold: fold A's rows
      # are calibrated by fold B's fit and vice versa.
      if (isTRUE(slope_gate)) {
        gated <- validate_wp_temporal_slope(cf_leverage$oof_preds, cf_leverage$oof_labels, cf_leverage$oof_meta,
                                            threshold = 0.10, threshold_q4close = 0.25)
        slope_all_after <- gated$slope_all
        slope_q4close_after <- gated$slope_q4close
      } else {
        slope_all_after <- wp_gate_slope(cf_leverage$oof_preds, cf_leverage$oof_labels, cf_leverage$oof_meta, "all")
        q4c_detail <- wp_gate_slope(cf_leverage$oof_preds, cf_leverage$oof_labels, cf_leverage$oof_meta, "q4close", detail = TRUE)
        slope_q4close_after <- q4c_detail$slope
        cli::cli_alert_warning(
          "Slope gate skipped (slope_gate = FALSE): pooled OOF all = {round(slope_all_after, 3)}, q4close = {round(slope_q4close_after, 3)} (n = {q4c_detail$n}, SE = {round(q4c_detail$se, 3)})"
        )
      }

      # ---- Shipped artifact (\enc{§}{Section}8 point 4): refit (a, b, c) on the FULL
      # gate season (both folds, deduped convention) -- the pooled OOF
      # slopes above are what the gate actually verified; this is "more
      # data, same recipe" now that the gate has passed (or been
      # overridden), not a second measurement.
      calib <- fit_wp_calibration_deduped(temporal$preds, temporal$labels, temporal$meta_cols,
                                          form = "leverage_interaction_v1")
      cli::cli_inform("Full gate-season refit (shipped): a = {round(calib$a, 4)}, b = {round(calib$b, 4)}, c = {round(calib$c, 4)}")

      # Cell-boundary discontinuity -- N/A for the leverage form by
      # construction: w is continuous in both match-time and margin (no
      # hard step the way q4close_interaction's binary I() has), so
      # calibrated WP has no boundary jump to measure. Report-only, never
      # gated -- kept at 0/NA rather than calling wp_calibration_boundary_jump()
      # (which assumes the binary is_q4close arm and would error without a
      # w argument).
      bj <- list(max_jump = 0, max_at_p = NA_real_, jump_at_p50 = 0)
      cli::cli_inform(
        "[report-only, not gated] Cell-boundary discontinuity: N/A for leverage_interaction_v1 (continuous weight, no boundary by construction)"
      )

      # Gate-cell sample-size + reliability diagnostics -- lets the log
      # answer "stable estimate or thin-cell noise?" on both the pooled OOF
      # set the gate measured and the full-season refit that ships.
      wp_gate_cell_diagnostics(cf_leverage$oof_preds, cf_leverage$oof_labels, cf_leverage$oof_meta, label = "pooled OOF (gate)")
      w_shipped <- wp_leverage_weight(temporal$meta_cols$est_match_remaining / 60, temporal$meta_cols$points_diff)
      calibrated_full <- apply_wp_calibration(temporal$preds, calib, w = w_shipped)
      wp_gate_cell_diagnostics(calibrated_full, temporal$labels, temporal$meta_cols, label = "full refit (shipped)")

      calib_formula <- "plogis(a + (b + c*w)*qlogis(p)), w = leverage_weight(minutes_remaining, points_diff; ramp_mins, margin_cap)"
      wp_calibration_obj <- list(
        a = calib$a, b = calib$b, c = calib$c,
        form = calib$form,
        ramp_mins = 20, margin_cap = 18,
        cell = list(period = 4, margin_abs_max = 12),
        formula = calib_formula,
        fitted_on = "temporal-oos-full-gate-season-refit",
        gate_validated_on = "temporal-oos-cross-fit-pooled-oof",
        gate_season = gate_season,
        n_fit = nrow(temporal$meta_cols),
        n_foldA_matches = n_foldA_matches,
        n_foldB_matches = n_foldB_matches,
        slope_before = slope_all_before, slope_after = slope_all_after,
        slope_q4close_before = slope_q4close_before, slope_q4close_after = slope_q4close_after,
        slope_all_global = slope_all_global, slope_q4close_global = slope_q4close_global,
        max_boundary_jump = bj$max_jump
      )
      wp_calibration_meta <- torpmodels:::build_model_meta(
        "wp_calibration", seasons, list(formula = calib_formula, form = calib$form),
        c("a", "b", "c"),
        n_rows = nrow(temporal$meta_cols),
        extra = list(
          script = "train_models.R",
          a = calib$a, b = calib$b, c = calib$c,
          form = calib$form, ramp_mins = 20, margin_cap = 18,
          gate_season = gate_season, n_fit = nrow(temporal$meta_cols),
          n_foldA_matches = n_foldA_matches,
          n_foldB_matches = n_foldB_matches,
          slope_before = slope_all_before, slope_after = slope_all_after,
          slope_q4close_before = slope_q4close_before, slope_q4close_after = slope_q4close_after,
          slope_all_global = slope_all_global, slope_q4close_global = slope_q4close_global,
          max_boundary_jump = bj$max_jump
        )
      )
      saveRDS(torpmodels:::stamp_model_meta(wp_calibration_obj, wp_calibration_meta),
              file.path(output_dir, "wp_calibration.rds"))
      results$wp_calibration <- wp_calibration_meta

      # Report-only diagnostics -- printed, never gated (D2/D5).
      row_folds_wp <- make_match_folds(model_data_wp$torp_match_id)
      folds_wp <- lapply(seq_len(max(row_folds_wp)), function(k) which(row_folds_wp == k))
      cv_oos <- cv_wp_oos_preds(wp_fit$X, wp_fit$y, folds_wp, wp_params(), wp_fit$optimal_nrounds)
      meta_full <- model_data_wp |> dplyr::select("period", "points_diff",
                                                   "est_match_elapsed", "est_match_remaining",
                                                   "match_id")
      w_full <- wp_leverage_weight(meta_full$est_match_remaining / 60, meta_full$points_diff)
      cv_oos_calibrated <- apply_wp_calibration(cv_oos, calib, w = w_full)
      cli::cli_inform(
        "[report-only, not gated] Random-CV calibrated slope: all = {round(wp_gate_slope(cv_oos_calibrated, model_data_wp$label_wp, meta_full, 'all'), 3)}, q4close = {round(wp_gate_slope(cv_oos_calibrated, model_data_wp$label_wp, meta_full, 'q4close'), 3)}"
      )

      in_progress_season <- gate_season + 1L
      ip_chains <- tryCatch(torp::load_chains(seasons = in_progress_season, rounds = TRUE), error = function(e) NULL)
      if (is.null(ip_chains) || nrow(ip_chains) == 0) {
        cli::cli_inform("[report-only, not gated] In-progress season {in_progress_season}: no rows available yet")
      } else {
        # tryCatch, not left to propagate: this diagnostic is explicitly
        # "report-only, not gated" (D2/D5) -- score_wp_rows() now aborts
        # loudly on a feature NA (see abort_on_feature_na()), which is the
        # right behavior for the gate-season call in
        # fit_wp_temporal_variant() but would wrongly turn THIS non-gating
        # diagnostic into a hard failure mid-run, after wp_calibration.rds
        # has already been saved and before wp_model.rds is.
        tryCatch({
          ip_epv <- torp::clean_pbp(ip_chains) |> torp:::clean_model_data_epv()
          ip_scored <- score_wp_rows(ip_epv, ep_fit$model, wp_fit$model)
          w_ip <- wp_leverage_weight(ip_scored$meta_cols$est_match_remaining / 60, ip_scored$meta_cols$points_diff)
          ip_preds_cal <- apply_wp_calibration(ip_scored$preds, calib, w = w_ip)
          cli::cli_inform(
            "[report-only, not gated] In-progress season {in_progress_season}: calibrated slope all = {round(wp_gate_slope(ip_preds_cal, ip_scored$labels, ip_scored$meta_cols, 'all'), 3)}, q4close = {round(wp_gate_slope(ip_preds_cal, ip_scored$labels, ip_scored$meta_cols, 'q4close'), 3)} (n = {length(ip_scored$labels)})"
          )
        }, error = function(e) {
          cli::cli_warn("[report-only, not gated] In-progress season {in_progress_season}: diagnostic failed ({conditionMessage(e)}) -- skipping (never gates the release)")
        })
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
