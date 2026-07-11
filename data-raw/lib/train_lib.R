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
# TRAINING-CONSOLIDATION-PLAN.md Step 3 -- the single canonical trainer for
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
#' @param output_dir Directory to save trained artifacts.
#' @return Named list of `torp_meta` objects, one per trained model.
#' @keywords internal
train_core_models <- function(models = c("ep", "wp", "shot"),
                              seasons = default_training_seasons(),
                              upload = TRUE,
                              wp_ep_source = c("cv", "insample"),
                              output_dir = file.path("inst", "models", "core")) {
  models <- match.arg(models, c("ep", "wp", "shot"), several.ok = TRUE)
  wp_ep_source <- match.arg(wp_ep_source, c("cv", "insample"))

  if (wp_ep_source == "insample" && upload) {
    cli::cli_alert_warning("wp_ep_source = 'insample' forces upload = FALSE (comparison/debug only -- the legacy train_wp_model.R semantics)")
    upload <- FALSE
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
