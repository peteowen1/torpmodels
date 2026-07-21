# WP sharpness experiment harness (FABLE-WP-SHARPNESS-PLAN.md, S1-S5)
# ======================================================================
# Experiment-local machinery ONLY -- never touches torp/R, torpmodels/R,
# WP_MODEL_FEATURES, or any serving/production path. Reuses the canonical
# trainer's building blocks (fit_ep, cv_ep_oos_preds, build_wp_data,
# make_match_folds, wp_gate_slope, wp_cell_flag, abort_on_feature_na,
# ep_params, wp_params, default_training_seasons, load_training_pbp) from
# torpmodels/data-raw/lib/train_lib.R -- sourced by the caller before this
# file. This file adds the HOOKS fit_wp_temporal_variant() doesn't expose
# (per-row weights, extra engineered feature columns, ablated monotone
# constraints, draw-row filtering) so each experiment stays "one knob" on
# top of the real protocol, per the plan's execution shape.
#
# Design for compute reuse: EP fitting (fit_ep + 5-fold OOS refits) is the
# expensive step and is IDENTICAL across every S1-S5 variant (none of them
# touch EP). So the caller fits EP once, builds the base WP train/gate
# frames once, and this file's run_wp_variant() only re-fits the (cheap,
# single 5-fold CV) WP model per variant.

# ---- Feature engineering (S3) ---------------------------------------------

#' Add the plan's 3 engineered decisiveness features to a WP data frame
#' (post build_wp_data(), i.e. has points_diff, exp_pts, time_left_scaler,
#' est_match_remaining already).
#'
#' - margin_per_remaining_poss: points_diff / expected remaining "possessions"
#'   (chain-events), using a pace constant derived empirically from the
#'   TRAINING data (rows per match-minute) -- passed in, not re-derived per
#'   call, so train/gate use the identical constant.
#' - insurmountable: pmax(0, |points_diff| - 6*sqrt(minutes_remaining)) --
#'   a magnitude bound on how "decisive" the lead already is.
#' - points_diff_time_scaled: points_diff * time_left_scaler (distinct from
#'   the EXISTING diff_time_ratio, which uses xpoints_diff, not raw
#'   points_diff).
add_decisiveness_features <- function(df, pace_rows_per_min) {
  minutes_remaining <- pmax(df$est_match_remaining, 0) / 60
  expected_remaining_poss <- pmax(minutes_remaining * pace_rows_per_min, 1)
  df$margin_per_remaining_poss <- df$points_diff / expected_remaining_poss
  df$insurmountable <- pmax(0, abs(df$points_diff) - 6 * sqrt(pmax(minutes_remaining, 0)))
  df$points_diff_time_scaled <- df$points_diff * df$time_left_scaler
  df
}

S3_EXTRA_FEATURES <- c("margin_per_remaining_poss", "insurmountable", "points_diff_time_scaled")
# monotone-increasing: margin_per_remaining_poss (bigger relative lead -> higher WP),
# points_diff_time_scaled (bigger lead late -> higher WP). insurmountable is a
# symmetric magnitude (same sign regardless of who's leading) -- unconstrained.
S3_EXTRA_MONOTONE <- c(margin_per_remaining_poss = 1L, insurmountable = 0L, points_diff_time_scaled = 1L)

#' Empirical pace constant: WP-model rows per match-minute, from the
#' training frame. Chain-event rows are the available proxy for
#' "possessions" here -- there's no dedicated pace constant in torp.
compute_pace_rows_per_min <- function(train_df) {
  n_matches <- length(unique(train_df$torp_match_id))
  total_minutes <- n_matches * (4 * 20)  # AFL_PLAY_QUARTER_SECONDS=1200s=20min * 4
  nrow(train_df) / total_minutes
}

# ---- Weighting hooks (S1, S4, S5) ------------------------------------------

#' S1 binary leverage weight: 1 + k * is_q4close
wp_weight_leverage_binary <- function(k) {
  function(df) 1 + k * as.numeric(wp_cell_flag(df))
}

#' S1 continuous leverage weight (bonus, "if cheap"): 1 + k * urgency_norm,
#' urgency_norm = |score_urgency| normalized to [0,1] via the 99th
#' percentile (score_urgency itself can be negative -- the plan's literal
#' `1 + k*score_urgency` would produce negative weights for a trailing
#' team, which xgboost rejects, so this is the sane reading of "continuous
#' form").
wp_weight_leverage_continuous <- function(k) {
  function(df) {
    u <- abs(df$score_urgency)
    cap <- stats::quantile(u, 0.99, na.rm = TRUE)
    u_norm <- pmin(u / cap, 1)
    1 + k * u_norm
  }
}

#' S4 recency weight: exponential half-life in seasons, anchored at the
#' most recent training season (so the most recent season always gets
#' weight 1).
wp_weight_recency <- function(half_life_seasons) {
  function(df) {
    max_season <- max(df$season)
    0.5^((max_season - df$season) / half_life_seasons)
  }
}

#' S5 draw down-weight (used instead of exclude_draws_train when
#' down-weighting rather than dropping)
wp_weight_downweight_draws <- function(w_draw = 0.5) {
  function(df) ifelse(df$label_wp == 0.5, w_draw, 1)
}

# ---- Core variant fit (the "one knob" experiment cell) ---------------------

#' Fit + score one WP variant. Mirrors fit_wp()/fit_wp_temporal_variant()'s
#' body exactly except for the hooks below; EP is NOT refit here (passed in
#' already scored onto train/gate).
#'
#' @param train_df Base WP training frame (build_wp_data() output on
#'   seasons < gate_season), before feature/weight/filter hooks.
#' @param gate_df Base WP gate-season frame (build_wp_data() output scored
#'   via the already-fitted temporal EP model).
#' @param extra_features Character vector of extra column names (already
#'   present in train_df/gate_df, e.g. via add_decisiveness_features()) to
#'   append to torp:::WP_MODEL_FEATURES.
#' @param extra_monotone Named integer vector (0/1) for extra_features, in
#'   torp:::WP_MODEL_FEATURES-then-extra_features column order.
#' @param drop_constraints Logical; TRUE zeroes every monotone constraint
#'   (S2a).
#' @param weight_fn NULL or function(train_df_after_filters) -> numeric
#'   weight vector.
#' @param exclude_draws_train Logical; drop label_wp==0.5 rows from
#'   TRAINING only (S5a). Gate scoring is untouched (slopes already
#'   exclude draws by convention).
#' @param params_wp Base WP params (monotone_constraints overwritten here).
#' @param nrounds_max Passed to xgb.cv.
#' @return list(variant, model, optimal_nrounds, cv_logloss, preds_gate,
#'   labels_gate, meta_gate, n_train, weights_summary)
run_wp_variant <- function(variant, train_df, gate_df,
                           extra_features = character(0),
                           extra_monotone = integer(0),
                           drop_constraints = FALSE,
                           weight_fn = NULL,
                           exclude_draws_train = FALSE,
                           params_wp = wp_params(),
                           nrounds_max = 500L) {
  cli::cli_h2("Variant: {variant}")
  t0 <- Sys.time()

  # train_df/gate_df come from build_wp_data() -> clean_model_data_wp(), which
  # runs on top of clean_model_data_epv()'s data.table pipeline and preserves
  # the data.table class. data.table's `[.data.table` uses NSE on j -- a bare
  # variable like `df[, feature_cols]` is looked up as a COLUMN NAME
  # ("feature_cols"), not evaluated as the character vector in scope (that
  # needs `..feature_cols` or `with = FALSE`). Coercing to a plain data.frame
  # here makes every bracket-index below behave with ordinary base-R
  # semantics, matching how select_wp_model_vars()/select_epv_model_vars()
  # sidestep the same trap via dplyr::select() instead of `[`.
  train_df <- as.data.frame(train_df)
  gate_df <- as.data.frame(gate_df)

  if (exclude_draws_train) {
    n_before <- nrow(train_df)
    train_df <- train_df[train_df$label_wp %in% c(0, 1), ]
    cli::cli_inform("Excluded {n_before - nrow(train_df)} draw-labeled rows from training ({nrow(train_df)} remain)")
  }

  feature_cols <- c(torp:::WP_MODEL_FEATURES, extra_features)
  wp_vars_train <- train_df[, feature_cols, drop = FALSE]
  abort_on_feature_na(wp_vars_train, paste0("WP variant [", variant, "] train"))
  X_train <- stats::model.matrix(~ . + 0, data = wp_vars_train, na.action = na.pass)
  stopifnot(nrow(X_train) == nrow(wp_vars_train))
  y_train <- train_df$label_wp

  monotone_vec <- if (drop_constraints) {
    rep(0L, length(feature_cols))
  } else {
    base_mono <- as.integer(torp:::WP_MODEL_FEATURES %in% torp:::WP_MONOTONE_INCREASING)
    extra_mono <- if (length(extra_features) > 0) unname(extra_monotone[extra_features]) else integer(0)
    c(base_mono, extra_mono)
  }
  stopifnot(length(monotone_vec) == ncol(X_train))
  params_wp$monotone_constraints <- paste0("(", paste(monotone_vec, collapse = ","), ")")

  weights <- if (!is.null(weight_fn)) weight_fn(train_df) else NULL
  weights_summary <- if (!is.null(weights)) {
    c(min = min(weights), mean = mean(weights), max = max(weights))
  } else {
    c(min = 1, mean = 1, max = 1)
  }

  row_folds <- make_match_folds(train_df$torp_match_id)
  folds <- lapply(seq_len(max(row_folds)), function(k) which(row_folds == k))

  full_train <- if (is.null(weights)) {
    xgboost::xgb.DMatrix(data = X_train, label = y_train)
  } else {
    xgboost::xgb.DMatrix(data = X_train, label = y_train, weight = weights)
  }

  cli::cli_inform("[{variant}] Running WP 5-fold CV (match-grouped), n = {nrow(X_train)}, {length(feature_cols)} features...")
  set.seed(1234)
  cv_result <- xgboost::xgb.cv(
    params = params_wp, data = full_train, nrounds = nrounds_max,
    folds = folds, early_stopping_rounds = 20, print_every_n = 50, verbose = 1
  )
  optimal_nrounds <- cv_result$best_iteration
  if (is.null(optimal_nrounds) || length(optimal_nrounds) == 0) {
    optimal_nrounds <- which.min(cv_result$evaluation_log$test_logloss_mean)
  }
  cv_logloss <- min(cv_result$evaluation_log$test_logloss_mean)
  cli::cli_inform("[{variant}] WP optimal nrounds: {optimal_nrounds} | CV logloss: {round(cv_logloss, 6)}")

  set.seed(1234)
  model <- xgboost::xgb.train(params = params_wp, data = full_train,
                              nrounds = optimal_nrounds, print_every_n = 50)

  wp_vars_gate <- gate_df[, feature_cols, drop = FALSE]
  abort_on_feature_na(wp_vars_gate, paste0("WP variant [", variant, "] gate"))
  X_gate <- stats::model.matrix(~ . + 0, data = wp_vars_gate, na.action = na.pass)
  stopifnot(nrow(X_gate) == nrow(wp_vars_gate))
  preds_gate <- predict(model, X_gate)
  if (is.matrix(preds_gate)) preds_gate <- as.vector(preds_gate)

  meta_gate <- gate_df[, c("period", "points_diff", "est_match_elapsed", "match_id"), drop = FALSE]
  labels_gate <- gate_df$label_wp

  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  cli::cli_alert_success("[{variant}] done in {round(elapsed, 1)} min")

  list(
    variant = variant, model = model, optimal_nrounds = optimal_nrounds, cv_logloss = cv_logloss,
    preds_gate = preds_gate, labels_gate = labels_gate, meta_gate = meta_gate,
    n_train = nrow(X_train), weights_summary = weights_summary,
    monotone_constraints = params_wp$monotone_constraints,
    elapsed_min = elapsed
  )
}

# ---- Scorecard (the plan's fixed, no-metric-shopping scorecard) ------------

#' logloss/Brier over ALL gate rows (draws included, label 0.5 handled
#' naturally by both formulas -- "kept for logloss" convention, same as
#' fit_wp_calibration()'s docstring).
wp_logloss_brier <- function(preds, labels) {
  p <- pmin(pmax(preds, 1e-6), 1 - 1e-6)
  logloss <- -mean(labels * log(p) + (1 - labels) * log(1 - p))
  brier <- mean((p - labels)^2)
  c(logloss = logloss, brier = brier)
}

#' The fixed scorecard for one variant's gate-season predictions.
wp_scorecard <- function(preds, labels, meta) {
  slope_all <- wp_gate_slope(preds, labels, meta, cell = "all")
  q4c <- wp_gate_slope(preds, labels, meta, cell = "q4close", detail = TRUE)
  lb <- wp_logloss_brier(preds, labels)
  list(
    slope_all = slope_all, slope_q4close = q4c$slope,
    se_q4close = q4c$se, n_q4close = q4c$n,
    logloss = unname(lb["logloss"]), brier = unname(lb["brier"])
  )
}

# ---- Monotonicity violation diagnostic (S2 risk note) ----------------------

#' Report-only: for a sample of Q4/close gate rows, perturb points_diff
#' over a grid (recomputing the points_diff-DERIVED features --
#' xpoints_diff, diff_time_ratio, score_urgency, and any S3 extras that
#' depend on points_diff -- holding everything else, incl. pos_lead_prob,
#' fixed) and measure how often WP decreases as points_diff increases.
#' Simplified (doesn't recompute pos_lead_prob from the EP class probs),
#' but is applied identically to the constrained baseline and the S2
#' variant so the CONTRAST between the two violation rates is meaningful
#' even if the absolute numbers carry that approximation.
monotonicity_violation_rate <- function(gate_df, model, feature_cols,
                                        n_sample = 300, delta_grid = seq(-15, 15, by = 3),
                                        seed = 777) {
  # Same data.table NSE trap as run_wp_variant() -- see comment there.
  gate_df <- as.data.frame(gate_df)
  cell <- gate_df[wp_cell_flag(gate_df), ]
  if (nrow(cell) == 0) return(list(violation_rate = NA_real_, n_rows = 0L, n_pairs = 0L))
  set.seed(seed)
  idx <- sample(seq_len(nrow(cell)), min(n_sample, nrow(cell)))
  rows <- cell[idx, , drop = FALSE]

  n_violations <- 0L
  n_pairs <- 0L
  for (i in seq_len(nrow(rows))) {
    r <- rows[i, ]
    preds_i <- numeric(length(delta_grid))
    for (j in seq_along(delta_grid)) {
      d <- delta_grid[j]
      rr <- r
      rr$points_diff <- r$points_diff + d
      rr$xpoints_diff <- rr$points_diff + r$exp_pts
      rr$diff_time_ratio <- rr$xpoints_diff * r$time_left_scaler
      rr$score_urgency <- rr$points_diff / pmax(r$est_match_remaining / 60, 1)
      if ("margin_per_remaining_poss" %in% feature_cols) {
        rr$margin_per_remaining_poss <- r$margin_per_remaining_poss * (rr$points_diff / ifelse(r$points_diff == 0, 1, r$points_diff))
      }
      if ("points_diff_time_scaled" %in% feature_cols) {
        rr$points_diff_time_scaled <- rr$points_diff * r$time_left_scaler
      }
      Xrr <- stats::model.matrix(~ . + 0, data = rr[, feature_cols, drop = FALSE], na.action = na.pass)
      preds_i[j] <- predict(model, Xrr)
    }
    diffs <- diff(preds_i)
    n_pairs <- n_pairs + length(diffs)
    n_violations <- n_violations + sum(diffs < -1e-9)
  }
  list(violation_rate = n_violations / n_pairs, n_rows = nrow(rows), n_pairs = n_pairs)
}
