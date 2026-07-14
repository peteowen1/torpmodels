# ws3_cv_stack.R — WS3: CV-stack the GAM chain (FABLE-MATCH-MAE-PLAN.md)
# =========================================================================
# Hypothesis: replacing in-sample intermediate GAM predictions
# (gam_pred_tot_xscore, gam_pred_xscore_diff, gam_pred_conv_diff) with
# cross-fitted (out-of-fold) ones for stages 1-3 teaches models 2-4 the
# correct (smaller) pass-through weight on their stacked inputs, reducing
# the over-dispersion that produces the observed margin slope < 1.
#
# Only stages 1-3 are cross-fit (their outputs feed 2+ later stages).
# Stage 4 (score_diff) trains once on the cross-fitted stage 1-3 features
# but is not itself cross-fit; stage 5 (win) is unchanged from production
# (in-sample link to stage 4, exactly as .train_match_gams() today) — this
# mirrors the plan's explicit scope ("Stage 4/5 train on the cross-fitted
# features" — i.e. consume cross-fitted upstream features, not be
# cross-fit themselves).
#
# Test-row (held-out round) predictions always flow through the FULL-FIT
# chain (each stage's model trained on all training rows), exactly as
# production does today — only the TRAINING features for stages 2-5 change.
#
# Run:
#   Rscript "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/ws3_cv_stack.R"

# Setup ----
library(tidyverse)
library(xgboost)
library(mgcv)
library(MLmetrics)
library(geosphere)
library(cli)

torp_paths <- c("../torp", "../../torp", "../../../torp", "C:/dev/torpverse/torp")
torp_loaded <- FALSE
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p)
    torp_loaded <- TRUE
    break
  }
}
if (!torp_loaded) stop("Cannot find torp package (dev). Run from torpverse workspace.")

rolling_lib_candidates <- c(
  "experiments/rolling_lib.R",
  "04-match-model/experiments/rolling_lib.R",
  "data-raw/04-match-model/experiments/rolling_lib.R",
  "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/rolling_lib.R"
)
rolling_lib_hits <- rolling_lib_candidates[file.exists(rolling_lib_candidates)]
if (length(rolling_lib_hits) == 0) stop("Cannot find experiments/rolling_lib.R")
source(rolling_lib_hits[1])

# .cross_fit_stage ----

#' K-fold cross-fit one GAM stage, by match_id-coherent folds
#'
#' Fits `formula` on data excluding each fold in turn, predicting the held
#' fold. Returns a vector of out-of-fold predictions covering every row of
#' `data`, aligned to `data`'s row order.
#'
#' @keywords internal
.cross_fit_stage <- function(formula, data, weights, family, fold_vec, K,
                              nthreads, gamma_arg) {
  # NSE gotcha (documentation only -- no eval() is written here; this
  # describes base R's own internal stats::model.frame.default mechanism,
  # invoked by mgcv::bam() on trusted in-repo formula/data objects, not user
  # input): mgcv::bam()'s weights argument is resolved via
  # model.frame.default's internal `eval(extras, data, env)` where `env =
  # environment(formula)` -- NOT the frame bam() was called from. In
  # .train_match_gams()'s production code this is invisible because the
  # formula is built in the SAME frame as the bam() call (so its locals,
  # e.g. gam_df, are visible via environment(formula)). Here `formula` is
  # built in a different (caller) frame, so `weights[train_idx]` -- an
  # expression referencing THIS function's locals -- cannot be resolved and
  # errors with "object 'train_idx' not found". Fix: attach weights as an
  # actual column of the `data` argument and reference it as a bare column
  # name; column lookup via `data` itself succeeds regardless of env.
  data$.cv_weight <- weights
  cv_pred <- rep(NA_real_, nrow(data))
  for (k in seq_len(K)) {
    train_idx <- which(fold_vec != k)
    test_idx  <- which(fold_vec == k)
    train_df <- data[train_idx, ]
    test_df  <- data[test_idx, ]
    fit_k <- mgcv::bam(
      formula,
      data = train_df, weights = .cv_weight,
      family = family, nthreads = nthreads, select = TRUE, discrete = TRUE,
      drop.unused.levels = FALSE, gamma = gamma_arg
    )
    cv_pred[test_idx] <- predict(fit_k, newdata = test_df, type = "response")
  }
  if (anyNA(cv_pred)) {
    cli::cli_abort(".cross_fit_stage: {sum(is.na(cv_pred))} rows never fell in a held-out fold")
  }
  cv_pred
}

# .train_match_gams_cv_stack ----

#' CV-stacked variant of torp:::.train_match_gams()
#'
#' Copied (per plan G5 — experiments only, torp/R/*.R untouched) from
#' torp/R/match_train.R and modified so stages 1-3's gam_pred_* features
#' are cross-fitted (K = 3 folds by match_id) before being consumed by
#' downstream stage training, instead of the production in-sample pull
#' (`gam_df$gam_pred_X <- team_mdl_df$gam_pred_X[train_mask]`).
#'
#' Formulas are byte-identical to .train_match_gams() — only the training
#' data fed into stages 2-5 differs.
#'
#' @keywords internal
.train_match_gams_cv_stack <- function(team_mdl_df, train_filter = NULL,
                                        nthreads = 4L, gamma_arg = 1.4,
                                        K = 3L, seed = 1234L) {
  loadNamespace("mgcv")

  if (is.null(train_filter)) {
    train_mask <- !is.na(team_mdl_df$win)
  } else {
    train_mask <- train_filter & !is.na(team_mdl_df$win)
  }

  gam_df <- team_mdl_df[train_mask, ]
  cli::cli_inform("[cv_stack] Training on {nrow(gam_df)} completed matches (K={K})")
  if (nrow(gam_df) == 0) {
    cli::cli_abort("Cannot train GAM models: 0 completed matches after filtering")
  }

  # Fold assignment: K folds by match_id (both team-rows of a match share a
  # fold) — deterministic given `seed`, independent of round order.
  match_ids <- unique(gam_df$match_id)
  set.seed(seed)
  fold_lookup <- stats::setNames(
    sample(rep(seq_len(K), length.out = length(match_ids))),
    as.character(match_ids)
  )
  gam_df$.fold <- unname(fold_lookup[as.character(gam_df$match_id)])
  stopifnot(
    !anyNA(gam_df$.fold),
    all(tapply(gam_df$.fold, gam_df$match_id, function(x) length(unique(x))) == 1)
  )
  # Every fold must be non-trivial or cross_fit_stage's held-out predict has nothing.
  if (any(table(gam_df$.fold) == 0)) {
    cli::cli_abort("[cv_stack] Empty fold in early rolling rounds (n_train too small for K={K})")
  }

  # Optional smooth terms guard — identical to .train_match_gams(), computed
  # once on the full training set (not per-fold) so per-fold and full-fit
  # formulas stay identical.
  optional_smooth_terms <- list(
    "s(psr.x, bs = \"ts\", k = 5)"           = list(var = "psr.x", k = 5),
    "s(psr.y, bs = \"ts\", k = 5)"           = list(var = "psr.y", k = 5),
    "s(log_wind, bs = \"ts\", k = 5)"        = list(var = "log_wind", k = 5),
    "s(log_precip, bs = \"ts\", k = 5)"      = list(var = "log_precip", k = 5),
    "s(temp_avg, bs = \"ts\", k = 5)"        = list(var = "temp_avg", k = 5),
    "s(humidity_avg, bs = \"ts\", k = 5)"     = list(var = "humidity_avg", k = 5),
    "s(abs(psr_diff), bs = \"ts\", k = 5)"   = list(var = "psr_diff", k = 5),
    "s(abs(osr_diff), bs = \"ts\", k = 5)"   = list(var = "osr_diff", k = 5),
    "s(abs(dsr_diff), bs = \"ts\", k = 5)"   = list(var = "dsr_diff", k = 5),
    "s(psr_diff, bs = \"ts\", k = 5)"        = list(var = "psr_diff", k = 5),
    "s(osr_diff, bs = \"ts\", k = 5)"        = list(var = "osr_diff", k = 5),
    "s(dsr_diff, bs = \"ts\", k = 5)"        = list(var = "dsr_diff", k = 5),
    "ti(psr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)" = list(var = "psr_diff", k = 4)
  )
  drop_terms <- character(0)
  for (term_str in names(optional_smooth_terms)) {
    info <- optional_smooth_terms[[term_str]]
    vals <- gam_df[[info$var]]
    n_unique <- length(unique(vals[!is.na(vals)]))
    if (n_unique < info$k) drop_terms <- c(drop_terms, term_str)
  }
  .add_optional <- function(base_terms, optional_terms) {
    keep <- setdiff(optional_terms, drop_terms)
    if (length(keep) > 0) paste(base_terms, "+", paste(keep, collapse = " + ")) else base_terms
  }

  # ---- Stage 1: total xPoints (no upstream gam_pred_* inputs -> cross-fit as-is) ----
  m1_base <- paste(
    "total_xpoints_adj ~",
    "s(team_type_fac, bs = \"re\")",
    "+ s(game_year_decimal.x, bs = \"ts\")",
    "+ s(game_prop_through_year.x, bs = \"cc\")",
    "+ s(game_prop_through_month.x, bs = \"cc\")",
    "+ s(game_wday_fac.x, bs = \"re\")",
    "+ s(game_prop_through_day.x, bs = \"cc\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    "+ s(abs(epr_diff), bs = \"ts\", k = 5)",
    "+ s(abs(epr_recv_diff), bs = \"ts\", k = 5)",
    "+ s(abs(epr_disp_diff), bs = \"ts\", k = 5)",
    "+ s(abs(epr_spoil_diff), bs = \"ts\", k = 5)",
    "+ s(abs(epr_hitout_diff), bs = \"ts\", k = 5)",
    "+ s(epr.x, bs = \"ts\", k = 5) + s(epr.y, bs = \"ts\", k = 5)",
    "+ s(abs(torp_diff), bs = \"ts\", k = 5)",
    "+ s(torp.x, bs = \"ts\", k = 5) + s(torp.y, bs = \"ts\", k = 5)",
    "+ s(venue_fac, bs = \"re\")",
    "+ s(log_dist.x, bs = \"ts\", k = 5) + s(log_dist.y, bs = \"ts\", k = 5)",
    "+ s(familiarity.x, bs = \"ts\", k = 5) + s(familiarity.y, bs = \"ts\", k = 5)",
    "+ s(log_dist_diff, bs = \"ts\", k = 5)",
    "+ s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m1_optional <- c(
    "s(psr.x, bs = \"ts\", k = 5)", "s(psr.y, bs = \"ts\", k = 5)",
    "s(abs(psr_diff), bs = \"ts\", k = 5)",
    "s(abs(osr_diff), bs = \"ts\", k = 5)", "s(abs(dsr_diff), bs = \"ts\", k = 5)",
    "s(log_wind, bs = \"ts\", k = 5)", "s(log_precip, bs = \"ts\", k = 5)",
    "s(temp_avg, bs = \"ts\", k = 5)", "s(humidity_avg, bs = \"ts\", k = 5)"
  )
  m1_formula <- stats::as.formula(.add_optional(m1_base, m1_optional))

  cli::cli_progress_step("[cv_stack] Cross-fitting stage 1 (total xPoints), {K} folds")
  gam_df$gam_pred_tot_xscore <- .cross_fit_stage(
    m1_formula, gam_df, gam_df$weightz, gaussian(), gam_df$.fold, K, nthreads, gamma_arg
  )
  afl_total_xpoints_mdl <- mgcv::bam(
    m1_formula, data = gam_df, weights = gam_df$weightz,
    family = gaussian(), nthreads = nthreads, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_tot_xscore <- predict(afl_total_xpoints_mdl, newdata = team_mdl_df, type = "response")

  # ---- Stage 2: xScore differential (consumes CV stage-1 feature) ----
  m2_base <- paste(
    "xscore_diff ~",
    "s(team_type_fac, bs = \"re\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    "+ ti(epr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
    "+ s(gam_pred_tot_xscore, bs = \"ts\", k = 5)",
    "+ s(epr_diff, bs = \"ts\", k = 5)",
    "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
    "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
    "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
    "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
    "+ s(torp_diff, bs = \"ts\", k = 5)",
    "+ ti(torp_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
    "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m2_optional <- c("s(psr_diff, bs = \"ts\", k = 5)",
                    "ti(psr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
                    "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  m2_formula <- stats::as.formula(.add_optional(m2_base, m2_optional))

  cli::cli_progress_step("[cv_stack] Cross-fitting stage 2 (xScore diff), {K} folds")
  gam_df$gam_pred_xscore_diff <- .cross_fit_stage(
    m2_formula, gam_df, gam_df$weightz, gaussian(), gam_df$.fold, K, nthreads, gamma_arg
  )
  afl_xscore_diff_mdl <- mgcv::bam(
    m2_formula, data = gam_df, weights = gam_df$weightz,
    family = gaussian(), nthreads = nthreads, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_xscore_diff <- predict(afl_xscore_diff_mdl, newdata = team_mdl_df, type = "response")

  # ---- Stage 3: conversion differential (consumes CV stage-1/2 features) ----
  m3_base <- paste(
    "shot_conv_diff ~",
    "s(team_type_fac, bs = \"re\")",
    "+ s(game_year_decimal.x, bs = \"ts\")",
    "+ s(game_prop_through_year.x, bs = \"cc\")",
    "+ s(game_prop_through_month.x, bs = \"cc\")",
    "+ s(game_wday_fac.x, bs = \"re\")",
    "+ s(game_prop_through_day.x, bs = \"cc\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    "+ ti(epr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
    "+ s(epr_diff, bs = \"ts\", k = 5)",
    "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
    "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
    "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
    "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
    "+ s(torp_diff, bs = \"ts\", k = 5)",
    "+ ti(torp_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
    "+ s(gam_pred_tot_xscore, bs = \"ts\", k = 5)",
    "+ s(gam_pred_xscore_diff, bs = \"ts\", k = 5)",
    "+ s(venue_fac, bs = \"re\")",
    "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m3_optional <- c("s(psr_diff, bs = \"ts\", k = 5)",
                    "ti(psr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
                    "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  m3_formula <- stats::as.formula(.add_optional(m3_base, m3_optional))

  cli::cli_progress_step("[cv_stack] Cross-fitting stage 3 (conversion diff), {K} folds")
  gam_df$gam_pred_conv_diff <- .cross_fit_stage(
    m3_formula, gam_df, gam_df$shot_weightz, gaussian(), gam_df$.fold, K, nthreads, gamma_arg
  )
  afl_conv_mdl <- mgcv::bam(
    m3_formula, data = gam_df, weights = gam_df$shot_weightz,
    family = gaussian(), nthreads = nthreads, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_conv_diff <- predict(afl_conv_mdl, newdata = team_mdl_df, type = "response")

  # ---- Stage 4: score differential — trains on cross-fitted 1-3 features,
  # NOT itself cross-fit (plan scope: only "stages 1..3"). Single full fit,
  # exactly as production, but its training frame carries CV inputs instead
  # of in-sample ones. ----
  m4_base <- paste(
    "score_diff ~",
    "s(team_type_fac, bs = \"re\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    "+ ti(epr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
    "+ ti(gam_pred_xscore_diff, gam_pred_conv_diff, bs = \"ts\", k = 5)",
    "+ ti(gam_pred_tot_xscore, gam_pred_conv_diff, bs = \"ts\", k = 5)",
    "+ s(gam_pred_xscore_diff)",
    "+ s(epr_diff, bs = \"ts\", k = 5)",
    "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
    "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
    "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
    "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
    "+ s(torp_diff, bs = \"ts\", k = 5)",
    "+ ti(torp_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
    "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m4_optional <- c("s(psr_diff, bs = \"ts\", k = 5)",
                    "ti(psr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
                    "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  m4_formula <- stats::as.formula(.add_optional(m4_base, m4_optional))

  cli::cli_progress_step("[cv_stack] Training stage 4 (score diff) on CV-stacked features")
  afl_score_mdl <- mgcv::bam(
    m4_formula, data = gam_df, weights = gam_df$weightz,
    family = "gaussian", nthreads = nthreads, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_score_diff <- predict(afl_score_mdl, newdata = team_mdl_df, type = "response")

  # ---- Stage 5: win probability — unchanged from production (in-sample
  # link to stage 4, out of WS3's scope). ----
  cli::cli_progress_step("[cv_stack] Training stage 5 (win probability)")
  gam_df$pred_tot_xscore  <- gam_df$gam_pred_tot_xscore
  gam_df$pred_score_diff  <- team_mdl_df$gam_pred_score_diff[train_mask]
  afl_win_mdl <- mgcv::bam(
    win ~
      +s(team_name.x, bs = "re") + s(team_name.y, bs = "re")
      + s(team_name_season.x, bs = "re") + s(team_name_season.y, bs = "re")
      + ti(pred_tot_xscore, pred_score_diff, bs = c("ts", "ts"), k = 4)
      + s(pred_score_diff, bs = "ts", k = 5)
      + s(log_dist_diff, bs = "ts", k = 5) + s(familiarity_diff, bs = "ts", k = 5)
      + s(days_rest_diff_fac, bs = "re"),
    data = gam_df, weights = gam_df$weightz,
    family = "binomial", nthreads = nthreads, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE, gamma = gamma_arg
  )

  team_mdl_df$pred_tot_xscore  <- team_mdl_df$gam_pred_tot_xscore
  team_mdl_df$pred_xscore_diff <- team_mdl_df$gam_pred_xscore_diff
  team_mdl_df$pred_conv_diff   <- team_mdl_df$gam_pred_conv_diff
  team_mdl_df$pred_score_diff  <- team_mdl_df$gam_pred_score_diff

  team_mdl_df$gam_pred_win <- predict(afl_win_mdl, newdata = team_mdl_df, type = "response")
  team_mdl_df$pred_win     <- team_mdl_df$gam_pred_win

  if (any(is.na(team_mdl_df$pred_win[!is.na(team_mdl_df$win)]))) {
    cli::cli_warn("[cv_stack] NA values in pred_win for completed matches")
  }

  sym_check <- team_mdl_df |>
    dplyr::group_by(match_id) |>
    dplyr::summarise(
      score_sum = sum(pred_score_diff), win_sum = sum(pred_win), n = dplyr::n(),
      .groups = "drop"
    ) |>
    dplyr::filter(n == 2)
  if (nrow(sym_check) > 0) {
    max_score_asym <- max(abs(sym_check$score_sum), na.rm = TRUE)
    if (max_score_asym > 5) {
      cli::cli_abort("[cv_stack] Home/away prediction asymmetry detected (max score_diff sum: {round(max_score_asym, 1)})")
    }
  }

  team_mdl_df$bits <- dplyr::case_when(
    team_mdl_df$win == 1   ~ 1 + log2(team_mdl_df$pred_win),
    team_mdl_df$win == 0   ~ 1 + log2(1 - team_mdl_df$pred_win),
    TRUE                   ~ 1 + 0.5 * log2(team_mdl_df$pred_win * (1 - team_mdl_df$pred_win))
  )
  team_mdl_df$mae <- abs(team_mdl_df$score_diff - team_mdl_df$pred_score_diff)

  models <- list(
    total_xpoints = afl_total_xpoints_mdl, xscore_diff = afl_xscore_diff_mdl,
    conv_diff = afl_conv_mdl, score_diff = afl_score_mdl, win = afl_win_mdl
  )
  cli::cli_alert_success("[cv_stack] GAM pipeline trained on {nrow(gam_df)} matches (K={K} cross-fit stages 1-3)")
  list(models = models, data = team_mdl_df)
}

# recal_margin / through-origin b readout (WS1-style, plan's key WS3 check) ----
# Lightweight stand-in for WS1's full expanding-window recalibration (not yet
# landed): a single pooled through-origin slope b = lm(margin ~ pred_margin
# + 0) fit on the OOS predictions from a completed rolling run. Reports
# whether cv_stack's fitted b moves toward 1.0 relative to baseline's — the
# plan's explicit WS3 interaction readout ("rerun WS1's V1a on top of the
# CV-stacked chain").
.pooled_through_origin_b <- function(preds) {
  unname(stats::coef(stats::lm(margin ~ pred_margin + 0, data = preds))[1])
}
