# ws_psr_epr_0_ablations.R -- G4 fresh baseline card + WS0 parts-vs-whole
# feature ablations (V0a-V0e), per docs/plans/FABLE-PSR-EPR-PLAN.md.
# =====================================================================
# WS0 hypothesis (plan Section 2): the composite features (torp_diff,
# epr_diff) are redundant given their summands (the four EPR component
# diffs, psr_diff). This is the falsifier for the h2o-importance lead
# documented in docs/reviews/2026-PSR-EPR-DIAGNOSIS.md -- the h2o GLM saw
# exact linear redundancy (torp_diff = 0.5*epr_diff + 0.5*psr_diff;
# epr_diff = sum of 4 components) and "components rank above composite" is
# the expected degenerate outcome of a regularised linear model facing
# that redundancy, not evidence the blend is wrong. WS0 tests whether the
# match GAM/XGB pipeline (which already owns these reweighting degrees of
# freedom -- diagnosis Section 3) is actually helped or hurt by dropping
# the composites/components, in the one true scoreboard: rolling
# week-by-week OOS (rolling_lib.R's run_rolling_eval()).
#
# Five pre-registered variants (plan Section 2, WS0 table -- closed list,
# do not add/remove/edit here):
#   V0a - drop TORP composite            (GAM: torp smooths; XGB: torp_diff)
#   V0b - components-only EPR            (V0a + drop epr_diff; keep epr.x/.y
#                                          level terms + all 4 components)
#   V0c - composites-only                (drop the 4 EPR component smooths;
#                                          keep epr_diff, torp_diff, psr_diff)
#   V0d - EPR-family only                (V0a + drop psr family entirely)
#   V0e - PSR-family only                (drop all EPR: components, epr_diff,
#                                          epr.x/.y level terms, + torp; keep
#                                          psr family)
#
# Implementation: each variant is a LOCAL flag-parametrised copy of
# torp:::.train_match_gams() (torp/R/match_train.R, the current V4b+elo
# "C6" production formula) and rolling_lib.R's .train_xgb_fixed() -- plan
# G5 ("variant trainers are local copies/edits inside the experiment
# file"; torp/R/*.R is never touched). The baseline (G4) uses the REAL
# production trainers (.train_match_gams unqualified via devtools::load_all,
# .train_xgb_fixed with extra_feature_cols="elo_diff") so the fresh
# baseline card is genuinely "what torp/R/ on dev now implements" (plan
# G4), not a hand-copied approximation.
#
# V1a recalibration post-pass (recal_expanding/.apply_recal/v1a_recal_own)
# is copied verbatim from ws6_decay_on_c6.R / ws7_elo_xgb_fix.R (which
# copied it from ws1_margin_recal.R) -- applied to baseline AND every
# variant, per plan Section 2 WS0 build note and G5 (each WS keeps its own
# copy rather than sourcing another WS's stage-gated top-level code).
#
# Run stage-by-stage (checkpoints to experiments/results/*.rds):
#   Rscript ws_psr_epr_0_ablations.R data              # build+cache team_mdl_df fresh
#   Rscript ws_psr_epr_0_ablations.R baseline_screen    # G4 baseline, TEST_SEASONS=2026
#   Rscript ws_psr_epr_0_ablations.R baseline_pool      # G4 baseline, pooled 2025:2026 + falsifier check
#   Rscript ws_psr_epr_0_ablations.R screen_v0a         # (...v0b, v0c, v0d, v0e) 2026 screen vs baseline
#   Rscript ws_psr_epr_0_ablations.R confirm_v0a        # (...) pooled 2025:2026 confirm, only if adoptable per G2
#   Rscript ws_psr_epr_0_ablations.R summary            # full table + WS0 interpretation branches

stage <- {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) >= 1) a[1] else "all"
}
cat("=== ws_psr_epr_0_ablations.R stage:", stage, "===\n")

# Setup ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(xgboost)
  library(mgcv)
  library(MLmetrics)
  library(geosphere)
  library(cli)
  library(data.table)
})

torp_paths <- c("../torp", "../../torp", "../../../torp", "C:/dev/torpverse/torp")
torp_loaded <- FALSE
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p, quiet = TRUE)
    torp_loaded <- TRUE
    break
  }
}
if (!torp_loaded) stop("Cannot find torp package (run from torpverse workspace).")

EXPERIMENTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RESULTS_DIR <- file.path(EXPERIMENTS_DIR, "results")
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR, recursive = TRUE)
.rds <- function(name) file.path(RESULTS_DIR, name)

source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))

TEST_SEASONS    <- 2026
CONFIRM_SEASONS <- 2025:2026

# ---- V1a recal: recal_expanding + .apply_recal + v1a_recal_own + printers
# (copied verbatim from ws6_decay_on_c6.R / ws7_elo_xgb_fix.R, which
# themselves copied from ws1_margin_recal.R -- plan G5: each WS keeps its
# own copy). ----
recal_expanding <- function(preds_all, score_idx, history_pool_idx,
                             mode = c("slope_only", "slope_intercept", "nonlinear"),
                             min_n = 30) {
  mode <- match.arg(mode)
  key <- preds_all$season * 1000L + preds_all$round

  score_idx <- score_idx[order(key[score_idx])]
  hist_key  <- key[history_pool_idx]

  out <- numeric(length(score_idx))
  b_trace <- vector("list", length(score_idx))

  for (k in seq_along(score_idx)) {
    i <- score_idx[k]
    cur_key <- key[i]
    hist_idx <- history_pool_idx[hist_key < cur_key]
    n_hist <- length(hist_idx)

    if (n_hist < min_n) {
      out[k] <- preds_all$pred_margin[i]
      b_trace[[k]] <- data.frame(season = preds_all$season[i], round = preds_all$round[i],
                                  b = 1, a = 0, n_hist = n_hist, applied = "identity")
      next
    }

    hist_df <- preds_all[hist_idx, c("pred_margin", "margin")]

    if (mode == "slope_only") {
      b <- unname(stats::coef(stats::lm(margin ~ pred_margin + 0, data = hist_df))[1])
      out[k] <- b * preds_all$pred_margin[i]
      b_trace[[k]] <- data.frame(season = preds_all$season[i], round = preds_all$round[i],
                                  b = b, a = 0, n_hist = n_hist, applied = mode)
    } else if (mode == "slope_intercept") {
      cf <- stats::coef(stats::lm(margin ~ pred_margin, data = hist_df))
      out[k] <- unname(cf[["(Intercept)"]] + cf[["pred_margin"]] * preds_all$pred_margin[i])
      b_trace[[k]] <- data.frame(season = preds_all$season[i], round = preds_all$round[i],
                                  b = unname(cf[["pred_margin"]]), a = unname(cf[["(Intercept)"]]),
                                  n_hist = n_hist, applied = mode)
    } else {
      fit <- tryCatch(mgcv::gam(margin ~ s(pred_margin, k = 4), data = hist_df), error = function(e) NULL)
      if (is.null(fit)) {
        b <- unname(stats::coef(stats::lm(margin ~ pred_margin + 0, data = hist_df))[1])
        out[k] <- b * preds_all$pred_margin[i]
        b_trace[[k]] <- data.frame(season = preds_all$season[i], round = preds_all$round[i],
                                    b = b, a = 0, n_hist = n_hist, applied = "slope_fallback")
      } else {
        center <- as.numeric(predict(fit, newdata = data.frame(pred_margin = 0)))
        val <- as.numeric(predict(fit, newdata = data.frame(pred_margin = preds_all$pred_margin[i]))) - center
        out[k] <- val
        b_trace[[k]] <- data.frame(season = preds_all$season[i], round = preds_all$round[i],
                                    b = NA_real_, a = NA_real_, n_hist = n_hist, applied = mode)
      }
    }
  }

  list(idx = score_idx, pred_margin_recal = out, b_trace = dplyr::bind_rows(b_trace))
}

.apply_recal <- function(preds_all, res) {
  out <- preds_all[res$idx, ]
  out$pred_margin <- res$pred_margin_recal
  out
}

v1a_recal_own <- function(preds) {
  idx <- seq_len(nrow(preds))
  res <- recal_expanding(preds, idx, idx, mode = "slope_only", min_n = 30)
  .apply_recal(preds, res)
}

.print_metrics <- function(m, label) {
  cat(sprintf(
    "%-48s MAE=%.3f RMSE=%.3f Brier=%.4f Bits=%.4f Slope=%.3f Cor=%.3f SDRatio=%.3f CloseMAE(n=%d)=%.3f\n",
    label, m$mae, m$rmse, m$brier, m$bits, m$slope, m$cor, m$sd_ratio, m$close_n, m$close_mae
  ))
}

.print_decomposition <- function(m, label) {
  implied_slope <- m$cor * (m$sd_actual / m$sd_pred)
  cat(sprintf(
    "  [decomp] %-42s slope=%.3f = cor(%.3f) * sd_actual/sd_pred(%.3f) [check: %.3f] sd_ratio(pred/actual)=%.3f\n",
    label, m$slope, m$cor, m$sd_actual / m$sd_pred, implied_slope, m$sd_ratio
  ))
}

# ================================================================
# WS0 variant GAM trainer: local flag-parametrised copy of
# torp:::.train_match_gams() (torp/R/match_train.R -- V4b structural
# formula + elo_diff optional smooth on models 2/4, the current C6
# production trainer). Only models 1-4 vary; model 5 (win) is byte-
# identical to production and untouched by every variant (plan WS0 table:
# "GAM changes (models 1-4)").
#
# Drop flags (composable -- see plan WS0 table -> flag mapping in this
# file's header comment and the PLAN.md results section):
#   drop_torp            m1: s(abs(torp_diff)), s(torp.x), s(torp.y)
#                         m2-4: s(torp_diff)
#   drop_epr_diff         m1: s(abs(epr_diff)); m2-4: s(epr_diff)
#   drop_epr_components   m1: 4x s(abs(epr_*_diff)); m2-4: 4x s(epr_*_diff)
#   drop_epr_level        m1 only: s(epr.x), s(epr.y)
#   drop_psr_family        m1: s(psr.x/.y), s(abs(psr/osr/dsr_diff))
#                          m2-4: s(psr_diff), s(osr_diff), s(dsr_diff)
#                          (force-dropped regardless of the unique-value
#                          optional-term guard, since dropping the family
#                          is the point of the ablation, not data scarcity)
# ================================================================
.gam_trainer_ablate <- function(team_mdl_df, train_filter = NULL, nthreads = 4L, gamma_arg = 1.4,
                                 drop_torp = FALSE, drop_epr_diff = FALSE,
                                 drop_epr_components = FALSE, drop_epr_level = FALSE,
                                 drop_psr_family = FALSE) {
  loadNamespace("mgcv")

  if (is.null(train_filter)) {
    train_mask <- !is.na(team_mdl_df$win)
  } else {
    train_mask <- train_filter & !is.na(team_mdl_df$win)
  }

  gam_df <- team_mdl_df[train_mask, ]
  cli::cli_inform("[ablate] Training on {nrow(gam_df)} completed matches")
  if (nrow(gam_df) == 0) cli::cli_abort("Cannot train GAM models: 0 completed matches after filtering")

  optional_smooth_terms <- list(
    "s(psr.x, bs = \"ts\", k = 5)"           = list(var = "psr.x", k = 5),
    "s(psr.y, bs = \"ts\", k = 5)"           = list(var = "psr.y", k = 5),
    "s(log_wind, bs = \"ts\", k = 5)"        = list(var = "log_wind", k = 5),
    "s(log_precip, bs = \"ts\", k = 5)"      = list(var = "log_precip", k = 5),
    "s(temp_avg, bs = \"ts\", k = 5)"        = list(var = "temp_avg", k = 5),
    "s(humidity_avg, bs = \"ts\", k = 5)"    = list(var = "humidity_avg", k = 5),
    "s(abs(psr_diff), bs = \"ts\", k = 5)"   = list(var = "psr_diff", k = 5),
    "s(abs(osr_diff), bs = \"ts\", k = 5)"   = list(var = "osr_diff", k = 5),
    "s(abs(dsr_diff), bs = \"ts\", k = 5)"   = list(var = "dsr_diff", k = 5),
    "s(psr_diff, bs = \"ts\", k = 5)"        = list(var = "psr_diff", k = 5),
    "s(osr_diff, bs = \"ts\", k = 5)"        = list(var = "osr_diff", k = 5),
    "s(dsr_diff, bs = \"ts\", k = 5)"        = list(var = "dsr_diff", k = 5),
    "s(elo_diff, bs = \"ts\", k = 5)"        = list(var = "elo_diff", k = 5)
  )
  drop_terms <- character(0)
  for (term_str in names(optional_smooth_terms)) {
    info <- optional_smooth_terms[[term_str]]
    vals <- gam_df[[info$var]]
    n_unique <- length(unique(vals[!is.na(vals)]))
    if (n_unique < info$k) drop_terms <- c(drop_terms, term_str)
  }
  if (drop_psr_family) {
    drop_terms <- c(drop_terms,
      "s(psr.x, bs = \"ts\", k = 5)", "s(psr.y, bs = \"ts\", k = 5)",
      "s(abs(psr_diff), bs = \"ts\", k = 5)",
      "s(abs(osr_diff), bs = \"ts\", k = 5)", "s(abs(dsr_diff), bs = \"ts\", k = 5)",
      "s(psr_diff, bs = \"ts\", k = 5)", "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)"
    )
  }
  .add_optional <- function(base_terms, optional_terms) {
    keep <- setdiff(optional_terms, drop_terms)
    if (length(keep) > 0) paste(base_terms, "+", paste(keep, collapse = " + ")) else base_terms
  }

  # Model 1: total xPoints ----
  cli::cli_progress_step("[ablate] Training total xPoints model")
  m1_parts <- c(
    "total_xpoints_adj ~",
    "s(team_type_fac, bs = \"re\")",
    "+ s(game_year_decimal.x, bs = \"ts\")",
    "+ s(game_prop_through_year.x, bs = \"cc\")",
    "+ s(game_prop_through_month.x, bs = \"cc\")",
    "+ s(game_wday_fac.x, bs = \"re\")",
    "+ s(game_prop_through_day.x, bs = \"cc\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    if (!drop_epr_diff) "+ s(abs(epr_diff), bs = \"ts\", k = 5)",
    if (!drop_epr_components) paste(
      "+ s(abs(epr_recv_diff), bs = \"ts\", k = 5)",
      "+ s(abs(epr_disp_diff), bs = \"ts\", k = 5)",
      "+ s(abs(epr_spoil_diff), bs = \"ts\", k = 5)",
      "+ s(abs(epr_hitout_diff), bs = \"ts\", k = 5)"
    ),
    if (!drop_epr_level) "+ s(epr.x, bs = \"ts\", k = 5) + s(epr.y, bs = \"ts\", k = 5)",
    if (!drop_torp) paste(
      "+ s(abs(torp_diff), bs = \"ts\", k = 5)",
      "+ s(torp.x, bs = \"ts\", k = 5) + s(torp.y, bs = \"ts\", k = 5)"
    ),
    "+ s(venue_fac, bs = \"re\")",
    "+ s(log_dist.x, bs = \"ts\", k = 5) + s(log_dist.y, bs = \"ts\", k = 5)",
    "+ s(familiarity.x, bs = \"ts\", k = 5) + s(familiarity.y, bs = \"ts\", k = 5)",
    "+ s(log_dist_diff, bs = \"ts\", k = 5)",
    "+ s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m1_base <- paste(m1_parts, collapse = " ")
  m1_optional <- c(
    "s(psr.x, bs = \"ts\", k = 5)", "s(psr.y, bs = \"ts\", k = 5)",
    "s(abs(psr_diff), bs = \"ts\", k = 5)",
    "s(abs(osr_diff), bs = \"ts\", k = 5)", "s(abs(dsr_diff), bs = \"ts\", k = 5)",
    "s(log_wind, bs = \"ts\", k = 5)", "s(log_precip, bs = \"ts\", k = 5)",
    "s(temp_avg, bs = \"ts\", k = 5)", "s(humidity_avg, bs = \"ts\", k = 5)"
  )
  m1_formula <- stats::as.formula(.add_optional(m1_base, m1_optional))
  afl_total_xpoints_mdl <- mgcv::bam(
    m1_formula, data = gam_df, weights = gam_df$weightz, family = gaussian(),
    nthreads = nthreads, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_tot_xscore <- predict(afl_total_xpoints_mdl, newdata = team_mdl_df, type = "response")

  # Model 2: xScore diff ----
  cli::cli_progress_step("[ablate] Training xScore diff model")
  gam_df$gam_pred_tot_xscore <- team_mdl_df$gam_pred_tot_xscore[train_mask]
  m2_parts <- c(
    "xscore_diff ~",
    "s(team_type_fac, bs = \"re\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    "+ s(gam_pred_tot_xscore, bs = \"ts\", k = 5)",
    if (!drop_epr_diff) "+ s(epr_diff, bs = \"ts\", k = 5)",
    if (!drop_epr_components) paste(
      "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
      "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
      "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
      "+ s(epr_hitout_diff, bs = \"ts\", k = 5)"
    ),
    if (!drop_torp) "+ s(torp_diff, bs = \"ts\", k = 5)",
    "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m2_base <- paste(m2_parts, collapse = " ")
  m2_optional <- c("s(psr_diff, bs = \"ts\", k = 5)", "s(osr_diff, bs = \"ts\", k = 5)",
                    "s(dsr_diff, bs = \"ts\", k = 5)", "s(elo_diff, bs = \"ts\", k = 5)")
  m2_formula <- stats::as.formula(.add_optional(m2_base, m2_optional))
  afl_xscore_diff_mdl <- mgcv::bam(
    m2_formula, data = gam_df, weights = gam_df$weightz, family = gaussian(),
    nthreads = nthreads, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_xscore_diff <- predict(afl_xscore_diff_mdl, newdata = team_mdl_df, type = "response")

  # Model 3: conversion diff (no elo_diff -- production only adds elo to m2/m4) ----
  cli::cli_progress_step("[ablate] Training conversion model")
  gam_df$gam_pred_xscore_diff <- team_mdl_df$gam_pred_xscore_diff[train_mask]
  m3_parts <- c(
    "shot_conv_diff ~",
    "s(team_type_fac, bs = \"re\")",
    "+ s(game_year_decimal.x, bs = \"ts\")",
    "+ s(game_prop_through_year.x, bs = \"cc\")",
    "+ s(game_prop_through_month.x, bs = \"cc\")",
    "+ s(game_wday_fac.x, bs = \"re\")",
    "+ s(game_prop_through_day.x, bs = \"cc\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    if (!drop_epr_diff) "+ s(epr_diff, bs = \"ts\", k = 5)",
    if (!drop_epr_components) paste(
      "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
      "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
      "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
      "+ s(epr_hitout_diff, bs = \"ts\", k = 5)"
    ),
    if (!drop_torp) "+ s(torp_diff, bs = \"ts\", k = 5)",
    "+ s(gam_pred_tot_xscore, bs = \"ts\", k = 5)",
    "+ s(gam_pred_xscore_diff, bs = \"ts\", k = 5)",
    "+ s(venue_fac, bs = \"re\")",
    "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m3_base <- paste(m3_parts, collapse = " ")
  m3_optional <- c("s(psr_diff, bs = \"ts\", k = 5)", "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  m3_formula <- stats::as.formula(.add_optional(m3_base, m3_optional))
  afl_conv_mdl <- mgcv::bam(
    m3_formula, data = gam_df, weights = gam_df$shot_weightz, family = gaussian(),
    nthreads = nthreads, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_conv_diff <- predict(afl_conv_mdl, newdata = team_mdl_df, type = "response")

  # Model 4: score diff ----
  cli::cli_progress_step("[ablate] Training score diff model")
  gam_df$gam_pred_conv_diff <- team_mdl_df$gam_pred_conv_diff[train_mask]
  m4_parts <- c(
    "score_diff ~",
    "s(team_type_fac, bs = \"re\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    "+ s(gam_pred_xscore_diff)",
    if (!drop_epr_diff) "+ s(epr_diff, bs = \"ts\", k = 5)",
    if (!drop_epr_components) paste(
      "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
      "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
      "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
      "+ s(epr_hitout_diff, bs = \"ts\", k = 5)"
    ),
    if (!drop_torp) "+ s(torp_diff, bs = \"ts\", k = 5)",
    "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m4_base <- paste(m4_parts, collapse = " ")
  m4_optional <- c("s(psr_diff, bs = \"ts\", k = 5)", "s(osr_diff, bs = \"ts\", k = 5)",
                    "s(dsr_diff, bs = \"ts\", k = 5)", "s(elo_diff, bs = \"ts\", k = 5)")
  m4_formula <- stats::as.formula(.add_optional(m4_base, m4_optional))
  afl_score_mdl <- mgcv::bam(
    m4_formula, data = gam_df, weights = gam_df$weightz, family = "gaussian",
    nthreads = nthreads, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_score_diff <- predict(afl_score_mdl, newdata = team_mdl_df, type = "response")

  # Model 5: win probability -- unchanged in every variant (plan WS0: only models 1-4 vary) ----
  cli::cli_progress_step("[ablate] Training win probability model")
  gam_df$pred_tot_xscore <- gam_df$gam_pred_tot_xscore
  gam_df$pred_score_diff <- team_mdl_df$gam_pred_score_diff[train_mask]
  afl_win_mdl <- mgcv::bam(
    win ~
      + s(team_name.x, bs = "re") + s(team_name.y, bs = "re")
      + s(team_name_season.x, bs = "re") + s(team_name_season.y, bs = "re")
      + ti(pred_tot_xscore, pred_score_diff, bs = c("ts", "ts"), k = 4)
      + s(pred_score_diff, bs = "ts", k = 5)
      + s(log_dist_diff, bs = "ts", k = 5) + s(familiarity_diff, bs = "ts", k = 5)
      + s(days_rest_diff_fac, bs = "re"),
    data = gam_df, weights = gam_df$weightz, family = "binomial",
    nthreads = nthreads, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$pred_tot_xscore  <- team_mdl_df$gam_pred_tot_xscore
  team_mdl_df$pred_xscore_diff <- team_mdl_df$gam_pred_xscore_diff
  team_mdl_df$pred_conv_diff   <- team_mdl_df$gam_pred_conv_diff
  team_mdl_df$pred_score_diff  <- team_mdl_df$gam_pred_score_diff
  team_mdl_df$gam_pred_win <- predict(afl_win_mdl, newdata = team_mdl_df, type = "response")
  team_mdl_df$pred_win     <- team_mdl_df$gam_pred_win

  team_mdl_df$bits <- dplyr::case_when(
    team_mdl_df$win == 1 ~ 1 + log2(team_mdl_df$pred_win),
    team_mdl_df$win == 0 ~ 1 + log2(1 - team_mdl_df$pred_win),
    TRUE ~ 1 + 0.5 * log2(team_mdl_df$pred_win * (1 - team_mdl_df$pred_win))
  )

  models <- list(total_xpoints = afl_total_xpoints_mdl, xscore_diff = afl_xscore_diff_mdl,
                 conv_diff = afl_conv_mdl, score_diff = afl_score_mdl, win = afl_win_mdl)
  list(models = models, data = team_mdl_df)
}

# ================================================================
# WS0 variant XGB trainer: local flag-parametrised copy of rolling_lib.R's
# .train_xgb_fixed() (itself mirroring torp:::.train_match_xgb()'s
# base_cols, minus per-round CV). Steps 1-4 base_cols drop the same
# columns the GAM formula drops; step 5 (win, diagnostic only) is
# untouched, matching the GAM model 5 invariance above.
# ================================================================
.xgb_trainer_ablate <- function(team_mdl_df, train_filter, nrounds_vec, extra_feature_cols = NULL,
                                 xgb_nthread = NULL,
                                 drop_torp = FALSE, drop_epr_diff = FALSE,
                                 drop_epr_components = FALSE, drop_psr_family = FALSE) {
  loadNamespace("xgboost")

  train_mask <- train_filter & !is.na(team_mdl_df$win) &
    !is.na(team_mdl_df$total_xpoints_adj) & !is.na(team_mdl_df$xscore_diff) &
    !is.na(team_mdl_df$shot_conv_diff) & !is.na(team_mdl_df$score_diff)

  xgb_df <- team_mdl_df[train_mask, ]
  if (nrow(xgb_df) == 0) return(team_mdl_df)

  osr_dsr_cols <- character(0)
  if (!drop_psr_family && all(c("osr_diff", "dsr_diff") %in% names(team_mdl_df)) &&
      !all(is.na(team_mdl_df$osr_diff))) {
    osr_dsr_cols <- c("osr_diff", "dsr_diff")
  }

  base_cols <- c(
    "team_type_fac",
    "game_year_decimal.x", "game_prop_through_year.x",
    "game_prop_through_month.x", "game_prop_through_day.x",
    if (!drop_epr_diff) "epr_diff",
    if (!drop_epr_components) c("epr_recv_diff", "epr_disp_diff", "epr_spoil_diff", "epr_hitout_diff"),
    if (!drop_torp) "torp_diff",
    if (!drop_psr_family) "psr_diff",
    osr_dsr_cols,
    "log_dist_diff", "familiarity_diff", "days_rest_diff_fac",
    extra_feature_cols
  )

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

  predict_all <- function(model, df, feature_cols) {
    mat <- stats::model.matrix(~ . - 1, data = df[, feature_cols, drop = FALSE])
    predict(model, xgboost::xgb.DMatrix(data = mat))
  }

  # Step 1: total xPoints
  m1 <- train_fixed(xgb_df, xgb_df$total_xpoints_adj, xgb_df$weightz,
                     base_cols, reg_params, nrounds_vec["total_xpoints"])
  xgb_df$xgb_pred_tot_xscore <- predict_all(m1, xgb_df, base_cols)
  team_mdl_df$xgb_pred_tot_xscore <- predict_all(m1, team_mdl_df, base_cols)

  # Step 2: xScore diff
  s2_cols <- c(base_cols, "xgb_pred_tot_xscore")
  m2 <- train_fixed(xgb_df, xgb_df$xscore_diff, xgb_df$weightz,
                     s2_cols, reg_params, nrounds_vec["xscore_diff"])
  xgb_df$xgb_pred_xscore_diff <- predict_all(m2, xgb_df, s2_cols)
  team_mdl_df$xgb_pred_xscore_diff <- predict_all(m2, team_mdl_df, s2_cols)

  # Step 3: conv diff
  s3_cols <- c(base_cols, "xgb_pred_tot_xscore", "xgb_pred_xscore_diff")
  m3 <- train_fixed(xgb_df, xgb_df$shot_conv_diff, xgb_df$shot_weightz,
                     s3_cols, reg_params, nrounds_vec["conv_diff"])
  xgb_df$xgb_pred_conv_diff <- predict_all(m3, xgb_df, s3_cols)
  team_mdl_df$xgb_pred_conv_diff <- predict_all(m3, team_mdl_df, s3_cols)

  # Step 4: score diff
  s4_cols <- c(base_cols, "xgb_pred_xscore_diff", "xgb_pred_conv_diff", "xgb_pred_tot_xscore")
  m4 <- train_fixed(xgb_df, xgb_df$score_diff, xgb_df$weightz,
                     s4_cols, reg_params, nrounds_vec["score_diff"])
  xgb_df$xgb_pred_score_diff <- predict_all(m4, xgb_df, s4_cols)
  team_mdl_df$xgb_pred_score_diff <- predict_all(m4, team_mdl_df, s4_cols)

  # Step 5: win probability -- untouched (diagnostic only, not fed by any dropped col directly)
  s5_cols <- c("team_type_fac", "xgb_pred_tot_xscore", "xgb_pred_score_diff",
               "log_dist_diff", "familiarity_diff", "days_rest_diff_fac")
  m5 <- train_fixed(xgb_df, as.numeric(xgb_df$win), xgb_df$weightz,
                     s5_cols, cls_params, nrounds_vec["win"])
  team_mdl_df$xgb_pred_win <- predict_all(m5, team_mdl_df, s5_cols)

  team_mdl_df
}

# ---- Variant registry (plan WS0 table -> flag mapping, see header) ----
make_variant <- function(label, drop_torp = FALSE, drop_epr_diff = FALSE,
                          drop_epr_components = FALSE, drop_epr_level = FALSE,
                          drop_psr_family = FALSE) {
  force(label); force(drop_torp); force(drop_epr_diff)
  force(drop_epr_components); force(drop_epr_level); force(drop_psr_family)
  list(
    label = label,
    gam_trainer = function(team_mdl_df, train_filter = NULL, nthreads = 4L) {
      .gam_trainer_ablate(team_mdl_df, train_filter = train_filter, nthreads = nthreads, gamma_arg = 1.4,
                           drop_torp = drop_torp, drop_epr_diff = drop_epr_diff,
                           drop_epr_components = drop_epr_components, drop_epr_level = drop_epr_level,
                           drop_psr_family = drop_psr_family)
    },
    xgb_trainer = function(team_mdl_df, train_filter, nrounds_vec, extra_feature_cols = NULL, xgb_nthread = NULL) {
      .xgb_trainer_ablate(team_mdl_df, train_filter = train_filter, nrounds_vec = nrounds_vec,
                           extra_feature_cols = extra_feature_cols, xgb_nthread = xgb_nthread,
                           drop_torp = drop_torp, drop_epr_diff = drop_epr_diff,
                           drop_epr_components = drop_epr_components, drop_psr_family = drop_psr_family)
    }
  )
}

VARIANTS <- list(
  v0a = make_variant("V0a - drop TORP composite", drop_torp = TRUE),
  v0b = make_variant("V0b - components-only EPR", drop_torp = TRUE, drop_epr_diff = TRUE),
  v0c = make_variant("V0c - composites-only", drop_epr_components = TRUE),
  v0d = make_variant("V0d - EPR-family only", drop_torp = TRUE, drop_psr_family = TRUE),
  v0e = make_variant("V0e - PSR-family only", drop_torp = TRUE, drop_epr_diff = TRUE,
                      drop_epr_components = TRUE, drop_epr_level = TRUE)
)

# ================================================================
# Stage: data -- fresh team_mdl_df (G4: "re-derive a fresh baseline card
# as step one; do NOT reuse the pre-C6 champion numbers", and per
# FABLE-C6-SHIP-PLAN.md the elo MOV-multiplier fix (ab2a548) means any
# team_mdl_df cached before that commit is stale -- rebuild from scratch).
# ================================================================
if (stage %in% c("data", "all")) {
  cli::cli_h1("G4: building team_mdl_df fresh (no stale cache reuse)")
  t0 <- Sys.time()
  team_mdl_df <- build_team_mdl_df()
  cli::cli_inform("build_team_mdl_df() completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  cat(sprintf("\nSanity check: team_mdl_df has %d rows, seasons: %s\n",
              nrow(team_mdl_df), paste(sort(unique(team_mdl_df$season.x)), collapse = ", ")))
  cat("elo_diff present:", "elo_diff" %in% names(team_mdl_df), "\n")
  print(summary(team_mdl_df$elo_diff))
  n_completed <- sum(!is.na(team_mdl_df$win)) / 2
  cat(sprintf("Completed matches: %d\n", n_completed))
  max_round <- team_mdl_df |> dplyr::filter(!is.na(win)) |>
    dplyr::summarise(max_season = max(season.x), .groups = "drop")
  print(team_mdl_df |> dplyr::filter(!is.na(win), season.x == max(season.x)) |>
          dplyr::summarise(max_round = max(round_number.x)))

  saveRDS(team_mdl_df, .rds("ws0_team_mdl_df.rds"))
  cli::cli_alert_success("Saved ws0_team_mdl_df.rds")
}

# ================================================================
# Stage: baseline_screen -- G4 baseline, production trainers, TEST_SEASONS=2026
# ================================================================
if (stage %in% c("baseline_screen", "all")) {
  cli::cli_h1("G4 baseline: PRODUCTION .train_match_gams + .train_xgb_fixed(elo_diff), 2026 screen")
  team_mdl_df <- readRDS(.rds("ws0_team_mdl_df.rds"))

  t0 <- Sys.time()
  roll_screen <- run_rolling_eval(
    team_mdl_df, TEST_SEASONS,
    gam_trainer = .train_match_gams,   # production, unqualified via devtools::load_all(torp)
    xgb_trainer = .train_xgb_fixed,    # rolling_lib.R's own
    extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("Baseline 2026 screen completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  base_screen_norecal <- roll_screen$input_blend_preds
  m_base_screen_norecal <- .compute_metrics(base_screen_norecal)
  base_screen <- v1a_recal_own(base_screen_norecal)
  m_base_screen <- .compute_metrics(base_screen)

  .print_metrics(m_base_screen_norecal, "G4 baseline, no recal, 2026 screen")
  .print_metrics(m_base_screen, "G4 baseline + V1a recal, 2026 screen")
  .print_decomposition(m_base_screen, "G4 baseline + V1a recal, 2026 screen")

  saveRDS(list(roll = roll_screen, preds_norecal = base_screen_norecal, metrics_norecal = m_base_screen_norecal,
               preds = base_screen, metrics = m_base_screen),
          .rds("ws0_baseline_screen.rds"))
  cli::cli_alert_success("Saved ws0_baseline_screen.rds")
}

# ================================================================
# Stage: baseline_pool -- G4 baseline, pooled 2025:2026 + Section 6.5
# falsifier check (pooled MAE ~= 25.59, slope ~= 1.00, within ~0.15 MAE noise)
# ================================================================
if (stage %in% c("baseline_pool", "all")) {
  cli::cli_h1("G4 baseline: PRODUCTION trainers, pooled 2025:2026")
  team_mdl_df <- readRDS(.rds("ws0_team_mdl_df.rds"))

  t0 <- Sys.time()
  roll_pool <- run_rolling_eval(
    team_mdl_df, CONFIRM_SEASONS,
    gam_trainer = .train_match_gams,
    xgb_trainer = .train_xgb_fixed,
    extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("Baseline pooled completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  base_pool_norecal <- roll_pool$input_blend_preds
  m_base_pool_norecal <- .compute_metrics(base_pool_norecal)
  base_pool <- v1a_recal_own(base_pool_norecal)
  m_base_pool <- .compute_metrics(base_pool)

  .print_metrics(m_base_pool_norecal, "G4 baseline, no recal, pooled")
  .print_metrics(m_base_pool, "G4 baseline + V1a recal, pooled")
  .print_decomposition(m_base_pool, "G4 baseline + V1a recal, pooled")

  cat(sprintf(
    "\n=== PLAN SECTION 6.5 FALSIFIER CHECK (baseline-drift) ===\nPooled MAE=%.3f (target ~25.59, tol ~0.15) | Slope=%.3f (target ~1.00)\n",
    m_base_pool$mae, m_base_pool$slope
  ))
  falsifier_mae_ok <- abs(m_base_pool$mae - 25.59) < 0.15
  cat(sprintf("Falsifier verdict: %s\n",
              ifelse(falsifier_mae_ok, "PASS -- within run-to-run noise, safe to proceed with variants",
                     "FIRES -- baseline has drifted beyond noise; STOP and diagnose before running any variant")))

  saveRDS(list(roll = roll_pool, preds_norecal = base_pool_norecal, metrics_norecal = m_base_pool_norecal,
               preds = base_pool, metrics = m_base_pool, falsifier_mae_ok = falsifier_mae_ok),
          .rds("ws0_baseline_pool.rds"))
  cli::cli_alert_success("Saved ws0_baseline_pool.rds")
}

# ================================================================
# Stage: screen_v0{a..e} -- 2026 screen for one variant, boot vs baseline screen
# ================================================================
run_variant_screen <- function(key) {
  v <- VARIANTS[[key]]
  cli::cli_h1("WS0 screen: {v$label} -- 2026")
  team_mdl_df <- readRDS(.rds("ws0_team_mdl_df.rds"))
  base <- readRDS(.rds("ws0_baseline_screen.rds"))

  t0 <- Sys.time()
  roll <- run_rolling_eval(
    team_mdl_df, TEST_SEASONS,
    gam_trainer = v$gam_trainer,
    xgb_trainer = v$xgb_trainer,
    extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("{v$label} 2026 screen completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  preds_norecal <- roll$input_blend_preds
  m_norecal <- .compute_metrics(preds_norecal)
  preds <- v1a_recal_own(preds_norecal)
  m <- .compute_metrics(preds)

  .print_metrics(base$metrics_norecal, "G4 baseline, no recal, 2026 screen")
  .print_metrics(base$metrics, "G4 baseline + V1a recal, 2026 screen")
  .print_metrics(m_norecal, sprintf("%s, no recal, 2026 screen", v$label))
  .print_metrics(m, sprintf("%s + V1a recal, 2026 screen", v$label))

  boot_vs_base <- boot_mae_diff(preds, base$preds, B = 2000)
  cat(sprintf("\nboot_mae_diff(%s+recal - G4 baseline+recal, 2026 screen): N=%d deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
              v$label, boot_vs_base$n_matches, boot_vs_base$mae_diff, boot_vs_base$mae_ci[1], boot_vs_base$mae_ci[2], boot_vs_base$brier_diff))

  adoptable <- m$mae <= (base$metrics$mae + 0.0005)  # G2: "better or equal point estimate"
  cat(sprintf("G2 screen verdict: %s point-estimate %s baseline (%.3f vs %.3f) -> %s\n",
              v$label, ifelse(adoptable, "<=", ">"), m$mae, base$metrics$mae,
              ifelse(adoptable, "ADOPTABLE -- run pooled confirm", "NOT adoptable on point estimate -- screen-level finding stands, no pooled confirm needed")))

  out <- list(label = v$label, roll = roll, preds_norecal = preds_norecal, metrics_norecal = m_norecal,
              preds = preds, metrics = m, boot_vs_baseline_screen = boot_vs_base, adoptable = adoptable)
  saveRDS(out, .rds(paste0("ws0_", key, "_screen.rds")))
  cli::cli_alert_success("Saved ws0_{key}_screen.rds")
  invisible(out)
}

if (stage %in% c("screen_v0a", "all")) run_variant_screen("v0a")
if (stage %in% c("screen_v0b", "all")) run_variant_screen("v0b")
if (stage %in% c("screen_v0c", "all")) run_variant_screen("v0c")
if (stage %in% c("screen_v0d", "all")) run_variant_screen("v0d")
if (stage %in% c("screen_v0e", "all")) run_variant_screen("v0e")

# ================================================================
# Stage: confirm_v0{a..e} -- pooled 2025:2026 confirm, only meaningful if
# the screen stage marked the variant adoptable (G2); runs unconditionally
# if invoked directly, but prints a warning if the cached screen said "no".
# ================================================================
run_variant_confirm <- function(key) {
  v <- VARIANTS[[key]]
  screen <- readRDS(.rds(paste0("ws0_", key, "_screen.rds")))
  if (!isTRUE(screen$adoptable)) {
    cli::cli_warn("{v$label}: screen stage marked NOT adoptable (point estimate worse than baseline) -- running pooled confirm anyway since explicitly requested, but this is off the G2-prescribed path.")
  }
  cli::cli_h1("WS0 confirm: {v$label} -- pooled 2025:2026")
  team_mdl_df <- readRDS(.rds("ws0_team_mdl_df.rds"))
  base <- readRDS(.rds("ws0_baseline_pool.rds"))

  t0 <- Sys.time()
  roll <- run_rolling_eval(
    team_mdl_df, CONFIRM_SEASONS,
    gam_trainer = v$gam_trainer,
    xgb_trainer = v$xgb_trainer,
    extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("{v$label} pooled confirm completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  preds_norecal <- roll$input_blend_preds
  m_norecal <- .compute_metrics(preds_norecal)
  preds <- v1a_recal_own(preds_norecal)
  m <- .compute_metrics(preds)

  .print_metrics(base$metrics, "G4 baseline + V1a recal, pooled")
  .print_metrics(m, sprintf("%s + V1a recal, pooled", v$label))

  boot_vs_base <- boot_mae_diff(preds, base$preds, B = 2000)
  ci_excl_0 <- (boot_vs_base$mae_ci[1] > 0 && boot_vs_base$mae_ci[2] > 0) ||
    (boot_vs_base$mae_ci[1] < 0 && boot_vs_base$mae_ci[2] < 0)
  brier_ok <- boot_vs_base$brier_diff <= 0.002
  non_worse <- m$mae <= base$metrics$mae + 1e-9
  # G3: simplifications (WS0 feature drops) adopt on CI-overlaps-0 + non-worse
  # point estimate; a strict win (CI excludes 0, favours variant) is the
  # stronger bar -- report both.
  ship_pass_strict <- ci_excl_0 && boot_vs_base$mae_diff < 0 && brier_ok
  ship_pass_simplification <- non_worse && brier_ok

  cat(sprintf(
    "\nboot_mae_diff(%s+recal - G4 baseline+recal, pooled): N=%d deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
    v$label, boot_vs_base$n_matches, boot_vs_base$mae_diff, boot_vs_base$mae_ci[1], boot_vs_base$mae_ci[2], boot_vs_base$brier_diff
  ))
  cat(sprintf("G3 STRICT ship gate (CI excludes 0, favours variant, Brier not worse by >0.002): %s\n", ship_pass_strict))
  cat(sprintf("G3 SIMPLIFICATION gate (non-worse point estimate, Brier not worse by >0.002): %s\n", ship_pass_simplification))

  out <- list(label = v$label, roll = roll, preds_norecal = preds_norecal, metrics_norecal = m_norecal,
              preds = preds, metrics = m, boot_vs_baseline_pool = boot_vs_base,
              ship_pass_strict = ship_pass_strict, ship_pass_simplification = ship_pass_simplification)
  saveRDS(out, .rds(paste0("ws0_", key, "_confirm.rds")))
  cli::cli_alert_success("Saved ws0_{key}_confirm.rds")
  invisible(out)
}

if (stage == "confirm_v0a") run_variant_confirm("v0a")
if (stage == "confirm_v0b") run_variant_confirm("v0b")
if (stage == "confirm_v0c") run_variant_confirm("v0c")
if (stage == "confirm_v0d") run_variant_confirm("v0d")
if (stage == "confirm_v0e") run_variant_confirm("v0e")

# ================================================================
# Stage: summary -- full metric table + WS0 interpretation branches (plan
# Section 2 WS0 "Success / interpretation")
# ================================================================
if (stage %in% c("summary", "all")) {
  cli::cli_h1("WS0 Final Summary: parts-vs-whole feature ablations")

  load_if <- function(f) if (file.exists(.rds(f))) readRDS(.rds(f)) else NULL
  base_screen <- load_if("ws0_baseline_screen.rds")
  base_pool   <- load_if("ws0_baseline_pool.rds")

  cat("\n=== G4 BASELINE ===\n")
  if (!is.null(base_screen)) .print_metrics(base_screen$metrics, "G4 baseline + V1a recal, 2026 screen")
  if (!is.null(base_pool)) {
    .print_metrics(base_pool$metrics, "G4 baseline + V1a recal, pooled 2025:2026")
    cat(sprintf("Section 6.5 falsifier: %s\n", ifelse(isTRUE(base_pool$falsifier_mae_ok), "PASS", "FIRED")))
  }

  cat("\n=== WS0 VARIANTS: 2026 SCREEN vs G4 baseline ===\n")
  keys <- c("v0a", "v0b", "v0c", "v0d", "v0e")
  screens <- setNames(lapply(keys, function(k) load_if(paste0("ws0_", k, "_screen.rds"))), keys)
  for (k in keys) {
    s <- screens[[k]]
    if (is.null(s)) { cat(sprintf("%s: screen not run\n", k)); next }
    .print_metrics(s$metrics, sprintf("%s + V1a recal, 2026 screen", s$label))
    cat(sprintf("  vs baseline: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f adoptable=%s\n",
                s$boot_vs_baseline_screen$mae_diff, s$boot_vs_baseline_screen$mae_ci[1], s$boot_vs_baseline_screen$mae_ci[2],
                s$boot_vs_baseline_screen$brier_diff, s$adoptable))
  }

  cat("\n=== WS0 VARIANTS: POOLED 2025:2026 CONFIRM (where run) ===\n")
  confirms <- setNames(lapply(keys, function(k) load_if(paste0("ws0_", k, "_confirm.rds"))), keys)
  for (k in keys) {
    cf <- confirms[[k]]
    if (is.null(cf)) { cat(sprintf("%s: no pooled confirm run\n", k)); next }
    .print_metrics(cf$metrics, sprintf("%s + V1a recal, pooled", cf$label))
    cat(sprintf("  vs baseline: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f strict_pass=%s simplification_pass=%s\n",
                cf$boot_vs_baseline_pool$mae_diff, cf$boot_vs_baseline_pool$mae_ci[1], cf$boot_vs_baseline_pool$mae_ci[2],
                cf$boot_vs_baseline_pool$brier_diff, cf$ship_pass_strict, cf$ship_pass_simplification))
  }

  cat("\n=== WS0 interpretation branches (plan Section 2) ===\n")
  if (!is.null(screens$v0a)) {
    a_within_noise <- screens$v0a$boot_vs_baseline_screen$mae_ci[1] < 0 && screens$v0a$boot_vs_baseline_screen$mae_ci[2] > 0
    cat(sprintf("V0a within noise of baseline (expected null): %s\n", a_within_noise))
  }
  if (!is.null(screens$v0c)) {
    c_worse <- screens$v0c$boot_vs_baseline_screen$mae_diff > 0 &&
      screens$v0c$boot_vs_baseline_screen$mae_ci[1] > 0
    cat(sprintf("V0c materially worse (components carry real signal, strengthens WS2 premise): %s\n", c_worse))
  }
  if (!is.null(screens$v0d) && !is.null(screens$v0e)) {
    d_near_base <- abs(screens$v0d$boot_vs_baseline_screen$mae_diff) < 0.3
    e_worse <- screens$v0e$metrics$mae > screens$v0d$metrics$mae
    cat(sprintf("V0d ~= baseline (%s) while V0e worse (%s) -- PSR adds little marginal signal, raises WS1 stakes: %s\n",
                d_near_base, e_worse, d_near_base && e_worse))
  }

  saveRDS(list(base_screen = base_screen, base_pool = base_pool, screens = screens, confirms = confirms),
          .rds("ws0_final_summary.rds"))
  cli::cli_alert_success("Saved ws0_final_summary.rds")
}
