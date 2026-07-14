# ws7_elo_xgb_fix.R — Round 2: fix XGB nrounds pre-optimisation to include
# elo_diff BEFORE tuning, then re-verify the round-1 "elo-as-feature helps
# GAM, hurts XGB" finding under a fair XGB tuning regime.
# =====================================================================
# Round 1 (ws2_team_elo.R / ws5_grid.R candidate 6) found: elo_diff as a
# GAM/XGB feature helped the GAM side (MAE 25.73 vs champion GAM 26.04) but
# HURT the XGB side (MAE 27.41 vs champion XGB 26.92). Diagnosed as an
# artifact: run_rolling_eval()'s nrounds pre-optimisation step called
# torp:::.train_match_xgb() on cv_train_df BEFORE elo_diff was ever added to
# the feature set, so .train_xgb_fixed()'s actual per-round training (which
# DOES receive elo_diff via extra_feature_cols) used a stopping point tuned
# for a model that never saw elo_diff. rolling_lib.R now has
# .train_match_xgb_ext() + run_rolling_eval(cv_extra_feature_cols=...) to
# fix this (additive change, byte-compatible default path preserved).
#
# This script reruns candidate 6 ("Everything": V4b GAM formula + elo_diff
# feature + V1a post-hoc recal) with cv_extra_feature_cols = "elo_diff", and
# reports whether the XGB-side gap closes and whether the Input Blend
# clears the round-1 C6 champion (pooled MAE=25.545, Brier=0.1741) by the
# G3 ship gate (bootstrap CI on deltaMAE excludes zero, Brier doesn't
# worsen by >0.002, bits/game doesn't worsen).
#
# Run stage-by-stage (checkpoints to experiments/results/*.rds):
#   Rscript ws7_elo_xgb_fix.R screen   # 2026 screen: fixed-CV C6 vs round-1 C6
#   Rscript ws7_elo_xgb_fix.R pool     # 2025:2026 pooled confirm + V1a recal + boot vs champion
#   Rscript ws7_elo_xgb_fix.R summary  # aggregate + ship-gate verdict

stage <- {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) >= 1) a[1] else "all"
}
cat("=== ws7_elo_xgb_fix.R stage:", stage, "===\n")

# Setup ----
library(tidyverse)
library(xgboost)
library(mgcv)
library(MLmetrics)
library(geosphere)
library(cli)
library(data.table)

torp_paths <- c("../torp", "../../torp", "../../../torp", "C:/dev/torpverse/torp")
torp_loaded <- FALSE
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p)
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
source(file.path(EXPERIMENTS_DIR, "elo_lib.R"))

TEST_SEASONS    <- 2026
CONFIRM_SEASONS <- 2025:2026

# ---- Shared: team_mdl_df (reuse round-1 cache) ----
team_mdl_df <- readRDS(.rds("team_mdl_df_cache.rds"))
cli::cli_inform("team_mdl_df: {nrow(team_mdl_df)} rows, seasons {paste(sort(unique(team_mdl_df$season.x)), collapse=', ')}")

# ---- Shared: Elo table (reuse WS2's tuned k=20/hga=45/carryover=0.75 table) ----
elo_stuff <- readRDS(.rds("ws2_elo_table.rds"))
elo_table <- elo_stuff$elo_table
best_elo  <- elo_stuff$best
cli::cli_inform("Elo params (WS2 tuned): k={best_elo$k}, hga={best_elo$hga}, carryover={best_elo$carryover}")

team_mdl_df_elo <- join_elo_diff_to_team_mdl_df(team_mdl_df, elo_table)
n_na_elo <- sum(is.na(team_mdl_df_elo$elo_diff))
n_incomplete_elo <- sum(is.na(team_mdl_df_elo$win))
if (n_na_elo > 0) {
  if (n_na_elo > n_incomplete_elo) {
    cli::cli_abort("NA elo_diff on {n_na_elo - n_incomplete_elo} completed match row(s) beyond expected future-fixture NAs")
  }
  team_mdl_df_elo$elo_diff[is.na(team_mdl_df_elo$elo_diff)] <- 0
}

# ---- Shared: recal_expanding + .apply_recal + v1a_recal_own (copied
# verbatim from ws1_margin_recal.R / ws5_grid.R, plan G5 -- each WS keeps
# its own copy) ----
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
    "%-42s MAE=%.3f RMSE=%.3f Brier=%.4f Bits=%.4f Slope=%.3f Cor=%.3f SDRatio=%.3f CloseMAE(n=%d)=%.3f\n",
    label, m$mae, m$rmse, m$brier, m$bits, m$slope, m$cor, m$sd_ratio, m$close_n, m$close_mae
  ))
}

# ---- .train_match_gams_v4b_elo: exact copy of ws5_grid.R's candidate-6
# GAM trainer (V4b structural formula + elo_diff optional smooth on models
# 2/4), reproduced here per plan G5 (each WS keeps its own copy; no
# production torp/R/*.R edits, and no dependency on sourcing ws5_grid.R's
# stage-gated top-level code). ----
.train_match_gams_v4b_elo <- function(team_mdl_df, train_filter = NULL, nthreads = 4L, gamma_arg = 1.4) {
  loadNamespace("mgcv")

  if (is.null(train_filter)) {
    train_mask <- !is.na(team_mdl_df$win)
  } else {
    train_mask <- train_filter & !is.na(team_mdl_df$win)
  }

  gam_df <- team_mdl_df[train_mask, ]
  cli::cli_inform("[ws7] Training on {nrow(gam_df)} completed matches")
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
  .add_optional <- function(base_terms, optional_terms) {
    keep <- setdiff(optional_terms, drop_terms)
    if (length(keep) > 0) paste(base_terms, "+", paste(keep, collapse = " + ")) else base_terms
  }

  # Model 1: unchanged (elo not added here, mirrors WS2(b)'s choice) ----
  cli::cli_progress_step("[ws7] Training total xPoints model")
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
  afl_total_xpoints_mdl <- mgcv::bam(
    m1_formula, data = gam_df, weights = gam_df$weightz, family = gaussian(),
    nthreads = nthreads, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_tot_xscore <- predict(afl_total_xpoints_mdl, newdata = team_mdl_df, type = "response")

  # Model 2: V4b formula (drop ti(*, gam_pred_tot_xscore)) + elo optional smooth ----
  cli::cli_progress_step("[ws7] Training xScore diff model")
  gam_df$gam_pred_tot_xscore <- team_mdl_df$gam_pred_tot_xscore[train_mask]
  m2_terms <- c(
    "xscore_diff ~ s(team_type_fac, bs = \"re\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    "+ s(gam_pred_tot_xscore, bs = \"ts\", k = 5)",
    "+ s(epr_diff, bs = \"ts\", k = 5)",
    "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
    "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
    "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
    "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
    "+ s(torp_diff, bs = \"ts\", k = 5)",
    "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m2_optional <- c("s(psr_diff, bs = \"ts\", k = 5)", "s(osr_diff, bs = \"ts\", k = 5)",
                    "s(dsr_diff, bs = \"ts\", k = 5)", "s(elo_diff, bs = \"ts\", k = 5)")
  m2_formula <- stats::as.formula(.add_optional(paste(m2_terms, collapse = " "), m2_optional))
  afl_xscore_diff_mdl <- mgcv::bam(
    m2_formula, data = gam_df, weights = gam_df$weightz, family = gaussian(),
    nthreads = nthreads, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_xscore_diff <- predict(afl_xscore_diff_mdl, newdata = team_mdl_df, type = "response")

  # Model 3: V4a/V4b formula (drop ti(*, gam_pred_tot_xscore)), no elo (WS2(b) only added elo to m2/m4) ----
  cli::cli_progress_step("[ws7] Training conversion model")
  gam_df$gam_pred_xscore_diff <- team_mdl_df$gam_pred_xscore_diff[train_mask]
  m3_terms <- c(
    "shot_conv_diff ~ s(team_type_fac, bs = \"re\")",
    "+ s(game_year_decimal.x, bs = \"ts\")",
    "+ s(game_prop_through_year.x, bs = \"cc\")",
    "+ s(game_prop_through_month.x, bs = \"cc\")",
    "+ s(game_wday_fac.x, bs = \"re\")",
    "+ s(game_prop_through_day.x, bs = \"cc\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    "+ s(epr_diff, bs = \"ts\", k = 5)",
    "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
    "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
    "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
    "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
    "+ s(torp_diff, bs = \"ts\", k = 5)",
    "+ s(gam_pred_tot_xscore, bs = \"ts\", k = 5)",
    "+ s(gam_pred_xscore_diff, bs = \"ts\", k = 5)",
    "+ s(venue_fac, bs = \"re\")",
    "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m3_optional <- c("s(psr_diff, bs = \"ts\", k = 5)", "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  m3_formula <- stats::as.formula(.add_optional(paste(m3_terms, collapse = " "), m3_optional))
  afl_conv_mdl <- mgcv::bam(
    m3_formula, data = gam_df, weights = gam_df$shot_weightz, family = gaussian(),
    nthreads = nthreads, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_conv_diff <- predict(afl_conv_mdl, newdata = team_mdl_df, type = "response")

  # Model 4: V4b formula (drop both second-order tensors, keep s(gam_pred_xscore_diff)) + elo optional smooth ----
  cli::cli_progress_step("[ws7] Training score diff model")
  gam_df$gam_pred_conv_diff <- team_mdl_df$gam_pred_conv_diff[train_mask]
  m4_terms <- c(
    "score_diff ~ s(team_type_fac, bs = \"re\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    "+ s(gam_pred_xscore_diff)",
    "+ s(epr_diff, bs = \"ts\", k = 5)",
    "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
    "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
    "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
    "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
    "+ s(torp_diff, bs = \"ts\", k = 5)",
    "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m4_optional <- c("s(psr_diff, bs = \"ts\", k = 5)", "s(osr_diff, bs = \"ts\", k = 5)",
                    "s(dsr_diff, bs = \"ts\", k = 5)", "s(elo_diff, bs = \"ts\", k = 5)")
  m4_formula <- stats::as.formula(.add_optional(paste(m4_terms, collapse = " "), m4_optional))
  afl_score_mdl <- mgcv::bam(
    m4_formula, data = gam_df, weights = gam_df$weightz, family = "gaussian",
    nthreads = nthreads, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_score_diff <- predict(afl_score_mdl, newdata = team_mdl_df, type = "response")

  # Model 5: win probability, unchanged ----
  cli::cli_progress_step("[ws7] Training win probability model")
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
# Stage: screen -- 2026 screen, fixed-CV C6 (elo included in nrounds tuning)
# vs round-1's cached C6 (buggy CV). Direct check of the diagnosed gap:
# tuned nrounds, and XGB-only MAE, fixed vs round-1.
# ================================================================
if (stage %in% c("screen", "all")) {
  cli::cli_h1("WS7 screen: C6 with FIXED nrounds CV (elo_diff included), TEST_SEASONS=2026")

  t0 <- Sys.time()
  roll_fixed_2026 <- run_rolling_eval(
    team_mdl_df_elo, TEST_SEASONS,
    gam_trainer = .train_match_gams_v4b_elo,
    xgb_trainer = .train_xgb_fixed,
    extra_feature_cols = "elo_diff",
    cv_extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("WS7 fixed-CV C6 (2026) completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  saveRDS(roll_fixed_2026, .rds("ws7_fixed_roll_2026.rds"))

  cat("\n=== Tuned XGBoost nrounds: FIXED (elo included in CV) ===\n")
  print(roll_fixed_2026$xgb_nrounds)

  if (file.exists(.rds("ws5_c6_roll_2026.rds"))) {
    roll_c6_r1 <- readRDS(.rds("ws5_c6_roll_2026.rds"))
    cat("\n=== Tuned XGBoost nrounds: ROUND 1 (buggy CV, elo-free) ===\n")
    print(roll_c6_r1$xgb_nrounds)
    cat(sprintf("\nnrounds identical fixed vs round-1: %s\n",
                paste(roll_fixed_2026$xgb_nrounds == roll_c6_r1$xgb_nrounds[names(roll_fixed_2026$xgb_nrounds)], collapse = ", ")))

    m_xgb_r1    <- .compute_metrics(roll_c6_r1$xgb_preds)
    m_xgb_fixed <- .compute_metrics(roll_fixed_2026$xgb_preds)
    cat("\n=== XGB-only MAE: fixed-CV vs round-1 (both with elo_diff feature) ===\n")
    .print_metrics(m_xgb_r1, "XGB-only, round-1 (buggy CV)")
    .print_metrics(m_xgb_fixed, "XGB-only, fixed CV")
    cat("Reference: round-1 elo-free champion XGB-only MAE = 26.92 (memory), round-1 elo-feature XGB-only MAE = 27.41\n")
  } else {
    cli::cli_warn("ws5_c6_roll_2026.rds not found -- cannot compare nrounds/XGB-only MAE vs round-1 directly")
  }

  m_ib_norecal <- .compute_metrics(roll_fixed_2026$input_blend_preds)
  ib_recal <- v1a_recal_own(roll_fixed_2026$input_blend_preds)
  m_ib_recal <- .compute_metrics(ib_recal)

  cat("\n=== Input Blend, 2026 screen ===\n")
  .print_metrics(m_ib_norecal, "Fixed-CV C6, no recal (Input Blend)")
  .print_metrics(m_ib_recal, "Fixed-CV C6 + V1a recal (Input Blend)")

  if (file.exists(.rds("ws5_c6_2026.rds"))) {
    r1_2026 <- readRDS(.rds("ws5_c6_2026.rds"))
    m_r1_norecal <- .compute_metrics(r1_2026$preds_norecal)
    m_r1_recal   <- .compute_metrics(r1_2026$preds)
    cat("\n=== Round-1 C6 (buggy CV), 2026 screen, for reference ===\n")
    .print_metrics(m_r1_norecal, "Round-1 C6, no recal (Input Blend)")
    .print_metrics(m_r1_recal, "Round-1 C6 + V1a recal (Input Blend)")
  }

  saveRDS(list(roll = roll_fixed_2026, m_norecal = m_ib_norecal,
               preds_recal = ib_recal, m_recal = m_ib_recal),
          .rds("ws7_screen_2026.rds"))
  cli::cli_alert_success("Saved ws7_screen_2026.rds")
}

# ================================================================
# Stage: pool -- 2025:2026 pooled confirmation, fixed-CV C6 + V1a recal,
# bootstrap vs round-1's C6 champion (pooled MAE=25.545, cached).
# ================================================================
if (stage %in% c("pool", "all")) {
  cli::cli_h1("WS7 pool: fixed-CV C6 pooled 2025:2026 confirmation")

  t0 <- Sys.time()
  roll_fixed_pool <- run_rolling_eval(
    team_mdl_df_elo, CONFIRM_SEASONS,
    gam_trainer = .train_match_gams_v4b_elo,
    xgb_trainer = .train_xgb_fixed,
    extra_feature_cols = "elo_diff",
    cv_extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("WS7 fixed-CV C6 (pooled) completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  saveRDS(roll_fixed_pool, .rds("ws7_fixed_roll_pool.rds"))

  cat("\n=== Tuned XGBoost nrounds: FIXED, pooled window (season < 2025) ===\n")
  print(roll_fixed_pool$xgb_nrounds)

  pool_ib_norecal <- roll_fixed_pool$input_blend_preds
  m_pool_norecal <- .compute_metrics(pool_ib_norecal)
  pool_ib_recal <- v1a_recal_own(pool_ib_norecal)
  m_pool_recal <- .compute_metrics(pool_ib_recal)

  cat("\n=== Fixed-CV C6, pooled 2025:2026 ===\n")
  .print_metrics(m_pool_norecal, "Fixed-CV C6, no recal, pooled")
  .print_metrics(m_pool_recal, "Fixed-CV C6 + V1a recal, pooled")

  # Champion baseline: round-1's cached C6 (this round's champion to beat,
  # pooled MAE=25.545 per task brief). Recompute .compute_metrics() on its
  # cached $preds fresh (the cached $metrics predate the bits addition).
  champion_ok <- file.exists(.rds("ws5_c6_pool_confirm.rds"))
  if (champion_ok) {
    champ <- readRDS(.rds("ws5_c6_pool_confirm.rds"))
    m_champion <- .compute_metrics(champ$preds)
    cat("\n=== Champion: round-1 C6 (buggy CV) + V1a recal, pooled (recomputed w/ bits) ===\n")
    .print_metrics(m_champion, "Champion (round-1 C6 + recal), pooled")

    boot_vs_champion <- boot_mae_diff(pool_ib_recal, champ$preds)
    cat(sprintf(
      "\nboot_mae_diff(Fixed-CV C6+recal - Champion): N=%d deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f 95%%CI[%+.5f,%+.5f]\n",
      boot_vs_champion$n_matches, boot_vs_champion$mae_diff, boot_vs_champion$mae_ci[1], boot_vs_champion$mae_ci[2],
      boot_vs_champion$brier_diff, boot_vs_champion$brier_ci[1], boot_vs_champion$brier_ci[2]
    ))

    ci_excludes_zero <- (boot_vs_champion$mae_ci[1] > 0) || (boot_vs_champion$mae_ci[2] < 0)
    brier_ok <- (m_pool_recal$brier - m_champion$brier) <= 0.002
    bits_ok  <- m_pool_recal$bits >= m_champion$bits
    ship <- ci_excludes_zero && boot_vs_champion$mae_diff < 0 && brier_ok && bits_ok

    cat(sprintf("\nShip gate: CI excludes zero = %s (mae_diff sign %s) | Brier delta <= 0.002 = %s (delta=%.5f) | bits not worse = %s (delta=%.4f)\n",
                ci_excludes_zero, ifelse(boot_vs_champion$mae_diff < 0, "favours fixed-CV", "favours champion"),
                brier_ok, m_pool_recal$brier - m_champion$brier,
                bits_ok, m_pool_recal$bits - m_champion$bits))
    cat(sprintf("SHIP RECOMMENDATION: %s\n", ifelse(ship, "YES -- ships", "NO -- does not clear the gate")))

    saveRDS(list(m_champion = m_champion, boot = boot_vs_champion, ship = ship,
                 ci_excludes_zero = ci_excludes_zero, brier_ok = brier_ok, bits_ok = bits_ok),
            .rds("ws7_pool_ship_gate.rds"))
  } else {
    cli::cli_warn("ws5_c6_pool_confirm.rds not found -- cannot run the ship-gate boot vs champion")
  }

  saveRDS(list(roll = roll_fixed_pool, preds_norecal = pool_ib_norecal, m_norecal = m_pool_norecal,
               preds_recal = pool_ib_recal, m_recal = m_pool_recal),
          .rds("ws7_pool_2025_2026.rds"))
  cli::cli_alert_success("Saved ws7_pool_2025_2026.rds")
}

# ================================================================
# Stage: summary
# ================================================================
if (stage %in% c("summary", "all")) {
  cli::cli_h1("WS7 Final Summary: does the CV-tuning fix change the elo-feature verdict?")

  load_if <- function(f) if (file.exists(.rds(f))) readRDS(.rds(f)) else NULL
  screen  <- load_if("ws7_screen_2026.rds")
  pool    <- load_if("ws7_pool_2025_2026.rds")
  gate    <- load_if("ws7_pool_ship_gate.rds")

  if (!is.null(screen)) {
    cat("\n--- 2026 screen ---\n")
    .print_metrics(screen$m_norecal, "Fixed-CV C6, no recal")
    .print_metrics(screen$m_recal, "Fixed-CV C6 + V1a recal")
  }
  if (!is.null(pool)) {
    cat("\n--- 2025:2026 pooled ---\n")
    .print_metrics(pool$m_norecal, "Fixed-CV C6, no recal, pooled")
    .print_metrics(pool$m_recal, "Fixed-CV C6 + V1a recal, pooled")
  }
  if (!is.null(gate)) {
    cat("\n--- Ship gate vs round-1 C6 champion (pooled MAE=25.545) ---\n")
    .print_metrics(gate$m_champion, "Champion (round-1 C6 + recal), pooled")
    cat(sprintf("deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
                gate$boot$mae_diff, gate$boot$mae_ci[1], gate$boot$mae_ci[2], gate$boot$brier_diff))
    cat(sprintf("SHIP: %s\n", ifelse(gate$ship, "YES", "NO")))
  }

  cat("\nReference -- Squiggle 2026 leaderboard (after round 18):\n")
  cat("  Aggregate: MAE=25.45 bits~=0.325 (minimum bar) | Punters: MAE=25.32 | Wheelo: MAE=24.89\n")
}
