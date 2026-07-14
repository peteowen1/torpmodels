# ws6_decay_on_c6.R — WS6: Recency/decay lever retested ON TOP OF the C6 champion
# =====================================================================
# Round 1 (ws2_team_elo.R stage "decay") tested MATCH_WEIGHT_DECAY_DAYS in
# {300, 500} atop the plain ORIGINAL champion and found decay=300 moved
# close-bucket MAE from 19.13 -> 17.12 but with slope moving the WRONG way
# (0.836 -> 0.783) and sd_ratio INCREASING (0.701 -> 0.726) -- i.e. more
# raw dispersion, not better correlation, so it was not trusted as a real
# fix and candidate 11 (decay atop C6) was explicitly DEFERRED in
# ws5_grid.R rather than run (see that file's header note + C11 line).
#
# This script finally runs candidate 11: decay in {300, 400, 600, 800}
# applied ON TOP OF the C6 "everything" trainer (V4b formula + elo_diff
# feature + V1a post-hoc recal) instead of the plain champion, to test
# whether V1a recal (which already corrects margin over-dispersion at the
# final stage) changes the decay/dispersion interaction.
#
# Independent copy of the C6 trainer + recal helpers (plan G5: each WS
# keeps its own copy rather than sourcing ws5_grid.R's expensive top-level
# code) -- copied verbatim from ws5_grid.R so results are bit-for-bit
# comparable to the cached C6 baseline (ws5_c6_2026.rds / ws5_c6_pool_confirm.rds).
#
# Stages (checkpoint to experiments/results/*.rds):
#   Rscript ws6_decay_on_c6.R screen   # decay in {300,400,600,800} atop C6, 2026 screen
#   Rscript ws6_decay_on_c6.R confirm  # pooled 2025:2026 confirm of the best screen decay + bootstrap vs C6
#   Rscript ws6_decay_on_c6.R summary  # full metric table + slope=cor*sd_actual/sd_pred decomposition + ship gate

stage <- {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) >= 1) a[1] else "all"
}
cat("=== ws6_decay_on_c6.R stage:", stage, "===\n")

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

source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))   # extended: .compute_metrics now returns $bits
source(file.path(EXPERIMENTS_DIR, "elo_lib.R"))

TEST_SEASONS    <- 2026
CONFIRM_SEASONS <- 2025:2026
DECAY_GRID      <- c(300, 400, 600, 800)

# ---- Shared: team_mdl_df + Elo (identical inputs to ws5_grid.R's C6 run) ----
team_mdl_df <- readRDS(.rds("team_mdl_df_cache.rds"))
cli::cli_inform("team_mdl_df: {nrow(team_mdl_df)} rows, seasons {paste(sort(unique(team_mdl_df$season.x)), collapse=', ')}")

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

# ---- Recency reweighting (verbatim from ws2_team_elo.R stage "decay") ----
anchor_date <- max(as.Date(team_mdl_df$utc_start_time), na.rm = TRUE)
cli::cli_inform("Weight anchor date (proxy = max match date in team_mdl_df): {anchor_date}")

.reweight_team_mdl_df <- function(df, decay_days, anchor_date) {
  df$weightz <- exp(-(as.numeric(anchor_date - as.Date(df$utc_start_time))) / decay_days)
  df$weightz <- df$weightz / mean(df$weightz, na.rm = TRUE)
  df$shot_weightz <- (df$harmean_shots / mean(df$harmean_shots, na.rm = TRUE)) * df$weightz
  df
}

# ---- recal_expanding + .apply_recal + v1a_recal_own + .print_metrics
# (copied verbatim from ws5_grid.R, which itself copied from
# ws1_margin_recal.R -- plan G5: each WS keeps its own copy) ----
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

# ---- Slope decomposition printer: slope = cor * sd_actual/sd_pred.
# Makes explicit whether a slope move is REAL signal (cor up) or just
# dispersion (sd_ratio moves, cor flat/worse) -- required every time by
# the task, not just for the final pick. ----
.print_decomposition <- function(m, label) {
  implied_slope <- m$cor * (m$sd_actual / m$sd_pred)
  cat(sprintf(
    "  [decomp] %-38s slope=%.3f = cor(%.3f) * sd_actual/sd_pred(%.3f) [check: %.3f] sd_ratio(pred/actual)=%.3f\n",
    label, m$slope, m$cor, m$sd_actual / m$sd_pred, implied_slope, m$sd_ratio
  ))
}

# ================================================================
# C6 trainer: V4b structural formula + elo_diff optional smooth on
# models 2 & 4 (verbatim copy of ws5_grid.R's .train_match_gams_v4b_elo /
# .c6_gam_trainer, so decay-reweighted runs are bit-comparable to the
# cached C6 baseline).
# ================================================================
.train_match_gams_v4b_elo <- function(team_mdl_df, train_filter = NULL, nthreads = 4L, gamma_arg = 1.4) {
  loadNamespace("mgcv")

  if (is.null(train_filter)) {
    train_mask <- !is.na(team_mdl_df$win)
  } else {
    train_mask <- train_filter & !is.na(team_mdl_df$win)
  }

  gam_df <- team_mdl_df[train_mask, ]
  cli::cli_inform("[c6] Training on {nrow(gam_df)} completed matches")
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
  cli::cli_progress_step("[c6] Training total xPoints model")
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
  cli::cli_progress_step("[c6] Training xScore diff model")
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
  cli::cli_progress_step("[c6] Training conversion model")
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
  cli::cli_progress_step("[c6] Training score diff model")
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
  cli::cli_progress_step("[c6] Training win probability model")
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

.c6_gam_trainer <- function(team_mdl_df, train_filter = NULL, nthreads = 4L) {
  .train_match_gams_v4b_elo(team_mdl_df, train_filter = train_filter, nthreads = nthreads, gamma_arg = 1.4)
}

# ================================================================
# Stage: check -- reconstruction checkpoint (advisor-recommended, pre-grid).
# Confirms this file's independently-copied .train_match_gams_v4b_elo /
# .c6_gam_trainer reproduces ws5_grid.R's cached C6 numbers EXACTLY when run
# on the un-reweighted team_mdl_df_elo (native weightz, i.e. decay=1000
# equivalent per the weightz-family check) -- if the copy drifted by even
# one smooth term, every decay-vs-baseline delta below would be confounded
# with copy-divergence and invisible (a cached baseline can't catch a bug
# in code it doesn't run). Aborts on mismatch instead of proceeding.
# ================================================================
if (stage %in% c("check", "all")) {
  cli::cli_h1("WS6 check: does this file's C6 trainer copy reproduce ws5_grid.R's cached C6?")

  roll_check <- run_rolling_eval(team_mdl_df_elo, TEST_SEASONS,
                                  gam_trainer = .c6_gam_trainer,
                                  xgb_trainer = .train_xgb_fixed,
                                  extra_feature_cols = "elo_diff")
  check_preds_norecal <- roll_check$input_blend_preds
  m_check_norecal <- .compute_metrics(check_preds_norecal)
  check_preds <- v1a_recal_own(check_preds_norecal)
  m_check <- .compute_metrics(check_preds)

  .print_metrics(m_check_norecal, "Reconstruction, no recal")
  .print_metrics(m_check, "Reconstruction, + V1a recal")
  cat(sprintf(
    "Target (ws5_grid.R cached C6): no-recal MAE=26.292 Brier=0.1750 Slope=0.830 | +recal MAE=25.996 Brier=0.1750 Slope=0.898 CloseMAE=16.412\n"
  ))

  ok_norecal <- abs(m_check_norecal$mae - 26.292) < 0.05 && abs(m_check_norecal$slope - 0.830) < 0.02
  ok_recal   <- abs(m_check$mae - 25.996) < 0.05 && abs(m_check$slope - 0.898) < 0.02 && abs(m_check$close_mae - 16.412) < 0.05
  if (ok_norecal && ok_recal) {
    cli::cli_alert_success("Reconstruction matches cached C6 within tolerance -- trainer copy verified, proceeding is safe")
  } else {
    cli::cli_abort("Reconstruction does NOT match cached C6 -- the copied trainer has drifted from ws5_grid.R's original. Fix before trusting any decay-vs-C6 delta below.")
  }

  saveRDS(list(preds_norecal = check_preds_norecal, metrics_norecal = m_check_norecal,
               preds = check_preds, metrics = m_check), .rds("ws6_check_reconstruction.rds"))
  cli::cli_alert_success("Saved ws6_check_reconstruction.rds")
}

# ================================================================
# Stage: screen -- decay in {300,400,600,800} atop C6, fresh TEST_SEASONS=2026
# ================================================================
if (stage %in% c("screen", "all")) {
  cli::cli_h1("WS6 screen: MATCH_WEIGHT_DECAY_DAYS in {paste(DECAY_GRID, collapse=', ')} atop C6, 2026")

  # Baseline C6 (no decay change): reuse cached round-1 preds, just recompute
  # metrics with the extended (bits-aware) .compute_metrics -- no retrain needed.
  c6_base <- readRDS(.rds("ws5_c6_2026.rds"))
  m_c6_base_norecal <- .compute_metrics(c6_base$preds_norecal)
  m_c6_base <- .compute_metrics(c6_base$preds)
  cat("\n--- Baseline: C6 (current production decay, i.e. MATCH_WEIGHT_DECAY_DAYS=1000), 2026 screen ---\n")
  .print_metrics(m_c6_base_norecal, "C6 baseline, no recal")
  .print_metrics(m_c6_base, "C6 baseline, + V1a recal")
  .print_decomposition(m_c6_base_norecal, "C6 baseline, no recal")
  .print_decomposition(m_c6_base, "C6 baseline, + V1a recal")

  screen_results <- list(baseline = list(metrics_norecal = m_c6_base_norecal, metrics = m_c6_base))

  for (dd in DECAY_GRID) {
    cli::cli_h2("Decay = {dd} days, atop C6")
    team_mdl_df_dd <- .reweight_team_mdl_df(team_mdl_df_elo, dd, anchor_date)

    t0 <- Sys.time()
    roll_dd <- run_rolling_eval(team_mdl_df_dd, TEST_SEASONS,
                                 gam_trainer = .c6_gam_trainer,
                                 xgb_trainer = .train_xgb_fixed,
                                 extra_feature_cols = "elo_diff")
    cli::cli_inform("Decay={dd} completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
    saveRDS(roll_dd, .rds(paste0("ws6_decay_", dd, "_c6_roll_2026.rds")))

    dd_preds_norecal <- roll_dd$input_blend_preds
    m_dd_norecal <- .compute_metrics(dd_preds_norecal)
    dd_preds <- v1a_recal_own(dd_preds_norecal)
    m_dd <- .compute_metrics(dd_preds)

    .print_metrics(m_dd_norecal, sprintf("Decay=%d atop C6, no recal", dd))
    .print_metrics(m_dd, sprintf("Decay=%d atop C6, + V1a recal", dd))
    .print_decomposition(m_dd_norecal, sprintf("Decay=%d, no recal", dd))
    .print_decomposition(m_dd, sprintf("Decay=%d, + V1a recal", dd))

    boot_dd <- boot_mae_diff(dd_preds, c6_base$preds, B = 2000)
    cat(sprintf("  boot_mae_diff(Decay=%d+recal - C6 baseline+recal, 2026): N=%d diff=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
                dd, boot_dd$n_matches, boot_dd$mae_diff, boot_dd$mae_ci[1], boot_dd$mae_ci[2], boot_dd$brier_diff))

    saveRDS(list(preds_norecal = dd_preds_norecal, metrics_norecal = m_dd_norecal,
                 preds = dd_preds, metrics = m_dd, boot_vs_c6 = boot_dd),
            .rds(paste0("ws6_decay_", dd, "_c6_2026.rds")))
    screen_results[[as.character(dd)]] <- list(metrics_norecal = m_dd_norecal, metrics = m_dd, boot_vs_c6 = boot_dd)
  }

  saveRDS(screen_results, .rds("ws6_screen_results.rds"))
  cli::cli_alert_success("Saved ws6_screen_results.rds")
}

# ================================================================
# Stage: confirm -- pooled 2025:2026 rerun of the best screen decay value
# (chosen by lowest recal'd screen MAE), bootstrap vs cached C6 pooled
# champion (ws5_c6_pool_confirm.rds$preds, the actual ship-gate denominator).
# ================================================================
if (stage %in% c("confirm", "all")) {
  cli::cli_h1("WS6 confirm: pooled 2025:2026 rerun of best screen decay atop C6")

  screen_results <- readRDS(.rds("ws6_screen_results.rds"))
  dd_keys <- setdiff(names(screen_results), "baseline")
  dd_maes <- vapply(dd_keys, function(k) screen_results[[k]]$metrics$mae, numeric(1))
  best_dd <- as.numeric(dd_keys[which.min(dd_maes)])
  cli::cli_inform("Best screen decay by recal'd MAE: {best_dd} (MAE={round(min(dd_maes),3)}) vs C6 baseline MAE={round(screen_results$baseline$metrics$mae,3)}")

  team_mdl_df_best <- .reweight_team_mdl_df(team_mdl_df_elo, best_dd, anchor_date)

  t0 <- Sys.time()
  roll_pool <- run_rolling_eval(team_mdl_df_best, CONFIRM_SEASONS,
                                 gam_trainer = .c6_gam_trainer,
                                 xgb_trainer = .train_xgb_fixed,
                                 extra_feature_cols = "elo_diff")
  cli::cli_inform("Decay={best_dd} pooled confirm completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  saveRDS(roll_pool, .rds(paste0("ws6_decay_", best_dd, "_c6_pool_roll.rds")))

  pool_preds_norecal <- roll_pool$input_blend_preds
  m_pool_norecal <- .compute_metrics(pool_preds_norecal)
  pool_preds <- v1a_recal_own(pool_preds_norecal)
  m_pool <- .compute_metrics(pool_preds)

  c6_pool <- readRDS(.rds("ws5_c6_pool_confirm.rds"))  # champion (C6+recal, pooled) -- the ship-gate denominator
  m_champ_pool <- .compute_metrics(c6_pool$preds)       # recompute with extended (bits-aware) metrics
  boot_pool <- boot_mae_diff(pool_preds, c6_pool$preds, B = 2000)

  cat(sprintf("\n=== Decay=%d atop C6, pooled 2025:2026 (n=%d) ===\n", best_dd, nrow(pool_preds)))
  .print_metrics(m_champ_pool, "C6 champion (recomputed w/ bits), pooled")
  .print_metrics(m_pool_norecal, sprintf("Decay=%d atop C6, no recal, pooled", best_dd))
  .print_metrics(m_pool, sprintf("Decay=%d atop C6, + V1a recal, pooled", best_dd))
  .print_decomposition(m_champ_pool, "C6 champion, pooled")
  .print_decomposition(m_pool, sprintf("Decay=%d atop C6, pooled", best_dd))
  cat(sprintf("boot_mae_diff(Decay=%d+recal - C6 champion, pooled): N=%d diff=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
              best_dd, boot_pool$n_matches, boot_pool$mae_diff, boot_pool$mae_ci[1], boot_pool$mae_ci[2], boot_pool$brier_diff))

  ci_excl_0 <- (boot_pool$mae_ci[1] > 0 && boot_pool$mae_ci[2] > 0) || (boot_pool$mae_ci[1] < 0 && boot_pool$mae_ci[2] < 0)
  brier_ok  <- boot_pool$brier_diff <= 0.002
  bits_ok   <- m_pool$bits >= m_champ_pool$bits
  improved  <- boot_pool$mae_diff < 0
  ship_pass <- ci_excl_0 && improved && brier_ok && bits_ok
  cat(sprintf("\nSHIP GATE (decay=%d atop C6 vs C6 champion): CIexcl0=%s improved=%s BrierOK=%s BitsOK=%s -> PASS=%s\n",
              best_dd, ci_excl_0, improved, brier_ok, bits_ok, ship_pass))

  saveRDS(list(best_dd = best_dd, preds_norecal = pool_preds_norecal, metrics_norecal = m_pool_norecal,
               preds = pool_preds, metrics = m_pool, champ_metrics = m_champ_pool,
               boot = boot_pool, ship_pass = ship_pass),
          .rds("ws6_confirm_result.rds"))
  cli::cli_alert_success("Saved ws6_confirm_result.rds")
}

# ================================================================
# Stage: summary -- full table + explicit real-signal-vs-dispersion read
# ================================================================
if (stage %in% c("summary", "all")) {
  cli::cli_h1("WS6 Final Summary: recency/decay atop C6")

  screen_results <- readRDS(.rds("ws6_screen_results.rds"))
  confirm_result <- if (file.exists(.rds("ws6_confirm_result.rds"))) readRDS(.rds("ws6_confirm_result.rds")) else NULL

  cat("\n=== 2026 SCREEN (n=153) ===\n")
  .print_metrics(screen_results$baseline$metrics_norecal, "C6 baseline (decay=1000 prod), no recal")
  .print_metrics(screen_results$baseline$metrics, "C6 baseline (decay=1000 prod), + V1a recal")
  for (dd in DECAY_GRID) {
    r <- screen_results[[as.character(dd)]]
    .print_metrics(r$metrics_norecal, sprintf("Decay=%d atop C6, no recal", dd))
    .print_metrics(r$metrics, sprintf("Decay=%d atop C6, + V1a recal", dd))
    cat(sprintf("  vs C6 baseline (recal'd, 2026 screen): deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
                r$boot_vs_c6$mae_diff, r$boot_vs_c6$mae_ci[1], r$boot_vs_c6$mae_ci[2], r$boot_vs_c6$brier_diff))
  }

  cat("\n=== Slope decomposition (slope = cor * sd_actual/sd_pred) -- real signal vs dispersion ===\n")
  .print_decomposition(screen_results$baseline$metrics, "C6 baseline, + recal")
  for (dd in DECAY_GRID) {
    .print_decomposition(screen_results[[as.character(dd)]]$metrics, sprintf("Decay=%d atop C6, + recal", dd))
  }

  if (!is.null(confirm_result)) {
    cat(sprintf("\n=== POOLED 2025:2026 CONFIRM: decay=%d atop C6 (best screen pick) ===\n", confirm_result$best_dd))
    .print_metrics(confirm_result$champ_metrics, "C6 champion, pooled")
    .print_metrics(confirm_result$metrics, sprintf("Decay=%d atop C6, + recal, pooled", confirm_result$best_dd))
    .print_decomposition(confirm_result$champ_metrics, "C6 champion, pooled")
    .print_decomposition(confirm_result$metrics, sprintf("Decay=%d atop C6, pooled", confirm_result$best_dd))
    cat(sprintf("deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f deltaBits=%+.4f -> SHIP_PASS=%s\n",
                confirm_result$boot$mae_diff, confirm_result$boot$mae_ci[1], confirm_result$boot$mae_ci[2],
                confirm_result$boot$brier_diff, confirm_result$metrics$bits - confirm_result$champ_metrics$bits,
                confirm_result$ship_pass))
  } else {
    cli::cli_warn("No confirm-stage result found -- run 'confirm' stage first.")
  }

  saveRDS(list(screen = screen_results, confirm = confirm_result), .rds("ws6_final_summary.rds"))
  cli::cli_alert_success("Saved ws6_final_summary.rds")
}
