# round3_wide_window_ci.R -- Extend the C6 x C7-fixed margin ensemble's
# confirmation window backward into already-available historical seasons,
# to test whether the ship-gate bootstrap CI (currently [-0.370, +0.184] on
# pooled 2025:2026, n=369) narrows enough to exclude zero without waiting
# for more of the 2026 season to play out.
#
# Reuses the EXACT trainer functions from ws5_grid.R (.train_match_gams_v4b_elo
# / .c6_gam_trainer, gamma=1.4) and round2_c7_winfix.R (direct-margin XGB +
# GAM win-head, c7_base_cols incl. elo_diff) -- copied verbatim (plan G5:
# "each WS keeps its own copy"), not hand-retranscribed, to avoid the
# subtly-different-reproduction risk the task flagged. Verified against the
# cached 2025:2026 numbers before trusting the wider-window runs (stage
# "verify").
#
# MATCH_MIN_DATA_SEASON/ROUND (torp/R/constants_match.R: 2021 / R14) is the
# existing floor already baked into team_mdl_df_cache.rds (2021 starts at
# R14). Widening test_seasons to 2023:2026 or 2022:2026 only changes which
# ALREADY-TRAINED-ON rounds get scored; training remains strictly prior-only
# (G1/G6) at every step -- this is not a shortcut.
#
# Run stage-by-stage (checkpoints to experiments/results/*.rds):
#   Rscript round3_wide_window_ci.R verify   # reproduce cached 2025:2026 numbers
#   Rscript round3_wide_window_ci.R w2023    # C6 + C7fix rolling on 2023:2026
#   Rscript round3_wide_window_ci.R w2022    # C6 + C7fix rolling on 2022:2026
#   Rscript round3_wide_window_ci.R summary  # ensembles, boot CIs, per-season table

stage <- {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) >= 1) a[1] else "all"
}
cat("=== round3_wide_window_ci.R stage:", stage, "===\n")

# Setup ----
suppressPackageStartupMessages({
  library(tidyverse); library(xgboost); library(mgcv); library(MLmetrics); library(cli)
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
.rds <- function(name) file.path(RESULTS_DIR, name)

source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))
source(file.path(EXPERIMENTS_DIR, "elo_lib.R"))

# ---- team_mdl_df + elo_diff: reuse the cached with-elo table (verified
# below to be exactly join_elo_diff_to_team_mdl_df(team_mdl_df_cache, ws2
# elo table) -- same nrow, same match_id order, elo_diff fully populated,
# no NAs) rather than rebuilding. ----
team_mdl_df <- readRDS(.rds("team_mdl_df_cache.rds"))
team_mdl_df_elo <- readRDS(.rds("team_mdl_df_cache_with_elo.rds"))
stopifnot(nrow(team_mdl_df) == nrow(team_mdl_df_elo),
          identical(team_mdl_df$match_id, team_mdl_df_elo$match_id),
          "elo_diff" %in% names(team_mdl_df_elo),
          sum(is.na(team_mdl_df_elo$elo_diff)) == 0)
cli::cli_inform("team_mdl_df_elo: {nrow(team_mdl_df_elo)} rows, seasons {paste(sort(unique(team_mdl_df_elo$season.x)), collapse=', ')} (cache reused + shape-verified)")

# ================================================================
# Copied VERBATIM from ws5_grid.R (lines 122-193, 518-718) -- C6 "Everything"
# champion trainer + shared recal helpers. Not re-derived.
# ================================================================
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
    "%-46s MAE=%.3f RMSE=%.3f Brier=%.4f Bits=%.4f Slope=%.3f Cor=%.3f SDRatio=%.3f CloseMAE(n=%d)=%.3f\n",
    label, m$mae, m$rmse, m$brier, m$bits, m$slope, m$cor, m$sd_ratio, m$close_n, m$close_mae
  ))
}

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
# Copied VERBATIM from round2_c7_winfix.R (lines 158-235, 201-214) -- C7-fixed
# candidate (direct-margin XGBoost + GAM win-head). Not re-derived.
# ================================================================
.simple_match_format <- function(df) {
  home <- df[df$team_type == "home", ]
  data.frame(
    season = home$season.x, round = home$round_number.x, match_id = home$match_id,
    home_team = as.character(home$team_name.x),
    pred_margin = home$pred_margin, pred_win = home$pred_win,
    margin = home$score_diff,
    home_win = ifelse(home$score_diff > 0, 1, ifelse(home$score_diff == 0, 0.5, 0)),
    stringsAsFactors = FALSE
  )
}

run_custom_rolling_eval <- function(df, test_seasons, fit_predict_fn, verbose = TRUE) {
  test_rounds <- df |>
    dplyr::filter(!is.na(win), season.x %in% test_seasons) |>
    dplyr::distinct(season.x, round_number.x) |>
    dplyr::arrange(season.x, round_number.x) |>
    dplyr::rename(season = season.x, round = round_number.x)

  all_preds <- list()
  for (i in seq_len(nrow(test_rounds))) {
    s <- test_rounds$season[i]; r <- test_rounds$round[i]
    train_filter <- (df$season.x < s) | (df$season.x == s & df$round_number.x < r)
    test_mask <- !is.na(df$win) & df$season.x == s & df$round_number.x == r
    n_test <- sum(test_mask) / 2
    if (n_test == 0) next
    n_train <- sum(train_filter & !is.na(df$win)) / 2
    if (verbose) cli::cli_progress_step("{s} R{r}: train={n_train}, test={n_test}")
    train_df <- df[train_filter & !is.na(df$win), ]
    test_df  <- df[test_mask, ]
    all_preds[[i]] <- fit_predict_fn(train_df, test_df)
  }
  if (verbose) cli::cli_alert_success("Custom rolling evaluation complete")
  dplyr::bind_rows(all_preds)
}

osr_dsr_ok <- all(c("osr_diff", "dsr_diff") %in% names(team_mdl_df_elo)) &&
  !all(is.na(team_mdl_df_elo$osr_diff))
c7_base_cols <- c(
  "team_type_fac", "game_year_decimal.x", "game_prop_through_year.x",
  "game_prop_through_month.x", "game_prop_through_day.x",
  "epr_diff", "epr_recv_diff", "epr_disp_diff", "epr_spoil_diff", "epr_hitout_diff",
  "torp_diff", "psr_diff",
  if (osr_dsr_ok) c("osr_diff", "dsr_diff"),
  "log_dist_diff", "familiarity_diff", "days_rest_diff_fac",
  "elo_diff", "temp_avg", "precipitation_total", "wind_avg", "humidity_avg"
)
reg_params <- list(objective = "reg:squarederror", eval_metric = "rmse", tree_method = "hist",
                    eta = 0.05, subsample = 0.7, colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15)

# ================================================================
# PINNED copy of torp:::.train_match_xgb() as it existed when the ws5_grid.R
# /round2_c7_winfix.R caches this script verifies against were built (git
# rev 1e04c86~1, i.e. immediately BEFORE today's "Integrate C6 match-margin
# improvement" commit). Used ONLY for run_rolling_eval()'s nrounds-CV
# pre-optimisation step, via the new cv_trainer= override (rolling_lib.R).
#
# WHY THIS EXISTS (discovered during this script's own "verify" stage, round
# 3): calling today's live torp:::.train_match_xgb() reproduced C7-fixed
# exactly (untouched by this) but NOT C6 (MAE 25.701 vs cached 25.545, a
# 0.157 gap) -- traced to two production changes to that function landing
# on disk between when the caches were built and when this script first ran,
# neither of which is a bug in this script's copied trainers:
#   1. (git rev 1e04c86, COMMITTED today) base_cols gained "elo_diff"
#      unconditionally -- a deliberate, documented production choice, but
#      different from the research-harness C6 definition, which
#      intentionally left the nrounds-CV step elo-blind (see
#      FABLE-MATCH-MAE-PLAN.md's own note on this being a ship-gated quirk
#      of C6, not a bug to silently fix here).
#   2. (UNCOMMITTED working-tree edit, not yet validated against any cached
#      number) reg_params/cls_params gained nthread = MATCH_XGB_NTHREAD
#      (4L), previously unset (uncapped -- xgboost grabs all logical cores).
#      xgboost's tree_method="hist" is documented (plan §8) as
#      non-deterministic across thread counts even with a fixed seed, and
#      that alone plausibly explains most of a 0.157 MAE swing given how
#      much the resulting nrounds moved (mine, live-torp: 60/44/12/85/79 vs
#      cached: 31/45/13/59/86).
# Pinning this local copy makes the wide-window runs below immune to BOTH
# of these (and any future production drift) -- byte-identical to the exact
# procedure that produced ws5_c6_pool_confirm.rds / ws5_c6_pool_roll.rds,
# regardless of what torp/R/match_train.R does next. Confirmed via the
# "verify" stage: with this pinned copy, nrounds reproduce 31/45/13/59/86
# exactly and MAE reproduces to <0.01.
# ================================================================
.train_match_xgb_pinned <- function(team_mdl_df, train_filter = NULL) {
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
  cli::cli_inform("XGBoost training on {nrow(xgb_df)} rows")
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
    "days_rest_diff_fac"
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
  if (length(train_seasons) >= 2) {
    folds <- lapply(train_seasons, function(s) which(xgb_df$season.x == s))
  } else {
    # w2022's pre-test prefix is 2021 only — season folds are impossible
    # (xgb.cv needs >=2). Fall back to 5 temporal round-chunks within the
    # season; both rows of a match share a round, so match pairs never
    # straddle folds (random row folds would leak).
    rounds <- sort(unique(xgb_df$round_number.x))
    chunk <- cut(match(xgb_df$round_number.x, rounds), breaks = 5, labels = FALSE)
    folds <- unname(split(seq_len(nrow(xgb_df)), chunk))
  }

  train_step <- function(df, label, weights, feature_cols, params, step_name) {
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

  predict_all <- function(model, df, feature_cols) {
    mat <- stats::model.matrix(~ . - 1, data = df[, feature_cols, drop = FALSE])
    predict(model, xgboost::xgb.DMatrix(data = mat))
  }

  s1 <- train_step(xgb_df, xgb_df$total_xpoints_adj, xgb_df$weightz, s1_cols, reg_params, "total_xpoints")
  xgb_df$xgb_pred_tot_xscore <- s1$preds
  team_mdl_df$xgb_pred_tot_xscore <- predict_all(s1$model, team_mdl_df, s1_cols)

  s2_cols <- c(base_cols, "xgb_pred_tot_xscore")
  s2 <- train_step(xgb_df, xgb_df$xscore_diff, xgb_df$weightz, s2_cols, reg_params, "xscore_diff")
  xgb_df$xgb_pred_xscore_diff <- s2$preds
  team_mdl_df$xgb_pred_xscore_diff <- predict_all(s2$model, team_mdl_df, s2_cols)

  s3_cols <- c(base_cols, "xgb_pred_tot_xscore", "xgb_pred_xscore_diff")
  s3 <- train_step(xgb_df, xgb_df$shot_conv_diff, xgb_df$shot_weightz, s3_cols, reg_params, "conv_diff")
  xgb_df$xgb_pred_conv_diff <- s3$preds
  team_mdl_df$xgb_pred_conv_diff <- predict_all(s3$model, team_mdl_df, s3_cols)

  s4_cols <- c(base_cols, "xgb_pred_xscore_diff", "xgb_pred_conv_diff", "xgb_pred_tot_xscore")
  s4 <- train_step(xgb_df, xgb_df$score_diff, xgb_df$weightz, s4_cols, reg_params, "score_diff")
  xgb_df$xgb_pred_score_diff <- s4$preds
  team_mdl_df$xgb_pred_score_diff <- predict_all(s4$model, team_mdl_df, s4_cols)

  s5_cols <- c(
    "team_type_fac",
    "xgb_pred_tot_xscore", "xgb_pred_score_diff",
    "log_dist_diff", "familiarity_diff", "days_rest_diff_fac"
  )
  s5 <- train_step(xgb_df, as.numeric(xgb_df$win), xgb_df$weightz, s5_cols, cls_params, "win")
  xgb_df$xgb_pred_win <- s5$preds
  team_mdl_df$xgb_pred_win <- predict_all(s5$model, team_mdl_df, s5_cols)

  cli::cli_alert_success("XGBoost pipeline trained (pinned pre-integration copy) ({s1$best_n}/{s2$best_n}/{s3$best_n}/{s4$best_n}/{s5$best_n} rounds)")

  steps <- list(
    total_xpoints = list(best_n = s1$best_n, cv_score = s1$cv_score),
    xscore_diff   = list(best_n = s2$best_n, cv_score = s2$cv_score),
    conv_diff      = list(best_n = s3$best_n, cv_score = s3$cv_score),
    score_diff     = list(best_n = s4$best_n, cv_score = s4$cv_score),
    win            = list(best_n = s5$best_n, cv_score = s5$cv_score)
  )
  list(steps = steps)
}

.fit_c7_winhead <- function(train_df, test_df, pred_margin_train, pred_margin_test) {
  train_df$pred_margin <- pred_margin_train
  win_gam <- mgcv::bam(
    win ~
      s(team_name.x, bs = "re") + s(team_name.y, bs = "re")
      + s(team_name_season.x, bs = "re") + s(team_name_season.y, bs = "re")
      + s(pred_margin, bs = "ts", k = 5)
      + s(log_dist_diff, bs = "ts", k = 5) + s(familiarity_diff, bs = "ts", k = 5)
      + s(days_rest_diff_fac, bs = "re"),
    data = train_df, weights = train_df$weightz,
    family = "binomial", nthreads = 4L, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE, gamma = 1.4
  )
  test_df$pred_margin <- pred_margin_test
  as.numeric(predict(win_gam, newdata = test_df, type = "response"))
}

# ================================================================
# Stage: verify -- reproduce the cached 2025:2026 pooled numbers (C6=25.545,
# C7fix-norecal=25.527, ensemble w=0.5=25.454) using THIS script's copied
# functions, before trusting them on a wider window. Cheap-ish (same 42-round
# window as everything already cached) but a fresh full retrain, not a
# cache read -- this is the "if you rebuild, verify it reproduces the cached
# numbers to 2-3dp" check the task asked for.
# ================================================================
if (stage %in% c("verify", "all")) {
  cli::cli_h1("VERIFY: reproduce cached 2025:2026 pooled numbers with this script's copied trainers")
  CONFIRM_SEASONS <- 2025:2026

  t0 <- Sys.time()
  roll_c6_v <- run_rolling_eval(team_mdl_df_elo, CONFIRM_SEASONS,
                                 gam_trainer = .c6_gam_trainer,
                                 xgb_trainer = .train_xgb_fixed,
                                 extra_feature_cols = "elo_diff",
                                 cv_trainer = .train_match_xgb_pinned)
  cli::cli_inform("C6 verify run: {round(difftime(Sys.time(), t0, units='mins'),2)} min")
  c6_v_norecal <- roll_c6_v$input_blend_preds
  c6_v <- v1a_recal_own(c6_v_norecal)
  m_c6_v <- .compute_metrics(c6_v)
  .print_metrics(m_c6_v, "VERIFY C6 (should match MAE=25.545)")

  cached_c6 <- readRDS(.rds("ws5_c6_pool_confirm.rds"))
  m_cached_c6 <- .compute_metrics(cached_c6$preds)
  cat(sprintf("Cached C6 pooled: MAE=%.3f | This script's C6: MAE=%.3f | diff=%.4f\n",
              m_cached_c6$mae, m_c6_v$mae, m_c6_v$mae - m_cached_c6$mae))

  # C7-fixed verify: reuse round-1's nrounds tuned on season.x<2025 (cached),
  # exactly as round2_c7_winfix.R's confirm stage does.
  c7_pool_cache <- readRDS(.rds("ws5_c7_pool_confirm.rds"))
  best_nrounds_margin_pool <- c7_pool_cache$nrounds_margin

  fit_predict_c7fix_pool <- function(train_df, test_df) {
    fmat_tr <- stats::model.matrix(~ . - 1, data = train_df[, c7_base_cols, drop = FALSE])
    fmat_te <- stats::model.matrix(~ . - 1, data = test_df[, c7_base_cols, drop = FALSE])
    set.seed(1234)
    m_reg <- xgboost::xgb.train(params = reg_params,
                                 data = xgboost::xgb.DMatrix(fmat_tr, label = train_df$score_diff, weight = train_df$weightz),
                                 nrounds = best_nrounds_margin_pool, verbose = 0)
    pred_margin_tr <- predict(m_reg, xgboost::xgb.DMatrix(fmat_tr))
    pred_margin_te <- predict(m_reg, xgboost::xgb.DMatrix(fmat_te))
    test_df$pred_margin <- pred_margin_te
    test_df$pred_win <- .fit_c7_winhead(train_df, test_df, pred_margin_tr, pred_margin_te)
    .simple_match_format(test_df)
  }
  t0 <- Sys.time()
  c7fix_v <- run_custom_rolling_eval(team_mdl_df_elo, CONFIRM_SEASONS, fit_predict_c7fix_pool)
  cli::cli_inform("C7fix verify run: {round(difftime(Sys.time(), t0, units='mins'),2)} min")
  m_c7fix_v <- .compute_metrics(c7fix_v)
  .print_metrics(m_c7fix_v, "VERIFY C7fix (should match MAE=25.527)")

  cached_c7fix <- readRDS(.rds("round2_c7fix_pool_confirm.rds"))
  m_cached_c7fix <- .compute_metrics(cached_c7fix$preds)
  cat(sprintf("Cached C7fix pooled: MAE=%.3f | This script's C7fix: MAE=%.3f | diff=%.4f\n",
              m_cached_c7fix$mae, m_c7fix_v$mae, m_c7fix_v$mae - m_cached_c7fix$mae))

  # Ensemble verify
  c6m <- c6_v[, c("match_id","pred_margin","pred_win","margin","home_win")]
  c7m <- c7fix_v[, c("match_id","pred_margin","pred_win","margin","home_win")]
  names(c6m)[2:3] <- c("pred_margin_c6","pred_win_c6")
  names(c7m)[2:5] <- c("pred_margin_c7","pred_win_c7","margin_c7","home_win_c7")
  merged_v <- merge(c6m, c7m, by = "match_id")
  ens_v <- data.frame(pred_margin = 0.5*merged_v$pred_margin_c6 + 0.5*merged_v$pred_margin_c7,
                       pred_win = merged_v$pred_win_c6, margin = merged_v$margin,
                       home_win = merged_v$home_win, match_id = merged_v$match_id)
  m_ens_v <- .compute_metrics(ens_v)
  .print_metrics(m_ens_v, "VERIFY Ensemble w=0.5 (should match MAE=25.454)")

  saveRDS(list(c6_v = c6_v, c7fix_v = c7fix_v, ens_v = ens_v,
               m_c6_v = m_c6_v, m_c7fix_v = m_c7fix_v, m_ens_v = m_ens_v,
               m_cached_c6 = m_cached_c6, m_cached_c7fix = m_cached_c7fix),
          .rds("round3_verify.rds"))
  cli::cli_alert_success("Saved round3_verify.rds")
}

# ================================================================
# Reusable: run C6 + C7fix on a wider test_seasons window, tag with a suffix
# ================================================================
run_window <- function(test_seasons, suffix) {
  cli::cli_h1("WIDE WINDOW {paste(range(test_seasons), collapse=':')}: C6 + C7fix")

  # ---- C6 ----
  t0 <- Sys.time()
  roll_c6 <- run_rolling_eval(team_mdl_df_elo, test_seasons,
                               gam_trainer = .c6_gam_trainer,
                               xgb_trainer = .train_xgb_fixed,
                               extra_feature_cols = "elo_diff",
                               cv_trainer = .train_match_xgb_pinned)
  cli::cli_inform("C6 [{suffix}] rolling run: {round(difftime(Sys.time(), t0, units='mins'),2)} min")
  c6_norecal <- roll_c6$input_blend_preds
  c6_recal <- v1a_recal_own(c6_norecal)
  m_c6 <- .compute_metrics(c6_recal)
  m_c6_norecal <- .compute_metrics(c6_norecal)
  .print_metrics(m_c6_norecal, paste0("C6 [", suffix, "] no recal"))
  .print_metrics(m_c6, paste0("C6 [", suffix, "] + V1a recal"))
  saveRDS(list(preds_norecal = c6_norecal, preds = c6_recal,
               metrics_norecal = m_c6_norecal, metrics = m_c6,
               test_seasons = test_seasons),
          .rds(paste0("round3_c6_", suffix, ".rds")))

  # ---- C7-fixed: nrounds must be re-tuned on THIS window's pre-test data
  # (G6) -- cannot reuse the 2025:2026-tuned value (nrounds-window trap,
  # already documented in ws5_grid.R/round2_c7_winfix.R). ----
  pretest_mask <- team_mdl_df_elo$season.x < min(test_seasons) & !is.na(team_mdl_df_elo$win)
  cv_df <- team_mdl_df_elo[pretest_mask, ]
  n_pretest_matches <- nrow(cv_df) / 2
  cli::cli_inform("C7fix [{suffix}] nrounds CV input: {n_pretest_matches} matches (seasons < {min(test_seasons)})")
  fmat_cv <- stats::model.matrix(~ . - 1, data = cv_df[, c7_base_cols, drop = FALSE])
  set.seed(1234)
  cv_reg <- xgboost::xgb.cv(params = reg_params,
                             data = xgboost::xgb.DMatrix(fmat_cv, label = cv_df$score_diff, weight = cv_df$weightz),
                             nrounds = 400, nfold = 5, early_stopping_rounds = 25, verbose = 0)
  best_nrounds_margin <- which.min(cv_reg$evaluation_log$test_rmse_mean)
  cli::cli_inform("C7fix [{suffix}] nrounds_margin={best_nrounds_margin} (n_pretest_matches={n_pretest_matches})")

  fit_predict_c7fix <- function(train_df, test_df) {
    fmat_tr <- stats::model.matrix(~ . - 1, data = train_df[, c7_base_cols, drop = FALSE])
    fmat_te <- stats::model.matrix(~ . - 1, data = test_df[, c7_base_cols, drop = FALSE])
    set.seed(1234)
    m_reg <- xgboost::xgb.train(params = reg_params,
                                 data = xgboost::xgb.DMatrix(fmat_tr, label = train_df$score_diff, weight = train_df$weightz),
                                 nrounds = best_nrounds_margin, verbose = 0)
    pred_margin_tr <- predict(m_reg, xgboost::xgb.DMatrix(fmat_tr))
    pred_margin_te <- predict(m_reg, xgboost::xgb.DMatrix(fmat_te))
    test_df$pred_margin <- pred_margin_te
    test_df$pred_win <- .fit_c7_winhead(train_df, test_df, pred_margin_tr, pred_margin_te)
    .simple_match_format(test_df)
  }

  t0 <- Sys.time()
  c7fix_preds <- run_custom_rolling_eval(team_mdl_df_elo, test_seasons, fit_predict_c7fix)
  cli::cli_inform("C7fix [{suffix}] rolling run: {round(difftime(Sys.time(), t0, units='mins'),2)} min")
  m_c7fix <- .compute_metrics(c7fix_preds)
  .print_metrics(m_c7fix, paste0("C7fix [", suffix, "] no recal"))
  saveRDS(list(preds = c7fix_preds, metrics = m_c7fix, nrounds_margin = best_nrounds_margin,
               n_pretest_matches = n_pretest_matches, test_seasons = test_seasons),
          .rds(paste0("round3_c7fix_", suffix, ".rds")))

  invisible(list(c6_recal = c6_recal, c7fix = c7fix_preds))
}

if (stage %in% c("w2023", "all")) run_window(2023:2026, "w2023")
if (stage %in% c("w2022", "all")) run_window(2022:2026, "w2022")

# ================================================================
# Stage: summary -- build the ensemble on each window, boot CI vs C6,
# per-season stability check, comparison vs the original 2025:2026 result.
# ================================================================
if (stage %in% c("summary", "all")) {
  cli::cli_h1("ROUND 3 SUMMARY: wide-window ensemble CI")

  build_ensemble_and_report <- function(suffix) {
    c6_obj <- readRDS(.rds(paste0("round3_c6_", suffix, ".rds")))
    c7_obj <- readRDS(.rds(paste0("round3_c7fix_", suffix, ".rds")))
    c6 <- c6_obj$preds  # recal applied
    c7 <- c7_obj$preds  # no recal (matches original ensemble construction)

    c6m <- c6[, c("match_id","season","pred_margin","pred_win","margin","home_win")]
    c7m <- c7[, c("match_id","pred_margin","pred_win","margin","home_win")]
    names(c6m)[3:4] <- c("pred_margin_c6","pred_win_c6")
    names(c7m)[2:5] <- c("pred_margin_c7","pred_win_c7","margin_c7","home_win_c7")
    merged <- merge(c6m, c7m, by = "match_id")
    stopifnot(all.equal(merged$margin, merged$margin_c7))

    ens <- data.frame(pred_margin = 0.5*merged$pred_margin_c6 + 0.5*merged$pred_margin_c7,
                       pred_win = merged$pred_win_c6, margin = merged$margin,
                       home_win = merged$home_win, match_id = merged$match_id,
                       season = merged$season)

    m_c6 <- .compute_metrics(c6)
    m_ens <- .compute_metrics(ens)
    boot <- boot_mae_diff(ens, c6)

    cat(sprintf("\n=== Window [%s] (n=%d) ===\n", suffix, nrow(ens)))
    .print_metrics(m_c6, paste0("C6 champion [", suffix, "]"))
    .print_metrics(m_ens, paste0("Ensemble w=0.5 [", suffix, "]"))
    cat(sprintf("  Ensemble vs C6: deltaMAE=%+.3f 95%%CI[%.3f, %.3f] deltaBrier=%+.5f 95%%CI[%.5f, %.5f]\n",
                boot$mae_diff, boot$mae_ci[1], boot$mae_ci[2], boot$brier_diff, boot$brier_ci[1], boot$brier_ci[2]))
    ci_excludes_0 <- (boot$mae_ci[1] > 0 && boot$mae_ci[2] > 0) || (boot$mae_ci[1] < 0 && boot$mae_ci[2] < 0)
    cat(sprintf("  CI excludes zero: %s\n", ci_excludes_0))

    cat(sprintf("\n  --- Per-season MAE, window [%s] ---\n", suffix))
    per_season <- merged |>
      dplyr::mutate(pm_ens = 0.5*pred_margin_c6 + 0.5*pred_margin_c7) |>
      dplyr::group_by(season) |>
      dplyr::summarise(
        n = dplyr::n(),
        mae_c6 = round(mean(abs(pred_margin_c6 - margin)), 3),
        mae_ens = round(mean(abs(pm_ens - margin)), 3),
        delta = round(mae_ens - mae_c6, 3),
        .groups = "drop"
      )
    print(as.data.frame(per_season))

    list(m_c6 = m_c6, m_ens = m_ens, boot = boot, ci_excludes_0 = ci_excludes_0,
         per_season = per_season, n = nrow(ens))
  }

  res_2023 <- build_ensemble_and_report("w2023")
  res_2022 <- build_ensemble_and_report("w2022")

  cat("\n=== COMPARISON TABLE (all windows) ===\n")
  cat(sprintf("%-16s %6s %8s %8s %20s %10s %8s\n",
              "Window", "n", "C6 MAE", "Ens MAE", "deltaMAE [CI]", "CI excl0", "dBrier"))
  cat(sprintf("%-16s %6d %8s %8s %20s %10s %8s\n",
              "2025:2026 (orig)", 369, "25.545", "25.454", "-0.091 [-0.370,+0.184]", "NO", "0.0000"))
  cat(sprintf("%-16s %6d %8.3f %8.3f %9.3f [%+.3f,%+.3f] %10s %+8.5f\n",
              "2023:2026", res_2023$n, res_2023$m_c6$mae, res_2023$m_ens$mae,
              res_2023$boot$mae_diff, res_2023$boot$mae_ci[1], res_2023$boot$mae_ci[2],
              ifelse(res_2023$ci_excludes_0, "YES", "NO"), res_2023$boot$brier_diff))
  cat(sprintf("%-16s %6d %8.3f %8.3f %9.3f [%+.3f,%+.3f] %10s %+8.5f\n",
              "2022:2026", res_2022$n, res_2022$m_c6$mae, res_2022$m_ens$mae,
              res_2022$boot$mae_diff, res_2022$boot$mae_ci[1], res_2022$boot$mae_ci[2],
              ifelse(res_2022$ci_excludes_0, "YES", "NO"), res_2022$boot$brier_diff))

  saveRDS(list(res_2023 = res_2023, res_2022 = res_2022), .rds("round3_summary.rds"))
  cli::cli_alert_success("Saved round3_summary.rds")
}
