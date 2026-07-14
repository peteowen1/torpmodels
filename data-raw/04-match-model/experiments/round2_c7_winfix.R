# round2_c7_winfix.R -- Round 2: fix candidate 7's win-probability path
# =====================================================================
# Round 1 (ws5_grid.R candidate 7): direct single-stage XGBoost margin
# regression, no GAM chain. Best raw margin accuracy of anything tested
# (pooled MAE=25.527, slope=1.051) but DISQUALIFIED because its Brier got
# WORSE than the champion -- traced to it using its own from-scratch flat
# XGBoost win classifier (binary:logistic on the raw feature set) instead
# of deriving WP from a smooth function of its own margin prediction, the
# way every other candidate does.
#
# Fix (plan option (a)): replace the flat classifier with a small GAM
# win-head, structurally identical to torp:::.train_match_gams()'s win
# model, EXCEPT the ti(pred_tot_xscore, pred_score_diff) tensor term is
# dropped (candidate 7 has no total-points prediction -- there's no chain)
# and pred_score_diff is replaced by candidate 7's OWN margin prediction.
# Everything else (team/team-season random effects, log_dist_diff,
# familiarity_diff, days_rest_diff_fac, family=binomial, select=TRUE,
# discrete=TRUE, gamma=1.4) is copied verbatim from match_train.R's
# afl_win_mdl formula.
#
# The margin path (base_cols, reg_params, nrounds, seed) is BYTE-IDENTICAL
# to round 1's candidate 7 -- only pred_win derivation changes. So MAE/
# RMSE/slope/cor/sd_ratio/close_mae should reproduce round 1 exactly;
# only brier/bits/logloss/accuracy can move.
#
# Run stage-by-stage (checkpoints to experiments/results/*.rds):
#   Rscript round2_c7_winfix.R screen   # 2026 screen (fast, n=153)
#   Rscript round2_c7_winfix.R confirm  # pooled 2025:2026 + bootstrap vs C6
#   Rscript round2_c7_winfix.R summary  # print final comparison table

stage <- {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) >= 1) a[1] else "all"
}
cat("=== round2_c7_winfix.R stage:", stage, "===\n")

# Setup ----
library(tidyverse)
library(xgboost)
library(mgcv)
library(MLmetrics)
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
if (!torp_loaded) stop("Cannot find torp package (run from torpverse workspace).")

EXPERIMENTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RESULTS_DIR <- file.path(EXPERIMENTS_DIR, "results")
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR, recursive = TRUE)
.rds <- function(name) file.path(RESULTS_DIR, name)

source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))
source(file.path(EXPERIMENTS_DIR, "elo_lib.R"))

TEST_SEASONS    <- 2026
CONFIRM_SEASONS <- 2025:2026

# ---- Shared: team_mdl_df + Elo diff (reuse round-1 caches verbatim) ----
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

# ---- Copied verbatim from ws1_margin_recal.R / ws5_grid.R (plan G5: each
# WS keeps its own copy) ----
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

# ---- Copied verbatim from ws5_grid.R ----
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

.print_metrics <- function(m, label) {
  cat(sprintf(
    "%-52s MAE=%.3f RMSE=%.3f Brier=%.4f Bits=%.4f Slope=%.3f Cor=%.3f SDRatio=%.3f CloseMAE(n=%d)=%.3f\n",
    label, m$mae, m$rmse, m$brier, m$bits, m$slope, m$cor, m$sd_ratio, m$close_n, m$close_mae
  ))
}

# ---- C7 base_cols: identical to ws5_grid.R candidate 7 ----
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

# ---- Win-head GAM: same small formula as torp:::.train_match_gams()'s
# afl_win_mdl, minus the ti(pred_tot_xscore, pred_score_diff) tensor (no
# tot_xscore prediction exists in this no-chain candidate), with
# pred_score_diff replaced by candidate 7's own margin prediction. ----
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

# ---- OOF-margin variant (post-hoc diagnostic follow-up): the in-sample
# XGB margin fed into s(pred_margin) above is more overfit than a GAM's own
# fitted values would be (champion's win model consumes a GAM stage-4 fit,
# not an XGB fit) -- risk is s(pred_margin) learns a too-steep curve on
# training rows, so OOS margins pushed through it come out overconfident.
# Mitigation: 5-fold inner CV within train_df to get out-of-fold margins as
# the win-head's TRAINING input (row-level folds -- consistent with how
# every stage in this codebase already treats team-rows independently with
# weightz weighting; test-set margins still come from the full-train model). ----
.get_oof_margin <- function(train_df, nrounds, k = 5, seed = 1234) {
  n <- nrow(train_df)
  withr::with_seed(seed, {
    folds <- sample(rep(seq_len(k), length.out = n))
  })
  oof <- numeric(n)
  for (fold in seq_len(k)) {
    idx_val <- which(folds == fold)
    idx_tr  <- which(folds != fold)
    fmat_tr  <- stats::model.matrix(~ . - 1, data = train_df[idx_tr, c7_base_cols, drop = FALSE])
    fmat_val <- stats::model.matrix(~ . - 1, data = train_df[idx_val, c7_base_cols, drop = FALSE])
    set.seed(seed)
    m <- xgboost::xgb.train(params = reg_params,
                             data = xgboost::xgb.DMatrix(fmat_tr, label = train_df$score_diff[idx_tr], weight = train_df$weightz[idx_tr]),
                             nrounds = nrounds, verbose = 0)
    oof[idx_val] <- predict(m, xgboost::xgb.DMatrix(fmat_val))
  }
  oof
}

.fit_c7_winhead_oof <- function(train_df, test_df, pred_margin_oof_train, pred_margin_test) {
  train_df$pred_margin <- pred_margin_oof_train
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
# Stage: screen -- 2026 screen (fast, n=153), reusing round-1's tuned
# nrounds (margin regression is unchanged, so nrounds tuning doesn't
# need to be redone; only pred_win derivation changes).
# ================================================================
if (stage %in% c("screen", "all")) {
  cli::cli_h1("Round 2 Candidate 7-fixed: GAM win-head, 2026 screen")

  c7_2026_cache <- readRDS(.rds("ws5_c7_2026.rds"))
  best_nrounds_margin <- c7_2026_cache$nrounds_margin
  cli::cli_inform("Reusing round-1 nrounds_margin={best_nrounds_margin} (2026 screen window)")

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
  c7fix_preds <- run_custom_rolling_eval(team_mdl_df_elo, TEST_SEASONS, fit_predict_c7fix)
  cli::cli_inform("C7-fixed (2026 screen) completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  m_c7fix <- .compute_metrics(c7fix_preds)
  .print_metrics(m_c7fix, "C7-fixed: direct XGB margin + GAM win-head (2026)")

  # Sanity: MAE/RMSE/slope should reproduce round-1 C7 exactly (margin path unchanged)
  cat(sprintf("Round-1 C7 (2026) reference : MAE=%.3f RMSE=%.3f Slope=%.3f\n",
              c7_2026_cache$metrics$mae, c7_2026_cache$metrics$rmse, c7_2026_cache$metrics$slope))
  cat(sprintf("Round-2 C7-fixed (2026)     : MAE=%.3f RMSE=%.3f Slope=%.3f\n",
              m_c7fix$mae, m_c7fix$rmse, m_c7fix$slope))

  saveRDS(list(preds = c7fix_preds, metrics = m_c7fix, nrounds_margin = best_nrounds_margin),
          .rds("round2_c7fix_2026.rds"))
  cli::cli_alert_success("Saved round2_c7fix_2026.rds")
}

# ================================================================
# Stage: confirm -- pooled 2025:2026, bootstrap vs C6 ("Everything",
# pooled MAE=25.545 -- the current champion per round-2 task framing).
# Also: C7-fixed + V1a recal bonus variant.
# ================================================================
if (stage %in% c("confirm", "all")) {
  cli::cli_h1("Round 2 Candidate 7-fixed: pooled 2025:2026 confirmation")

  c7_pool_cache <- readRDS(.rds("ws5_c7_pool_confirm.rds"))
  best_nrounds_margin_pool <- c7_pool_cache$nrounds_margin
  cli::cli_inform("Reusing round-1 nrounds_margin={best_nrounds_margin_pool} (pooled window, seasons < {min(CONFIRM_SEASONS)})")

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
  c7fix_pool_preds <- run_custom_rolling_eval(team_mdl_df_elo, CONFIRM_SEASONS, fit_predict_c7fix_pool)
  cli::cli_inform("C7-fixed pooled confirm completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  m_c7fix_pool <- .compute_metrics(c7fix_pool_preds)

  # Comparator: C6 "Everything" (V4b + elo_diff + V1a recal), pooled MAE=25.545
  # -- the current champion per this round's task framing (NOT round 1's
  # base production champion, MAE=26.026, which round-1 bootstrapped C7 against).
  c6_pool_confirm <- readRDS(.rds("ws5_c6_pool_confirm.rds"))
  champion_preds <- c6_pool_confirm$preds
  m_champion <- .compute_metrics(champion_preds)

  boot_vs_c6 <- boot_mae_diff(c7fix_pool_preds, champion_preds)

  .print_metrics(m_champion, "C6 'Everything' (current champion), pooled")
  .print_metrics(m_c7fix_pool, "C7-fixed, pooled 2025:2026")
  cat(sprintf("  C7-fixed vs C6: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f 95%%CI[%+.5f,%+.5f]\n",
              boot_vs_c6$mae_diff, boot_vs_c6$mae_ci[1], boot_vs_c6$mae_ci[2],
              boot_vs_c6$brier_diff, boot_vs_c6$brier_ci[1], boot_vs_c6$brier_ci[2]))

  # Sanity: margin metrics should reproduce round-1 C7 pooled exactly
  cat(sprintf("Round-1 C7 (pooled) reference: MAE=%.3f RMSE=%.3f Slope=%.3f\n",
              c7_pool_cache$metrics$mae, c7_pool_cache$metrics$rmse, c7_pool_cache$metrics$slope))
  cat(sprintf("Round-2 C7-fixed (pooled)    : MAE=%.3f RMSE=%.3f Slope=%.3f\n",
              m_c7fix_pool$mae, m_c7fix_pool$rmse, m_c7fix_pool$slope))

  # ---- Bonus: C7-fixed + V1a recal (check whether recal helps or is now
  # unnecessary/harmful given C7's raw slope is already ~1.05, near-ideal) ----
  c7fix_recal_pool_preds <- v1a_recal_own(c7fix_pool_preds)
  m_c7fix_recal_pool <- .compute_metrics(c7fix_recal_pool_preds)
  boot_recal_vs_c6 <- boot_mae_diff(c7fix_recal_pool_preds, champion_preds)

  .print_metrics(m_c7fix_recal_pool, "C7-fixed + V1a recal, pooled 2025:2026")
  cat(sprintf("  C7-fixed+recal vs C6: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f 95%%CI[%+.5f,%+.5f]\n",
              boot_recal_vs_c6$mae_diff, boot_recal_vs_c6$mae_ci[1], boot_recal_vs_c6$mae_ci[2],
              boot_recal_vs_c6$brier_diff, boot_recal_vs_c6$brier_ci[1], boot_recal_vs_c6$brier_ci[2]))

  saveRDS(list(
    preds = c7fix_pool_preds, metrics = m_c7fix_pool, boot_vs_c6 = boot_vs_c6,
    preds_recal = c7fix_recal_pool_preds, metrics_recal = m_c7fix_recal_pool, boot_recal_vs_c6 = boot_recal_vs_c6,
    champion_metrics = m_champion, nrounds_margin = best_nrounds_margin_pool
  ), .rds("round2_c7fix_pool_confirm.rds"))
  cli::cli_alert_success("Saved round2_c7fix_pool_confirm.rds")
}

# ================================================================
# Stage: oof -- mitigation follow-up (advisor-recommended diagnostic
# showed C7-fixed's Brier/bits trail C6 despite decile-level calibration
# looking reasonable; hypothesis is in-sample XGB margins teach
# s(pred_margin) a too-steep curve). Feed the win-head 5-fold OOF margins
# instead of in-sample fits for its TRAINING rows only (test-round margins
# still come from the full-train model, as they must).
# ================================================================
if (stage %in% c("oof", "all")) {
  cli::cli_h1("Round 2 Candidate 7-fixed: OOF-margin win-head mitigation, pooled 2025:2026")

  c7_pool_cache <- readRDS(.rds("ws5_c7_pool_confirm.rds"))
  best_nrounds_margin_pool <- c7_pool_cache$nrounds_margin

  fit_predict_c7fix_oof <- function(train_df, test_df) {
    fmat_tr <- stats::model.matrix(~ . - 1, data = train_df[, c7_base_cols, drop = FALSE])
    fmat_te <- stats::model.matrix(~ . - 1, data = test_df[, c7_base_cols, drop = FALSE])

    set.seed(1234)
    m_reg <- xgboost::xgb.train(params = reg_params,
                                 data = xgboost::xgb.DMatrix(fmat_tr, label = train_df$score_diff, weight = train_df$weightz),
                                 nrounds = best_nrounds_margin_pool, verbose = 0)
    pred_margin_te <- predict(m_reg, xgboost::xgb.DMatrix(fmat_te))

    pred_margin_oof_tr <- .get_oof_margin(train_df, best_nrounds_margin_pool)

    test_df$pred_margin <- pred_margin_te
    test_df$pred_win <- .fit_c7_winhead_oof(train_df, test_df, pred_margin_oof_tr, pred_margin_te)
    .simple_match_format(test_df)
  }

  t0 <- Sys.time()
  c7fix_oof_preds <- run_custom_rolling_eval(team_mdl_df_elo, CONFIRM_SEASONS, fit_predict_c7fix_oof)
  cli::cli_inform("C7-fixed OOF pooled confirm completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  m_c7fix_oof <- .compute_metrics(c7fix_oof_preds)

  c6_pool_confirm <- readRDS(.rds("ws5_c6_pool_confirm.rds"))
  champion_preds <- c6_pool_confirm$preds
  m_champion <- .compute_metrics(champion_preds)
  boot_oof_vs_c6 <- boot_mae_diff(c7fix_oof_preds, champion_preds)

  c7fix_pool <- readRDS(.rds("round2_c7fix_pool_confirm.rds"))

  .print_metrics(m_champion, "C6 'Everything' (current champion), pooled")
  .print_metrics(c7fix_pool$metrics, "C7-fixed (in-sample margin -> win-head)")
  .print_metrics(m_c7fix_oof, "C7-fixed (OOF margin -> win-head)")
  cat(sprintf("  C7-fixed(OOF) vs C6: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f 95%%CI[%+.5f,%+.5f]\n",
              boot_oof_vs_c6$mae_diff, boot_oof_vs_c6$mae_ci[1], boot_oof_vs_c6$mae_ci[2],
              boot_oof_vs_c6$brier_diff, boot_oof_vs_c6$brier_ci[1], boot_oof_vs_c6$brier_ci[2]))

  # Sanity: MAE/RMSE/slope must still reproduce round-1 C7 exactly (only the
  # win-head's TRAINING input changed; test-round margin path is untouched)
  cat(sprintf("Round-1 C7 (pooled) reference: MAE=%.3f RMSE=%.3f Slope=%.3f\n",
              c7_pool_cache$metrics$mae, c7_pool_cache$metrics$rmse, c7_pool_cache$metrics$slope))
  cat(sprintf("Round-2 C7-fixed OOF (pooled): MAE=%.3f RMSE=%.3f Slope=%.3f\n",
              m_c7fix_oof$mae, m_c7fix_oof$rmse, m_c7fix_oof$slope))

  saveRDS(list(preds = c7fix_oof_preds, metrics = m_c7fix_oof, boot_vs_c6 = boot_oof_vs_c6,
               champion_metrics = m_champion),
          .rds("round2_c7fix_oof_pool_confirm.rds"))
  cli::cli_alert_success("Saved round2_c7fix_oof_pool_confirm.rds")
}

# ================================================================
# Stage: summary -- final comparison table vs C6 champion + Squiggle field
# ================================================================
if (stage %in% c("summary", "all")) {
  cli::cli_h1("Round 2: Final summary")

  c7fix_2026 <- readRDS(.rds("round2_c7fix_2026.rds"))
  c7fix_pool <- readRDS(.rds("round2_c7fix_pool_confirm.rds"))
  c6_2026 <- tryCatch(readRDS(.rds("ws5_c6_2026.rds")), error = function(e) NULL)
  c7_orig_2026 <- tryCatch(readRDS(.rds("ws5_c7_2026.rds")), error = function(e) NULL)
  reused_2026 <- tryCatch(readRDS(.rds("ws5_reused_2026.rds")), error = function(e) NULL)

  # Round-1 caches' $metrics predate the bits field -- recompute fresh from
  # their stored raw preds so every row in this table is bits-comparable.
  cat("\n=== 2026 SCREEN (n=153) ===\n")
  if (!is.null(reused_2026)) .print_metrics(.compute_metrics(reused_2026$champ_ib), "Base champion (Input Blend)")
  if (!is.null(c6_2026)) .print_metrics(.compute_metrics(c6_2026$preds), "C6 'Everything' (V4b+Elo+V1a recal)")
  if (!is.null(c7_orig_2026)) .print_metrics(.compute_metrics(c7_orig_2026$preds), "C7 original (round-1 flat classifier)")
  .print_metrics(c7fix_2026$metrics, "C7-fixed: direct XGB margin + GAM win-head")

  cat("\n=== POOLED 2025:2026 (n=369) -- ship-gate confirmation ===\n")
  .print_metrics(c7fix_pool$champion_metrics, "C6 'Everything' (current champion)")
  .print_metrics(c7fix_pool$metrics, "C7-fixed (no recal)")
  .print_metrics(c7fix_pool$metrics_recal, "C7-fixed + V1a recal")

  g3_gate <- function(label, mae_diff, mae_ci, brier_diff, bits_diff) {
    ci_excl_0 <- (mae_ci[1] > 0 && mae_ci[2] > 0) || (mae_ci[1] < 0 && mae_ci[2] < 0)
    mae_pass <- ci_excl_0 && mae_diff < 0
    brier_pass <- brier_diff <= 0.002
    bits_pass <- bits_diff >= 0
    pass <- mae_pass && brier_pass && bits_pass
    cat(sprintf("  %-28s CI-excl-0=%s (diff=%+.3f) | Brier-ok=%s (diff=%+.5f) | Bits-ok=%s (diff=%+.4f) -> %s\n",
                label, ci_excl_0, mae_diff, brier_pass, brier_diff, bits_pass, bits_diff, if (pass) "PASS" else "FAIL"))
    pass
  }

  cat("\n=== G3 ship gate vs C6 champion (MAE=25.545, Bits=0.2471, pooled) ===\n")
  pass_norecal <- g3_gate("C7-fixed (no recal)", c7fix_pool$boot_vs_c6$mae_diff, c7fix_pool$boot_vs_c6$mae_ci, c7fix_pool$boot_vs_c6$brier_diff,
                           c7fix_pool$metrics$bits - c7fix_pool$champion_metrics$bits)
  pass_recal   <- g3_gate("C7-fixed + V1a recal", c7fix_pool$boot_recal_vs_c6$mae_diff, c7fix_pool$boot_recal_vs_c6$mae_ci, c7fix_pool$boot_recal_vs_c6$brier_diff,
                           c7fix_pool$metrics_recal$bits - c7fix_pool$champion_metrics$bits)

  oof_cache <- tryCatch(readRDS(.rds("round2_c7fix_oof_pool_confirm.rds")), error = function(e) NULL)
  if (!is.null(oof_cache)) {
    cat("\n=== OOF-margin mitigation ===\n")
    .print_metrics(oof_cache$metrics, "C7-fixed (OOF margin -> win-head)")
    pass_oof <- g3_gate("C7-fixed (OOF margin)", oof_cache$boot_vs_c6$mae_diff, oof_cache$boot_vs_c6$mae_ci, oof_cache$boot_vs_c6$brier_diff,
                         oof_cache$metrics$bits - oof_cache$champion_metrics$bits)
  }

  cat("\n=== vs Squiggle 2026 field (round-2 task framing; MAE not directly comparable, screen n=153) ===\n")
  cat(sprintf("Aggregate     : MAE=25.45 bits~=0.325\n"))
  cat(sprintf("Wheelo        : MAE=24.89 bits~=0.318\n"))
  cat(sprintf("Punters       : MAE=25.32 bits~=0.347\n"))
  cat(sprintf("C7-fixed (2026 screen): MAE=%.3f Bits=%.4f\n", c7fix_2026$metrics$mae, c7fix_2026$metrics$bits))

  cat(sprintf("\nBrier rescue vs round-1 C7 (round-1 disqualifier): %s\n",
              if (!is.null(c6_2026)) "see brier deltas above" else "n/a"))
}
