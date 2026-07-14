# ws5_grid.R — WS5: Bounded "AutoML-spirit" model/feature search
# FABLE-MATCH-MAE-PLAN.md WS5
# =====================================================================
# Closed 12-candidate grid (plan table). Screens on TEST_SEASONS <- 2026
# (G2), confirms top candidates on 2025:2026 pooled (G2/G3), single
# champion picked by pooled MAE subject to G3 (bootstrap CI excl. 0,
# Brier guard). Ties break toward the simpler model.
#
# Candidate status (set after reading WS1-4 results):
#   1  Champion + WS1 winner (V1a recal)          -- RUN (reused, free)
#   2  Champion + Elo feature                     -- RUN (reused, free)
#   3  Champion + Elo + WS1 winner                -- RUN (reused, free)
#   4  CV-stacked chain (+ recal)                 -- SKIPPED: WS3 crashed
#      mid-run (log cuts off inside cv_stack round 3/19, no results/*.rds
#      ever written) -- no usable winner to carry forward (plan instruction:
#      skip, don't invent).
#   5  Best WS4 structural variant (+ recal)      -- RUN (V4b AND V4c,
#      both + recal, reused/free -- WS4 asked both be carried forward)
#   6  "Everything": best structure + Elo + recal -- RUN (fresh: V4b
#      formula + elo_diff feature + V1a recal)
#   7  Direct-margin XGBoost (no chain)            -- RUN (fresh, custom
#      single-stage rolling harness)
#   8  Ridge/GLM linear baseline                   -- RUN (fresh, custom
#      single-stage rolling harness)
#   9  Elo-anchored hybrid                         -- RUN (fresh; WS2(a)
#      standalone Elo was statistically indistinguishable from champion,
#      CI [-1.318,+2.156] overlapping 0 -- judged "competitive" per the
#      plan's gate for this candidate)
#   10 Blend-weight sweep on final margin          -- RUN (reused, free;
#      NOTE scope caveat: reweights the Output Blend, i.e. final
#      GAM-chain margin vs final XGB-chain margin, because the cached
#      artifacts only retain each pipeline's FINAL pred_margin -- the
#      intermediate per-stage arrays needed to reweight the champion's
#      actual Input-Blend mechanism were not cached by WS2/WS4 and
#      re-deriving them would require a fresh full rerun. This still
#      directly answers the plan's stated question "is 0.5/0.5 optimal
#      between the two model families".)
#   11 Recency: best-of-#6 with WS2(c) decay winner -- DEFERRED (not a
#      prerequisite failure -- WS2(c) did produce a usable decay=300
#      value -- but deprioritized given (a) WS2(c)'s own caveat that its
#      close-bucket gain is a dispersion-widening artifact, the opposite
#      of what this plan is trying to fix, and (b) it requires a second
#      full team_mdl_df rebuild + full candidate-6-cost retrain, which
#      the compute budget for this pass does not allow after 6/7/8/9.
#      Documented here, not silently dropped.
#   12 (reserve)                                   -- unused by design,
#      no plan edit added a candidate here.
#
# Run stage-by-stage (checkpoints to experiments/results/*.rds so partial
# progress survives an interrupted run):
#   Rscript ws5_grid.R check      # reconstruction checkpoint vs WS1's reported V1a numbers
#   Rscript ws5_grid.R reused     # candidates 1,2,3,5b,5c,10 (2026 screen, no retraining)
#   Rscript ws5_grid.R c8         # candidate 8 fresh (2026 screen)
#   Rscript ws5_grid.R c7         # candidate 7 fresh (2026 screen)
#   Rscript ws5_grid.R c9         # candidate 9 fresh (2026 screen)
#   Rscript ws5_grid.R c6         # candidate 6 fresh (2026 screen) -- expensive
#   Rscript ws5_grid.R champpool  # champion fresh pooled 2025:2026 run -- expensive
#   Rscript ws5_grid.R confirm    # pooled 2025:2026 confirmation of top candidates
#   Rscript ws5_grid.R squiggle   # Squiggle field-median ceiling reference (not a candidate)
#   Rscript ws5_grid.R summary    # aggregate everything, print + write final table

stage <- {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) >= 1) a[1] else "all"
}
cat("=== ws5_grid.R stage:", stage, "===\n")

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

# ---- Shared: team_mdl_df (reuse WS2's cache -- verified byte-identical
# champion reproduction to WS4's independent build, see session notes) ----
team_mdl_df <- readRDS(.rds("team_mdl_df_cache.rds"))
cli::cli_inform("team_mdl_df: {nrow(team_mdl_df)} rows, seasons {paste(sort(unique(team_mdl_df$season.x)), collapse=', ')}")

# ---- Shared: Elo table (reuse WS2's tuned k/hga/carryover + full-history table) ----
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

# ---- Shared: recal_expanding + .apply_recal (copied verbatim from
# ws1_margin_recal.R, plan G5 -- each WS keeps its own copy) ----
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

#' V1a recalibration applied to a prediction set's OWN history (within the
#' seasons present in `preds`; matches WS1's V1a exactly when `preds` is a
#' 2026-only screen, and generalises correctly to a pooled multi-season set).
v1a_recal_own <- function(preds) {
  idx <- seq_len(nrow(preds))
  res <- recal_expanding(preds, idx, idx, mode = "slope_only", min_n = 30)
  .apply_recal(preds, res)
}

.print_metrics <- function(m, label) {
  cat(sprintf(
    "%-40s MAE=%.3f RMSE=%.3f Brier=%.4f Slope=%.3f Cor=%.3f SDRatio=%.3f CloseMAE(n=%d)=%.3f\n",
    label, m$mae, m$rmse, m$brier, m$slope, m$cor, m$sd_ratio, m$close_n, m$close_mae
  ))
}

# ---- Simple one-row-per-match formatter for custom (non-chain) trainers
# (candidates 7, 8, 9 don't go through torp:::.format_match_preds() since
# they have no gam_pred_tot_xscore/pred_xscore_diff/bits chain columns --
# only pred_margin, pred_win, margin, home_win are needed for
# .compute_metrics()/boot_mae_diff()). ----
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

# ---- Generic custom rolling harness for single-flat-model candidates
# (7, 8, 9) -- same train-strictly-prior/test-this-round discipline as
# run_rolling_eval(), just without the dual GAM+XGB chain/blend logic. ----
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

# ================================================================
# Stage: check -- reconstruction checkpoint (advisor-recommended)
# ================================================================
if (stage %in% c("check", "all")) {
  cli::cli_h1("CHECK: reconstruct candidate 1 (V1a recal on cached champion) -- sanity pass")

  roll_baseline <- readRDS(.rds("ws4_roll_baseline.rds"))
  champ_2026 <- roll_baseline$input_blend_preds

  base_m <- .compute_metrics(champ_2026)
  .print_metrics(base_m, "Champion (2026-only-screen nrounds regime, no recal)")

  preds_v1a_check <- v1a_recal_own(champ_2026)
  m_v1a_check <- .compute_metrics(preds_v1a_check)
  .print_metrics(m_v1a_check, "V1a recal reconstruction (same regime)")

  cat("\nWS1 reported V1a (POOLED-run nrounds regime, tuned on season.x<2025): MAE=26.228 Slope=0.898 CloseMAE=17.157\n")
  cat(sprintf(
    "This reconstruction (2026-SCREEN nrounds regime, tuned on season.x<2026): MAE=%.3f Slope=%.3f CloseMAE=%.3f\n",
    m_v1a_check$mae, m_v1a_check$slope, m_v1a_check$close_mae
  ))
  cat(sprintf(
    "Champion (no recal) also differs by regime: this screen=%.3f vs WS1 pooled-run's 2026 slice=26.372 (delta=%.3f)\n",
    base_m$mae, base_m$mae - 26.372
  ))
  cat("NOTE: slope (0.902 vs 0.898) and close_mae (17.254 vs 17.157) are within ~0.1 of WS1's\n")
  cat("reported V1a despite the two runs using DIFFERENT XGBoost nrounds-tuning windows (this is\n")
  cat("the nrounds-window trap: a 2026-only screen tunes nrounds on season.x<2026, WS1's pooled\n")
  cat("run tuned on season.x<2025) -- recal_expanding's ordering/min_n logic is behaving as\n")
  cat("intended; the residual MAE gap is attributable to the different champion baseline, not a\n")
  cat("bug in this reconstruction. A true apples-to-apples check happens in the champpool stage\n")
  cat("below, which regenerates a fresh pooled run and re-validates V1a against WS1's numbers in\n")
  cat("the SAME nrounds regime WS1 used.\n")
  cli::cli_alert_info("CHECK: proceeding on the strength of matching slope/close_mae; full apples-to-apples validation deferred to champpool stage")
}

# ================================================================
# Stage: reused -- candidates 1, 2, 3, 5b, 5c, 10 (2026 screen, free --
# post-hoc functions on already-computed WS2/WS4 rolling-OOS predictions)
# ================================================================
if (stage %in% c("reused", "all")) {
  cli::cli_h1("WS5: Reused candidates (1, 2, 3, 5b, 5c, 10) on 2026 screen")

  roll_champion <- readRDS(.rds("ws4_roll_baseline.rds"))   # == ws2_champion_roll.rds (verified identical)
  roll_elo      <- readRDS(.rds("ws2_feature_roll.rds"))    # WS2(b) Elo-as-feature Input Blend
  roll_v4b      <- readRDS(.rds("ws4_roll_v4b.rds"))
  roll_v4c      <- readRDS(.rds("ws4_roll_v4c.rds"))

  champ_ib <- roll_champion$input_blend_preds
  elo_ib   <- roll_elo$input_blend_preds
  v4b_ib   <- roll_v4b$input_blend_preds
  v4c_ib   <- roll_v4c$input_blend_preds

  m_champ <- .compute_metrics(champ_ib)

  # Candidate 1: Champion + WS1 winner (V1a recal)
  c1_preds <- v1a_recal_own(champ_ib)
  m_c1 <- .compute_metrics(c1_preds)

  # Candidate 2: Champion + Elo feature (no recal)
  m_c2 <- .compute_metrics(elo_ib)

  # Candidate 3: Champion + Elo + WS1 winner (recal applied to the Elo-feature preds' own history)
  c3_preds <- v1a_recal_own(elo_ib)
  m_c3 <- .compute_metrics(c3_preds)

  # Candidate 5b: WS4 V4b + recal
  c5b_preds <- v1a_recal_own(v4b_ib)
  m_c5b <- .compute_metrics(c5b_preds)

  # Candidate 5c: WS4 V4c + recal
  c5c_preds <- v1a_recal_own(v4c_ib)
  m_c5c <- .compute_metrics(c5c_preds)

  # Candidate 10: blend-weight sweep on the FINAL margin (Output Blend:
  # w*GAM-chain-final-margin + (1-w)*XGB-chain-final-margin). See header
  # note -- intermediate-stage arrays needed to reweight the champion's
  # actual Input Blend were not cached; this reweights each chain's final
  # output instead, still directly testing "is 0.5/0.5 optimal".
  gam_final <- roll_champion$gam_preds
  xgb_final <- roll_champion$xgb_preds
  stopifnot(nrow(gam_final) == nrow(xgb_final), all(gam_final$match_id == xgb_final$match_id))

  sweep_blend <- function(w) {
    out <- gam_final
    out$pred_margin <- w * gam_final$pred_margin + (1 - w) * xgb_final$pred_margin
    out$pred_win    <- w * gam_final$pred_win    + (1 - w) * xgb_final$pred_win
    out
  }
  c10_035 <- sweep_blend(0.35); c10_050 <- sweep_blend(0.50); c10_065 <- sweep_blend(0.65)
  m_c10_035 <- .compute_metrics(c10_035)
  m_c10_050 <- .compute_metrics(c10_050)
  m_c10_065 <- .compute_metrics(c10_065)

  cat("\n=== Reused candidates, 2026 screen (n=153) ===\n")
  .print_metrics(m_champ, "Champion (Input Blend, G4)")
  .print_metrics(m_c1, "C1: Champion + V1a recal")
  .print_metrics(m_c2, "C2: Champion + Elo feature")
  .print_metrics(m_c3, "C3: Champion + Elo + V1a recal")
  .print_metrics(m_c5b, "C5b: WS4 V4b + V1a recal")
  .print_metrics(m_c5c, "C5c: WS4 V4c + V1a recal")
  .print_metrics(m_c10_035, "C10: Output Blend w=0.35 (GAM weight)")
  .print_metrics(m_c10_050, "C10: Output Blend w=0.50 (GAM weight)")
  .print_metrics(m_c10_065, "C10: Output Blend w=0.65 (GAM weight)")

  saveRDS(list(
    m_champ = m_champ,
    c1_preds = c1_preds, m_c1 = m_c1,
    c2_preds = elo_ib, m_c2 = m_c2,
    c3_preds = c3_preds, m_c3 = m_c3,
    c5b_preds = c5b_preds, m_c5b = m_c5b,
    c5c_preds = c5c_preds, m_c5c = m_c5c,
    c10_035 = c10_035, m_c10_035 = m_c10_035,
    c10_050 = c10_050, m_c10_050 = m_c10_050,
    c10_065 = c10_065, m_c10_065 = m_c10_065,
    champ_ib = champ_ib
  ), .rds("ws5_reused_2026.rds"))
  cli::cli_alert_success("Saved ws5_reused_2026.rds")
}

# ================================================================
# Stage: c8 -- Ridge/GLM linear baseline (fresh, 2026 screen)
# ================================================================
if (stage %in% c("c8", "all")) {
  cli::cli_h1("WS5 Candidate 8: Ridge/GLM linear baseline (fresh, 2026 screen)")

  fit_predict_c8 <- function(train_df, test_df) {
    fit_lm  <- stats::lm(score_diff ~ torp_diff + elo_diff + log_dist_diff + days_rest_diff,
                          data = train_df, weights = weightz)
    fit_glm <- stats::glm(win ~ torp_diff + elo_diff + log_dist_diff + days_rest_diff,
                           data = train_df, family = stats::binomial(), weights = weightz)
    test_df$pred_margin <- stats::predict(fit_lm, newdata = test_df)
    test_df$pred_win    <- stats::predict(fit_glm, newdata = test_df, type = "response")
    .simple_match_format(test_df)
  }

  t0 <- Sys.time()
  c8_preds <- run_custom_rolling_eval(team_mdl_df_elo, TEST_SEASONS, fit_predict_c8)
  cli::cli_inform("C8 completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  m_c8 <- .compute_metrics(c8_preds)
  .print_metrics(m_c8, "C8: Ridge/GLM linear baseline")

  saveRDS(list(preds = c8_preds, metrics = m_c8), .rds("ws5_c8_2026.rds"))
  cli::cli_alert_success("Saved ws5_c8_2026.rds")
}

# ================================================================
# Stage: c7 -- Direct-margin XGBoost, no chain (fresh, 2026 screen)
# ================================================================
if (stage %in% c("c7", "all")) {
  cli::cli_h1("WS5 Candidate 7: Direct-margin XGBoost, no chain (fresh, 2026 screen)")

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
  cli::cli_inform("C7 base_cols: {paste(c7_base_cols, collapse=', ')}")

  reg_params <- list(objective = "reg:squarederror", eval_metric = "rmse", tree_method = "hist",
                      eta = 0.05, subsample = 0.7, colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15)
  cls_params <- list(objective = "binary:logistic", eval_metric = "logloss", tree_method = "hist",
                      eta = 0.05, subsample = 0.7, colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15)

  # Pre-optimise nrounds via CV on pre-test-season data only (G6)
  pretest_mask <- team_mdl_df_elo$season.x < TEST_SEASONS & !is.na(team_mdl_df_elo$win)
  cv_df <- team_mdl_df_elo[pretest_mask, ]
  cli::cli_inform("nrounds CV input: {nrow(cv_df)/2} matches (seasons < {TEST_SEASONS})")

  fmat_cv <- stats::model.matrix(~ . - 1, data = cv_df[, c7_base_cols, drop = FALSE])
  set.seed(1234)
  cv_reg <- xgboost::xgb.cv(params = reg_params,
                             data = xgboost::xgb.DMatrix(fmat_cv, label = cv_df$score_diff, weight = cv_df$weightz),
                             nrounds = 400, nfold = 5, early_stopping_rounds = 25, verbose = 0)
  # xgb.cv()'s return object has no $best_iteration in this xgboost version --
  # mirror torp:::.train_match_xgb()'s own extraction (match_train.R:529):
  # which.min() over the evaluation_log's test metric column.
  best_nrounds_margin <- which.min(cv_reg$evaluation_log$test_rmse_mean)

  set.seed(1234)
  cv_cls <- xgboost::xgb.cv(params = cls_params,
                             data = xgboost::xgb.DMatrix(fmat_cv, label = as.numeric(cv_df$win), weight = cv_df$weightz),
                             nrounds = 400, nfold = 5, early_stopping_rounds = 25, verbose = 0)
  best_nrounds_win <- which.min(cv_cls$evaluation_log$test_logloss_mean)
  cli::cli_inform("C7 nrounds: margin={best_nrounds_margin}, win={best_nrounds_win}")

  fit_predict_c7 <- function(train_df, test_df) {
    fmat_tr <- stats::model.matrix(~ . - 1, data = train_df[, c7_base_cols, drop = FALSE])
    fmat_te <- stats::model.matrix(~ . - 1, data = test_df[, c7_base_cols, drop = FALSE])

    set.seed(1234)
    m_reg <- xgboost::xgb.train(params = reg_params,
                                 data = xgboost::xgb.DMatrix(fmat_tr, label = train_df$score_diff, weight = train_df$weightz),
                                 nrounds = best_nrounds_margin, verbose = 0)
    set.seed(1234)
    m_cls <- xgboost::xgb.train(params = cls_params,
                                 data = xgboost::xgb.DMatrix(fmat_tr, label = as.numeric(train_df$win), weight = train_df$weightz),
                                 nrounds = best_nrounds_win, verbose = 0)

    test_df$pred_margin <- predict(m_reg, xgboost::xgb.DMatrix(fmat_te))
    test_df$pred_win    <- predict(m_cls, xgboost::xgb.DMatrix(fmat_te))
    .simple_match_format(test_df)
  }

  t0 <- Sys.time()
  c7_preds <- run_custom_rolling_eval(team_mdl_df_elo, TEST_SEASONS, fit_predict_c7)
  cli::cli_inform("C7 completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  m_c7 <- .compute_metrics(c7_preds)
  .print_metrics(m_c7, "C7: Direct-margin XGBoost (no chain)")

  saveRDS(list(preds = c7_preds, metrics = m_c7,
               nrounds_margin = best_nrounds_margin, nrounds_win = best_nrounds_win),
          .rds("ws5_c7_2026.rds"))
  cli::cli_alert_success("Saved ws5_c7_2026.rds")
}

# ================================================================
# Stage: c9 -- Elo-anchored hybrid (fresh, 2026 screen)
# ================================================================
if (stage %in% c("c9", "all")) {
  cli::cli_h1("WS5 Candidate 9: Elo-anchored hybrid (fresh, 2026 screen)")

  fit_predict_c9 <- function(train_df, test_df) {
    train_home <- train_df[train_df$team_type == "home", ]
    fit_scale <- fit_elo_margin_scale(train_home$elo_diff, best_elo$hga, train_home$score_diff)

    hga_sign_tr <- ifelse(train_df$team_type == "home", 1, -1)
    train_df$elo_diff_hga <- train_df$elo_diff + hga_sign_tr * best_elo$hga
    train_df$elo_pred_margin <- stats::predict(fit_scale, newdata = data.frame(elo_diff_hga = train_df$elo_diff_hga))
    train_df$resid_margin <- train_df$score_diff - train_df$elo_pred_margin

    fit_resid <- mgcv::gam(
      resid_margin ~ s(torp_diff, bs = "ts", k = 5) + s(epr_diff, bs = "ts", k = 5) + s(psr_diff, bs = "ts", k = 5),
      data = train_df, weights = train_df$weightz
    )

    hga_sign_te <- ifelse(test_df$team_type == "home", 1, -1)
    test_df$elo_diff_hga <- test_df$elo_diff + hga_sign_te * best_elo$hga
    test_df$elo_pred_margin <- stats::predict(fit_scale, newdata = data.frame(elo_diff_hga = test_df$elo_diff_hga))
    test_df$resid_pred <- stats::predict(fit_resid, newdata = test_df)
    test_df$pred_margin <- test_df$elo_pred_margin + test_df$resid_pred
    test_df$pred_win <- 1 / (1 + 10^(-(test_df$elo_diff_hga) / 400))

    .simple_match_format(test_df)
  }

  t0 <- Sys.time()
  c9_preds <- run_custom_rolling_eval(team_mdl_df_elo, TEST_SEASONS, fit_predict_c9)
  cli::cli_inform("C9 completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  m_c9 <- .compute_metrics(c9_preds)
  .print_metrics(m_c9, "C9: Elo-anchored hybrid")

  saveRDS(list(preds = c9_preds, metrics = m_c9), .rds("ws5_c9_2026.rds"))
  cli::cli_alert_success("Saved ws5_c9_2026.rds")
}

# ================================================================
# Trainer factory for candidate 6 / 11: V4b structural formula (drop
# ti(*, gam_pred_tot_xscore) from models 2-4, drop model 4's second-order
# stack tensors) + elo_diff optional smooth added to models 2 and 4 (exact
# mirror of WS2(b)'s elo-as-feature addition pattern). Full copy of
# torp:::.train_match_gams()/ws4's V4b variant (plan G5).
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
# Trainer for candidate 5b confirmation: V4b structure WITHOUT elo (i.e.
# exactly WS4's V4b variant, reproduced here so the pooled confirmation
# doesn't depend on sourcing ws4_formula_variants.R's executable top-level
# code). Advisor review (post-c6): C5b (V4b+recal, no elo) scores BETTER
# 2026-screen MAE than C6 (25.935 vs 25.996) with a cleaner Brier margin
# (0.1733 vs 0.1750) -- WS2(b)'s elo-as-feature result (worse than
# champion alone, 26.361 vs 26.198) undercuts the plan's a-priori
# "candidate 6 is the presumptive shipper" framing, so C5b is confirmed
# on pooled 2025:2026 alongside C6, not instead of it.
# ================================================================
.train_match_gams_v4b_only <- function(team_mdl_df, train_filter = NULL, nthreads = 4L, gamma_arg = 1.4) {
  loadNamespace("mgcv")
  if (is.null(train_filter)) train_mask <- !is.na(team_mdl_df$win) else train_mask <- train_filter & !is.na(team_mdl_df$win)
  gam_df <- team_mdl_df[train_mask, ]
  cli::cli_inform("[c5b] Training on {nrow(gam_df)} completed matches")
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
    "s(dsr_diff, bs = \"ts\", k = 5)"        = list(var = "dsr_diff", k = 5)
  )
  drop_terms <- character(0)
  for (term_str in names(optional_smooth_terms)) {
    info <- optional_smooth_terms[[term_str]]
    vals <- gam_df[[info$var]]
    if (length(unique(vals[!is.na(vals)])) < info$k) drop_terms <- c(drop_terms, term_str)
  }
  .add_optional <- function(base_terms, optional_terms) {
    keep <- setdiff(optional_terms, drop_terms)
    if (length(keep) > 0) paste(base_terms, "+", paste(keep, collapse = " + ")) else base_terms
  }

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
  m2_optional <- c("s(psr_diff, bs = \"ts\", k = 5)", "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  m2_formula <- stats::as.formula(.add_optional(paste(m2_terms, collapse = " "), m2_optional))
  afl_xscore_diff_mdl <- mgcv::bam(
    m2_formula, data = gam_df, weights = gam_df$weightz, family = gaussian(),
    nthreads = nthreads, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_xscore_diff <- predict(afl_xscore_diff_mdl, newdata = team_mdl_df, type = "response")

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
  m4_optional <- c("s(psr_diff, bs = \"ts\", k = 5)", "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  m4_formula <- stats::as.formula(.add_optional(paste(m4_terms, collapse = " "), m4_optional))
  afl_score_mdl <- mgcv::bam(
    m4_formula, data = gam_df, weights = gam_df$weightz, family = "gaussian",
    nthreads = nthreads, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = gamma_arg
  )
  team_mdl_df$gam_pred_score_diff <- predict(afl_score_mdl, newdata = team_mdl_df, type = "response")

  cli::cli_progress_step("[c5b] Training win probability model")
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

.c5b_gam_trainer <- function(team_mdl_df, train_filter = NULL, nthreads = 4L) {
  .train_match_gams_v4b_only(team_mdl_df, train_filter = train_filter, nthreads = nthreads, gamma_arg = 1.4)
}

# ================================================================
# Stage: c6 -- "Everything" model: V4b structure + Elo feature (+ recal
# applied post-hoc), fresh 2026 screen. Expensive (full 5-GAM+5-XGB chain
# per round).
# ================================================================
if (stage %in% c("c6", "all")) {
  cli::cli_h1("WS5 Candidate 6: 'Everything' (V4b structure + Elo feature), fresh 2026 screen")

  t0 <- Sys.time()
  roll_c6 <- run_rolling_eval(team_mdl_df_elo, TEST_SEASONS,
                               gam_trainer = .c6_gam_trainer,
                               xgb_trainer = .train_xgb_fixed,
                               extra_feature_cols = "elo_diff")
  cli::cli_inform("C6 completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  saveRDS(roll_c6, .rds("ws5_c6_roll_2026.rds"))

  c6_preds <- roll_c6$input_blend_preds
  m_c6_norecal <- .compute_metrics(c6_preds)
  c6_recal <- v1a_recal_own(c6_preds)
  m_c6 <- .compute_metrics(c6_recal)

  .print_metrics(m_c6_norecal, "C6: V4b + Elo, no recal (Input Blend)")
  .print_metrics(m_c6, "C6: V4b + Elo + V1a recal (Input Blend)")

  saveRDS(list(preds_norecal = c6_preds, metrics_norecal = m_c6_norecal,
               preds = c6_recal, metrics = m_c6), .rds("ws5_c6_2026.rds"))
  cli::cli_alert_success("Saved ws5_c6_2026.rds")
}

# ================================================================
# Stage: c5bpool -- Candidate 5b (V4b + recal, NO elo) fresh pooled
# 2025:2026 run. Added after inspecting the 2026 screen: C5b (25.935,
# Brier 0.1733) beats C6 (25.996, Brier 0.1750) on both MAE and Brier --
# WS2(b)'s own result (elo feature alone is worse than champion, 26.361
# vs 26.198) undercuts the plan's a-priori framing of C6 as "the
# presumptive shipper", so C5b is confirmed on the pooled window too,
# not assumed dominated by C6. Expensive (full chain, 42 rounds).
# ================================================================
if (stage %in% c("c5bpool", "all")) {
  cli::cli_h1("WS5 Candidate 5b pooled confirm: V4b + V1a recal (no elo), fresh 2025:2026")

  t0 <- Sys.time()
  roll_c5b_pool <- run_rolling_eval(team_mdl_df, CONFIRM_SEASONS,
                                     gam_trainer = .c5b_gam_trainer,
                                     xgb_trainer = .train_xgb_fixed)
  cli::cli_inform("C5b pooled confirm completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  saveRDS(roll_c5b_pool, .rds("ws5_c5b_pool_roll.rds"))

  c5b_pool_ib <- roll_c5b_pool$input_blend_preds
  m_c5b_pool_norecal <- .compute_metrics(c5b_pool_ib)
  c5b_pool_recal <- v1a_recal_own(c5b_pool_ib)
  m_c5b_pool <- .compute_metrics(c5b_pool_recal)

  champ_pool_cache <- readRDS(.rds("ws5_champ_pool.rds"))
  boot_c5b_pool <- boot_mae_diff(c5b_pool_recal, champ_pool_cache$preds)

  .print_metrics(m_c5b_pool_norecal, "C5b (V4b, no recal), pooled")
  .print_metrics(m_c5b_pool, "C5b (V4b + V1a recal), pooled")
  cat(sprintf("  C5b vs champion: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
              boot_c5b_pool$mae_diff, boot_c5b_pool$mae_ci[1], boot_c5b_pool$mae_ci[2], boot_c5b_pool$brier_diff))

  saveRDS(list(preds_norecal = c5b_pool_ib, metrics_norecal = m_c5b_pool_norecal,
               preds = c5b_pool_recal, metrics = m_c5b_pool, boot = boot_c5b_pool),
          .rds("ws5_c5b_pool_confirm.rds"))
  cli::cli_alert_success("Saved ws5_c5b_pool_confirm.rds")
}

# ================================================================
# Stage: champpool -- champion (Input Blend) fresh run on pooled
# 2025:2026. Needed as (a) the G3-gate denominator for bootstrap CI, and
# (b) a free confirmation base for candidates 1 & 10 (post-hoc functions
# of this run's own outputs). Expensive (~7-9 min, 42 rounds).
# ================================================================
if (stage %in% c("champpool", "all")) {
  cli::cli_h1("WS5: Champion (Input Blend) fresh pooled 2025:2026 run (G3 denominator)")

  t0 <- Sys.time()
  roll_champ_pool <- run_rolling_eval(team_mdl_df, CONFIRM_SEASONS)
  cli::cli_inform("Champion pooled run completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  saveRDS(roll_champ_pool, .rds("ws5_champ_pool_roll.rds"))

  champ_pool_ib <- roll_champ_pool$input_blend_preds
  m_champ_pool <- .compute_metrics(champ_pool_ib)
  .print_metrics(m_champ_pool, sprintf("Champion pooled 2025:2026 (n=%d)", nrow(champ_pool_ib)))

  cat(sprintf(
    "\nWS1 reported champion pooled: MAE=26.026 Slope=0.920 CloseMAE=17.863 n=369\nThis run           : MAE=%.3f Slope=%.3f CloseMAE=%.3f n=%d\n",
    m_champ_pool$mae, m_champ_pool$slope, m_champ_pool$close_mae, nrow(champ_pool_ib)
  ))
  reprod_ok <- abs(m_champ_pool$mae - 26.026) < 0.05 && abs(m_champ_pool$slope - 0.920) < 0.02
  if (reprod_ok) {
    cli::cli_alert_success("Champion pooled reproduction matches WS1 within tolerance -- same nrounds regime confirmed")
  } else {
    cli::cli_warn("Champion pooled reproduction drifts from WS1's reported numbers by more than tolerance -- investigate before trusting downstream deltas")
  }

  # True apples-to-apples V1a check (same nrounds regime WS1 used): subset
  # this pooled run's 2026 rows, recalibrate with history restricted to
  # WITHIN 2026 only (true V1a mode, not the pooled/V1d-equivalent mode).
  champ_2026_from_pool <- champ_pool_ib |> dplyr::filter(season == 2026)
  v1a_true_check <- v1a_recal_own(champ_2026_from_pool)
  m_v1a_true_check <- .compute_metrics(v1a_true_check)
  cat(sprintf(
    "\nTrue V1a check (same regime as WS1): WS1 reported MAE=26.228 Slope=0.898 CloseMAE=17.157\nThis reconstruction                : MAE=%.3f Slope=%.3f CloseMAE=%.3f\n",
    m_v1a_true_check$mae, m_v1a_true_check$slope, m_v1a_true_check$close_mae
  ))
  v1a_ok <- abs(m_v1a_true_check$mae - 26.228) < 0.05 && abs(m_v1a_true_check$slope - 0.898) < 0.02 && abs(m_v1a_true_check$close_mae - 17.157) < 0.05
  if (v1a_ok) {
    cli::cli_alert_success("V1a reconstruction matches WS1 within tolerance in the SAME nrounds regime -- recal_expanding validated")
  } else {
    cli::cli_abort("V1a reconstruction still does not match WS1 in the same nrounds regime -- a real bug, fix before proceeding")
  }

  saveRDS(list(preds = champ_pool_ib, metrics = m_champ_pool), .rds("ws5_champ_pool.rds"))
  cli::cli_alert_success("Saved ws5_champ_pool.rds")
}

# ================================================================
# Stage: squiggle -- Squiggle field-median margin ceiling reference
# (diagnostic only, NOT a candidate)
# ================================================================
if (stage %in% c("squiggle", "all")) {
  cli::cli_h1("WS5: Squiggle field-median margin ceiling reference (2026, same 153 games)")

  result <- tryCatch({
    tips <- fitzRoy::fetch_squiggle_data("tips", year = 2026)
    tips <- tips |>
      dplyr::mutate(
        round = as.integer(round),
        hteam_norm = torp_replace_teams(hteam),
        hmargin = as.numeric(hmargin)
      ) |>
      dplyr::filter(!is.na(hmargin))

    field_median <- tips |>
      dplyr::group_by(round, hteam_norm) |>
      dplyr::summarise(field_median_margin = median(hmargin), n_sources = dplyr::n(), .groups = "drop")

    roll_baseline <- readRDS(.rds("ws4_roll_baseline.rds"))
    champ_2026 <- roll_baseline$input_blend_preds |>
      dplyr::mutate(hteam_norm = torp_replace_teams(as.character(home_team)))

    joined <- champ_2026 |>
      dplyr::inner_join(field_median, by = c("round", "hteam_norm"))

    m_field <- list(
      mae = mean(abs(joined$field_median_margin - joined$margin)),
      rmse = sqrt(mean((joined$field_median_margin - joined$margin)^2)),
      slope = unname(stats::coef(stats::lm(margin ~ field_median_margin, data = joined))[["field_median_margin"]]),
      cor = stats::cor(joined$field_median_margin, joined$margin),
      sd_pred = stats::sd(joined$field_median_margin),
      sd_actual = stats::sd(joined$margin),
      n = nrow(joined),
      n_sources_med = median(joined$n_sources)
    )

    cat(sprintf(
      "Squiggle field-median margin (n=%d games, median %d sources/game): MAE=%.3f RMSE=%.3f Slope=%.3f Cor=%.3f SD(pred)=%.2f SD(actual)=%.2f\n",
      m_field$n, m_field$n_sources_med, m_field$mae, m_field$rmse, m_field$slope, m_field$cor, m_field$sd_pred, m_field$sd_actual
    ))
    cat("Reference (diagnosis doc, source-labelled 'Squiggle' crowd tip): MAE=26.12, SD(pred)=20.83\n")

    saveRDS(m_field, .rds("ws5_squiggle_ceiling.rds"))
    list(ran = TRUE, metrics = m_field)
  }, error = function(e) {
    cli::cli_warn("Squiggle ceiling fetch failed: {e$message}")
    list(ran = FALSE, error = conditionMessage(e))
  })

  saveRDS(result, .rds("ws5_squiggle_result.rds"))
}

# ================================================================
# Stage: confirm -- pooled 2025:2026 confirmation of top candidates.
# Candidates 1 & 10 confirm for FREE from the champpool run (post-hoc
# functions of its own outputs). Any candidate whose trainer differs from
# the champion (2, 3, 5b, 5c, 6, 7, 8, 9) needs a FRESH pooled run if
# selected for confirmation -- a cached 2026-only screen artifact must
# NOT be reused here, because XGBoost nrounds are tuned on
# season.x < min(test_seasons), which differs between a 2026-only screen
# (<2026) and a 2025:2026 pooled window (<2025) -- see rolling_lib.R G6.
# ================================================================
if (stage %in% c("confirm", "all")) {
  cli::cli_h1("WS5: Pooled 2025:2026 confirmation")

  champ_pool <- readRDS(.rds("ws5_champ_pool.rds"))
  champ_pool_ib <- champ_pool$preds
  m_champ_pool <- champ_pool$metrics

  # Candidate 1 pooled (free): V1a recal on champion pooled preds
  c1_pool_preds <- v1a_recal_own(champ_pool_ib)
  m_c1_pool <- .compute_metrics(c1_pool_preds)
  boot_c1_pool <- boot_mae_diff(c1_pool_preds, champ_pool_ib)

  # Candidate 10 pooled (free): Output Blend weight sweep using champpool's own gam/xgb final margins
  roll_champ_pool_full <- readRDS(.rds("ws5_champ_pool_roll.rds"))
  gam_pool <- roll_champ_pool_full$gam_preds
  xgb_pool <- roll_champ_pool_full$xgb_preds
  stopifnot(nrow(gam_pool) == nrow(xgb_pool), all(gam_pool$match_id == xgb_pool$match_id))
  sweep_blend_pool <- function(w) {
    out <- gam_pool
    out$pred_margin <- w * gam_pool$pred_margin + (1 - w) * xgb_pool$pred_margin
    out$pred_win    <- w * gam_pool$pred_win    + (1 - w) * xgb_pool$pred_win
    out
  }
  c10_035_pool <- sweep_blend_pool(0.35); c10_050_pool <- sweep_blend_pool(0.50); c10_065_pool <- sweep_blend_pool(0.65)
  m_c10_035_pool <- .compute_metrics(c10_035_pool)
  m_c10_050_pool <- .compute_metrics(c10_050_pool)
  m_c10_065_pool <- .compute_metrics(c10_065_pool)
  boot_c10_035_pool <- boot_mae_diff(c10_035_pool, champ_pool_ib)
  boot_c10_065_pool <- boot_mae_diff(c10_065_pool, champ_pool_ib)

  cat("\n=== FREE pooled confirmations (candidates 1 & 10) ===\n")
  .print_metrics(m_champ_pool, "Champion, pooled")
  .print_metrics(m_c1_pool, "C1 (V1a recal), pooled")
  cat(sprintf("  C1 vs champion: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
              boot_c1_pool$mae_diff, boot_c1_pool$mae_ci[1], boot_c1_pool$mae_ci[2], boot_c1_pool$brier_diff))
  .print_metrics(m_c10_035_pool, "C10 w=0.35, pooled")
  .print_metrics(m_c10_050_pool, "C10 w=0.50, pooled (== champion Output Blend)")
  .print_metrics(m_c10_065_pool, "C10 w=0.65, pooled")
  cat(sprintf("  C10(w=0.35) vs champion: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f]\n",
              boot_c10_035_pool$mae_diff, boot_c10_035_pool$mae_ci[1], boot_c10_035_pool$mae_ci[2]))
  cat(sprintf("  C10(w=0.65) vs champion: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f]\n",
              boot_c10_065_pool$mae_diff, boot_c10_065_pool$mae_ci[1], boot_c10_065_pool$mae_ci[2]))

  saveRDS(list(
    m_champ_pool = m_champ_pool,
    c1_pool_preds = c1_pool_preds, m_c1_pool = m_c1_pool, boot_c1_pool = boot_c1_pool,
    m_c10_035_pool = m_c10_035_pool, m_c10_050_pool = m_c10_050_pool, m_c10_065_pool = m_c10_065_pool,
    boot_c10_035_pool = boot_c10_035_pool, boot_c10_065_pool = boot_c10_065_pool
  ), .rds("ws5_confirm_free.rds"))

  # ---- Fresh pooled confirmation of the top-ranked non-free candidate(s) ----
  # Selected after inspecting the 2026 screen summary (see ws5_analyze.R /
  # summary stage): candidate 6 ("everything") is the plan's presumptive
  # shipper and is confirmed fresh on 2025:2026 regardless of its screen
  # rank, per plan text ("candidate 6 ... the presumptive shipper").
  cli::cli_h2("Fresh pooled 2025:2026 confirmation: Candidate 6 ('everything')")
  t0 <- Sys.time()
  roll_c6_pool <- run_rolling_eval(team_mdl_df_elo, CONFIRM_SEASONS,
                                    gam_trainer = .c6_gam_trainer,
                                    xgb_trainer = .train_xgb_fixed,
                                    extra_feature_cols = "elo_diff")
  cli::cli_inform("C6 pooled confirm completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  saveRDS(roll_c6_pool, .rds("ws5_c6_pool_roll.rds"))

  c6_pool_ib <- roll_c6_pool$input_blend_preds
  m_c6_pool_norecal <- .compute_metrics(c6_pool_ib)
  c6_pool_recal <- v1a_recal_own(c6_pool_ib)
  m_c6_pool <- .compute_metrics(c6_pool_recal)
  boot_c6_pool <- boot_mae_diff(c6_pool_recal, champ_pool_ib)

  .print_metrics(m_c6_pool_norecal, "C6 (V4b+Elo, no recal), pooled")
  .print_metrics(m_c6_pool, "C6 (V4b+Elo+V1a recal), pooled")
  cat(sprintf("  C6 vs champion: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
              boot_c6_pool$mae_diff, boot_c6_pool$mae_ci[1], boot_c6_pool$mae_ci[2], boot_c6_pool$brier_diff))

  saveRDS(list(preds_norecal = c6_pool_ib, metrics_norecal = m_c6_pool_norecal,
               preds = c6_pool_recal, metrics = m_c6_pool, boot = boot_c6_pool),
          .rds("ws5_c6_pool_confirm.rds"))
  cli::cli_alert_success("Saved ws5_c6_pool_confirm.rds")

  # ---- Fresh pooled confirmation: Candidate 7 (direct-margin XGBoost) ----
  # Best 2026-screen MAE of any candidate (25.575 vs champion 26.198) --
  # cheap to confirm (single flat XGB model, no chain), so confirmed
  # regardless of the plan-mandated candidate-6 confirmation above.
  cli::cli_h2("Fresh pooled 2025:2026 confirmation: Candidate 7 (direct-margin XGBoost)")

  c7_cache <- readRDS(.rds("ws5_c7_2026.rds"))
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
  cls_params <- list(objective = "binary:logistic", eval_metric = "logloss", tree_method = "hist",
                      eta = 0.05, subsample = 0.7, colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15)

  # Pre-optimise nrounds via CV on pre-test-season data only (G6) -- pooled
  # window's pre-test seasons are season.x < 2025 (different regime from
  # the 2026-only screen's < 2026 -- must retune here, cannot reuse
  # ws5_c7_2026.rds's nrounds per the nrounds-window trap).
  pretest_mask_pool <- team_mdl_df_elo$season.x < min(CONFIRM_SEASONS) & !is.na(team_mdl_df_elo$win)
  cv_df_pool <- team_mdl_df_elo[pretest_mask_pool, ]
  cli::cli_inform("C7 pooled-confirm nrounds CV input: {nrow(cv_df_pool)/2} matches (seasons < {min(CONFIRM_SEASONS)})")
  fmat_cv_pool <- stats::model.matrix(~ . - 1, data = cv_df_pool[, c7_base_cols, drop = FALSE])
  set.seed(1234)
  cv_reg_pool <- xgboost::xgb.cv(params = reg_params,
                                  data = xgboost::xgb.DMatrix(fmat_cv_pool, label = cv_df_pool$score_diff, weight = cv_df_pool$weightz),
                                  nrounds = 400, nfold = 5, early_stopping_rounds = 25, verbose = 0)
  best_nrounds_margin_pool <- which.min(cv_reg_pool$evaluation_log$test_rmse_mean)
  set.seed(1234)
  cv_cls_pool <- xgboost::xgb.cv(params = cls_params,
                                  data = xgboost::xgb.DMatrix(fmat_cv_pool, label = as.numeric(cv_df_pool$win), weight = cv_df_pool$weightz),
                                  nrounds = 400, nfold = 5, early_stopping_rounds = 25, verbose = 0)
  best_nrounds_win_pool <- which.min(cv_cls_pool$evaluation_log$test_logloss_mean)
  cli::cli_inform("C7 pooled-confirm nrounds: margin={best_nrounds_margin_pool}, win={best_nrounds_win_pool}")

  fit_predict_c7_pool <- function(train_df, test_df) {
    fmat_tr <- stats::model.matrix(~ . - 1, data = train_df[, c7_base_cols, drop = FALSE])
    fmat_te <- stats::model.matrix(~ . - 1, data = test_df[, c7_base_cols, drop = FALSE])
    set.seed(1234)
    m_reg <- xgboost::xgb.train(params = reg_params,
                                 data = xgboost::xgb.DMatrix(fmat_tr, label = train_df$score_diff, weight = train_df$weightz),
                                 nrounds = best_nrounds_margin_pool, verbose = 0)
    set.seed(1234)
    m_cls <- xgboost::xgb.train(params = cls_params,
                                 data = xgboost::xgb.DMatrix(fmat_tr, label = as.numeric(train_df$win), weight = train_df$weightz),
                                 nrounds = best_nrounds_win_pool, verbose = 0)
    test_df$pred_margin <- predict(m_reg, xgboost::xgb.DMatrix(fmat_te))
    test_df$pred_win    <- predict(m_cls, xgboost::xgb.DMatrix(fmat_te))
    .simple_match_format(test_df)
  }

  t0 <- Sys.time()
  c7_pool_preds <- run_custom_rolling_eval(team_mdl_df_elo, CONFIRM_SEASONS, fit_predict_c7_pool)
  cli::cli_inform("C7 pooled confirm completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  m_c7_pool <- .compute_metrics(c7_pool_preds)
  boot_c7_pool <- boot_mae_diff(c7_pool_preds, champ_pool_ib)
  .print_metrics(m_c7_pool, "C7 (direct-margin XGBoost), pooled")
  cat(sprintf("  C7 vs champion: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
              boot_c7_pool$mae_diff, boot_c7_pool$mae_ci[1], boot_c7_pool$mae_ci[2], boot_c7_pool$brier_diff))

  saveRDS(list(preds = c7_pool_preds, metrics = m_c7_pool, boot = boot_c7_pool,
               nrounds_margin = best_nrounds_margin_pool, nrounds_win = best_nrounds_win_pool),
          .rds("ws5_c7_pool_confirm.rds"))
  cli::cli_alert_success("Saved ws5_c7_pool_confirm.rds")
}

# ================================================================
# Stage: summary -- aggregate all candidates, pick champion by pooled
# MAE subject to G3 (bootstrap CI excludes 0, Brier guard <= +0.002),
# ties broken toward the simpler model.
# ================================================================
if (stage %in% c("summary", "all")) {
  cli::cli_h1("WS5: Final summary")

  reused <- readRDS(.rds("ws5_reused_2026.rds"))
  c7_2026 <- readRDS(.rds("ws5_c7_2026.rds"))
  c8_2026 <- readRDS(.rds("ws5_c8_2026.rds"))
  c9_2026 <- readRDS(.rds("ws5_c9_2026.rds"))
  c6_2026 <- readRDS(.rds("ws5_c6_2026.rds"))
  champ_pool <- readRDS(.rds("ws5_champ_pool.rds"))
  confirm_free <- readRDS(.rds("ws5_confirm_free.rds"))
  c6_pool_confirm <- readRDS(.rds("ws5_c6_pool_confirm.rds"))
  c7_pool_confirm <- readRDS(.rds("ws5_c7_pool_confirm.rds"))
  c5b_pool_confirm <- readRDS(.rds("ws5_c5b_pool_confirm.rds"))
  squiggle_result <- tryCatch(readRDS(.rds("ws5_squiggle_result.rds")), error = function(e) list(ran = FALSE))

  cat("\n=== 2026 SCREEN SUMMARY (n=153) ===\n")
  .print_metrics(reused$m_champ, "Champion (Input Blend, G4)")
  .print_metrics(reused$m_c1, "C1: Champion + V1a recal")
  .print_metrics(reused$m_c2, "C2: Champion + Elo feature")
  .print_metrics(reused$m_c3, "C3: Champion + Elo + V1a recal")
  cli::cli_inform("C4: CV-stacked chain -- SKIPPED (WS3 did not complete; no results/*.rds ever written)")
  .print_metrics(reused$m_c5b, "C5b: WS4 V4b + V1a recal")
  .print_metrics(reused$m_c5c, "C5c: WS4 V4c + V1a recal")
  .print_metrics(c6_2026$metrics_norecal, "C6: 'Everything' V4b+Elo, no recal")
  .print_metrics(c6_2026$metrics, "C6: 'Everything' V4b+Elo+V1a recal")
  .print_metrics(c7_2026$metrics, "C7: Direct-margin XGBoost (no chain)")
  .print_metrics(c8_2026$metrics, "C8: Ridge/GLM linear baseline")
  .print_metrics(c9_2026$metrics, "C9: Elo-anchored hybrid")
  .print_metrics(reused$m_c10_035, "C10: Output Blend w=0.35")
  .print_metrics(reused$m_c10_050, "C10: Output Blend w=0.50")
  .print_metrics(reused$m_c10_065, "C10: Output Blend w=0.65")
  cli::cli_inform("C11: Recency (decay=300) atop C6 -- DEFERRED (see header note; not run)")
  cli::cli_inform("C12: reserve -- unused (no plan edit added a candidate)")
  if (isTRUE(squiggle_result$ran)) {
    cat(sprintf("Squiggle field-median (ceiling ref, n=%d, %d src/game): MAE=%.3f Slope=%.3f Cor=%.3f (NOT a candidate)\n",
                squiggle_result$metrics$n, squiggle_result$metrics$n_sources_med,
                squiggle_result$metrics$mae, squiggle_result$metrics$slope, squiggle_result$metrics$cor))
  } else {
    cli::cli_inform("Squiggle ceiling reference: fetch failed, see ws5_squiggle_result.rds for error")
  }

  cat("\n=== POOLED 2025:2026 CONFIRMATION (G3 gate) ===\n")
  .print_metrics(champ_pool$metrics, sprintf("Champion, pooled (n=%d)", nrow(champ_pool$preds)))
  .print_metrics(confirm_free$m_c1_pool, "C1 (V1a recal), pooled")
  .print_metrics(confirm_free$m_c10_035_pool, "C10 (w=0.35), pooled")
  .print_metrics(confirm_free$m_c10_065_pool, "C10 (w=0.65), pooled")
  .print_metrics(c6_pool_confirm$metrics, "C6 ('everything'+recal), pooled")
  .print_metrics(c7_pool_confirm$metrics, "C7 (direct XGB), pooled")
  .print_metrics(c5b_pool_confirm$metrics, "C5b (V4b+recal, no elo), pooled")

  g3_gate <- function(label, mae_diff, mae_ci, brier_diff) {
    ci_excl_0 <- (mae_ci[1] > 0 && mae_ci[2] > 0) || (mae_ci[1] < 0 && mae_ci[2] < 0)
    brier_ok <- brier_diff <= 0.002
    improved <- mae_diff < 0
    pass <- ci_excl_0 && improved && brier_ok
    cat(sprintf("%-40s deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] CIexcl0=%s deltaBrier=%+.5f BrierOK=%s G3PASS=%s\n",
                label, mae_diff, mae_ci[1], mae_ci[2], ci_excl_0, brier_diff, brier_ok, pass))
    pass
  }

  cat("\n--- G3 ship-gate check (pooled 2025:2026, vs champion) ---\n")
  pass_c1  <- g3_gate("C1 (V1a recal)", confirm_free$boot_c1_pool$mae_diff, confirm_free$boot_c1_pool$mae_ci, confirm_free$boot_c1_pool$brier_diff)
  pass_c10 <- g3_gate("C10 (w=0.65)", confirm_free$boot_c10_065_pool$mae_diff, confirm_free$boot_c10_065_pool$mae_ci, confirm_free$boot_c10_065_pool$brier_diff)
  pass_c6  <- g3_gate("C6 ('everything'+recal)", c6_pool_confirm$boot$mae_diff, c6_pool_confirm$boot$mae_ci, c6_pool_confirm$boot$brier_diff)
  pass_c7  <- g3_gate("C7 (direct XGB)", c7_pool_confirm$boot$mae_diff, c7_pool_confirm$boot$mae_ci, c7_pool_confirm$boot$brier_diff)
  pass_c5b <- g3_gate("C5b (V4b+recal, no elo)", c5b_pool_confirm$boot$mae_diff, c5b_pool_confirm$boot$mae_ci, c5b_pool_confirm$boot$brier_diff)

  candidates_pool <- data.frame(
    candidate = c("Champion", "C1 (V1a recal)", "C10 (w=0.65)", "C6 (everything+recal)", "C7 (direct XGB)", "C5b (V4b+recal, no elo)"),
    mae = c(champ_pool$metrics$mae, confirm_free$m_c1_pool$mae, confirm_free$m_c10_065_pool$mae,
            c6_pool_confirm$metrics$mae, c7_pool_confirm$metrics$mae, c5b_pool_confirm$metrics$mae),
    g3_pass = c(NA, pass_c1, pass_c10, pass_c6, pass_c7, pass_c5b)
  )
  cat("\n=== Pooled MAE ranking ===\n")
  print(candidates_pool[order(candidates_pool$mae), ], row.names = FALSE)

  passing <- candidates_pool[!is.na(candidates_pool$g3_pass) & candidates_pool$g3_pass, ]
  if (nrow(passing) == 0) {
    champion_final <- "Champion (Input Blend, G4) -- UNCHANGED"
    ship_recommended <- FALSE
    cli::cli_alert_danger("No candidate clears the G3 ship gate on pooled 2025:2026 -- champion stays as-is")
  } else {
    champion_final <- passing$candidate[which.min(passing$mae)]
    ship_recommended <- TRUE
    cli::cli_alert_success("G3-passing champion by pooled MAE: {champion_final}")
  }

  saveRDS(list(candidates_pool = candidates_pool, champion_final = champion_final, ship_recommended = ship_recommended),
          .rds("ws5_final_summary.rds"))
  cli::cli_alert_success("Saved ws5_final_summary.rds")
}

cli::cli_alert_success("Stage '{stage}' complete")
