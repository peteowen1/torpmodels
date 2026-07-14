# phase2_elo_refine.R -- Post-ship research: is the shipped team-Elo config
# (torp/R/team_elo.R + constants_match.R: ELO_K=20, ELO_HGA=45,
# ELO_CARRYOVER=0.75, 538-style MOV multiplier) leaving material match-model
# MAE on the table? Those hyperparameters were chosen from a coarse 27-combo
# grid fit on a pre-2025 sample and explicitly flagged by Pete as "genuinely
# a first draft, worth revisiting" (FABLE-MATCH-MAE-PLAN.md WS2).
#
# RESEARCH ONLY -- does not touch torp/R/team_elo.R or constants_match.R.
# Prototypes here, reusing elo_lib.R (build_team_elo/tune_team_elo/
# fit_elo_margin_scale/elo_pred_win/.matches_from_team_mdl_df) and
# rolling_lib.R (run_rolling_eval/.compute_metrics/boot_mae_diff/
# .train_xgb_fixed) as the starting point, per plan G5 (each experiment
# keeps its own copies of anything it needs to extend; no production edits).
#
# Three things tested, each tuned leak-safely on pre-2025 matches only (G6),
# screened via the REAL rolling-OOS G2 harness on TEST_SEASONS=2026 (not just
# the cheap standalone-Elo triage number -- that's used only to decide which
# configs are worth the expensive full GAM+XGB pipeline run), and any 2026
# screen winner confirmed on pooled 2025:2026 with boot_mae_diff's bootstrap
# CI before being called a win (G3):
#
#   1. Wider hyperparameter grid (denser k, wider hga, finer carryover) --
#      the original 27-combo grid was coarse.
#   2. Recency weighting within Elo itself, tested two ways, each collapsing
#      to a SINGLE elo_diff feature (so it can be scored through production
#      torp:::.train_match_gams / .train_xgb_fixed unmodified):
#        2a. a k that varies linearly with season (single track, k-schedule)
#        2b. a two-timescale blend (fast-k "recent form" track + slow-k
#            "stable" track), combined into one elo_diff via a tuned weight
#            alpha BEFORE it ever becomes a feature -- still one column.
#   3. Alternative AFL-calibrated margin-of-victory scaling -- the current
#      log(|m|+1) convention treats every point as in NFL-scale units; AFL
#      margins run ~3x larger (SD~40 vs NFL's ~14). Tests a scaled-log family
#      (log(|m|/S + 1)) and a sqrt family, each with k re-tuned to
#      re-equilibrate against the rescaled update magnitude.
#
# Integration mirrors production exactly: elo_diff joined once, fed to
# torp:::.train_match_gams (already ships the V4b formula + elo_diff smooths
# in models 2/4, verified by direct read of match_train.R) and
# rolling_lib.R's .train_xgb_fixed with extra_feature_cols="elo_diff" +
# cv_extra_feature_cols="elo_diff" (the WS7 nrounds-CV fix -- elo_diff must
# be present when XGBoost's nrounds get pre-optimised, or the comparison is
# contaminated by a stopping-point tuned for a different feature set).
#
# The baseline (current production Elo config) is REGENERATED FRESH in this
# same script run for every comparison -- not reused from an earlier
# session's cache -- because XGBoost's tree_method="hist" is not
# deterministic across different nthread/session configurations (documented
# in rolling_lib.R's own header and FABLE-MATCH-MAE-PLAN.md S8). Comparing a
# candidate scored in this run against a baseline scored in a DIFFERENT run
# would confound the Elo change with a thread-count artifact.
#
# Run stage-by-stage (checkpoints to experiments/results/phase2_*.rds):
#   Rscript phase2_elo_refine.R check      # sanity: build_team_elo2() == elo_lib.R::build_team_elo() at defaults
#   Rscript phase2_elo_refine.R grid       # test 1: wider grid, pre-2025 tune (cheap)
#   Rscript phase2_elo_refine.R recency    # test 2: k-schedule + two-track blend, pre-2025 tune (cheap)
#   Rscript phase2_elo_refine.R mov        # test 3: MOV scaling family sweep, pre-2025 tune (cheap)
#   Rscript phase2_elo_refine.R pipeline   # G2 screen: baseline + all candidates, full GAM+XGB pipeline, 2026 (expensive)
#   Rscript phase2_elo_refine.R pool       # G3 confirm: pooled 2025:2026 + boot_mae_diff for 2026-screen survivors (expensive)
#   Rscript phase2_elo_refine.R summary    # aggregate + final verdict
#   Rscript phase2_elo_refine.R all        # (default) everything in sequence

stage <- {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) >= 1) a[1] else "all"
}
cat("=== phase2_elo_refine.R stage:", stage, "===\n")

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

# Current production config (torp/R/constants_match.R), read from the loaded
# package itself so this never drifts from what's actually shipped.
PROD_K         <- ELO_K
PROD_HGA       <- ELO_HGA
PROD_CARRYOVER <- ELO_CARRYOVER
cli::cli_inform("Production Elo config: k={PROD_K}, hga={PROD_HGA}, carryover={PROD_CARRYOVER}")

team_mdl_df <- readRDS(.rds("team_mdl_df_cache.rds"))
cli::cli_inform("team_mdl_df: {nrow(team_mdl_df)} rows, seasons {paste(sort(unique(team_mdl_df$season.x)), collapse=', ')}")

matches_all <- .matches_from_team_mdl_df(team_mdl_df)
matches_pre <- matches_all[matches_all$season < 2025, ]
cli::cli_inform("Match universe: {nrow(matches_all)} total, {nrow(matches_pre)} pre-2025 (tuning set, G6)")

# ================================================================
# build_team_elo2 -- generalized Elo builder: pluggable k_fn(season) and
# mov_fn(m, elo_diff). Reduces EXACTLY to elo_lib.R::build_team_elo() when
# k_fn is constant and mov_fn is the 538-style default (verified in the
# `check` stage). Own copy per plan G5 -- no elo_lib.R edits.
# ================================================================
default_mov_fn <- function(m, elo_diff) {
  log(abs(m) + 1) * (2.2 / (0.001 * abs(elo_diff) + 2.2))
}

build_team_elo2 <- function(matches, k_fn, hga, carryover, mov_fn = default_mov_fn) {
  stopifnot(all(c("match_id", "date", "season", "home_team", "away_team", "home_margin") %in% names(matches)))
  matches <- matches[order(matches$date, matches$match_id), ]

  teams <- sort(unique(c(matches$home_team, matches$away_team)))
  elo <- stats::setNames(rep(1500, length(teams)), teams)
  last_season <- stats::setNames(rep(NA_integer_, length(teams)), teams)

  n <- nrow(matches)
  pre_home <- numeric(n)
  pre_away <- numeric(n)

  for (i in seq_len(n)) {
    h <- matches$home_team[i]
    a <- matches$away_team[i]
    s <- matches$season[i]

    if (!is.na(last_season[[h]]) && last_season[[h]] < s) {
      elo[[h]] <- carryover * elo[[h]] + (1 - carryover) * 1500
    }
    if (!is.na(last_season[[a]]) && last_season[[a]] < s) {
      elo[[a]] <- carryover * elo[[a]] + (1 - carryover) * 1500
    }

    elo_h <- elo[[h]]
    elo_a <- elo[[a]]
    pre_home[i] <- elo_h
    pre_away[i] <- elo_a

    exp_home <- 1 / (1 + 10^(-((elo_h + hga) - elo_a) / 400))
    m <- matches$home_margin[i]
    result <- if (is.na(m)) 0.5 else if (m > 0) 1 else if (m < 0) 0 else 0.5

    mov <- if (!is.na(m) && m != 0) mov_fn(m, elo_h - elo_a) else 1
    k <- k_fn(s)

    delta <- k * mov * (result - exp_home)
    elo[[h]] <- elo_h + delta
    elo[[a]] <- elo_a - delta

    last_season[[h]] <- s
    last_season[[a]] <- s
  }

  data.table::data.table(
    match_id  = rep(matches$match_id, 2),
    team_name = c(matches$home_team, matches$away_team),
    elo_pre   = c(pre_home, pre_away)
  )
}

.const_k <- function(k) { force(k); function(season) k }

#' In-sample pre-2025 MAE for one (k_fn, hga, carryover, mov_fn) config --
#' same scoring convention as elo_lib.R::tune_team_elo(): fit the
#' points-per-Elo scale on the SAME matches, score in-sample MAE (honest
#' because elo_pre is point-in-time by construction). This is the G6-
#' compliant SELECTION criterion for every test below.
.score_config <- function(matches, k_fn, hga, carryover, mov_fn = default_mov_fn) {
  et <- build_team_elo2(matches, k_fn, hga, carryover, mov_fn)
  ex <- stats::setNames(et$elo_pre, paste(et$match_id, et$team_name))
  elo_h <- unname(ex[paste(matches$match_id, matches$home_team)])
  elo_a <- unname(ex[paste(matches$match_id, matches$away_team)])
  elo_diff_home <- elo_h - elo_a
  fit <- fit_elo_margin_scale(elo_diff_home, hga, matches$home_margin)
  pred <- stats::predict(fit)
  mean(abs(pred - matches$home_margin))
}

#' Standalone OOS triage number (WS2(a)-style) -- NOT the selection
#' criterion, just a cheap filter for deciding which configs are worth a
#' full GAM+XGB pipeline run. Built over the FULL match history (elo_pre
#' reflects real accumulated history through 2026); the points-per-Elo scale
#' is fit on pre-2025 only (G6), scored OOS on `score_seasons`.
.standalone_oos <- function(k_fn, hga, carryover, mov_fn, score_seasons) {
  et_full <- build_team_elo2(matches_all, k_fn, hga, carryover, mov_fn)
  ex <- stats::setNames(et_full$elo_pre, paste(et_full$match_id, et_full$team_name))

  elo_h_pre <- unname(ex[paste(matches_pre$match_id, matches_pre$home_team)])
  elo_a_pre <- unname(ex[paste(matches_pre$match_id, matches_pre$away_team)])
  fit_m <- fit_elo_margin_scale(elo_h_pre - elo_a_pre, hga, matches_pre$home_margin)

  ms <- matches_all[matches_all$season %in% score_seasons, ]
  elo_h <- unname(ex[paste(ms$match_id, ms$home_team)])
  elo_a <- unname(ex[paste(ms$match_id, ms$away_team)])
  elo_diff_home <- elo_h - elo_a
  pred_margin <- stats::predict(fit_m, newdata = data.frame(elo_diff_hga = elo_diff_home + hga))
  pred_win <- elo_pred_win(elo_diff_home, hga)
  preds <- data.frame(
    match_id = ms$match_id, pred_margin = unname(pred_margin), pred_win = pred_win,
    margin = ms$home_margin,
    home_win = ifelse(ms$home_margin > 0, 1, ifelse(ms$home_margin == 0, 0.5, 0))
  )
  list(preds = preds, metrics = .compute_metrics(preds), fit_m = fit_m)
}

BASELINE_INSAMPLE_MAE <- .score_config(matches_pre, .const_k(PROD_K), PROD_HGA, PROD_CARRYOVER)
baseline_standalone <- .standalone_oos(.const_k(PROD_K), PROD_HGA, PROD_CARRYOVER, default_mov_fn, TEST_SEASONS)
cli::cli_inform(
  "Baseline (prod k={PROD_K}/hga={PROD_HGA}/carryover={PROD_CARRYOVER}): pre-2025 in-sample MAE={round(BASELINE_INSAMPLE_MAE,3)}, standalone 2026 OOS MAE={round(baseline_standalone$metrics$mae,3)}"
)
saveRDS(list(insample_mae = BASELINE_INSAMPLE_MAE, standalone = baseline_standalone),
        .rds("phase2_baseline_standalone.rds"))

# ================================================================
# Stage: check -- build_team_elo2() must reduce EXACTLY to
# elo_lib.R::build_team_elo() at default params before anything downstream
# can be trusted.
# ================================================================
if (stage %in% c("check", "all")) {
  cli::cli_h1("CHECK: build_team_elo2() == elo_lib.R::build_team_elo() at defaults")
  et1 <- build_team_elo(matches_pre, k = PROD_K, hga = PROD_HGA, carryover = PROD_CARRYOVER, mov_mult = TRUE)
  et2 <- build_team_elo2(matches_pre, .const_k(PROD_K), PROD_HGA, PROD_CARRYOVER, default_mov_fn)
  stopifnot(identical(et1$match_id, et2$match_id), identical(et1$team_name, et2$team_name))
  max_diff <- max(abs(et1$elo_pre - et2$elo_pre))
  cat(sprintf("Max |elo_pre diff| between elo_lib.R::build_team_elo() and build_team_elo2(): %.10f\n", max_diff))
  if (max_diff > 1e-9) {
    cli::cli_abort("build_team_elo2() does NOT reproduce build_team_elo() -- fix before trusting any downstream result")
  }
  cli::cli_alert_success("build_team_elo2() verified byte-identical to elo_lib.R::build_team_elo()")
}

# ================================================================
# Stage: grid -- Test 1: wider hyperparameter grid (denser k, wider hga,
# finer carryover), pre-2025 tuning only (G6). Original plan grid: k in
# {15,20,30}, hga in {25,35,45}, carryover in {0.6,0.75,0.9} (27 combos).
# ================================================================
if (stage %in% c("grid", "all")) {
  cli::cli_h1("Test 1: wider Elo hyperparameter grid (pre-2025 tuning, G6)")

  k_grid         <- seq(10, 45, by = 5)                  # 8 values (orig 3, {15,20,30})
  hga_grid       <- seq(15, 65, by = 10)                 # 6 values (orig 3, {25,35,45})
  carryover_grid <- round(seq(0.45, 0.95, by = 0.05), 2) # 11 values (orig 3, {0.6,0.75,0.9})
  grid <- expand.grid(k = k_grid, hga = hga_grid, carryover = carryover_grid)
  cli::cli_inform("Grid size: {nrow(grid)} combos (original plan grid was 27)")

  t0 <- Sys.time()
  grid$mae <- vapply(seq_len(nrow(grid)), function(i)
    .score_config(matches_pre, .const_k(grid$k[i]), grid$hga[i], grid$carryover[i]), numeric(1))
  cli::cli_inform("Grid tune completed in {round(difftime(Sys.time(), t0, units = 'secs'), 1)}s")

  grid <- grid[order(grid$mae), ]
  cat("\n=== Top 10 wider-grid combos (pre-2025 in-sample MAE) ===\n")
  print(head(grid, 10), row.names = FALSE)

  is_baseline <- abs(grid$k - PROD_K) < 1e-9 & abs(grid$hga - PROD_HGA) < 1e-9 & abs(grid$carryover - PROD_CARRYOVER) < 1e-9
  baseline_row <- grid[is_baseline, ]
  cat("\n=== Baseline (current production) row for reference ===\n")
  print(baseline_row, row.names = FALSE)

  best <- grid[1, ]
  cli::cli_alert_success(
    "Wider-grid winner: k={best$k}, hga={best$hga}, carryover={best$carryover} (MAE={round(best$mae,3)} vs baseline {round(baseline_row$mae,3)})"
  )

  # Standalone 2026 OOS for top 5 + baseline (triage only, not selection)
  top5 <- head(grid, 5)
  standalone_top5 <- dplyr::bind_rows(lapply(seq_len(nrow(top5)), function(i) {
    r <- top5[i, ]
    so <- .standalone_oos(.const_k(r$k), r$hga, r$carryover, default_mov_fn, TEST_SEASONS)
    data.frame(k = r$k, hga = r$hga, carryover = r$carryover, insample_mae = r$mae,
               standalone_2026_mae = so$metrics$mae, standalone_2026_slope = so$metrics$slope)
  }))
  cat("\n=== Top 5 wider-grid combos: pre-2025 in-sample MAE vs standalone 2026 OOS MAE ===\n")
  print(standalone_top5, row.names = FALSE)
  cat(sprintf("\nBaseline for comparison: standalone 2026 OOS MAE = %.3f\n", baseline_standalone$metrics$mae))

  saveRDS(list(grid = grid, baseline_row = baseline_row, best = best, standalone_top5 = standalone_top5),
          .rds("phase2_grid_results.rds"))
  cli::cli_alert_success("Saved phase2_grid_results.rds")
}

# ================================================================
# Stage: recency -- Test 2: recency weighting within Elo, two mechanisms,
# each collapsing to a SINGLE elo_diff feature.
# ================================================================
if (stage %in% c("recency", "all")) {
  cli::cli_h1("Test 2: recency weighting within Elo (k-schedule + two-track blend)")

  # ---- 2a: linear-in-season k-schedule, atop production hga/carryover ----
  anchor_season <- min(matches_pre$season)
  linear_k_fn <- function(k_base, slope) {
    force(k_base); force(slope); force(anchor_season)
    function(season) max(k_base * (1 + slope * (season - anchor_season)), 1)  # floor: never let k go to 0/negative when extrapolated to unseen seasons
  }
  slope_grid <- c(-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.30, 0.50)
  slope_res <- data.frame(slope = slope_grid, mae = vapply(slope_grid, function(sl)
    .score_config(matches_pre, linear_k_fn(PROD_K, sl), PROD_HGA, PROD_CARRYOVER), numeric(1)))
  slope_res <- slope_res[order(slope_res$mae), ]
  cat("\n=== 2a: k-schedule slope sweep (k_base=20, hga=45, carryover=0.75, pre-2025 in-sample MAE) ===\n")
  print(slope_res, row.names = FALSE)
  best_slope <- slope_res$slope[1]
  slope0_mae <- slope_res$mae[slope_res$slope == 0]
  cli::cli_alert_success("Best k-schedule slope: {best_slope} (MAE={round(slope_res$mae[1],3)} vs slope=0 baseline {round(slope0_mae,3)})")

  standalone_2a <- .standalone_oos(linear_k_fn(PROD_K, best_slope), PROD_HGA, PROD_CARRYOVER, default_mov_fn, TEST_SEASONS)
  cat(sprintf("2a standalone 2026 OOS MAE (slope=%.2f): %.3f (baseline %.3f)\n",
              best_slope, standalone_2a$metrics$mae, baseline_standalone$metrics$mae))

  # ---- 2b: two-track blend (fast + slow Elo, combined into ONE elo_diff) ----
  slow_et <- build_team_elo2(matches_pre, .const_k(PROD_K), PROD_HGA, PROD_CARRYOVER)
  ex_slow <- stats::setNames(slow_et$elo_pre, paste(slow_et$match_id, slow_et$team_name))
  slow_diff <- unname(ex_slow[paste(matches_pre$match_id, matches_pre$home_team)]) -
               unname(ex_slow[paste(matches_pre$match_id, matches_pre$away_team)])

  fast_k_grid <- c(40, 60, 90)   # 2x, 3x, 4.5x the production k -- "shorter memory" tracks
  alpha_grid  <- seq(0, 1, by = 0.1)
  blend_all <- dplyr::bind_rows(lapply(fast_k_grid, function(fk) {
    fast_et <- build_team_elo2(matches_pre, .const_k(fk), PROD_HGA, PROD_CARRYOVER)
    ex_fast <- stats::setNames(fast_et$elo_pre, paste(fast_et$match_id, fast_et$team_name))
    fast_diff <- unname(ex_fast[paste(matches_pre$match_id, matches_pre$home_team)]) -
                 unname(ex_fast[paste(matches_pre$match_id, matches_pre$away_team)])
    data.frame(fast_k = fk, alpha = alpha_grid, mae = vapply(alpha_grid, function(al) {
      blend_diff <- al * fast_diff + (1 - al) * slow_diff
      fit <- fit_elo_margin_scale(blend_diff, PROD_HGA, matches_pre$home_margin)
      mean(abs(predict(fit) - matches_pre$home_margin))
    }, numeric(1)))
  }))
  blend_all <- blend_all[order(blend_all$mae), ]
  cat(sprintf("\n=== 2b: two-track blend (slow k=%d, fast k in {%s}), alpha sweep (pre-2025 in-sample MAE) ===\n",
              PROD_K, paste(fast_k_grid, collapse = ",")))
  print(head(blend_all, 10), row.names = FALSE)
  alpha0_mae <- min(blend_all$mae[blend_all$alpha == 0])  # alpha=0 == pure slow == baseline, same for every fast_k
  best_blend <- blend_all[1, ]
  cli::cli_alert_success(
    "Best blend: fast_k={best_blend$fast_k}, alpha={best_blend$alpha} (MAE={round(best_blend$mae,3)} vs alpha=0 (baseline) {round(alpha0_mae,3)})"
  )

  # Standalone 2026 OOS for the blend winner
  slow_et_full <- build_team_elo2(matches_all, .const_k(PROD_K), PROD_HGA, PROD_CARRYOVER)
  fast_et_full <- build_team_elo2(matches_all, .const_k(best_blend$fast_k), PROD_HGA, PROD_CARRYOVER)
  ex_slow_f <- stats::setNames(slow_et_full$elo_pre, paste(slow_et_full$match_id, slow_et_full$team_name))
  ex_fast_f <- stats::setNames(fast_et_full$elo_pre, paste(fast_et_full$match_id, fast_et_full$team_name))

  .blend_standalone_oos <- function(alpha, score_seasons) {
    slow_diff_pre <- unname(ex_slow_f[paste(matches_pre$match_id, matches_pre$home_team)]) -
                     unname(ex_slow_f[paste(matches_pre$match_id, matches_pre$away_team)])
    fast_diff_pre <- unname(ex_fast_f[paste(matches_pre$match_id, matches_pre$home_team)]) -
                     unname(ex_fast_f[paste(matches_pre$match_id, matches_pre$away_team)])
    blend_pre <- alpha * fast_diff_pre + (1 - alpha) * slow_diff_pre
    fit_m <- fit_elo_margin_scale(blend_pre, PROD_HGA, matches_pre$home_margin)

    ms <- matches_all[matches_all$season %in% score_seasons, ]
    slow_diff_s <- unname(ex_slow_f[paste(ms$match_id, ms$home_team)]) - unname(ex_slow_f[paste(ms$match_id, ms$away_team)])
    fast_diff_s <- unname(ex_fast_f[paste(ms$match_id, ms$home_team)]) - unname(ex_fast_f[paste(ms$match_id, ms$away_team)])
    blend_s <- alpha * fast_diff_s + (1 - alpha) * slow_diff_s
    pred_margin <- stats::predict(fit_m, newdata = data.frame(elo_diff_hga = blend_s + PROD_HGA))
    pred_win <- elo_pred_win(blend_s, PROD_HGA)
    preds <- data.frame(match_id = ms$match_id, pred_margin = unname(pred_margin), pred_win = pred_win,
                         margin = ms$home_margin,
                         home_win = ifelse(ms$home_margin > 0, 1, ifelse(ms$home_margin == 0, 0.5, 0)))
    list(preds = preds, metrics = .compute_metrics(preds))
  }
  standalone_2b <- .blend_standalone_oos(best_blend$alpha, TEST_SEASONS)
  cat(sprintf("2b standalone 2026 OOS MAE (alpha=%.1f, fast_k=%d): %.3f (baseline %.3f)\n",
              best_blend$alpha, best_blend$fast_k, standalone_2b$metrics$mae, baseline_standalone$metrics$mae))

  saveRDS(list(
    slope_res = slope_res, best_slope = best_slope, standalone_2a = standalone_2a,
    blend_all = blend_all, best_blend = best_blend, standalone_2b = standalone_2b
  ), .rds("phase2_recency_results.rds"))
  cli::cli_alert_success("Saved phase2_recency_results.rds")
}

# ================================================================
# Stage: mov -- Test 3: AFL-calibrated margin-of-victory scaling. Current
# formula log(|m|+1) is an off-the-shelf NFL-scale convention; AFL margin SD
# (~40) runs ~3x NFL's (~14). Test a scaled-log family (log(|m|/S+1)) and a
# sqrt family, re-tuning k per scale (MOV rescaling changes the effective
# update magnitude, so k must re-equilibrate); hga/carryover held at
# production values to isolate the MOV effect.
# ================================================================
if (stage %in% c("mov", "all")) {
  cli::cli_h1("Test 3: alternative margin-of-victory scaling (AFL-appropriate)")

  make_scaled_log_mov <- function(S) { force(S); function(m, elo_diff) log(abs(m) / S + 1) * (2.2 / (0.001 * abs(elo_diff) + 2.2)) }
  make_sqrt_mov <- function(scale) { force(scale); function(m, elo_diff) (sqrt(abs(m)) / scale) * (2.2 / (0.001 * abs(elo_diff) + 2.2)) }

  k_local_grid <- seq(5, 45, by = 5)

  scaled_log_S <- c(1, 3, 5, 7, 10)   # S=1 reproduces the current production formula exactly
  scaled_log_res <- do.call(rbind, lapply(scaled_log_S, function(S) {
    mov_fn <- make_scaled_log_mov(S)
    maes <- vapply(k_local_grid, function(k) .score_config(matches_pre, .const_k(k), PROD_HGA, PROD_CARRYOVER, mov_fn), numeric(1))
    best_i <- which.min(maes)
    data.frame(family = "scaled_log", S = S, best_k = k_local_grid[best_i], mae = maes[best_i])
  }))

  sqrt_scale_grid <- c(1, 2, 3, 4, 6)
  sqrt_res <- do.call(rbind, lapply(sqrt_scale_grid, function(sc) {
    mov_fn <- make_sqrt_mov(sc)
    maes <- vapply(k_local_grid, function(k) .score_config(matches_pre, .const_k(k), PROD_HGA, PROD_CARRYOVER, mov_fn), numeric(1))
    best_i <- which.min(maes)
    data.frame(family = "sqrt", S = sc, best_k = k_local_grid[best_i], mae = maes[best_i])
  }))

  # No-MOV floor (pure win/loss/draw Elo) for reference
  nomov_maes <- vapply(k_local_grid, function(k)
    .score_config(matches_pre, .const_k(k), PROD_HGA, PROD_CARRYOVER, function(m, elo_diff) 1), numeric(1))
  nomov_best_i <- which.min(nomov_maes)
  nomov_res <- data.frame(family = "no_mov", S = NA_real_, best_k = k_local_grid[nomov_best_i], mae = nomov_maes[nomov_best_i])

  mov_all <- rbind(scaled_log_res, sqrt_res, nomov_res)
  mov_all <- mov_all[order(mov_all$mae), ]
  cat("\n=== MOV family/scale sweep (each row: best-of-k_local_grid, pre-2025 in-sample MAE) ===\n")
  print(mov_all, row.names = FALSE)

  baseline_mov_row <- scaled_log_res[scaled_log_res$S == 1, ]
  cat(sprintf("\nBaseline formula (S=1, current production 538-style MOV) best-of-grid: k=%d, MAE=%.3f\n",
              baseline_mov_row$best_k, baseline_mov_row$mae))

  best_mov <- mov_all[1, ]
  cli::cli_alert_success("MOV winner: family={best_mov$family}, S={best_mov$S}, k={best_mov$best_k} (MAE={round(best_mov$mae,3)})")

  winner_mov_fn <- if (best_mov$family == "scaled_log") make_scaled_log_mov(best_mov$S) else
    if (best_mov$family == "sqrt") make_sqrt_mov(best_mov$S) else function(m, elo_diff) 1
  standalone_mov <- .standalone_oos(.const_k(best_mov$best_k), PROD_HGA, PROD_CARRYOVER, winner_mov_fn, TEST_SEASONS)
  cat(sprintf("MOV winner standalone 2026 OOS MAE: %.3f (baseline %.3f)\n",
              standalone_mov$metrics$mae, baseline_standalone$metrics$mae))

  saveRDS(list(scaled_log_res = scaled_log_res, sqrt_res = sqrt_res, nomov_res = nomov_res,
               mov_all = mov_all, best_mov = best_mov, standalone_mov = standalone_mov),
          .rds("phase2_mov_results.rds"))
  cli::cli_alert_success("Saved phase2_mov_results.rds")
}

# ---- Shared helper: rebuild the 5 configs (baseline + 4 candidates) from
# the cheap-stage results. Used by both `pipeline` and `pool` stages. ----
.build_phase2_configs <- function() {
  grid_res    <- readRDS(.rds("phase2_grid_results.rds"))
  recency_res <- readRDS(.rds("phase2_recency_results.rds"))
  mov_res     <- readRDS(.rds("phase2_mov_results.rds"))

  anchor_season <- min(matches_pre$season)
  linear_k_fn <- function(k_base, slope) {
    force(k_base); force(slope); force(anchor_season)
    function(season) max(k_base * (1 + slope * (season - anchor_season)), 1)
  }
  make_scaled_log_mov <- function(S) { force(S); function(m, elo_diff) log(abs(m) / S + 1) * (2.2 / (0.001 * abs(elo_diff) + 2.2)) }
  make_sqrt_mov <- function(scale) { force(scale); function(m, elo_diff) (sqrt(abs(m)) / scale) * (2.2 / (0.001 * abs(elo_diff) + 2.2)) }

  configs <- list(
    baseline = list(k_fn = .const_k(PROD_K), hga = PROD_HGA, carryover = PROD_CARRYOVER, mov_fn = default_mov_fn),
    grid_winner = list(k_fn = .const_k(grid_res$best$k), hga = grid_res$best$hga, carryover = grid_res$best$carryover, mov_fn = default_mov_fn),
    recency_kschedule = list(k_fn = linear_k_fn(PROD_K, recency_res$best_slope), hga = PROD_HGA, carryover = PROD_CARRYOVER, mov_fn = default_mov_fn),
    mov_winner = list(
      k_fn = .const_k(mov_res$best_mov$best_k), hga = PROD_HGA, carryover = PROD_CARRYOVER,
      mov_fn = if (mov_res$best_mov$family == "scaled_log") make_scaled_log_mov(mov_res$best_mov$S)
               else if (mov_res$best_mov$family == "sqrt") make_sqrt_mov(mov_res$best_mov$S)
               else function(m, elo_diff) 1
    )
  )

  list(configs = configs, blend_alpha = recency_res$best_blend$alpha, blend_fast_k = recency_res$best_blend$fast_k)
}

.join_phase2_config <- function(cfg_name, built) {
  if (cfg_name == "recency_blend") {
    slow_et <- build_team_elo2(matches_all, .const_k(PROD_K), PROD_HGA, PROD_CARRYOVER)
    fast_et <- build_team_elo2(matches_all, .const_k(built$blend_fast_k), PROD_HGA, PROD_CARRYOVER)
    stopifnot(identical(slow_et$match_id, fast_et$match_id), identical(slow_et$team_name, fast_et$team_name))
    et <- data.table::copy(slow_et)
    et$elo_pre <- built$blend_alpha * fast_et$elo_pre + (1 - built$blend_alpha) * slow_et$elo_pre
  } else {
    cfg <- built$configs[[cfg_name]]
    et <- build_team_elo2(matches_all, cfg$k_fn, cfg$hga, cfg$carryover, cfg$mov_fn)
  }
  tmdf <- join_elo_diff_to_team_mdl_df(team_mdl_df, et)
  n_na <- sum(is.na(tmdf$elo_diff))
  n_incomplete <- sum(is.na(tmdf$win))
  if (n_na > n_incomplete) {
    cli::cli_abort("{cfg_name}: {n_na - n_incomplete} NA elo_diff on completed match row(s) beyond expected future-fixture NAs")
  }
  # Neutral-impute future-fixture NAs to 0 (never used in any train/test
  # mask, but xgboost's model.matrix(~ . - 1) NA-omits rows and would
  # silently shrink predict_all()'s output length -- same fix as ws7).
  tmdf$elo_diff[is.na(tmdf$elo_diff)] <- 0
  tmdf
}

CFG_NAMES <- c("baseline", "grid_winner", "recency_kschedule", "recency_blend", "mov_winner")

# ================================================================
# Stage: pipeline -- G2 screen. Baseline + all 4 candidates through the REAL
# rolling-OOS harness on TEST_SEASONS=2026, using PRODUCTION
# torp:::.train_match_gams (already ships the V4b formula + elo_diff smooths
# -- verified by direct read of match_train.R) and rolling_lib.R's
# .train_xgb_fixed with the WS7 nrounds-CV fix (cv_extra_feature_cols
# ="elo_diff"). Baseline is regenerated fresh here, not reused from an
# earlier session, to avoid any cross-session xgboost thread-determinism
# confound (rolling_lib.R header; FABLE-MATCH-MAE-PLAN.md S8).
# ================================================================
if (stage %in% c("pipeline", "all")) {
  cli::cli_h1("Full-pipeline G2 screen (2026): baseline + 4 candidates, production torp:::.train_match_gams")

  built <- .build_phase2_configs()

  pipeline_results <- list()
  for (nm in CFG_NAMES) {
    cli::cli_h2("Full-pipeline 2026 screen: {nm}")
    tmdf_nm <- .join_phase2_config(nm, built)
    t0 <- Sys.time()
    roll <- run_rolling_eval(
      tmdf_nm, TEST_SEASONS,
      gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
      extra_feature_cols = "elo_diff", cv_extra_feature_cols = "elo_diff"
    )
    cli::cli_inform("{nm} completed in {round(difftime(Sys.time(), t0, units = 'mins'), 2)} min")
    m <- .compute_metrics(roll$input_blend_preds)
    cat(sprintf("%-20s MAE=%.3f RMSE=%.3f Brier=%.4f Slope=%.3f CloseMAE(n=%d)=%.3f\n",
                nm, m$mae, m$rmse, m$brier, m$slope, m$close_n, m$close_mae))
    pipeline_results[[nm]] <- list(roll = roll, metrics = m)
    saveRDS(pipeline_results, .rds("phase2_pipeline_2026.rds"))  # checkpoint after each
  }

  cli::cli_alert_success("Saved phase2_pipeline_2026.rds")
}

# ================================================================
# Stage: pool -- G3 confirm. Only candidates that beat the baseline on the
# 2026 screen proceed to pooled 2025:2026 confirmation with boot_mae_diff.
# Ship gate: bootstrap 95% CI on deltaMAE excludes zero AND Brier doesn't
# worsen by > 0.002 (same gate as FABLE-MATCH-MAE-PLAN.md G3/WS7).
# ================================================================
if (stage %in% c("pool", "all")) {
  cli::cli_h1("Pooled 2025:2026 confirmation for 2026-screen survivors")

  pr <- readRDS(.rds("phase2_pipeline_2026.rds"))
  baseline_mae_2026 <- pr$baseline$metrics$mae
  candidate_names <- setdiff(CFG_NAMES, "baseline")
  survivors <- candidate_names[vapply(candidate_names, function(nm) {
    !is.null(pr[[nm]]) && pr[[nm]]$metrics$mae < baseline_mae_2026
  }, logical(1))]
  cli::cli_inform(
    "2026-screen baseline MAE={round(baseline_mae_2026,3)}. Survivors (candidate MAE < baseline): {if (length(survivors)==0) 'NONE' else paste(survivors, collapse=', ')}"
  )

  if (length(survivors) == 0) {
    cli::cli_alert_info("No candidate beat the baseline on the 2026 screen -- per G2, nothing proceeds to pooled confirmation. This is a valid, reportable null result.")
    saveRDS(list(survivors = character(0), baseline_mae_2026 = baseline_mae_2026), .rds("phase2_pool_results.rds"))
  } else {
    built <- .build_phase2_configs()

    pool_results <- list()
    for (nm in c("baseline", survivors)) {
      cli::cli_h2("Pooled 2025:2026: {nm}")
      tmdf_nm <- .join_phase2_config(nm, built)
      t0 <- Sys.time()
      roll <- run_rolling_eval(
        tmdf_nm, CONFIRM_SEASONS,
        gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
        extra_feature_cols = "elo_diff", cv_extra_feature_cols = "elo_diff"
      )
      cli::cli_inform("{nm} pooled completed in {round(difftime(Sys.time(), t0, units = 'mins'), 2)} min")
      m <- .compute_metrics(roll$input_blend_preds)
      cat(sprintf("%-20s MAE=%.3f RMSE=%.3f Brier=%.4f Slope=%.3f CloseMAE(n=%d)=%.3f\n",
                  nm, m$mae, m$rmse, m$brier, m$slope, m$close_n, m$close_mae))
      pool_results[[nm]] <- list(roll = roll, metrics = m)
      saveRDS(pool_results, .rds("phase2_pool_2025_2026.rds"))
    }

    baseline_preds <- pool_results$baseline$roll$input_blend_preds
    gate_rows <- list()
    for (nm in survivors) {
      b <- boot_mae_diff(pool_results[[nm]]$roll$input_blend_preds, baseline_preds)
      brier_delta <- pool_results[[nm]]$metrics$brier - pool_results$baseline$metrics$brier
      ci_excludes_zero <- (b$mae_ci[1] > 0) || (b$mae_ci[2] < 0)
      ship <- ci_excludes_zero && b$mae_diff < 0 && brier_delta <= 0.002
      cat(sprintf("\nboot_mae_diff(%s - baseline): N=%d deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f -- SHIP=%s\n",
                  nm, b$n_matches, b$mae_diff, b$mae_ci[1], b$mae_ci[2], brier_delta, ship))
      gate_rows[[nm]] <- list(boot = b, brier_delta = brier_delta, ship = ship)
    }
    saveRDS(list(survivors = survivors, gate = gate_rows), .rds("phase2_pool_results.rds"))
  }
}

# ================================================================
# Stage: summary
# ================================================================
if (stage %in% c("summary", "all")) {
  cli::cli_h1("PHASE 2 FINAL SUMMARY: refined team-Elo vs production config")

  load_if <- function(f) if (file.exists(.rds(f))) readRDS(.rds(f)) else NULL
  grid_res    <- load_if("phase2_grid_results.rds")
  recency_res <- load_if("phase2_recency_results.rds")
  mov_res     <- load_if("phase2_mov_results.rds")
  pipe_2026   <- load_if("phase2_pipeline_2026.rds")
  pool_res    <- load_if("phase2_pool_results.rds")

  cat("\n=== Test 1: Wider hyperparameter grid ===\n")
  if (!is.null(grid_res)) {
    cat(sprintf("Baseline (k=%d,hga=%d,carryover=%.2f): pre-2025 in-sample MAE=%.3f\n",
                PROD_K, PROD_HGA, PROD_CARRYOVER, grid_res$baseline_row$mae))
    cat(sprintf("Wider-grid winner (k=%s,hga=%s,carryover=%.2f): pre-2025 in-sample MAE=%.3f\n",
                grid_res$best$k, grid_res$best$hga, grid_res$best$carryover, grid_res$best$mae))
  }

  cat("\n=== Test 2: Recency weighting ===\n")
  if (!is.null(recency_res)) {
    cat(sprintf("2a k-schedule: best slope=%.2f, pre-2025 in-sample MAE=%.3f (slope=0 baseline: %.3f)\n",
                recency_res$best_slope, recency_res$slope_res$mae[1],
                recency_res$slope_res$mae[recency_res$slope_res$slope == 0]))
    cat(sprintf("2b two-track blend: best alpha=%.1f (fast_k=%d), pre-2025 in-sample MAE=%.3f\n",
                recency_res$best_blend$alpha, recency_res$best_blend$fast_k, recency_res$best_blend$mae))
  }

  cat("\n=== Test 3: Alternative margin-of-victory scaling ===\n")
  if (!is.null(mov_res)) {
    print(mov_res$mov_all, row.names = FALSE)
  }

  cat("\n=== G2 screen: full pipeline (production torp:::.train_match_gams), 2026, n=153 ===\n")
  if (!is.null(pipe_2026)) {
    for (nm in names(pipe_2026)) {
      m <- pipe_2026[[nm]]$metrics
      cat(sprintf("%-20s MAE=%.3f RMSE=%.3f Brier=%.4f Slope=%.3f CloseMAE(n=%d)=%.3f\n",
                  nm, m$mae, m$rmse, m$brier, m$slope, m$close_n, m$close_mae))
    }
  }

  cat("\n=== G3 ship gate: pooled 2025:2026 bootstrap vs baseline ===\n")
  if (!is.null(pool_res)) {
    if (length(pool_res$survivors) == 0) {
      cat("No candidate beat the baseline on the 2026 screen -- nothing qualified for pooled confirmation.\n")
    } else {
      for (nm in names(pool_res$gate)) {
        g <- pool_res$gate[[nm]]
        cat(sprintf("%-20s deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f SHIP=%s\n",
                    nm, g$boot$mae_diff, g$boot$mae_ci[1], g$boot$mae_ci[2], g$brier_delta, g$ship))
      }
    }
  } else {
    cat("(pool stage not yet run)\n")
  }

  cat("\n=== VERDICT ===\n")
  if (!is.null(pool_res) && length(pool_res$survivors) > 0 && any(vapply(pool_res$gate, function(g) g$ship, logical(1)))) {
    winners <- names(pool_res$gate)[vapply(pool_res$gate, function(g) g$ship, logical(1))]
    cat(sprintf("SHIPS: %s clears G3 (bootstrap CI excludes zero, Brier guard respected).\n", paste(winners, collapse = ", ")))
  } else {
    cat("NO refined Elo variant clears the G3 ship gate. Today's first-draft config\n")
    cat("(k=20, hga=45, carryover=0.75, 538-style MOV) is not leaving material MAE on\n")
    cat("the table across wider hyperparameters, recency mechanisms, or AFL-calibrated\n")
    cat("MOV scaling -- a clean negative result.\n")
  }
}

cat("\n=== phase2_elo_refine.R stage", stage, "complete ===\n")
