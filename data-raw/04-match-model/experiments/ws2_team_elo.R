# ws2_team_elo.R — Team-Elo feature and baseline (FABLE-MATCH-MAE-PLAN.md WS2)
# ================================================================================
# (a) Standalone Elo baseline, scored through the rolling window (no GAM/XGB) --
#     THE most important number in the plan: does plain team-Elo beat torp's
#     2026 margin MAE outright?
# (b) Elo as a GAM/XGB feature variant (elo_diff joined onto team_mdl_df).
# (c) MATCH_WEIGHT_DECAY_DAYS recency sweep (300, 500 vs current 1000).
#
# G6: Elo hyperparameters (k, hga, carryover) are tuned via grid search on
# pre-2025 seasons only. G2: screen on TEST_SEASONS <- 2026.
#
# Run stage-by-stage to respect tool timeouts -- each stage checkpoints its
# output to experiments/results/*.rds immediately so partial progress
# survives an interrupted run:
#   Rscript ws2_team_elo.R data       # build + cache team_mdl_df
#   Rscript ws2_team_elo.R tune       # Elo grid tune + standalone baseline (cheap, fast)
#   Rscript ws2_team_elo.R champion   # default rolling harness (Input Blend) on 2026 -- expensive
#   Rscript ws2_team_elo.R feature    # Elo-as-GAM/XGB-feature rolling harness on 2026 -- expensive
#   Rscript ws2_team_elo.R decay      # MATCH_WEIGHT_DECAY_DAYS=300/500 rolling harness runs -- expensive
#   Rscript ws2_team_elo.R summary    # aggregate all of the above, print final report
#   Rscript ws2_team_elo.R all        # (default) run everything in sequence

stage <- {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) >= 1) a[1] else "all"
}
cat("=== ws2_team_elo.R stage:", stage, "===\n")

# Locate this script's own directory regardless of the caller's working
# directory (avoids setwd(), which segfaults under some Rscript invocations
# -- r-datatable-gotchas.md), by re-using the same candidate-path search as
# the rolling_lib.R / elo_lib.R sourcing below.
.find_experiments_dir <- function() {
  cands <- c(
    "experiments",
    "04-match-model/experiments",
    "data-raw/04-match-model/experiments",
    "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
  )
  hit <- cands[file.exists(file.path(cands, "rolling_lib.R"))]
  if (length(hit) == 0) stop("Cannot locate experiments/ directory (rolling_lib.R not found via any candidate path)")
  normalizePath(hit[1])
}
EXPERIMENTS_DIR <- .find_experiments_dir()
RESULTS_DIR <- file.path(EXPERIMENTS_DIR, "results")
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR, recursive = TRUE)
.rds <- function(name) file.path(RESULTS_DIR, name)

TEST_SEASONS    <- 2026      # G2 screen
CONFIRM_SEASONS <- 2025:2026 # confirmation window for any real candidate

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

source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))
source(file.path(EXPERIMENTS_DIR, "elo_lib.R"))

# ---- Stage: data ----
build_or_load_team_mdl_df <- function() {
  cache_path <- .rds("team_mdl_df_cache.rds")
  if (file.exists(cache_path) && Sys.getenv("WS2_REBUILD_DATA", "0") != "1") {
    cli::cli_inform("Loading cached team_mdl_df from {cache_path}")
    return(readRDS(cache_path))
  }
  cli::cli_h1("Building Match Prediction Training Data")
  tictoc::tic("build_team_mdl_df")
  team_mdl_df <- build_team_mdl_df()
  tictoc::toc()
  saveRDS(team_mdl_df, cache_path)
  team_mdl_df
}

if (stage %in% c("data", "all", "tune", "champion", "feature", "decay")) {
  team_mdl_df <- build_or_load_team_mdl_df()
  cli::cli_inform("Seasons: {paste(sort(unique(team_mdl_df$season.x)), collapse = ', ')}")
}

# ---- Stage: tune (Elo grid + standalone baseline) ----
if (stage %in% c("tune", "all")) {
  cli::cli_h1("WS2(a): Elo hyperparameter grid tune (pre-2025 only, G6)")

  matches_all <- .matches_from_team_mdl_df(team_mdl_df)
  cli::cli_inform("Match universe for Elo: {nrow(matches_all)} completed matches, seasons {paste(range(matches_all$season), collapse='-')}")

  matches_pre <- matches_all[matches_all$season < 2025, ]
  cli::cli_inform("Pre-2025 tuning set: {nrow(matches_pre)} matches")

  tune_result <- tune_team_elo(
    matches_pre,
    k_grid = c(15, 20, 30), hga_grid = c(25, 35, 45), carryover_grid = c(0.6, 0.75, 0.9),
    mov_mult = TRUE
  )
  cat("\n=== Elo hyperparameter grid (sorted by pre-2025 MAE) ===\n")
  print(head(tune_result, 10), row.names = FALSE)

  best <- tune_result[1, ]
  cli::cli_alert_success("Best Elo hyperparameters: k={best$k}, hga={best$hga}, carryover={best$carryover} (pre-2025 MAE={round(best$mae, 3)})")

  # Final Elo table over the FULL match history using the tuned combo --
  # leak-safe by construction (elo_pre only reflects strictly-earlier matches).
  elo_table <- build_team_elo(matches_all, k = best$k, hga = best$hga,
                               carryover = best$carryover, mov_mult = TRUE)
  saveRDS(list(elo_table = elo_table, best = best, tune_result = tune_result,
               matches_all = matches_all, matches_pre = matches_pre),
          .rds("ws2_elo_table.rds"))

  # Fit points-per-Elo scale on pre-2025 matches only (G6)
  ex <- stats::setNames(elo_table$elo_pre, paste(elo_table$match_id, elo_table$team_name))
  elo_diff_home_pre <- unname(ex[paste(matches_pre$match_id, matches_pre$home_team)] -
                                 ex[paste(matches_pre$match_id, matches_pre$away_team)])
  fit_m <- fit_elo_margin_scale(elo_diff_home_pre, best$hga, matches_pre$home_margin)
  cli::cli_inform("Points-per-Elo scale (pre-2025 fit): {round(coef(fit_m)[[1]], 4)}")
  saveRDS(fit_m, .rds("ws2_fit_m.rds"))

  .standalone_elo_preds <- function(matches_subset) {
    elo_h <- unname(ex[paste(matches_subset$match_id, matches_subset$home_team)])
    elo_a <- unname(ex[paste(matches_subset$match_id, matches_subset$away_team)])
    elo_diff_home <- elo_h - elo_a
    pred_margin <- stats::predict(fit_m, newdata = data.frame(elo_diff_hga = elo_diff_home + best$hga))
    pred_win <- elo_pred_win(elo_diff_home, best$hga)
    data.frame(
      match_id = matches_subset$match_id,
      pred_margin = unname(pred_margin),
      pred_win = pred_win,
      margin = matches_subset$home_margin,
      home_win = ifelse(matches_subset$home_margin > 0, 1,
                         ifelse(matches_subset$home_margin == 0, 0.5, 0))
    )
  }

  standalone_2026 <- .standalone_elo_preds(matches_all[matches_all$season %in% TEST_SEASONS, ])
  standalone_confirm <- .standalone_elo_preds(matches_all[matches_all$season %in% CONFIRM_SEASONS, ])

  m_2026 <- .compute_metrics(standalone_2026)
  m_confirm <- .compute_metrics(standalone_confirm)

  cat("\n", strrep("=", 78), "\n", sep = "")
  cat("=== WS2(a) STANDALONE TEAM-ELO BASELINE -- THE headline number ===\n")
  cat(strrep("=", 78), "\n")
  cat(sprintf("2026 screen        (N=%d): MAE=%.2f RMSE=%.2f Brier=%.4f Slope=%.3f Cor=%.3f SDRatio=%.3f CloseMAE(n=%d)=%.2f\n",
              nrow(standalone_2026), m_2026$mae, m_2026$rmse, m_2026$brier, m_2026$slope,
              m_2026$cor, m_2026$sd_ratio, m_2026$close_n, m_2026$close_mae))
  cat(sprintf("2025-2026 confirm  (N=%d): MAE=%.2f RMSE=%.2f Brier=%.4f Slope=%.3f Cor=%.3f SDRatio=%.3f CloseMAE(n=%d)=%.2f\n",
              nrow(standalone_confirm), m_confirm$mae, m_confirm$rmse, m_confirm$brier, m_confirm$slope,
              m_confirm$cor, m_confirm$sd_ratio, m_confirm$close_n, m_confirm$close_mae))
  cat("Reference: torp submitted-tips 2026 MAE = 27.09 (2026-MATCH-MAE-DIAGNOSIS.md); Wheelo (best) 24.89\n")
  cat(strrep("=", 78), "\n")

  saveRDS(list(standalone_2026 = standalone_2026, standalone_confirm = standalone_confirm,
               m_2026 = m_2026, m_confirm = m_confirm, best = best),
          .rds("ws2_standalone.rds"))
}

# ---- Stage: champion (default harness baseline, our own G4 reproduction) ----
if (stage %in% c("champion", "all")) {
  cli::cli_h1("Champion baseline: default rolling harness on TEST_SEASONS (G4 reproduction)")
  tictoc::tic("champion_run")
  roll_champion <- run_rolling_eval(team_mdl_df, TEST_SEASONS)
  tictoc::toc()
  saveRDS(roll_champion, .rds("ws2_champion_roll.rds"))

  champ_m <- list(
    gam   = .compute_metrics(roll_champion$gam_preds),
    xgb   = .compute_metrics(roll_champion$xgb_preds),
    outb  = .compute_metrics(roll_champion$blend_preds),
    inb   = .compute_metrics(roll_champion$input_blend_preds)
  )
  cat("\n=== Champion baseline (2026 screen) ===\n")
  for (nm in names(champ_m)) {
    m <- champ_m[[nm]]
    cat(sprintf("%-10s N=%d MAE=%.2f RMSE=%.2f Brier=%.4f Slope=%.3f SDRatio=%.3f CloseMAE=%.2f\n",
                nm, nrow(roll_champion$gam_preds), m$mae, m$rmse, m$brier, m$slope, m$sd_ratio, m$close_mae))
  }
  saveRDS(champ_m, .rds("ws2_champion_metrics.rds"))
}

# ---- Stage: feature (Elo as GAM/XGB feature) ----
if (stage %in% c("feature", "all")) {
  cli::cli_h1("WS2(b): Elo as GAM/XGB feature")

  elo_stuff <- readRDS(.rds("ws2_elo_table.rds"))
  elo_table <- elo_stuff$elo_table

  team_mdl_df_elo <- join_elo_diff_to_team_mdl_df(team_mdl_df, elo_table)
  n_na <- sum(is.na(team_mdl_df_elo$elo_diff))
  n_incomplete <- sum(is.na(team_mdl_df_elo$win))
  cli::cli_inform("elo_diff joined: {sum(!is.na(team_mdl_df_elo$elo_diff))} of {nrow(team_mdl_df_elo)} rows non-NA ({n_na} NA; {n_incomplete} rows are incomplete/future fixtures)")
  if (n_na > 0) {
    if (n_na > n_incomplete) {
      cli::cli_abort("NA elo_diff on {n_na - n_incomplete} COMPLETED match row(s) beyond the {n_incomplete} expected future-fixture NAs -- likely a team-name join mismatch, investigate before proceeding")
    }
    # Expected: future/incomplete fixtures have no Elo history yet (never in
    # train_filter or test_mask, so this is inert for scoring) -- unlike the
    # GAM path (predict.gam silently NA-preserves row count), xgboost's
    # model.matrix(~ . - 1) NA-omits rows, which would silently shrink the
    # predict_all() output vs team_mdl_df and corrupt the column assignment.
    # Neutral-impute so .train_xgb_fixed()'s predict_all() keeps full length.
    cli::cli_inform("Neutral-imputing {n_na} future-fixture NA elo_diff rows to 0 (never used in train/test masks)")
    team_mdl_df_elo$elo_diff[is.na(team_mdl_df_elo$elo_diff)] <- 0
  }

  # .train_match_gams_elo: exact copy of torp:::.train_match_gams (match_train.R)
  # with elo_diff added as an optional smooth to models 2 and 4 (plan WS2 step 3).
  # Copied per plan G5 (no production torp/R/*.R edits in this pass).
  .train_match_gams_elo <- function(team_mdl_df, train_filter = NULL, nthreads = 4L, gamma_arg = 1.4) {
    loadNamespace("mgcv")

    if (is.null(train_filter)) {
      train_mask <- !is.na(team_mdl_df$win)
    } else {
      train_mask <- train_filter & !is.na(team_mdl_df$win)
    }

    gam_df <- team_mdl_df[train_mask, ]
    cli::cli_inform("Training on {nrow(gam_df)} completed matches")
    if (nrow(gam_df) == 0) {
      cli::cli_abort("Cannot train GAM models: 0 completed matches after filtering")
    }

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
      "ti(psr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)" = list(var = "psr_diff", k = 4),
      # WS2 addition: elo_diff optional smooth (models 2 and 4)
      "s(elo_diff, bs = \"ts\", k = 5)"        = list(var = "elo_diff", k = 5)
    )
    drop_terms <- character(0)
    for (term_str in names(optional_smooth_terms)) {
      info <- optional_smooth_terms[[term_str]]
      vals <- gam_df[[info$var]]
      n_unique <- length(unique(vals[!is.na(vals)]))
      if (n_unique < info$k) {
        drop_terms <- c(drop_terms, term_str)
        cli::cli_warn("Dropping smooth {.code {term_str}}: only {n_unique} unique value{?s} (need >= {info$k})")
      }
    }

    .add_optional <- function(base_terms, optional_terms) {
      keep <- setdiff(optional_terms, drop_terms)
      if (length(keep) > 0) {
        paste(base_terms, "+", paste(keep, collapse = " + "))
      } else {
        base_terms
      }
    }

    # Model 1: Total expected points (unchanged -- elo_diff not added here per plan)
    cli::cli_progress_step("Training total xPoints model")
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
      m1_formula, data = gam_df, weights = gam_df$weightz,
      family = gaussian(), nthreads = nthreads, select = TRUE, discrete = TRUE,
      drop.unused.levels = FALSE, gamma = gamma_arg
    )
    team_mdl_df$gam_pred_tot_xscore <- predict(afl_total_xpoints_mdl, newdata = team_mdl_df, type = "response")

    # Model 2: xScore differential -- WS2 adds elo_diff
    cli::cli_progress_step("Training xScore diff model")
    gam_df$gam_pred_tot_xscore <- team_mdl_df$gam_pred_tot_xscore[train_mask]
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
                      "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)",
                      "s(elo_diff, bs = \"ts\", k = 5)")
    m2_formula <- stats::as.formula(.add_optional(m2_base, m2_optional))

    afl_xscore_diff_mdl <- mgcv::bam(
      m2_formula, data = gam_df, weights = gam_df$weightz,
      family = gaussian(), nthreads = nthreads, select = TRUE, discrete = TRUE,
      drop.unused.levels = FALSE, gamma = gamma_arg
    )
    team_mdl_df$gam_pred_xscore_diff <- predict(afl_xscore_diff_mdl, newdata = team_mdl_df, type = "response")

    # Model 3: Conversion differential (unchanged)
    cli::cli_progress_step("Training conversion model")
    gam_df$gam_pred_xscore_diff <- team_mdl_df$gam_pred_xscore_diff[train_mask]
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

    afl_conv_mdl <- mgcv::bam(
      m3_formula, data = gam_df, weights = gam_df$shot_weightz,
      family = gaussian(), nthreads = nthreads, select = TRUE, discrete = TRUE,
      drop.unused.levels = FALSE, gamma = gamma_arg
    )
    team_mdl_df$gam_pred_conv_diff <- predict(afl_conv_mdl, newdata = team_mdl_df, type = "response")

    # Model 4: Score differential -- WS2 adds elo_diff
    cli::cli_progress_step("Training score diff model")
    gam_df$gam_pred_conv_diff <- team_mdl_df$gam_pred_conv_diff[train_mask]
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
                      "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)",
                      "s(elo_diff, bs = \"ts\", k = 5)")
    m4_formula <- stats::as.formula(.add_optional(m4_base, m4_optional))

    afl_score_mdl <- mgcv::bam(
      m4_formula, data = gam_df, weights = gam_df$weightz,
      family = "gaussian", nthreads = nthreads, select = TRUE, discrete = TRUE,
      drop.unused.levels = FALSE, gamma = gamma_arg
    )
    team_mdl_df$gam_pred_score_diff <- predict(afl_score_mdl, newdata = team_mdl_df, type = "response")

    # Model 5: Win probability (unchanged)
    cli::cli_progress_step("Training win probability model")
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

    # Scoring metrics (unchanged from torp:::.train_match_gams -- .format_match_preds()
    # selects `bits`, so this block is required, not just diagnostic)
    team_mdl_df$bits <- dplyr::case_when(
      team_mdl_df$win == 1   ~ 1 + log2(team_mdl_df$pred_win),
      team_mdl_df$win == 0   ~ 1 + log2(1 - team_mdl_df$pred_win),
      TRUE                   ~ 1 + 0.5 * log2(team_mdl_df$pred_win * (1 - team_mdl_df$pred_win))
    )
    team_mdl_df$tips <- dplyr::case_when(
      round(team_mdl_df$pred_win) == team_mdl_df$win ~ 1,
      team_mdl_df$win == 0.5                         ~ 1,
      TRUE                                            ~ 0
    )
    team_mdl_df$mae <- abs(team_mdl_df$score_diff - team_mdl_df$pred_score_diff)

    models <- list(
      total_xpoints = afl_total_xpoints_mdl, xscore_diff = afl_xscore_diff_mdl,
      conv_diff = afl_conv_mdl, score_diff = afl_score_mdl, win = afl_win_mdl
    )
    cli::cli_alert_success("GAM+Elo pipeline trained on {nrow(gam_df)} matches")
    list(models = models, data = team_mdl_df)
  }

  tictoc::tic("elo_feature_run")
  roll_elo_feature <- run_rolling_eval(
    team_mdl_df_elo, TEST_SEASONS,
    gam_trainer = .train_match_gams_elo,
    xgb_trainer = .train_xgb_fixed,
    extra_feature_cols = "elo_diff"
  )
  tictoc::toc()
  saveRDS(roll_elo_feature, .rds("ws2_feature_roll.rds"))

  feat_m <- list(
    gam  = .compute_metrics(roll_elo_feature$gam_preds),
    xgb  = .compute_metrics(roll_elo_feature$xgb_preds),
    outb = .compute_metrics(roll_elo_feature$blend_preds),
    inb  = .compute_metrics(roll_elo_feature$input_blend_preds)
  )
  cat("\n=== WS2(b) Elo-as-feature (2026 screen) ===\n")
  for (nm in names(feat_m)) {
    m <- feat_m[[nm]]
    cat(sprintf("%-10s N=%d MAE=%.2f RMSE=%.2f Brier=%.4f Slope=%.3f SDRatio=%.3f CloseMAE=%.2f\n",
                nm, nrow(roll_elo_feature$gam_preds), m$mae, m$rmse, m$brier, m$slope, m$sd_ratio, m$close_mae))
  }
  saveRDS(feat_m, .rds("ws2_feature_metrics.rds"))

  if (file.exists(.rds("ws2_champion_roll.rds"))) {
    roll_champion <- readRDS(.rds("ws2_champion_roll.rds"))
    boot_inb <- boot_mae_diff(roll_elo_feature$input_blend_preds, roll_champion$input_blend_preds, B = 2000)
    cat(sprintf("\nboot_mae_diff(Elo-feature InputBlend - Champion InputBlend): N=%d diff=%.3f 95%%CI[%.3f, %.3f]\n",
                boot_inb$n_matches, boot_inb$mae_diff, boot_inb$mae_ci[1], boot_inb$mae_ci[2]))
    saveRDS(boot_inb, .rds("ws2_feature_vs_champion_boot.rds"))
  }
}

# ---- Stage: decay (MATCH_WEIGHT_DECAY_DAYS sweep) ----
if (stage %in% c("decay", "all")) {
  cli::cli_h1("WS2(c): MATCH_WEIGHT_DECAY_DAYS recency sweep (300, 500 vs current 1000)")

  current_decay <- tryCatch(MATCH_WEIGHT_DECAY_DAYS, error = function(e) 1000)
  cli::cli_inform("Current production MATCH_WEIGHT_DECAY_DAYS = {current_decay}")

  anchor_date <- max(as.Date(team_mdl_df$utc_start_time), na.rm = TRUE)
  cli::cli_inform("Weight anchor date (proxy = max match date in team_mdl_df): {anchor_date}")

  .reweight_team_mdl_df <- function(df, decay_days, anchor_date) {
    df$weightz <- exp(-(as.numeric(anchor_date - as.Date(df$utc_start_time))) / decay_days)
    df$weightz <- df$weightz / mean(df$weightz, na.rm = TRUE)
    df$shot_weightz <- (df$harmean_shots / mean(df$harmean_shots, na.rm = TRUE)) * df$weightz
    df
  }

  decay_results <- list()
  for (dd in c(300, 500)) {
    cli::cli_h2("Decay = {dd} days")
    team_mdl_df_dd <- .reweight_team_mdl_df(team_mdl_df, dd, anchor_date)
    tictoc::tic(paste0("decay_", dd))
    roll_dd <- run_rolling_eval(team_mdl_df_dd, TEST_SEASONS)
    tictoc::toc()
    saveRDS(roll_dd, .rds(paste0("ws2_decay_", dd, "_roll.rds")))

    m_dd <- list(
      gam  = .compute_metrics(roll_dd$gam_preds),
      xgb  = .compute_metrics(roll_dd$xgb_preds),
      outb = .compute_metrics(roll_dd$blend_preds),
      inb  = .compute_metrics(roll_dd$input_blend_preds)
    )
    cat(sprintf("\n=== Decay=%d Input Blend (2026 screen) ===\n", dd))
    m <- m_dd$inb
    cat(sprintf("N=%d MAE=%.2f RMSE=%.2f Brier=%.4f Slope=%.3f SDRatio=%.3f CloseMAE=%.2f\n",
                nrow(roll_dd$input_blend_preds), m$mae, m$rmse, m$brier, m$slope, m$sd_ratio, m$close_mae))

    if (file.exists(.rds("ws2_champion_roll.rds"))) {
      roll_champion <- readRDS(.rds("ws2_champion_roll.rds"))
      boot_dd <- boot_mae_diff(roll_dd$input_blend_preds, roll_champion$input_blend_preds, B = 2000)
      cat(sprintf("boot_mae_diff(Decay=%d InputBlend - Champion InputBlend): N=%d diff=%.3f 95%%CI[%.3f, %.3f]\n",
                  dd, boot_dd$n_matches, boot_dd$mae_diff, boot_dd$mae_ci[1], boot_dd$mae_ci[2]))
      decay_results[[as.character(dd)]] <- list(metrics = m_dd, boot = boot_dd)
    } else {
      decay_results[[as.character(dd)]] <- list(metrics = m_dd, boot = NULL)
    }
  }
  saveRDS(decay_results, .rds("ws2_decay_metrics.rds"))
}

# ---- Stage: summary ----
if (stage %in% c("summary", "all")) {
  cli::cli_h1("WS2 Final Summary")

  load_if <- function(f) if (file.exists(.rds(f))) readRDS(.rds(f)) else NULL

  standalone   <- load_if("ws2_standalone.rds")
  champ_m      <- load_if("ws2_champion_metrics.rds")
  feat_m       <- load_if("ws2_feature_metrics.rds")
  feat_boot    <- load_if("ws2_feature_vs_champion_boot.rds")
  decay_m      <- load_if("ws2_decay_metrics.rds")

  rows <- list()
  if (!is.null(standalone)) {
    rows[["Elo standalone (2026)"]] <- standalone$m_2026
    rows[["Elo standalone (2025-2026 confirm)"]] <- standalone$m_confirm
  }
  if (!is.null(champ_m)) rows[["Champion Input Blend (2026)"]] <- champ_m$inb
  if (!is.null(feat_m)) rows[["Elo-feature Input Blend (2026)"]] <- feat_m$inb
  if (!is.null(decay_m)) {
    for (nm in names(decay_m)) rows[[paste0("Decay=", nm, " Input Blend (2026)")]] <- decay_m[[nm]]$metrics$inb
  }

  if (length(rows) > 0) {
    summary_df <- do.call(rbind, lapply(names(rows), function(nm) {
      m <- rows[[nm]]
      data.frame(Variant = nm, MAE = round(m$mae, 2), RMSE = round(m$rmse, 2),
                 Brier = round(m$brier, 4), Slope = round(m$slope, 3),
                 SDRatio = round(m$sd_ratio, 3), Cor = round(m$cor, 3),
                 CloseMAE = round(m$close_mae, 2), stringsAsFactors = FALSE)
    }))
    cat("\n=== WS2 Comparison Table (2026 screen unless noted) ===\n")
    print(summary_df, row.names = FALSE)
    write.csv(summary_df, .rds("ws2_summary_table.csv"), row.names = FALSE)
  } else {
    cli::cli_warn("No stage outputs found -- run earlier stages first.")
  }

  # Falsifier #3 (plan §6): does plain Elo beat the harness champion outright?
  # The decision-relevant comparison is standalone Elo vs the HARNESS champion
  # (this script's own Input Blend reproduction), not vs the submitted tips
  # (which carry serve-path effects) -- compute and print explicitly.
  if (!is.null(standalone) && !is.null(champ_m)) {
    roll_champion <- load_if("ws2_champion_roll.rds")
    if (!is.null(roll_champion)) {
      champ_2026 <- roll_champion$input_blend_preds
      boot_standalone_vs_champ <- boot_mae_diff(standalone$standalone_2026, champ_2026, B = 2000)
      cat(sprintf(
        "\n=== FALSIFIER CHECK: Standalone Elo vs Champion Input Blend (2026, N=%d matched) ===\n",
        boot_standalone_vs_champ$n_matches
      ))
      cat(sprintf("diff (Elo - Champion) = %.3f MAE, 95%% CI [%.3f, %.3f] %s\n",
                  boot_standalone_vs_champ$mae_diff,
                  boot_standalone_vs_champ$mae_ci[1], boot_standalone_vs_champ$mae_ci[2],
                  if (boot_standalone_vs_champ$mae_ci[2] < 0) "-- CI EXCLUDES 0: plain Elo beats champion outright (priority inverts per plan S6.3)" else
                    if (boot_standalone_vs_champ$mae_ci[1] > 0) "-- CI EXCLUDES 0: champion beats plain Elo" else
                      "-- CI overlaps 0: inconclusive"))
      saveRDS(boot_standalone_vs_champ, .rds("ws2_standalone_vs_champion_boot.rds"))
    }
  }

  if (!is.null(feat_boot)) {
    cat(sprintf("\nElo-feature vs Champion: diff=%.3f 95%%CI[%.3f, %.3f] %s\n",
                feat_boot$mae_diff, feat_boot$mae_ci[1], feat_boot$mae_ci[2],
                if (feat_boot$mae_ci[2] < 0) "-- CI EXCLUDES 0, Elo-feature wins" else
                  if (feat_boot$mae_ci[1] > 0) "-- CI EXCLUDES 0, Elo-feature LOSES" else
                    "-- CI overlaps 0, inconclusive"))
  }
  if (!is.null(decay_m)) {
    for (nm in names(decay_m)) {
      b <- decay_m[[nm]]$boot
      if (!is.null(b)) {
        cat(sprintf("Decay=%s vs Champion: diff=%.3f 95%%CI[%.3f, %.3f] %s\n",
                    nm, b$mae_diff, b$mae_ci[1], b$mae_ci[2],
                    if (b$mae_ci[2] < 0) "-- CI EXCLUDES 0, decay wins" else
                      if (b$mae_ci[1] > 0) "-- CI EXCLUDES 0, decay LOSES" else
                        "-- CI overlaps 0, inconclusive"))
      }
    }
  }
}

cat("\n=== ws2_team_elo.R stage", stage, "complete ===\n")
