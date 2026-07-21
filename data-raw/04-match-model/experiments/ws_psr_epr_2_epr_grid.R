# ws_psr_epr_2_epr_grid.R -- WS2: EPR aggregation-parameter sensitivity grid
# (measurement, not optimization), per docs/plans/FABLE-PSR-EPR-PLAN.md
# Section 2 WS2 + Section 6 falsifier 4.
# =====================================================================
# Hypothesis to MEASURE, not assume: rolling-OOS margin MAE is sensitive to
# EPR's aggregation-time hyperparameters (per-component decays 273/630/523/545;
# prior_games=3). If not, the April optimizer's constants are ratified for the
# match-model purpose and "re-run the optimizer against the harness" closes
# for free -- a flat grid is a FULLY SUCCESSFUL outcome (it closes the plan).
#
# Pre-registered 3x3 grid (closed, plan Section 2 WS2):
#   decay multiplier m in {0.7, 1.0, 1.3} applied jointly to all four
#     component decays (EPR_DECAY_RECV/DISP/SPOIL/HITOUT = 273/630/523/545)
#   prior_games p in {3, 6, 9} applied jointly to all four components
#     (3 is the current value AND the April optimizer's lower bound -- this
#     grid deliberately probes the direction the bound may have censored)
# (m=1.0, p=3) is the CONTROL cell -- current production constants exactly.
# Scope: aggregation-time params ONLY. EPV credit weights (player_credit.R,
# baked into released player_game_data at daily-pipeline time) are explicitly
# OUT OF SCOPE (plan non-goal 4) -- untouched here.
#
# Design (binding, plan Section 2 WS2 + Section 6 falsifier 4):
#   1. Control-cell fidelity FIRST (hard stop): (m=1.0, p=3) must reproduce
#      the fresh WS0 baseline team_mdl_df's EPR columns to numerical noise --
#      same logic as WS1's provenance check. If it doesn't, STOP and report.
#   2. Per cell: rebuild EPR history via calculate_epr_stats_batch() over all
#      (season, round) cutoffs (2021-current) -> .prepare_final_dataframe()
#      per round (mirrors torp/data-raw/03-ratings/run_ratings_pipeline.R's
#      get_epr_df(), extracted+parameterised locally per G5) -> rebuild
#      team_mdl_df (component diffs, epr_diff, torp.x/.y recompute
#      automatically in .build_team_mdl_df()) -> 2026 screen ONLY (G2 permits
#      screen-only for sensitivity grids; only an interesting cell gets a
#      pooled confirm).
#   3. Pooled confirmation ONLY for a cell with screen deltaMAE <= -0.2 vs the
#      fresh baseline (25.132 screen, +V1a recal).
#   4. Section 6.4 flatness falsifier: if the screen MAE range across all 8
#      non-control cells is smaller than the baseline's own bootstrap MAE CI
#      width, EPR aggregation params are NOT a margin lever at team-sum
#      granularity -- ratify the April constants, no optimizer re-run
#      authorized.
#   5. PSR is untouched throughout (WS1's default psr_df, shared across every
#      cell) -- this workstream varies EPR aggregation only.
#
# Run stage-by-stage (checkpoints to experiments/results/ws2_*):
#   Rscript ws_psr_epr_2_epr_grid.R data                # load+cache shared inputs (pgd, stat_ratings, fixtures, psr_df)
#   Rscript ws_psr_epr_2_epr_grid.R control              # control cell (m=1.0,p=3): fidelity check, HARD STOP if it fails
#   Rscript ws_psr_epr_2_epr_grid.R cell_c1 ... cell_c8   # 8 non-control cells, corners first
#   Rscript ws_psr_epr_2_epr_grid.R confirm_<key>          # pooled confirm, only if warranted
#   Rscript ws_psr_epr_2_epr_grid.R summary                 # full table + falsifier verdicts + closure evidence

stage <- {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) >= 1) a[1] else "all"
}
cat("=== ws_psr_epr_2_epr_grid.R stage:", stage, "===\n")

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
TORP_PATH <- NULL
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p, quiet = TRUE)
    TORP_PATH <- normalizePath(p)
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

# ---- V1a recal + printers (copied verbatim, plan G5) ----
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

# ---- Raw single-MAE block bootstrap CI (Section 6.4 flatness falsifier
# needs "the control's [baseline's] own bootstrap CI width", not a delta) ----
.boot_mae_ci <- function(preds, B = 2000, seed = 1234) {
  mae_i <- abs(preds$pred_margin - preds$margin)
  match_id <- preds$match_id
  n_ids <- length(unique(match_id))
  ids <- unique(match_id)
  withr::with_seed(seed, {
    boots <- replicate(B, {
      s <- sample(ids, n_ids, replace = TRUE)
      mean(mae_i[match(s, match_id)])
    })
  })
  ci <- stats::quantile(boots, c(0.025, 0.975))
  list(mae = mean(mae_i), ci = ci, width = unname(ci[2] - ci[1]))
}

# ================================================================
# .get_epr_df_variant(): local extraction of
# torp/data-raw/03-ratings/run_ratings_pipeline.R's get_epr_df(), the
# production full-history EPR builder, parameterised by decay/prior_games
# (plan G5: extracted into the experiment file, torp/R/*.R untouched).
# Identical logic to production: calculate_epr_stats_batch() over all
# rounds in one vectorised pass, then per-round roster join + TOG-weighted
# centering via .prepare_final_dataframe().
# ================================================================
.get_epr_df_variant <- function(year, rounds, pgd, stat_ratings, fixtures,
                                 decay_recv, decay_disp, decay_spoil, decay_hitout,
                                 prior_games_recv, prior_games_disp, prior_games_spoil, prior_games_hitout,
                                 prior_rate_recv = EPR_PRIOR_RATE_RECV, prior_rate_disp = EPR_PRIOR_RATE_DISP,
                                 prior_rate_spoil = EPR_PRIOR_RATE_SPOIL, prior_rate_hitout = EPR_PRIOR_RATE_HITOUT,
                                 loading = EPR_LOADING_DEFAULT) {
  plyr_tm_df <- load_player_details(year)
  if (nrow(plyr_tm_df) == 0 || !"season" %in% names(plyr_tm_df)) {
    plyr_tm_df <- load_player_details(year - 1)
  }

  fix_dt <- data.table::as.data.table(fixtures)
  fix_dates <- fix_dt[
    season == year & round_number %in% rounds,
    .(date_val = lubridate::as_date(min(utc_start_time))),
    by = .(round_val = round_number)
  ]

  round_info <- data.table::data.table(
    round_val = rounds,
    match_ref = paste0("CD_M", year, "014", sprintf("%02d", rounds))
  )
  round_info <- round_info[fix_dates, on = "round_val", nomatch = NULL]

  if (nrow(round_info) == 0) {
    cli::cli_alert_danger("No fixtures found for {year}")
    return(data.frame())
  }

  batch_stats <- calculate_epr_stats_batch(
    pgd, round_info,
    decay_recv = decay_recv, decay_disp = decay_disp, decay_spoil = decay_spoil, decay_hitout = decay_hitout,
    loading = loading,
    prior_games_recv = prior_games_recv, prior_games_disp = prior_games_disp,
    prior_games_spoil = prior_games_spoil, prior_games_hitout = prior_games_hitout,
    prior_rate_recv = prior_rate_recv, prior_rate_disp = prior_rate_disp,
    prior_rate_spoil = prior_rate_spoil, prior_rate_hitout = prior_rate_hitout
  )

  batch_stats[, pred_tog := NA_real_]
  batch_stats[, pred_selection := NA_real_]
  batch_stats[, pred_cond_tog := NA_real_]
  if (!is.null(stat_ratings)) {
    stat_ratings_dt <- data.table::as.data.table(stat_ratings)
    batch_stats[stat_ratings_dt, `:=`(
      pred_selection = i.squad_selection_rating,
      pred_cond_tog = i.cond_tog_rating
    ), on = "player_id"]
    batch_stats[is.na(pred_selection), pred_selection := 0]
    batch_stats[is.na(pred_cond_tog), pred_cond_tog := 0]
    batch_stats[, pred_tog := pred_selection * pred_cond_tog]
  }

  fix_summary <- fixtures |>
    dplyr::group_by(season = .data$season, round = .data$round_number) |>
    dplyr::summarise(ref_date = lubridate::as_date(min(.data$utc_start_time)), .groups = "drop")

  results <- lapply(round_info$round_val, function(rv) {
    round_dt <- batch_stats[round_val == rv]
    final_df <- .prepare_final_dataframe(plyr_tm_df, round_dt, year, rv, fixtures, fix_summary = fix_summary)

    if (!is.null(stat_ratings) && nrow(final_df) > 0) {
      final_df$pred_tog[is.na(final_df$pred_tog)] <- 0
      tot_tog <- sum(final_df$pred_tog)
      if (tot_tog > 0) {
        n_teams <- length(unique(final_df$team))
        target_tog <- n_teams * 18L
        final_df$pred_tog <- final_df$pred_tog * (target_tog / tot_tog)
        comps <- c("recv_epr", "disp_epr", "spoil_epr", "hitout_epr")
        for (comp in comps) {
          avg_val <- sum(final_df[[comp]] * final_df$pred_tog, na.rm = TRUE) / sum(final_df$pred_tog)
          final_df[[comp]] <- final_df[[comp]] - avg_val
        }
        final_df$epr <- round(final_df$recv_epr + final_df$disp_epr + final_df$spoil_epr + final_df$hitout_epr, 2)
        for (comp in comps) final_df[[comp]] <- round(final_df[[comp]], 2)
      }
    }
    final_df
  })

  n_empty <- sum(vapply(results, function(x) nrow(x) == 0, logical(1)))
  if (n_empty == length(round_info$round_val) && length(round_info$round_val) > 1) {
    cli::cli_abort("All {length(round_info$round_val)} rounds empty for {year}")
  }

  dplyr::bind_rows(results)
}

# ================================================================
# .build_epr_history_grid(): loops all seasons (2021-current), same
# start_round/max_round convention as production run_ratings_pipeline.R
# (round 0 from 2024 onward, else round 1; current season stops at next
# unplayed round, else 28), building one full-history torp_df variant for
# a given (decay_mult, prior_games) grid cell.
# ================================================================
.build_epr_history_grid <- function(decay_mult, prior_games, pgd, stat_ratings, fixtures) {
  cur_season <- get_afl_season()
  cur_round <- get_afl_week(type = "next")
  seasons <- 2021:cur_season

  decay_recv   <- EPR_DECAY_RECV * decay_mult
  decay_disp   <- EPR_DECAY_DISP * decay_mult
  decay_spoil  <- EPR_DECAY_SPOIL * decay_mult
  decay_hitout <- EPR_DECAY_HITOUT * decay_mult

  parts <- lapply(seasons, function(s) {
    start_round <- if (s >= 2024) 0 else 1
    max_round <- if (s == cur_season) cur_round else 28
    .get_epr_df_variant(
      s, start_round:max_round, pgd, stat_ratings, fixtures,
      decay_recv = decay_recv, decay_disp = decay_disp, decay_spoil = decay_spoil, decay_hitout = decay_hitout,
      prior_games_recv = prior_games, prior_games_disp = prior_games,
      prior_games_spoil = prior_games, prior_games_hitout = prior_games
    )
  })
  dplyr::bind_rows(parts)
}

# ================================================================
# .rebuild_team_mdl_df_with_torp(): copy of build_team_mdl_df()'s body
# with the torp_df injection point swapped for a caller-supplied EPR
# history instead of load_torp_ratings()'s default (published) read.
# psr_df is shared/fixed across every WS2 cell (this workstream varies EPR
# only) -- plan G5, no torp/R/*.R edits.
# ================================================================
.rebuild_team_mdl_df_with_torp <- function(torp_df_custom, psr_df_shared, season = NULL, target_weeks = NULL) {
  if (is.null(season)) season <- get_afl_season()

  all_grounds <- file_reader("stadium_data", "reference-data")
  xg_df <- load_xg(TRUE)
  fixtures <- load_fixtures(TRUE)
  results <- load_results(TRUE)
  teams <- load_teams(TRUE)

  if (nrow(fixtures) < 100) cli::cli_abort("Fixtures too small ({nrow(fixtures)} rows)")
  if (nrow(torp_df_custom) < 100) cli::cli_abort("Ratings too small ({nrow(torp_df_custom)} rows)")
  if (nrow(teams) < 100) cli::cli_abort("Teams too small ({nrow(teams)} rows)")

  fix_df <- .build_fixtures_df(fixtures)
  team_rt_df <- .build_team_ratings_df(teams, torp_df_custom, psr_df_shared)
  team_rt_fix_df <- .build_match_features(fix_df, team_rt_df, all_grounds)
  weather_df <- .load_match_weather(fixtures, all_grounds, target_weeks, season)

  weight_anchor_date <- if (!is.null(target_weeks) && !is.null(season)) {
    target_fix <- fixtures |> dplyr::filter(season == .env$season, round_number %in% target_weeks)
    if (nrow(target_fix) > 0) as.Date(min(target_fix$utc_start_time)) else Sys.Date()
  } else {
    max(as.Date(fix_df$utc_start_time), na.rm = TRUE)
  }

  .build_team_mdl_df(team_rt_fix_df, results, xg_df, weather_df, weight_anchor_date)
}

# ---- Grid registry (plan-closed 3x3, corners-first ordering) ----
GRID <- list(
  control = list(m = 1.0, p = 3, label = "Control (m=1.0, p=3) -- current production constants"),
  c1 = list(m = 0.7, p = 3, label = "m=0.7, p=3 (corner)"),
  c2 = list(m = 0.7, p = 9, label = "m=0.7, p=9 (corner)"),
  c3 = list(m = 1.3, p = 3, label = "m=1.3, p=3 (corner)"),
  c4 = list(m = 1.3, p = 9, label = "m=1.3, p=9 (corner)"),
  c5 = list(m = 0.7, p = 6, label = "m=0.7, p=6 (edge)"),
  c6 = list(m = 1.0, p = 6, label = "m=1.0, p=6 (edge)"),
  c7 = list(m = 1.0, p = 9, label = "m=1.0, p=9 (edge)"),
  c8 = list(m = 1.3, p = 6, label = "m=1.3, p=6 (edge)")
)

# ================================================================
# Stage: data -- load + cache shared inputs ONCE (plan G7: the rating
# rebuild is the cheap part once these are loaded; EPV/pbp-level data is
# NOT touched, only aggregation-time params vary per cell).
# ================================================================
if (stage %in% c("data", "all")) {
  cli::cli_h1("WS2 data: loading shared inputs (pgd, stat_ratings, fixtures, teams, psr_df)")
  t0 <- Sys.time()

  pgd <- load_player_game_data(TRUE)
  data.table::setDT(pgd)
  # Opponent adjustment (production run_ratings_pipeline.R Stage 3, lines
  # ~163-173): adds the _oadj columns calculate_epr_stats_batch() prefers
  # over _adj -- load_player_game_data() does NOT persist these, they are
  # computed fresh at pipeline-run time. Omitting this call was diagnosed as
  # the root cause of a first control-cell fidelity failure (cor 0.985-0.998
  # but non-trivial systematic mean_abs_diff, e.g. epr.x=1.13) -- without it,
  # calculate_epr_stats_batch() silently falls back to non-opponent-adjusted
  # _adj columns, a real structural mismatch vs baseline, not noise.
  pgd <- adjust_epv_for_opponents(pgd)
  data.table::setkey(pgd, match_id)

  stat_ratings <- tryCatch(get_player_stat_ratings(current = FALSE), error = function(e) NULL)
  stat_ratings <- .normalise_skills_columns(stat_ratings, strict = FALSE)

  fixtures <- load_fixtures(TRUE)
  teams <- load_teams(TRUE)

  skills_for_psr <- load_player_stat_ratings(TRUE)
  psr_df_shared <- .compute_psr_from_stat_ratings(skills_for_psr)

  cli::cli_inform("Shared data loaded in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  cat(sprintf("pgd: %d rows | stat_ratings: %s | fixtures: %d rows | teams: %d rows | psr_df_shared: %d rows\n",
              nrow(pgd), ifelse(is.null(stat_ratings), "NULL", paste0(nrow(stat_ratings), " rows")),
              nrow(fixtures), nrow(teams), nrow(psr_df_shared)))

  saveRDS(list(pgd = pgd, stat_ratings = stat_ratings, fixtures = fixtures, teams = teams,
               psr_df_shared = psr_df_shared),
          .rds("ws2_shared_data.rds"))
  cli::cli_alert_success("Saved ws2_shared_data.rds")
}

# ================================================================
# Stage: control -- (m=1.0, p=3) cell. FIDELITY CHECK (hard stop): must
# reproduce the fresh WS0 baseline team_mdl_df's EPR columns to numerical
# noise. Section 6.4's flatness falsifier also needs the baseline's own
# bootstrap MAE CI width -- computed here from the cached WS0 baseline
# screen predictions (no extra harness run needed).
# ================================================================
run_epr_cell <- function(key) {
  g <- GRID[[key]]
  shared <- readRDS(.rds("ws2_shared_data.rds"))

  cli::cli_h2("Building EPR history: {g$label}")
  t0 <- Sys.time()
  torp_df_variant <- .build_epr_history_grid(g$m, g$p, shared$pgd, shared$stat_ratings, shared$fixtures)
  cli::cli_inform("EPR history rebuild ({key}) completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min ({nrow(torp_df_variant)} rows)")

  t0 <- Sys.time()
  team_mdl_df_variant <- .rebuild_team_mdl_df_with_torp(torp_df_variant, shared$psr_df_shared)
  cli::cli_inform("team_mdl_df rebuild ({key}) completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min ({nrow(team_mdl_df_variant)} rows)")

  list(torp_df_variant = torp_df_variant, team_mdl_df_variant = team_mdl_df_variant)
}

if (stage %in% c("control", "all")) {
  cli::cli_h1("WS2 control cell: (m=1.0, p=3) -- fidelity check vs fresh WS0 baseline")

  built <- run_epr_cell("control")
  team_mdl_df_control <- built$team_mdl_df_variant
  saveRDS(team_mdl_df_control, .rds("ws2_team_mdl_df_control.rds"))

  base_team_mdl_df <- readRDS(.rds("ws0_team_mdl_df.rds"))

  epr_cols <- c("epr.x", "recv_epr.x", "disp_epr.x", "spoil_epr.x", "hitout_epr.x", "torp.x")
  j_control <- data.table::as.data.table(team_mdl_df_control)[, c("match_id", "team_name.x", epr_cols), with = FALSE]
  j_base    <- data.table::as.data.table(base_team_mdl_df)[, c("match_id", "team_name.x", epr_cols), with = FALSE]
  data.table::setnames(j_control, epr_cols, paste0(epr_cols, "_control"))
  data.table::setnames(j_base, epr_cols, paste0(epr_cols, "_base"))
  joined <- merge(j_control, j_base, by = c("match_id", "team_name.x"))
  cat(sprintf("\nFidelity join: %d rows matched (control has %d, baseline has %d)\n",
              nrow(joined), nrow(team_mdl_df_control), nrow(base_team_mdl_df)))

  fid_results <- list()
  for (col in epr_cols) {
    a <- joined[[paste0(col, "_control")]]
    b <- joined[[paste0(col, "_base")]]
    ok <- !is.na(a) & !is.na(b)
    cor_val <- if (sum(ok) >= 3 && stats::sd(a[ok]) > 0 && stats::sd(b[ok]) > 0) stats::cor(a[ok], b[ok]) else NA_real_
    mean_abs_diff <- mean(abs(a[ok] - b[ok]))
    max_abs_diff <- max(abs(a[ok] - b[ok]))
    fid_results[[col]] <- list(cor = cor_val, mean_abs_diff = mean_abs_diff, max_abs_diff = max_abs_diff, n = sum(ok))
    cat(sprintf("%-15s cor=%.4f mean_abs_diff=%.4f max_abs_diff=%.4f (n=%d)\n",
                col, cor_val, mean_abs_diff, max_abs_diff, sum(ok)))
  }

  fidelity_pass <- all(vapply(fid_results, function(r) !is.na(r$cor) && r$cor > 0.98, logical(1)))
  cat(sprintf("\n=== CONTROL-CELL FIDELITY VERDICT (Section 6.4-adjacent hard stop, same logic as WS1 Section 6.3): %s ===\n",
              ifelse(fidelity_pass, "PASS -- control reproduces baseline EPR columns to noise, safe to proceed to the grid",
                     "FAIL -- STOP: control cell does not reproduce baseline; diagnose before running any grid cell")))

  # Baseline's own bootstrap MAE CI width (for Section 6.4 flatness falsifier)
  base_screen <- readRDS(.rds("ws0_baseline_screen.rds"))
  base_ci <- .boot_mae_ci(base_screen$preds, B = 2000)
  cat(sprintf("\nBaseline (control-equivalent) screen MAE=%.3f, bootstrap 95%% CI[%.3f,%.3f], width=%.3f\n",
              base_ci$mae, base_ci$ci[1], base_ci$ci[2], base_ci$width))

  saveRDS(list(fid_results = fid_results, fidelity_pass = fidelity_pass, base_ci = base_ci,
               team_mdl_df_control = team_mdl_df_control),
          .rds("ws2_control_summary.rds"))
  cli::cli_alert_success("Saved ws2_control_summary.rds")
}

# ================================================================
# Stage: cell_c1 .. cell_c8 -- 8 non-control cells, 2026 screen only,
# boot vs cached WS0 baseline screen predictions.
# ================================================================
run_grid_cell_screen <- function(key) {
  g <- GRID[[key]]
  base <- readRDS(.rds("ws0_baseline_screen.rds"))

  built <- run_epr_cell(key)
  team_mdl_df_variant <- built$team_mdl_df_variant
  saveRDS(team_mdl_df_variant, .rds(paste0("ws2_team_mdl_df_", key, ".rds")))

  t0 <- Sys.time()
  roll <- run_rolling_eval(
    team_mdl_df_variant, TEST_SEASONS,
    gam_trainer = .train_match_gams,
    xgb_trainer = .train_xgb_fixed,
    extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("{g$label} 2026 screen completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  preds_norecal <- roll$input_blend_preds
  m_norecal <- .compute_metrics(preds_norecal)
  preds <- v1a_recal_own(preds_norecal)
  m <- .compute_metrics(preds)

  .print_metrics(base$metrics, "G4 baseline + V1a recal, 2026 screen")
  .print_metrics(m, sprintf("%s + V1a recal, 2026 screen", g$label))

  boot_vs_base <- boot_mae_diff(preds, base$preds, B = 2000)
  cat(sprintf("\nboot_mae_diff(%s+recal - G4 baseline+recal, 2026 screen): N=%d deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
              g$label, boot_vs_base$n_matches, boot_vs_base$mae_diff, boot_vs_base$mae_ci[1], boot_vs_base$mae_ci[2], boot_vs_base$brier_diff))

  confirm_worthy <- boot_vs_base$mae_diff <= -0.2
  cat(sprintf("Pooled-confirm threshold (deltaMAE <= -0.2): %s (%.3f)\n", confirm_worthy, boot_vs_base$mae_diff))

  out <- list(key = key, m = g$m, p = g$p, label = g$label, roll = roll,
              preds_norecal = preds_norecal, metrics_norecal = m_norecal,
              preds = preds, metrics = m, boot_vs_baseline_screen = boot_vs_base,
              confirm_worthy = confirm_worthy)
  saveRDS(out, .rds(paste0("ws2_", key, "_screen.rds")))
  cli::cli_alert_success("Saved ws2_{key}_screen.rds")
  invisible(out)
}

for (k in c("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")) {
  if (stage %in% c(paste0("cell_", k), "all")) {
    cli::cli_h1("WS2 grid cell: {GRID[[k]]$label} -- 2026 screen")
    run_grid_cell_screen(k)
  }
}

# ================================================================
# Stage: control_screen -- run the control cell (m=1.0, p=3) through the
# SAME 2026-screen harness as every grid cell, using the already-cached
# ws2_team_mdl_df_control.rds (no re-rebuild needed). Added after observing
# that all 8 grid cells showed a negative deltaMAE vs the *published*
# baseline (ws0_baseline_screen.rds) with no clear m/p pattern -- consistent
# with the control-cell fidelity check's own small systematic offset (a
# fresh full-batch EPR recompute via this script's pipeline vs baseline's
# load_torp_ratings() published/incrementally-upserted data), not with an
# aggregation-parameter effect. Isolating "does m/p matter" requires
# comparing grid cells against a control run through the IDENTICAL
# recompute pipeline (varying only m/p), not against the published
# baseline, which mixes in a pipeline-vs-published confound.
# ================================================================
if (stage %in% c("control_screen", "all")) {
  cli::cli_h1("WS2 control cell: 2026 screen (same harness as every grid cell)")
  team_mdl_df_control <- readRDS(.rds("ws2_team_mdl_df_control.rds"))
  base <- readRDS(.rds("ws0_baseline_screen.rds"))

  t0 <- Sys.time()
  roll <- run_rolling_eval(
    team_mdl_df_control, TEST_SEASONS,
    gam_trainer = .train_match_gams,
    xgb_trainer = .train_xgb_fixed,
    extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("Control 2026 screen completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  preds_norecal <- roll$input_blend_preds
  m_norecal <- .compute_metrics(preds_norecal)
  preds <- v1a_recal_own(preds_norecal)
  m <- .compute_metrics(preds)

  .print_metrics(base$metrics, "Published baseline (load_torp_ratings) + V1a recal, 2026 screen")
  .print_metrics(m, "Control (m=1.0,p=3, this pipeline) + V1a recal, 2026 screen")

  boot_vs_base <- boot_mae_diff(preds, base$preds, B = 2000)
  cat(sprintf("\nboot_mae_diff(Control+recal - Published baseline+recal, 2026 screen): N=%d deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
              boot_vs_base$n_matches, boot_vs_base$mae_diff, boot_vs_base$mae_ci[1], boot_vs_base$mae_ci[2], boot_vs_base$brier_diff))
  cat("(This delta isolates the pipeline-recompute-vs-published-data confound; the grid's own analysis below should use THIS control as its zero-point, not the published baseline.)\n")

  control_ci <- .boot_mae_ci(preds, B = 2000)
  cat(sprintf("\nControl's own bootstrap MAE CI: MAE=%.3f, 95%% CI[%.3f,%.3f], width=%.3f (this is the correct Section 6.4 noise band)\n",
              control_ci$mae, control_ci$ci[1], control_ci$ci[2], control_ci$width))

  out <- list(roll = roll, preds_norecal = preds_norecal, metrics_norecal = m_norecal,
              preds = preds, metrics = m, boot_vs_published_baseline = boot_vs_base, control_ci = control_ci)
  saveRDS(out, .rds("ws2_control_screen.rds"))
  cli::cli_alert_success("Saved ws2_control_screen.rds")
}

# ================================================================
# Stage: confirm_<key> -- pooled 2025:2026 confirm, only for a cell that
# cleared the screen deltaMAE <= -0.2 threshold (plan step 4).
# ================================================================
run_grid_cell_confirm <- function(key) {
  g <- GRID[[key]]
  screen <- readRDS(.rds(paste0("ws2_", key, "_screen.rds")))
  if (!isTRUE(screen$confirm_worthy)) {
    cli::cli_warn("{g$label}: screen deltaMAE did not clear the -0.2 threshold -- running pooled confirm anyway since explicitly requested, but this is off the plan-prescribed path.")
  }
  base <- readRDS(.rds("ws0_baseline_pool.rds"))
  team_mdl_df_variant <- readRDS(.rds(paste0("ws2_team_mdl_df_", key, ".rds")))

  t0 <- Sys.time()
  roll <- run_rolling_eval(
    team_mdl_df_variant, CONFIRM_SEASONS,
    gam_trainer = .train_match_gams,
    xgb_trainer = .train_xgb_fixed,
    extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("{g$label} pooled confirm completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  preds_norecal <- roll$input_blend_preds
  m_norecal <- .compute_metrics(preds_norecal)
  preds <- v1a_recal_own(preds_norecal)
  m <- .compute_metrics(preds)

  .print_metrics(base$metrics, "G4 baseline + V1a recal, pooled")
  .print_metrics(m, sprintf("%s + V1a recal, pooled", g$label))

  boot_vs_base <- boot_mae_diff(preds, base$preds, B = 2000)
  ci_excl_0 <- (boot_vs_base$mae_ci[1] > 0 && boot_vs_base$mae_ci[2] > 0) ||
    (boot_vs_base$mae_ci[1] < 0 && boot_vs_base$mae_ci[2] < 0)
  brier_ok <- boot_vs_base$brier_diff <= 0.002
  ship_pass <- ci_excl_0 && boot_vs_base$mae_diff < 0 && brier_ok

  cat(sprintf("\nboot_mae_diff(%s+recal - G4 baseline+recal, pooled): N=%d deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
              g$label, boot_vs_base$n_matches, boot_vs_base$mae_diff, boot_vs_base$mae_ci[1], boot_vs_base$mae_ci[2], boot_vs_base$brier_diff))
  cat(sprintf("G3 ship gate: %s\n", ship_pass))
  cat("NOTE: per plan (WS2 is a measurement workstream, not authorized to re-optimize) -- even a G3 pass here does NOT authorize\n")
  cat("an EPR optimizer re-run; that requires a separate plan edit per Section 2 WS2's own success criteria.\n")

  out <- list(key = key, roll = roll, preds_norecal = preds_norecal, metrics_norecal = m_norecal,
              preds = preds, metrics = m, boot_vs_baseline_pool = boot_vs_base, ship_pass = ship_pass)
  saveRDS(out, .rds(paste0("ws2_", key, "_confirm.rds")))
  cli::cli_alert_success("Saved ws2_{key}_confirm.rds")
  invisible(out)
}

for (k in c("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")) {
  if (stage == paste0("confirm_", k)) run_grid_cell_confirm(k)
}

# ================================================================
# Stage: summary -- full grid table + Section 6.4 flatness falsifier
# ================================================================
if (stage %in% c("summary", "all")) {
  cli::cli_h1("WS2 Final Summary: EPR aggregation-parameter sensitivity grid")

  load_if <- function(f) if (file.exists(.rds(f))) readRDS(.rds(f)) else NULL
  control_summary <- load_if("ws2_control_summary.rds")
  control_screen <- load_if("ws2_control_screen.rds")
  base_screen <- load_if("ws0_baseline_screen.rds")

  cat("\n=== Control-cell fidelity ===\n")
  if (!is.null(control_summary)) {
    cat(sprintf("Fidelity: %s\n", ifelse(control_summary$fidelity_pass, "PASS", "FAIL")))
  }
  if (!is.null(control_screen)) {
    .print_metrics(control_screen$metrics, "Control (m=1.0,p=3, this pipeline) + V1a recal, 2026 screen")
    cat(sprintf("vs published baseline: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] (this is the pipeline-vs-published confound, NOT an m/p effect)\n",
                control_screen$boot_vs_published_baseline$mae_diff, control_screen$boot_vs_published_baseline$mae_ci[1],
                control_screen$boot_vs_published_baseline$mae_ci[2]))
    cat(sprintf("Control's own bootstrap MAE CI width: %.3f (MAE=%.3f, CI[%.3f,%.3f]) -- the correct Section 6.4 noise band\n",
                control_screen$control_ci$width, control_screen$control_ci$mae,
                control_screen$control_ci$ci[1], control_screen$control_ci$ci[2]))
  } else if (!is.null(control_summary)) {
    cat(sprintf("(No control harness screen run -- using published-baseline bootstrap CI width as fallback noise band: %.3f)\n",
                control_summary$base_ci$width))
  }

  keys <- c("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")
  screens <- setNames(lapply(keys, function(k) load_if(paste0("ws2_", k, "_screen.rds"))), keys)

  # Use the control cell's own screen as the zero-point when available (this
  # is what isolates the m/p effect from the pipeline-vs-published confound);
  # fall back to the published baseline's boot_mae_diff otherwise.
  control_mae <- if (!is.null(control_screen)) control_screen$metrics$mae else NA_real_
  noise_width <- if (!is.null(control_screen)) control_screen$control_ci$width else control_summary$base_ci$width

  cat("\n=== 2026 SCREEN grid (all 8 non-control cells), vs published baseline AND vs control ===\n")
  mae_vals <- c(if (!is.null(base_screen)) base_screen$metrics$mae else NA_real_, control_mae)
  for (k in keys) {
    s <- screens[[k]]
    if (is.null(s)) { cat(sprintf("%s: not run\n", k)); next }
    .print_metrics(s$metrics, sprintf("%s (m=%.1f,p=%d) + V1a recal, 2026 screen", s$label, s$m, s$p))
    cat(sprintf("  vs published baseline: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f confirm_worthy(vs baseline)=%s\n",
                s$boot_vs_baseline_screen$mae_diff, s$boot_vs_baseline_screen$mae_ci[1], s$boot_vs_baseline_screen$mae_ci[2],
                s$boot_vs_baseline_screen$brier_diff, s$confirm_worthy))
    if (!is.na(control_mae)) {
      cat(sprintf("  vs control (isolates m/p): deltaMAE=%+.3f\n", s$metrics$mae - control_mae))
    }
    mae_vals <- c(mae_vals, s$metrics$mae)
  }

  confirms <- setNames(lapply(keys, function(k) load_if(paste0("ws2_", k, "_confirm.rds"))), keys)
  cat("\n=== POOLED confirms (where run) ===\n")
  any_confirm <- FALSE
  for (k in keys) {
    cf <- confirms[[k]]
    if (is.null(cf)) next
    any_confirm <- TRUE
    .print_metrics(cf$metrics, sprintf("%s + V1a recal, pooled", k))
    cat(sprintf("  vs baseline: deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] SHIP_PASS=%s\n",
                cf$boot_vs_baseline_pool$mae_diff, cf$boot_vs_baseline_pool$mae_ci[1], cf$boot_vs_baseline_pool$mae_ci[2], cf$ship_pass))
  }
  if (!any_confirm) cat("No cell cleared the deltaMAE <= -0.2 screen threshold -- no pooled confirms run.\n")

  # Section 6.4 flatness falsifier -- computed vs CONTROL (the correct
  # isolation of the m/p effect), not vs published baseline.
  cat("\n=== Section 6.4 flatness falsifier ===\n")
  grid_mae_vals <- vapply(keys, function(k) if (!is.null(screens[[k]])) screens[[k]]$metrics$mae else NA_real_, numeric(1))
  grid_mae_vals <- grid_mae_vals[!is.na(grid_mae_vals)]
  if (length(grid_mae_vals) >= 2 && !is.na(control_mae)) {
    all_vals <- c(control_mae, grid_mae_vals)
    grid_range <- max(all_vals) - min(all_vals)
    flat <- grid_range < noise_width
    cat(sprintf("Screen MAE range across grid (incl. control): %.3f (min=%.3f, max=%.3f, control=%.3f)\n",
                grid_range, min(all_vals), max(all_vals), control_mae))
    cat(sprintf("Control's own bootstrap MAE CI width (noise band): %.3f\n", noise_width))
    cat(sprintf("Verdict: %s\n", ifelse(flat,
                "FLAT -- grid range < control's own noise band -> EPR aggregation params are NOT a margin lever at team-sum granularity. Ratify April constants. No optimizer re-run authorized.",
                "NOT FLAT -- grid range exceeds control's own noise band -> some sensitivity detected, see per-cell deltas vs control for which cell(s) moved.")))
  } else {
    cat("Insufficient data to evaluate flatness falsifier (missing control screen or grid results).\n")
  }

  saveRDS(list(control_summary = control_summary, control_screen = control_screen,
               screens = screens, confirms = confirms, base_screen = base_screen),
          .rds("ws2_final_summary.rds"))
  cli::cli_alert_success("Saved ws2_final_summary.rds")
}
