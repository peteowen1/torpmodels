# ws_psr_epr_1_psr_vintage.R -- WS1: PSR coefficient vintage refresh
# (walk-forward annual policy), per docs/plans/FABLE-PSR-EPR-PLAN.md Section 2 WS1.
# =====================================================================
# Hypothesis: psr_diff/osr_diff/dsr_diff built from glmnet betas frozen at the
# 2021-2024 vintage (anchor 2024-12-31, shipped inst/extdata/psr_coefficients.csv,
# trained 2026-04-18) underperform features built from the newest legally-
# available vintage. WS0 (2026-07-21) found V0d (EPR-family only) ~= baseline
# while V0e (PSR-family only) was clearly worse -- PSR carries little marginal
# signal at its current frozen vintage, which is exactly the staleness this
# workstream tests: is that "little signal" a property of PSR, or a property
# of a PSR beta vintage that's ~18 months stale on its newest training outcome?
#
# Design (binding, plan Section 2 WS1 + Section 6 falsifiers 2/3):
#   1. fit_psr_vintage(train_through_season): local extraction of
#      06_train_psr_model.R's core fit (same alpha grid, lambda.min, same
#      exclusions, weights anchored to Dec-31 of the vintage year), per G5
#      (each WS keeps its own copy, torp/R/*.R untouched).
#   2. FIDELITY CHECK (hard stop): vintage_2024 (train <=2024) must
#      approximately reproduce the shipped CSVs. If it doesn't, the shipped
#      coefficients' provenance is unclear -- treat that as the finding and
#      stop before anything downstream.
#   3. vintage_2025 (train <=2025); report per-stat beta drift regardless of
#      outcome (feeds the TORP explainer later).
#   4. Variant PSR histories via calculate_psr_components() over the released
#      player_stat_ratings history (point-in-time inputs; only betas change).
#   5. Walk-forward assembly (the deployable candidate): 2025 rows use
#      vintage_2024 betas, 2026 rows use vintage_2025 betas (G6-legal on the
#      whole pooled 2025:2026 window). Also score vintage_2025-everywhere on
#      the 2026 screen ALONE (2025 rows there would be in-sample at the beta
#      level -- vintage_2025 saw 2025 outcomes in training).
#   6. Rebuild team_mdl_df per candidate (.build_team_ratings_df() takes
#      psr_df -- swap it in; torp.x/.y and *_diff columns recompute
#      automatically in .build_team_mdl_df()), run the harness, V1a recal
#      post-pass, boot_mae_diff() vs the fresh WS0 G4 baseline (25.132 screen
#      / 25.370 pooled -- reused from cached ws0_baseline_*.rds, since the
#      harness trainers are byte-identical to WS0's baseline; only team_mdl_df
#      differs).
#   7. Ship shape reminder: even a G3 pass here is PROPOSAL-ONLY (published-
#      rating input -> Pete decision, plan Section 4 shape 2). This script
#      produces evidence, never touches torp/R/*, never re-publishes
#      inst/extdata/*.csv, and performs no git operations.
#
# Run stage-by-stage (checkpoints to experiments/results/ws1_*):
#   Rscript ws_psr_epr_1_psr_vintage.R fit                 # fit vintage_2024 + vintage_2025, fidelity check, beta drift
#   Rscript ws_psr_epr_1_psr_vintage.R psr_histories        # full-history psr_df per vintage + walk-forward/v2025-everywhere assembly
#   Rscript ws_psr_epr_1_psr_vintage.R rebuild_walkforward   # team_mdl_df rebuild, walk-forward psr_df
#   Rscript ws_psr_epr_1_psr_vintage.R rebuild_v2025         # team_mdl_df rebuild, vintage_2025-everywhere psr_df
#   Rscript ws_psr_epr_1_psr_vintage.R screen_walkforward     # 2026 screen, walk-forward candidate
#   Rscript ws_psr_epr_1_psr_vintage.R screen_v2025            # 2026 screen, vintage_2025-everywhere candidate
#   Rscript ws_psr_epr_1_psr_vintage.R confirm_walkforward      # pooled 2025:2026 confirm, walk-forward candidate ONLY (G6 legality)
#   Rscript ws_psr_epr_1_psr_vintage.R summary                   # full table + interpretation

stage <- {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) >= 1) a[1] else "all"
}
cat("=== ws_psr_epr_1_psr_vintage.R stage:", stage, "===\n")

# Setup ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(xgboost)
  library(mgcv)
  library(MLmetrics)
  library(geosphere)
  library(cli)
  library(data.table)
  library(glmnet)
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

# ---- V1a recal: recal_expanding + .apply_recal + v1a_recal_own + printers
# (copied verbatim from ws_psr_epr_0_ablations.R / ws6_decay_on_c6.R /
# ws7_elo_xgb_fix.R -- plan G5: each WS keeps its own copy). ----
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
# fit_psr_vintage(): local extraction of 06_train_psr_model.R's core fit,
# parameterised by train_through_season. Same alpha grid, lambda.min, same
# exclusions (defensive/scoring stat lists + the exclude_stats blocklist),
# recency weights anchored to Dec-31 of train_through_season. Train window
# = all matches with season <= train_through_season (train_through_season=
# 2024 reproduces production's original "train 2021-2024" split exactly;
# 2025 extends the window by one season, matching the plan's walk-forward
# policy: "score each season with the newest vintage that excludes it").
# ================================================================
fit_psr_vintage <- function(train_through_season, verbose = TRUE) {
  rmse <- function(a, p) sqrt(mean((a - p)^2))

  if (verbose) cli::cli_h2("fit_psr_vintage(train_through_season = {train_through_season})")

  stat_ratings <- as.data.table(load_player_stat_ratings(TRUE))
  teams <- as.data.table(load_teams(TRUE))
  fixtures <- as.data.table(load_fixtures(all = TRUE))

  fixtures_margin <- fixtures[
    !is.na(home_score) & !is.na(away_score),
    .(match_id,
      season = as.integer(season),
      round = as.integer(round_number),
      home_score = as.numeric(home_score),
      away_score = as.numeric(away_score),
      home_margin = home_score - away_score,
      match_date = as.Date(substr(utc_start_time, 1, 10)))
  ]

  teams <- teams[is.na(lineup_position) | (lineup_position != "EMERG" & lineup_position != "SUB")]
  teams[, round := as.integer(round_number)]
  teams[, player_id := as.character(player_id)]
  teams[, season := as.integer(season)]
  stat_ratings[, player_id := as.character(player_id)]
  stat_ratings[, season := as.integer(season)]
  stat_ratings[, round := as.integer(round)]

  stat_defs <- stat_rating_definitions()
  exclude_stats <- c("cond_tog", "squad_selection", "dream_team_points", "rating_points",
                      "centre_bounce_attendances", "ruck_contests", "kickins", "bounces")
  all_rating_names <- setdiff(stat_defs$stat_name, exclude_stats)

  adj_rating_cols <- intersect(paste0(all_rating_names, "_adj_rating"), names(stat_ratings))
  raw_rating_cols <- intersect(paste0(all_rating_names, "_rating"), names(stat_ratings))

  if (length(adj_rating_cols) >= length(raw_rating_cols) * 0.8) {
    for (i in seq_along(adj_rating_cols)) {
      raw_col <- sub("_adj_rating$", "_rating", adj_rating_cols[i])
      if (raw_col %in% names(stat_ratings)) {
        stat_ratings[, (raw_col) := get(adj_rating_cols[i])]
      }
    }
  }
  rating_cols <- intersect(paste0(all_rating_names, "_rating"), names(stat_ratings))

  ratings_join <- stat_ratings[, c("player_id", "season", "round", "pos_group", rating_cols), with = FALSE]
  merged <- merge(teams, ratings_join, by = c("player_id", "season", "round"), all.x = TRUE)

  pos_means <- merged[!is.na(pos_group), lapply(.SD, mean, na.rm = TRUE),
                      by = pos_group, .SDcols = rating_cols]
  global_means <- merged[, lapply(.SD, mean, na.rm = TRUE), .SDcols = rating_cols]
  for (sc in rating_cols) {
    na_idx <- which(is.na(merged[[sc]]))
    if (length(na_idx) > 0) {
      for (pg in unique(pos_means$pos_group)) {
        pg_idx <- na_idx[merged$pos_group[na_idx] == pg & !is.na(merged$pos_group[na_idx])]
        if (length(pg_idx) > 0) merged[pg_idx, (sc) := pos_means[pos_group == pg, get(sc)]]
      }
      still_na <- which(is.na(merged[[sc]]))
      if (length(still_na) > 0) merged[still_na, (sc) := global_means[[sc]]]
    }
  }

  merged[, .total_rating := rowSums(.SD, na.rm = TRUE), .SDcols = rating_cols]
  team_ratings <- merged[order(-.total_rating)][
    , head(.SD, 22), by = .(match_id, team_id)
  ][, {
    out <- list(n_players = .N)
    for (sc in rating_cols) out[[sc]] <- sum(get(sc), na.rm = TRUE)
    out
  }, by = .(match_id, team_id, season, round)]

  team_ratings <- merge(team_ratings,
    fixtures[, .(match_id, home_team_id, away_team_id)],
    by = "match_id", all.x = TRUE)
  team_ratings[, team_type := fifelse(team_id == home_team_id, "home", "away")]

  home <- team_ratings[team_type == "home"]
  away <- team_ratings[team_type == "away"]

  home_cols <- paste0("home_", rating_cols)
  away_cols <- paste0("away_", rating_cols)

  setnames(home, rating_cols, home_cols)
  setnames(away, rating_cols, away_cols)

  match_df <- merge(
    home[, c("match_id", "season", "round", home_cols), with = FALSE],
    away[, c("match_id", away_cols), with = FALSE],
    by = "match_id"
  )
  match_df <- merge(match_df,
    fixtures_margin[, .(match_id, home_score, away_score, home_margin, match_date)],
    by = "match_id")

  # Weights: anchored to Dec-31 of the vintage year (plan step 1)
  anchor_date <- as.Date(paste0(train_through_season, "-12-31"))
  match_df[, weightz := exp(as.numeric(-(anchor_date - match_date)) / MATCH_WEIGHT_DECAY_DAYS)]
  match_df[, weightz := weightz / mean(weightz, na.rm = TRUE)]

  # Train window: season <= train_through_season (train_through_season=2024
  # reproduces the original "season < 2025" split exactly)
  train_idx <- which(match_df$season <= train_through_season)
  test_idx  <- which(match_df$season > train_through_season)

  all_feat_cols <- c(home_cols, away_cols)
  X_raw <- as.matrix(match_df[, all_feat_cols, with = FALSE])
  train_sds <- apply(X_raw[train_idx, ], 2, sd)
  train_sds[train_sds == 0] <- 1
  X <- sweep(X_raw, 2, train_sds, "/")

  X_train <- X[train_idx, ]
  X_test  <- if (length(test_idx) > 0) X[test_idx, , drop = FALSE] else X[integer(0), , drop = FALSE]
  w_train <- match_df$weightz[train_idx]

  y_margin_train <- match_df$home_margin[train_idx]
  y_margin_test  <- match_df$home_margin[test_idx]
  y_off_train <- match_df$home_score[train_idx]
  y_off_test  <- match_df$home_score[test_idx]
  y_def_train <- match_df$away_score[train_idx]
  y_def_test  <- match_df$away_score[test_idx]

  if (verbose) cli::cli_inform("Train: {length(train_idx)} (season <= {train_through_season}), Test: {length(test_idx)} (season > {train_through_season})")

  train_seasons <- match_df$season[train_idx]
  foldid <- as.integer(factor(train_seasons, levels = sort(unique(train_seasons))))

  scoring_stats <- c("goals", "behinds", "shots_at_goal", "score_involvements",
                      "goal_assists", "goal_accuracy", "score_launches")
  defensive_stats <- c("tackles", "spoils", "intercepts", "one_percenters",
                        "intercept_marks", "tackles_inside50")

  osr_exclude_cols <- which(colnames(X) %in% c(
    paste0("home_", defensive_stats, "_rating"),
    paste0("away_", defensive_stats, "_rating")
  ))
  dsr_exclude_cols <- which(colnames(X) %in% c(
    paste0("home_", scoring_stats, "_rating"),
    paste0("away_", scoring_stats, "_rating")
  ))

  X_train_osr <- X_train[, -osr_exclude_cols, drop = FALSE]
  X_test_osr  <- X_test[, -osr_exclude_cols, drop = FALSE]
  X_train_dsr <- X_train[, -dsr_exclude_cols, drop = FALSE]
  X_test_dsr  <- X_test[, -dsr_exclude_cols, drop = FALSE]

  alpha_grid <- c(0, 0.25, 0.5, 0.75, 1)

  fit_model <- function(X_tr, X_te, y_tr, y_te, label) {
    best_cvm <- Inf
    best_fit <- NULL
    best_a <- NULL

    for (a in alpha_grid) {
      set.seed(42)
      cv_f <- cv.glmnet(X_tr, y_tr, weights = w_train, alpha = a,
                         foldid = foldid, type.measure = "mse", standardize = FALSE)
      if (min(cv_f$cvm) < best_cvm) {
        best_cvm <- min(cv_f$cvm)
        best_fit <- cv_f
        best_a <- a
      }
    }

    mdl <- glmnet(X_tr, y_tr, weights = w_train, alpha = best_a,
                   lambda = best_fit$lambda.min, standardize = FALSE)

    p_tr <- as.numeric(predict(mdl, X_tr))
    test_rmse <- NA_real_
    if (nrow(X_te) > 0) {
      p_te <- as.numeric(predict(mdl, X_te))
      test_rmse <- rmse(y_te, p_te)
    }

    if (verbose) cli::cli_inform("{label}: alpha={best_a}, CV RMSE={round(sqrt(best_cvm), 2)}, Train RMSE={round(rmse(y_tr, p_tr), 2)}, Test RMSE={ifelse(is.na(test_rmse), 'n/a', round(test_rmse, 2))}")

    cs <- as.matrix(coef(mdl))
    list(model = mdl, coefs = cs, best_alpha = best_a, test_rmse = test_rmse, feature_cols = colnames(X_tr))
  }

  margin_fit <- fit_model(X_train, X_test, y_margin_train, y_margin_test, "Margin (PSR)")
  off_fit    <- fit_model(X_train_osr, X_test_osr, y_off_train, y_off_test, "Offense (OSR)")
  def_fit    <- fit_model(X_train_dsr, X_test_dsr, y_def_train, y_def_test, "Defense (DSR)")

  extract_betas <- function(coefs, prefix, rating_cols) {
    full_names <- paste0(prefix, rating_cols)
    coef_names <- rownames(coefs)
    betas <- numeric(length(full_names))
    names(betas) <- full_names
    for (i in seq_along(full_names)) {
      if (full_names[i] %in% coef_names) betas[i] <- coefs[full_names[i], 1]
    }
    betas
  }

  off_cs <- off_fit$coefs
  off_home_beta <- extract_betas(off_cs, "home_", rating_cols)
  off_away_beta <- extract_betas(off_cs, "away_", rating_cols)

  def_cs <- def_fit$coefs
  def_home_beta <- extract_betas(def_cs, "home_", rating_cols)
  def_away_beta <- extract_betas(def_cs, "away_", rating_cols)

  osr_beta <- (off_home_beta + def_away_beta) / 2
  dsr_beta <- -(def_home_beta + off_away_beta) / 2
  osr_beta[paste0("home_", defensive_stats, "_rating")] <- 0
  dsr_beta[paste0("home_", scoring_stats, "_rating")] <- 0

  home_sds <- train_sds[paste0("home_", rating_cols)]

  osr_coef_df <- data.frame(stat_name = sub("_rating$", "", rating_cols),
                             beta = as.numeric(osr_beta), sd = as.numeric(home_sds),
                             stringsAsFactors = FALSE)
  dsr_coef_df <- data.frame(stat_name = sub("_rating$", "", rating_cols),
                             beta = as.numeric(dsr_beta), sd = as.numeric(home_sds),
                             stringsAsFactors = FALSE)

  margin_cs <- margin_fit$coefs
  margin_home_beta <- margin_cs[paste0("home_", rating_cols), 1]
  margin_away_beta <- margin_cs[paste0("away_", rating_cols), 1]
  psr_beta <- (margin_home_beta - margin_away_beta) / 2
  psr_coef_df <- data.frame(stat_name = sub("_rating$", "", rating_cols),
                             beta = as.numeric(psr_beta), sd = as.numeric(home_sds),
                             stringsAsFactors = FALSE)

  list(
    psr_coef_df = psr_coef_df, osr_coef_df = osr_coef_df, dsr_coef_df = dsr_coef_df,
    train_through_season = train_through_season, anchor_date = anchor_date,
    n_train = length(train_idx), n_test = length(test_idx),
    margin_test_rmse = margin_fit$test_rmse,
    margin_alpha = margin_fit$best_alpha, osr_alpha = off_fit$best_alpha, dsr_alpha = def_fit$best_alpha
  )
}

# ================================================================
# .rebuild_team_mdl_df_with_psr(): copy of build_team_mdl_df()'s body
# (torp/R/match_model.R) with the psr_df injection point swapped for a
# caller-supplied psr_df instead of .compute_psr_from_stat_ratings()'s
# default (shipped CSV) read -- per plan step 5: ".build_team_ratings_df()
# takes psr_df as an argument -- swap it in; torp.x/.y recompute
# automatically in .build_team_mdl_df()". No torp/R/*.R edits (G5).
# ================================================================
.rebuild_team_mdl_df_with_psr <- function(psr_df_custom, season = NULL, target_weeks = NULL) {
  if (is.null(season)) season <- get_afl_season()

  cli::cli_h2("[ws1] Loading data (psr_df injected)")
  all_grounds <- file_reader("stadium_data", "reference-data")
  xg_df <- load_xg(TRUE)
  fixtures <- load_fixtures(TRUE)
  results <- load_results(TRUE)
  teams <- load_teams(TRUE)
  torp_df <- load_torp_ratings()

  cli::cli_inform(paste0(
    "Loaded: fixtures=", nrow(fixtures), ", results=", nrow(results),
    ", teams=", nrow(teams), ", ratings=", nrow(torp_df), ", psr_df(custom)=", nrow(psr_df_custom)
  ))

  if (nrow(fixtures) < 100) cli::cli_abort("Fixtures too small ({nrow(fixtures)} rows)")
  if (nrow(torp_df) < 100) cli::cli_abort("Ratings too small ({nrow(torp_df)} rows)")
  if (nrow(teams) < 100) cli::cli_abort("Teams too small ({nrow(teams)} rows)")

  cli::cli_h2("[ws1] Building fixture features")
  fix_df <- .build_fixtures_df(fixtures)

  cli::cli_h2("[ws1] Processing lineups (custom psr_df)")
  team_rt_df <- .build_team_ratings_df(teams, torp_df, psr_df_custom)

  cli::cli_h2("[ws1] Computing features")
  team_rt_fix_df <- .build_match_features(fix_df, team_rt_df, all_grounds)

  cli::cli_h2("[ws1] Loading weather")
  weather_df <- .load_match_weather(fixtures, all_grounds, target_weeks, season)

  weight_anchor_date <- if (!is.null(target_weeks) && !is.null(season)) {
    target_fix <- fixtures |> dplyr::filter(season == .env$season, round_number %in% target_weeks)
    if (nrow(target_fix) > 0) as.Date(min(target_fix$utc_start_time)) else Sys.Date()
  } else {
    max(as.Date(fix_df$utc_start_time), na.rm = TRUE)
  }
  cli::cli_inform("Weight anchor date: {weight_anchor_date}")

  cli::cli_h2("[ws1] Building model dataset")
  team_mdl_df <- .build_team_mdl_df(team_rt_fix_df, results, xg_df, weather_df, weight_anchor_date)
  team_mdl_df
}

# ================================================================
# Stage: fit -- fit vintage_2024 (train <=2024, anchor 2024-12-31, should
# reproduce inst/extdata/*.csv) and vintage_2025 (train <=2025, anchor
# 2025-12-31). FIDELITY CHECK (plan Section 6 falsifier 3, hard stop):
# compares vintage_2024 betas against the shipped CSVs.
# ================================================================
if (stage %in% c("fit", "all")) {
  cli::cli_h1("WS1 fit: vintage_2024 + vintage_2025")

  t0 <- Sys.time()
  vintage_2024 <- fit_psr_vintage(2024)
  vintage_2025 <- fit_psr_vintage(2025)
  cli::cli_inform("Both vintages fit in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  saveRDS(vintage_2024, .rds("ws1_vintage_2024.rds"))
  saveRDS(vintage_2025, .rds("ws1_vintage_2025.rds"))
  write.csv(vintage_2024$psr_coef_df, .rds("ws1_vintage_2024_psr_coefficients.csv"), row.names = FALSE)
  write.csv(vintage_2024$osr_coef_df, .rds("ws1_vintage_2024_osr_coefficients.csv"), row.names = FALSE)
  write.csv(vintage_2024$dsr_coef_df, .rds("ws1_vintage_2024_dsr_coefficients.csv"), row.names = FALSE)
  write.csv(vintage_2025$psr_coef_df, .rds("ws1_vintage_2025_psr_coefficients.csv"), row.names = FALSE)
  write.csv(vintage_2025$osr_coef_df, .rds("ws1_vintage_2025_osr_coefficients.csv"), row.names = FALSE)
  write.csv(vintage_2025$dsr_coef_df, .rds("ws1_vintage_2025_dsr_coefficients.csv"), row.names = FALSE)

  # ---- FIDELITY CHECK vs shipped CSVs (Section 6 falsifier 3) ----
  cli::cli_h2("Fidelity check: vintage_2024 vs shipped inst/extdata/*.csv")
  shipped_psr <- utils::read.csv(file.path(TORP_PATH, "inst", "extdata", "psr_coefficients.csv"))
  shipped_osr <- utils::read.csv(file.path(TORP_PATH, "inst", "extdata", "osr_coefficients.csv"))
  shipped_dsr <- utils::read.csv(file.path(TORP_PATH, "inst", "extdata", "dsr_coefficients.csv"))

  .fidelity <- function(shipped, mine, label) {
    m <- merge(shipped, mine, by = "stat_name", suffixes = c("_shipped", "_mine"))
    m$abs_diff <- abs(m$beta_shipped - m$beta_mine)
    ok_n <- sum(!is.na(m$beta_shipped) & !is.na(m$beta_mine))
    cor_val <- if (ok_n >= 3 && stats::sd(m$beta_shipped) > 0 && stats::sd(m$beta_mine) > 0) {
      stats::cor(m$beta_shipped, m$beta_mine)
    } else NA_real_
    cat(sprintf("\n%s: n_matched=%d cor=%.4f mean_abs_diff=%.4f max_abs_diff=%.4f (stat=%s)\n",
                label, nrow(m), cor_val, mean(m$abs_diff, na.rm = TRUE), max(m$abs_diff, na.rm = TRUE),
                m$stat_name[which.max(m$abs_diff)]))
    cat("  Top 5 largest diffs:\n")
    print(head(m[order(-m$abs_diff), c("stat_name", "beta_shipped", "beta_mine", "abs_diff")], 5), row.names = FALSE)
    list(cor = cor_val, mean_abs_diff = mean(m$abs_diff, na.rm = TRUE), n = nrow(m))
  }

  fid_psr <- .fidelity(shipped_psr, vintage_2024$psr_coef_df, "PSR fidelity")
  fid_osr <- .fidelity(shipped_osr, vintage_2024$osr_coef_df, "OSR fidelity")
  fid_dsr <- .fidelity(shipped_dsr, vintage_2024$dsr_coef_df, "DSR fidelity")

  fidelity_pass <- all(c(fid_psr$cor, fid_osr$cor, fid_dsr$cor) > 0.9, na.rm = TRUE) &&
    !any(is.na(c(fid_psr$cor, fid_osr$cor, fid_dsr$cor)))
  cat(sprintf("\n=== FIDELITY CHECK VERDICT (Section 6 falsifier 3): %s ===\n",
              ifelse(fidelity_pass, "PASS -- vintage_2024 approximately reproduces shipped coefficients, safe to proceed",
                     "FAIL -- STOP: provenance of shipped coefficients is unclear, diagnose before proceeding")))

  # ---- Beta drift table (report regardless of fidelity outcome, per plan) ----
  cli::cli_h2("Beta drift: vintage_2024 (anchor 2024-12-31) vs vintage_2025 (anchor 2025-12-31)")
  drift <- merge(vintage_2024$psr_coef_df[, c("stat_name", "beta")],
                 vintage_2025$psr_coef_df[, c("stat_name", "beta")],
                 by = "stat_name", suffixes = c("_v2024", "_v2025"))
  drift$delta <- drift$beta_v2025 - drift$beta_v2024
  drift$abs_delta <- abs(drift$delta)
  drift <- drift[order(-drift$abs_delta), ]
  cat("\nTop 15 PSR beta movers, 2024-vintage -> 2025-vintage:\n")
  print(head(drift, 15), row.names = FALSE)
  cat(sprintf("\nMargin-model test RMSE: v2024 (on 2025 holdout) = %.3f | v2025 (on 2026 holdout, n=%d) = %.3f\n",
              vintage_2024$margin_test_rmse, vintage_2025$n_test, vintage_2025$margin_test_rmse))

  write.csv(drift, .rds("ws1_beta_drift.csv"), row.names = FALSE)
  saveRDS(list(fid_psr = fid_psr, fid_osr = fid_osr, fid_dsr = fid_dsr,
               fidelity_pass = fidelity_pass, drift = drift),
          .rds("ws1_fit_summary.rds"))
  cli::cli_alert_success("Saved ws1_fit_summary.rds, ws1_beta_drift.csv, vintage coefficient CSVs")
}

# ================================================================
# Stage: psr_histories -- full-history psr/osr/dsr via
# calculate_psr_components() for each vintage (point-in-time-safe stat
# ratings already; only betas change), then assemble the walk-forward
# candidate (2025 rows <- vintage_2024, 2026 rows <- vintage_2025, earlier
# seasons <- vintage_2024) and the vintage_2025-everywhere candidate.
# ================================================================
if (stage %in% c("psr_histories", "all")) {
  cli::cli_h1("WS1 psr_histories: full-history PSR/OSR/DSR per vintage")

  vintage_2024 <- readRDS(.rds("ws1_vintage_2024.rds"))
  vintage_2025 <- readRDS(.rds("ws1_vintage_2025.rds"))

  skills <- load_player_stat_ratings(TRUE)
  cli::cli_inform("player_stat_ratings: {nrow(skills)} rows, seasons {paste(sort(unique(skills$season)), collapse=', ')}")

  psr_v2024 <- calculate_psr_components(skills, vintage_2024$psr_coef_df, vintage_2024$osr_coef_df,
                                         vintage_2024$dsr_coef_df, center = TRUE)
  psr_v2025 <- calculate_psr_components(skills, vintage_2025$psr_coef_df, vintage_2025$osr_coef_df,
                                         vintage_2025$dsr_coef_df, center = TRUE)

  cli::cli_inform("psr_v2024: {nrow(psr_v2024)} player-rounds | psr_v2025: {nrow(psr_v2025)} player-rounds")

  # Walk-forward: season <= 2025 -> vintage_2024, season >= 2026 -> vintage_2025
  psr_walkforward <- rbind(
    psr_v2024[psr_v2024$season <= 2025, ],
    psr_v2025[psr_v2025$season >= 2026, ]
  )
  cli::cli_inform("psr_walkforward: {nrow(psr_walkforward)} player-rounds ({sum(psr_walkforward$season <= 2025)} from v2024, {sum(psr_walkforward$season >= 2026)} from v2025)")

  # vintage_2025-everywhere: all rows from vintage_2025 (2026-screen-only legal)
  psr_v2025_everywhere <- psr_v2025

  saveRDS(list(psr_v2024 = psr_v2024, psr_v2025 = psr_v2025,
               psr_walkforward = psr_walkforward, psr_v2025_everywhere = psr_v2025_everywhere),
          .rds("ws1_psr_histories.rds"))
  cli::cli_alert_success("Saved ws1_psr_histories.rds")
}

# ================================================================
# Stage: rebuild_walkforward / rebuild_v2025 -- team_mdl_df rebuilds
# ================================================================
if (stage %in% c("rebuild_walkforward", "all")) {
  cli::cli_h1("WS1 rebuild: team_mdl_df with walk-forward psr_df")
  histories <- readRDS(.rds("ws1_psr_histories.rds"))
  t0 <- Sys.time()
  team_mdl_df_wf <- .rebuild_team_mdl_df_with_psr(histories$psr_walkforward)
  cli::cli_inform("Rebuild (walk-forward) completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  cat(sprintf("team_mdl_df_wf: %d rows, seasons %s\n", nrow(team_mdl_df_wf),
              paste(sort(unique(team_mdl_df_wf$season.x)), collapse = ", ")))
  saveRDS(team_mdl_df_wf, .rds("ws1_team_mdl_df_walkforward.rds"))
  cli::cli_alert_success("Saved ws1_team_mdl_df_walkforward.rds")
}

if (stage %in% c("rebuild_v2025", "all")) {
  cli::cli_h1("WS1 rebuild: team_mdl_df with vintage_2025-everywhere psr_df")
  histories <- readRDS(.rds("ws1_psr_histories.rds"))
  t0 <- Sys.time()
  team_mdl_df_v2025 <- .rebuild_team_mdl_df_with_psr(histories$psr_v2025_everywhere)
  cli::cli_inform("Rebuild (v2025-everywhere) completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  cat(sprintf("team_mdl_df_v2025: %d rows, seasons %s\n", nrow(team_mdl_df_v2025),
              paste(sort(unique(team_mdl_df_v2025$season.x)), collapse = ", ")))
  saveRDS(team_mdl_df_v2025, .rds("ws1_team_mdl_df_v2025.rds"))
  cli::cli_alert_success("Saved ws1_team_mdl_df_v2025.rds")
}

# ================================================================
# Stage: screen_walkforward / screen_v2025 -- 2026 screen (production
# trainers, identical to WS0 baseline), boot vs cached WS0 baseline screen
# preds (harness setup is byte-identical -- reuse per coordinator note).
# ================================================================
.baseline_screen <- function() {
  b <- .rds("ws0_baseline_screen.rds")
  if (!file.exists(b)) cli::cli_abort("WS0 ws0_baseline_screen.rds not found -- run WS0 first")
  readRDS(b)
}
.baseline_pool <- function() {
  b <- .rds("ws0_baseline_pool.rds")
  if (!file.exists(b)) cli::cli_abort("WS0 ws0_baseline_pool.rds not found -- run WS0 first")
  readRDS(b)
}

run_ws1_screen <- function(team_mdl_df, label, out_key) {
  base <- .baseline_screen()
  t0 <- Sys.time()
  roll <- run_rolling_eval(
    team_mdl_df, TEST_SEASONS,
    gam_trainer = .train_match_gams,
    xgb_trainer = .train_xgb_fixed,
    extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("{label} 2026 screen completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  preds_norecal <- roll$input_blend_preds
  m_norecal <- .compute_metrics(preds_norecal)
  preds <- v1a_recal_own(preds_norecal)
  m <- .compute_metrics(preds)

  .print_metrics(base$metrics, "G4 baseline + V1a recal, 2026 screen")
  .print_metrics(m_norecal, sprintf("%s, no recal, 2026 screen", label))
  .print_metrics(m, sprintf("%s + V1a recal, 2026 screen", label))

  boot_vs_base <- boot_mae_diff(preds, base$preds, B = 2000)
  cat(sprintf("\nboot_mae_diff(%s+recal - G4 baseline+recal, 2026 screen): N=%d deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
              label, boot_vs_base$n_matches, boot_vs_base$mae_diff, boot_vs_base$mae_ci[1], boot_vs_base$mae_ci[2], boot_vs_base$brier_diff))

  improved <- m$mae < base$metrics$mae
  cat(sprintf("Section 6 falsifier 2 check (screen improvement over baseline): %s (%.3f vs %.3f)\n",
              ifelse(improved, "IMPROVED", "NO IMPROVEMENT"), m$mae, base$metrics$mae))

  out <- list(label = label, roll = roll, preds_norecal = preds_norecal, metrics_norecal = m_norecal,
              preds = preds, metrics = m, boot_vs_baseline_screen = boot_vs_base, improved = improved)
  saveRDS(out, .rds(paste0("ws1_", out_key, "_screen.rds")))
  cli::cli_alert_success("Saved ws1_{out_key}_screen.rds")
  invisible(out)
}

if (stage %in% c("screen_walkforward", "all")) {
  cli::cli_h1("WS1 screen: walk-forward candidate -- 2026")
  team_mdl_df_wf <- readRDS(.rds("ws1_team_mdl_df_walkforward.rds"))
  run_ws1_screen(team_mdl_df_wf, "Walk-forward (v2024 on 2025, v2025 on 2026)", "walkforward")
}

if (stage %in% c("screen_v2025", "all")) {
  cli::cli_h1("WS1 screen: vintage_2025-everywhere candidate -- 2026 (screen ONLY, G6 legality)")
  team_mdl_df_v2025 <- readRDS(.rds("ws1_team_mdl_df_v2025.rds"))
  run_ws1_screen(team_mdl_df_v2025, "Vintage_2025-everywhere", "v2025")
}

# ================================================================
# Stage: confirm_walkforward -- pooled 2025:2026 confirm, walk-forward
# candidate ONLY (vintage_2025-everywhere is NOT G6-legal on the pooled
# window's 2025 portion -- those betas saw 2025 outcomes in training).
# ================================================================
if (stage %in% c("confirm_walkforward", "all")) {
  cli::cli_h1("WS1 confirm: walk-forward candidate -- pooled 2025:2026")
  team_mdl_df_wf <- readRDS(.rds("ws1_team_mdl_df_walkforward.rds"))
  base <- .baseline_pool()

  t0 <- Sys.time()
  roll <- run_rolling_eval(
    team_mdl_df_wf, CONFIRM_SEASONS,
    gam_trainer = .train_match_gams,
    xgb_trainer = .train_xgb_fixed,
    extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("Walk-forward pooled confirm completed in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  preds_norecal <- roll$input_blend_preds
  m_norecal <- .compute_metrics(preds_norecal)
  preds <- v1a_recal_own(preds_norecal)
  m <- .compute_metrics(preds)

  .print_metrics(base$metrics, "G4 baseline + V1a recal, pooled")
  .print_metrics(m, "Walk-forward + V1a recal, pooled")

  boot_vs_base <- boot_mae_diff(preds, base$preds, B = 2000)
  ci_excl_0 <- (boot_vs_base$mae_ci[1] > 0 && boot_vs_base$mae_ci[2] > 0) ||
    (boot_vs_base$mae_ci[1] < 0 && boot_vs_base$mae_ci[2] < 0)
  brier_ok <- boot_vs_base$brier_diff <= 0.002
  improved <- boot_vs_base$mae_diff < 0
  ship_pass <- ci_excl_0 && improved && brier_ok

  cat(sprintf(
    "\nboot_mae_diff(Walk-forward+recal - G4 baseline+recal, pooled): N=%d deltaMAE=%+.3f 95%%CI[%+.3f,%+.3f] deltaBrier=%+.5f\n",
    boot_vs_base$n_matches, boot_vs_base$mae_diff, boot_vs_base$mae_ci[1], boot_vs_base$mae_ci[2], boot_vs_base$brier_diff
  ))
  cat(sprintf("G3 ship gate (CI excludes 0, favours walk-forward, Brier not worse by >0.002): %s\n", ship_pass))
  cat("NOTE: even if this passes G3, plan Section 4 shape 2 applies -- this is a published-rating INPUT change (PSR coefficient vintage).\n")
  cat("Ship shape = PROPOSAL ONLY to Pete via DECISIONS.md; no integration performed by this script.\n")

  out <- list(roll = roll, preds_norecal = preds_norecal, metrics_norecal = m_norecal,
              preds = preds, metrics = m, boot_vs_baseline_pool = boot_vs_base, ship_pass = ship_pass)
  saveRDS(out, .rds("ws1_walkforward_confirm.rds"))
  cli::cli_alert_success("Saved ws1_walkforward_confirm.rds")
}

# ================================================================
# Stage: summary
# ================================================================
if (stage %in% c("summary", "all")) {
  cli::cli_h1("WS1 Final Summary: PSR coefficient vintage refresh")

  load_if <- function(f) if (file.exists(.rds(f))) readRDS(.rds(f)) else NULL
  fit_summary <- load_if("ws1_fit_summary.rds")
  screen_wf   <- load_if("ws1_walkforward_screen.rds")
  screen_v25  <- load_if("ws1_v2025_screen.rds")
  confirm_wf  <- load_if("ws1_walkforward_confirm.rds")
  base_screen <- if (file.exists(.rds("ws0_baseline_screen.rds"))) readRDS(.rds("ws0_baseline_screen.rds")) else NULL
  base_pool   <- if (file.exists(.rds("ws0_baseline_pool.rds"))) readRDS(.rds("ws0_baseline_pool.rds")) else NULL

  if (!is.null(fit_summary)) {
    cat(sprintf("\n=== Fidelity check: %s ===\n", ifelse(fit_summary$fidelity_pass, "PASS", "FAIL")))
    cat(sprintf("PSR cor=%.4f OSR cor=%.4f DSR cor=%.4f\n",
                fit_summary$fid_psr$cor, fit_summary$fid_osr$cor, fit_summary$fid_dsr$cor))
    cat("\nTop beta movers (2024-vintage -> 2025-vintage):\n")
    print(head(fit_summary$drift, 10), row.names = FALSE)
  }

  if (!is.null(base_screen)) cat(sprintf("\nG4 baseline, 2026 screen: MAE=%.3f slope=%.3f\n", base_screen$metrics$mae, base_screen$metrics$slope))
  if (!is.null(base_pool)) cat(sprintf("G4 baseline, pooled: MAE=%.3f slope=%.3f\n", base_pool$metrics$mae, base_pool$metrics$slope))

  if (!is.null(screen_wf)) {
    cat(sprintf("\nWalk-forward, 2026 screen: MAE=%.3f deltaMAE=%+.3f CI[%+.3f,%+.3f]\n",
                screen_wf$metrics$mae, screen_wf$boot_vs_baseline_screen$mae_diff,
                screen_wf$boot_vs_baseline_screen$mae_ci[1], screen_wf$boot_vs_baseline_screen$mae_ci[2]))
  }
  if (!is.null(screen_v25)) {
    cat(sprintf("Vintage_2025-everywhere, 2026 screen: MAE=%.3f deltaMAE=%+.3f CI[%+.3f,%+.3f]\n",
                screen_v25$metrics$mae, screen_v25$boot_vs_baseline_screen$mae_diff,
                screen_v25$boot_vs_baseline_screen$mae_ci[1], screen_v25$boot_vs_baseline_screen$mae_ci[2]))
  }
  if (!is.null(confirm_wf)) {
    cat(sprintf("\nWalk-forward, POOLED: MAE=%.3f deltaMAE=%+.3f CI[%+.3f,%+.3f] SHIP_PASS=%s\n",
                confirm_wf$metrics$mae, confirm_wf$boot_vs_baseline_pool$mae_diff,
                confirm_wf$boot_vs_baseline_pool$mae_ci[1], confirm_wf$boot_vs_baseline_pool$mae_ci[2],
                confirm_wf$ship_pass))
  }

  if (!is.null(screen_wf) && !is.null(screen_v25)) {
    cat(sprintf("\nSection 6 falsifier 2 (staleness-is-the-issue check): vintage_2025-on-2026 %s vs vintage_2024-on-2026(walk-forward uses v2024 through 2025, but both candidates use v2025 FOR 2026, so this is the sharpest single-vintage read): %s\n",
                ifelse(screen_v25$improved, "improved", "did not improve"),
                ifelse(screen_v25$improved, "staleness IS plausibly an issue -- refresh has room", "staleness is NOT the issue this season -- do not recommend annual refresh on tidiness grounds alone")))
  }

  saveRDS(list(fit_summary = fit_summary, screen_wf = screen_wf, screen_v25 = screen_v25,
               confirm_wf = confirm_wf, base_screen = base_screen, base_pool = base_pool),
          .rds("ws1_final_summary.rds"))
  cli::cli_alert_success("Saved ws1_final_summary.rds")
}
