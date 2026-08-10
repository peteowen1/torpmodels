# Do the MEASURED shrinkage and decay constants beat production?
# ==============================================================
# prior_games = 3.0 is live on every channel. Measured off 56,576 player-games,
# the requirement is 7.8-12.2 for v2's channels and 11.4-24.4 for v3's, because
# a single game is a far noisier read on a player than production assumes
# (reliability 0.13-0.15 for v2's channels, 0.06-0.10 for v3's). Production also
# forgets 1.8-3.5x too fast on three of four channels.
#
# These are MEASUREMENTS, not search results -- a variance ratio and a decay
# curve, both estimated without reference to any match outcome. So they carry no
# overfitting risk from this gate, and the gate is a genuine out-of-sample test
# rather than a re-scoring of something fitted to it.
#
# ARM ORDER IS DELIBERATE. v2 is production, so the v2 arms come first: if the
# corrected constants improve v2, that is shippable NOW and independent of
# whether v3 ever ships. Splitting prior_games from decay isolates which of the
# two corrections does the work.
#
# PERFORMANCE REVIEW -- MEASURED, after two wrong guesses.
# epv3_profile_ratings_build.R instruments the whole thing:
#
#   calculate_epr_stats_batch    26.6s   93.4%  of a season
#   per-round loop (29 calls)     1.7s    6.0%
#   load_player_details           0.2s    0.7%
#   => a full 6-season rebuild is ~3 MINUTES, not the 12 I asserted.
#
# So the rating builds were never the cost. The ROLLING EVAL is, at ~20 min per
# arm -- 50 test rounds x (5 GAMs + XGBoost), which is inherent. xgb_nrounds CV
# already runs once per eval rather than per round, so there is no win there.
#
# The only 3x lever is run_rolling_eval_parallel(), which is flagged NOT
# ship-gate-safe (xgboost thread nondeterminism), and this is effectively a ship
# gate for production constants. Swapping the proven fast path into the batch
# stage would save ~6 min of ~90 and is not worth any fidelity risk.
#
# So the optimisation is SCOPE, not speed: three arms instead of four. The v3
# arm is dropped -- v3's story is already known and it is not the shippable
# question. Rating builds are cached so a re-run is eval-only.
#
# And the unavoidable cost buys something: the fresh v2-production arm is
# compared against ws17's stored v2 predictions, which quantifies the harness's
# run-to-run noise. If that noise is comparable to the 0.184 MAE effects being
# chased, every comparison made today needs re-reading -- and nobody has
# measured it.
#
# Run detached: Start-Process Rscript -ArgumentList '"<this file>"'

suppressMessages({
  library(dplyr); library(data.table)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "epv3_shrinkage_gate.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con) }
say_dt <- function(x, n = 40) for (l in capture.output(print(utils::head(x, n)))) say(l)

say("=== Gate: measured shrinkage and decay vs production ===")
say("run at ", format(Sys.time()), " | test seasons ", paste(TEST_SEASONS, collapse = "-"))

est <- as.data.table(readRDS(file.path(OUT_DIR, "epv3_params_both_engines.rds")))
say("")
say("--- the constants under test ---")
say_dt(est[, .(engine, channel, reliability_lag1, prior_games, decay_days, decay_fit_r2)], 10)

pset <- function(eng, use_pg, use_decay) {
  e <- est[engine == eng]
  p <- list(loading = EPR_LOADING_DEFAULT)
  for (c in c("recv", "disp", "spoil", "hitout")) {
    row <- e[channel == c]
    p[[paste0("prior_games_", c)]] <- if (use_pg) row$prior_games else
      get(paste0("EPR_PRIOR_GAMES_", toupper(c)))
    p[[paste0("decay_", c)]] <- if (use_decay) row$decay_days else
      get(paste0("EPR_DECAY_", toupper(c)))
    p[[paste0("prior_rate_", c)]] <- get(paste0("EPR_PRIOR_RATE_", toupper(c)))
  }
  p
}

shared_stat_ratings <- get_player_stat_ratings(current = FALSE)
shared_fixtures     <- load_fixtures(TRUE)
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                   error = function(e) NULL)
teams <- load_teams(TRUE)

build_ratings <- function(pgd_file, label, epr_params) {
  cli::cli_h2("Ratings: {label}")
  d <- as.data.table(arrow::read_parquet(file.path(OUT_DIR, pgd_file)))
  d <- adjust_epv_for_opponents(d)
  if (isTRUE(EPV_LEVEL_CENTRE)) d <- centre_epv_by_position(d)
  seasons <- sort(unique(d$season))
  out <- rbindlist(lapply(seasons, function(s) {
    sr <- if (s >= 2024) 0 else 1
    mr <- if (s == get_afl_season()) get_afl_week(type = "next") else 28
    torp:::.build_epr_season(s, sr:mr, d, shared_stat_ratings, shared_fixtures,
                             epr_params = epr_params)
  }), use.names = TRUE, fill = TRUE)
  if (isTRUE(EPR_POSITION_CENTRE)) out <- centre_epr_by_position(out)
  if (!is.null(psr_df) && nrow(psr_df) > 0 && "psr" %in% names(psr_df)) {
    out <- calculate_torp(out, psr_df)
  }
  out <- as.data.table(out)
  cli::cli_inform("{label}: epr sd {round(sd(out$epr, na.rm = TRUE), 3)}")
  out
}

build_with_ratings <- function(torp_df) {
  ag <- file_reader("stadium_data", "reference-data")
  fx <- .build_fixtures_df(shared_fixtures)
  trt <- .build_team_ratings_df(teams, torp_df, psr_df)
  trf <- .build_match_features(fx, trt, ag)
  wx <- .load_match_weather(shared_fixtures, ag, NULL, get_afl_season())
  .build_team_mdl_df(trf, load_results(TRUE), load_xg(TRUE), wx,
                     max(as.Date(fx$utc_start_time), na.rm = TRUE))
}

.bits <- function(pw, hw) mean(ifelse(hw == 1, 1 + log2(pw),
                              ifelse(hw == 0, 1 + log2(1 - pw),
                                     1 + 0.5 * log2(pw * (1 - pw)))))

run_arm <- function(label, torp_df) {
  tm <- build_with_ratings(torp_df)
  ft <- grep("^(epr|psr|torp|elo|xelo).*_diff$|^(epr|psr|torp)\\.[xy]$", names(tm), value = TRUE)
  keep <- stats::complete.cases(tm[, ft, drop = FALSE])
  if (any(!keep)) tm <- tm[keep, , drop = FALSE]
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff", cv_extra_feature_cols = "xelo_diff")
  p <- unique(as.data.table(roll$input_blend_preds), by = "match_id")
  p[, arm := label]; p
}

cached_build <- function(tag, pgd_file, label, epr_params) {
  f <- file.path(OUT_DIR, paste0("epv3_ratings_", tag, ".parquet"))
  if (file.exists(f)) {
    cli::cli_alert_info("Reusing cached {tag} ratings")
    return(as.data.table(arrow::read_parquet(f)))
  }
  r <- build_ratings(pgd_file, label, epr_params)
  arrow::write_parquet(r, f)
  r
}

# Three arms, not four. The v3 arm is dropped: v3's story is known and it is not
# the shippable question. Scope is the only safe lever -- see the performance
# note at the top.
arms <- list()
arms[["v2 production"]] <- as.data.table(arrow::read_parquet(
  file.path(OUT_DIR, "epv3_ratings_v2.parquet")))
arms[["v2 + est prior_games"]] <- cached_build(
  "v2_pg", "epv3_player_game_v2.parquet", "v2 pg", pset("v2", TRUE, FALSE))
arms[["v2 + est pg + decay"]] <- cached_build(
  "v2_pgdecay", "epv3_player_game_v2.parquet", "v2 pg+decay", pset("v2", TRUE, TRUE))

say("")
say("--- ARMS GUARD ---")
base <- arms[["v2 production"]]
k <- c("player_id", "season", "round")
for (nm in setdiff(names(arms), "v2 production")) {
  cm <- merge(base[, c(k, "epr"), with = FALSE], arms[[nm]][, c(k, "epr"), with = FALSE],
              by = k, suffixes = c("_a", "_b"))
  d <- mean(abs(cm$epr_a - cm$epr_b), na.rm = TRUE)
  say(sprintf("  %-24s mean|diff| epr %.5f  cor %.4f", nm, d,
              cor(cm$epr_a, cm$epr_b, use = "complete.obs")))
  if (d < 1e-9) say("    !! identical to production -- arm not live")
}

preds <- rbindlist(lapply(names(arms), function(nm) run_arm(nm, arms[[nm]])),
                   use.names = TRUE, fill = TRUE)
common <- Reduce(intersect, split(preds$match_id, preds$arm))
preds <- preds[match_id %in% common]
say("")
say("--- same-games: ", length(common), " matches in every arm ---")

metrics <- function(p, seasons = NULL) {
  d <- if (is.null(seasons)) p else p[season %in% seasons]
  d <- d[is.finite(margin) & is.finite(pred_margin) & is.finite(pred_win)]
  hw <- ifelse(d$margin > 0, 1, ifelse(d$margin == 0, 0.5, 0))
  data.table(n = nrow(d),
             MAE = round(mean(abs(d$pred_margin - d$margin)), 4),
             RMSE = round(sqrt(mean((d$pred_margin - d$margin)^2)), 4),
             bits = round(.bits(pmin(pmax(d$pred_win, 1e-6), 1 - 1e-6), hw), 4),
             Brier = round(mean((d$pred_win - hw)^2), 4),
             tips = sum((d$pred_margin > 0) == (d$margin > 0), na.rm = TRUE))
}
say("")
say("=== GATE: pooled 2025-26, all metrics ===")
say_dt(preds[, metrics(.SD), by = arm], 12)
say("")
say("--- by season (reported, not decided on) ---")
say_dt(preds[, metrics(.SD, 2025), by = arm], 12)
say_dt(preds[, metrics(.SD, 2026), by = arm], 12)

say("")
say("--- paired against v2 production ---")
ref <- preds[arm == "v2 production", .(match_id, e0 = abs(pred_margin - margin))]
for (nm in setdiff(names(arms), "v2 production")) {
  m <- merge(ref, preds[arm == nm, .(match_id, e = abs(pred_margin - margin))], by = "match_id")
  d <- m$e - m$e0; ci <- t.test(d)$conf.int
  say(sprintf("  %-24s dMAE %+.4f  95%% CI [%+.4f, %+.4f]  (negative = BETTER)",
              nm, mean(d), ci[1], ci[2]))
}
say("")
say("The v2 arms are the shippable question: production's prior_games = 3.0")
say("against a measured 7.8-12.2. If those arms win, it ships regardless of v3.")

# ---- Harness run-to-run noise, harvested from a cost we cannot avoid --------
# The v2-production arm was already evaluated by ws17 on identical ratings,
# seasons and trainers. Any difference between that run and this one is pure
# harness nondeterminism (xgboost threading). Nobody has quantified it, and it
# matters: if the noise floor is comparable to the 0.184 MAE effects being
# chased, every comparison made today needs re-reading.
old <- tryCatch(as.data.table(arrow::read_parquet(
  file.path(OUT_DIR, "epv3_match_preds.parquet"))), error = function(e) NULL)
if (!is.null(old) && "arm" %in% names(old) && any(old$arm == "v2")) {
  o <- unique(old[arm == "v2"], by = "match_id")[, .(match_id, pm_old = pred_margin,
                                                     pw_old = pred_win)]
  n <- preds[arm == "v2 production", .(match_id, pm_new = pred_margin,
                                       pw_new = pred_win, margin)]
  cm <- merge(o, n, by = "match_id")
  say("")
  say("=== HARNESS NOISE FLOOR (same arm, two independent runs) ===")
  say("matches compared: ", nrow(cm))
  say("mean |pred_margin difference| : ", round(mean(abs(cm$pm_new - cm$pm_old)), 4))
  say("max  |pred_margin difference| : ", round(max(abs(cm$pm_new - cm$pm_old)), 4))
  mae_o <- mean(abs(cm$pm_old - cm$margin)); mae_n <- mean(abs(cm$pm_new - cm$margin))
  say("MAE run 1 (ws17) ", round(mae_o, 4), " | MAE run 2 (here) ", round(mae_n, 4),
      " | difference ", round(mae_n - mae_o, 4))
  say("")
  say("Read that last number against the effects this session has been weighing:")
  say("v3 vs v2 was +0.184 MAE, and 3-vs-4 channels was +0.0057. If the noise")
  say("floor is of that order, those comparisons are not resolvable by one run.")
} else {
  say("")
  say("(ws17 predictions unavailable -- harness noise not quantified this run.)")
}

arrow::write_parquet(preds, file.path(OUT_DIR, "epv3_shrinkage_gate_preds.parquet"))
close(con)
cat("\nDone\n")
