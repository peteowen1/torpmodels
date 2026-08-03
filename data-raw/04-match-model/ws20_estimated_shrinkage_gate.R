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
# PERFORMANCE REVIEW (per Pete's standing instruction). Cost is dominated by
# three production rating rebuilds at ~12 min each, and that is deliberate: the
# fast path exists for an optimiser's inner loop, but a GATE must run the real
# .build_epr_season() or it is not testing what would ship. The v2 baseline
# ratings are reused from cache. Nothing quadratic; nothing else worth
# shortcutting.
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

arms <- list()
arms[["v2 production"]] <- as.data.table(arrow::read_parquet(
  file.path(OUT_DIR, "epv3_ratings_v2.parquet")))
arms[["v2 + est prior_games"]] <- build_ratings(
  "epv3_player_game_v2.parquet", "v2 pg", pset("v2", TRUE, FALSE))
arms[["v2 + est pg + decay"]] <- build_ratings(
  "epv3_player_game_v2.parquet", "v2 pg+decay", pset("v2", TRUE, TRUE))
arms[["v3 + est pg + decay"]] <- build_ratings(
  "epv3_player_game_v3.parquet", "v3 pg+decay", pset("v3", TRUE, TRUE))

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

arrow::write_parquet(preds, file.path(OUT_DIR, "epv3_shrinkage_gate_preds.parquet"))
close(con)
cat("\nDone\n")
