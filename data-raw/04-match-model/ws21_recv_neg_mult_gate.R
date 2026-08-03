# Does dropping the negative reception term improve match prediction?
# ==================================================================
# EPV_RECV_NEG_MULT = 1.0 credits a player the full delta_epv when he is the
# lead_player on a possession CHANGE -- i.e. when the opposition turned it over
# to him. Measured at player level over 56,576 player-games, that term is noise:
# setting the multiplier to 0 raises reception reliability from 0.2507 to 0.3086
# (+23%) while count-dependence FALLS 0.53 -> 0.297, and the gain survives at
# lag 8 (0.2409 -> 0.2878).
#
# WHY THIS ONE IS DIFFERENT FROM EVERYTHING ELSE GATED TONIGHT. The match model
# already consumes all four channel diffs plus epr_diff in every GAM, so it
# reweights channels itself -- which is why shrinkage, points scaling and 3-vs-4
# channels were all absorbed. Those were rescalings of information it already
# had. This changes WHAT IS MEASURED: a whole class of events stops contributing.
# The model cannot reweight its way to that.
#
# Run on BOTH engines. v2 is production, so a win there is shippable now; v3 is
# where the reception channel is chain-only and the effect should be largest.
#
# PERFORMANCE, measured not assumed (epv3_profile_ratings_build.R): a rating
# rebuild is ~3 min, the rolling eval ~20 min per arm, and the eval has no safe
# speedup -- the 3x parallel path is not ship-gate-safe. So cost is set by ARM
# COUNT. Four arms here: two engines x {baseline, neg_mult 0}. ~90 min.
# Rating builds are cached.
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

con <- file(file.path(OUT_DIR, "epv3_recv_neg_gate.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con) }
say_dt <- function(x, n = 40) for (l in capture.output(print(utils::head(x, n)))) say(l)

say("=== Gate: EPV_RECV_NEG_MULT 1.0 -> 0 ===")
say("run at ", format(Sys.time()), " | test seasons ", paste(TEST_SEASONS, collapse = "-"))

pbp    <- load_pbp(TRUE)
stats_ <- load_player_stats(TRUE)
teams  <- load_teams(TRUE)
chains <- load_chains(TRUE)
shared_stat_ratings <- get_player_stat_ratings(current = FALSE)
shared_fixtures     <- load_fixtures(TRUE)
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                   error = function(e) NULL)

build_pgd <- function(engine, neg_mult, tag) {
  f <- file.path(OUT_DIR, paste0("epv3_pgd_", tag, ".parquet"))
  if (file.exists(f)) {
    cli::cli_alert_info("Reusing cached pgd {tag}")
    return(as.data.table(arrow::read_parquet(f)))
  }
  p <- default_epv_params()
  p$recv_neg_mult <- neg_mult
  d <- create_player_game_data(pbp, stats_, teams, chains,
                               epv_params = p, epv_engine = engine)
  arrow::write_parquet(d, f)
  as.data.table(d)
}

build_ratings <- function(pgd, tag) {
  f <- file.path(OUT_DIR, paste0("epv3_rt_", tag, ".parquet"))
  if (file.exists(f)) {
    cli::cli_alert_info("Reusing cached ratings {tag}")
    return(as.data.table(arrow::read_parquet(f)))
  }
  d <- adjust_epv_for_opponents(as.data.table(copy(pgd)))
  if (isTRUE(EPV_LEVEL_CENTRE)) d <- centre_epv_by_position(d)
  out <- rbindlist(lapply(sort(unique(d$season)), function(s) {
    sr <- if (s >= 2024) 0 else 1
    mr <- if (s == get_afl_season()) get_afl_week(type = "next") else 28
    torp:::.build_epr_season(s, sr:mr, d, shared_stat_ratings, shared_fixtures)
  }), use.names = TRUE, fill = TRUE)
  if (isTRUE(EPR_POSITION_CENTRE)) out <- centre_epr_by_position(out)
  if (!is.null(psr_df) && nrow(psr_df) > 0 && "psr" %in% names(psr_df)) {
    out <- calculate_torp(out, psr_df)
  }
  out <- as.data.table(out)
  arrow::write_parquet(out, f)
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

specs <- list(
  list(tag = "v2_base", engine = "v2", nm = 1.0, label = "v2 baseline"),
  list(tag = "v2_neg0", engine = "v2", nm = 0.0, label = "v2 recv_neg_mult=0"),
  list(tag = "v3_base", engine = "v3", nm = 1.0, label = "v3 baseline"),
  list(tag = "v3_neg0", engine = "v3", nm = 0.0, label = "v3 recv_neg_mult=0")
)

rts <- list()
for (s in specs) {
  cli::cli_h1(s$label)
  pgd <- build_pgd(s$engine, s$nm, s$tag)
  rts[[s$label]] <- build_ratings(pgd, s$tag)
}

say("")
say("--- ARMS GUARD: each neg_mult=0 arm must differ from its baseline ---")
k <- c("player_id", "season", "round")
for (e in c("v2", "v3")) {
  b <- rts[[paste0(e, " baseline")]]; n <- rts[[paste0(e, " recv_neg_mult=0")]]
  cm <- merge(b[, c(k, "epr"), with = FALSE], n[, c(k, "epr"), with = FALSE],
              by = k, suffixes = c("_b", "_n"))
  d <- mean(abs(cm$epr_b - cm$epr_n), na.rm = TRUE)
  say(sprintf("  %-4s mean|diff| epr %.5f  cor %.4f", e, d,
              cor(cm$epr_b, cm$epr_n, use = "complete.obs")))
  if (d < 1e-9) say("    !! identical -- arm not live")
}

preds <- rbindlist(lapply(names(rts), function(nm) run_arm(nm, rts[[nm]])),
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
say("=== GATE: pooled 2025-26 ===")
say_dt(preds[, metrics(.SD), by = arm], 12)
say("")
say("--- by season (reported, not decided on) ---")
say_dt(preds[, metrics(.SD, 2025), by = arm], 12)
say_dt(preds[, metrics(.SD, 2026), by = arm], 12)

say("")
say("--- paired, each engine against its own baseline ---")
for (e in c("v2", "v3")) {
  b <- preds[arm == paste0(e, " baseline"), .(match_id, e0 = abs(pred_margin - margin))]
  n <- preds[arm == paste0(e, " recv_neg_mult=0"), .(match_id, e1 = abs(pred_margin - margin))]
  m <- merge(b, n, by = "match_id"); d <- m$e1 - m$e0
  ci <- t.test(d)$conf.int
  say(sprintf("  %-4s dMAE %+.4f  95%% CI [%+.4f, %+.4f]  (negative = BETTER)",
              e, mean(d), ci[1], ci[2]))
}
say("")
say("This is the one change tonight that alters WHAT IS MEASURED rather than")
say("rescaling it, so unlike shrinkage or points-scaling the model cannot")
say("reweight it away. If it is still neutral, that is strong evidence the")
say("match model is insensitive to rating quality per se.")

arrow::write_parquet(preds, file.path(OUT_DIR, "epv3_recv_neg_gate_preds.parquet"))
close(con)
cat("\nDone\n")
