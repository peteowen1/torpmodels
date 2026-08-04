# The match gate for the FINISHED v3, against what is live today.
#
# Everything gated before this compared v3-as-built against v2. v3 has since
# changed in four ways that all had to be settled first, so the old +0.184 MAE
# is not the number any more:
#
#   EPV_STANDARDISE_CHANNELS  contest channel EXCLUDED -- standardising a partly
#                             ruck-exclusive channel normalises away what it
#                             measures (it read 0.0% of variance when merged in
#                             points and standardised)
#   EPV3_STOP_ZERO_SUM        TRUE -- the stoppage term becomes a win/loss
#                             ledger instead of an attendance count
#   EPV3_POINTS_SCALE         applied -- one unit of each channel = one point
#   EPR_PRIOR_GAMES_*         3.0 -> measured
#
# Two changes that were tried this session and are NOT here, both reverted on
# measurement: EPV_RECV_NEG_MULT -> 0 (breaks conservation, +72.26 -> +38.69 EPV
# per team-match, and year-over-year repeatability FALLS) and the 3.14x ruck
# amplification (its justifying swing figure is ~93% centre-bounce reset).
#
# WHAT THIS GATE IS AND IS NOT FOR. It is not being used to choose v3 -- the
# rating-quality case is made elsewhere, and five separate changes have already
# come back neutral here because the match model consumes every channel diff
# plus epr_diff in all five GAMs and reweights them itself. It is here to bound
# the PRICE: how much MAE the finished rebuild costs against production. Pete's
# tolerance is "somewhat predictive, slightly worse is fine".
#
# Three arms so the price is attributable rather than a lump:
#   1  v2 production    -- exactly what is live: engine v2, neg_mult 1.0,
#                          prior_games 3.0, the global points scale
#   2  v3 final         -- everything above applied
#   3  v3 final, prior_games 3.0 -- isolates the shrinkage change, which is the
#                          one piece with no independent rating-quality gate
#
# PERFORMANCE: the rolling eval is ~20 min per arm and has no ship-gate-safe
# speedup, so cost is set by arm count. Rating builds are cached and shared with
# epv3_calibrate_final.R. ~75 min. Run detached:
#   Start-Process Rscript -ArgumentList '"<this file>"'

suppressMessages({
  library(dplyr); library(data.table); library(arrow)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "epv3_final_gate.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con); flush(con) }
say_dt <- function(x, n = 40) for (l in capture.output(print(utils::head(x, n)))) say(l)

set_const <- function(l) for (nm in names(l)) assignInNamespace(nm, l[[nm]], ns = "torp")

FIN <- readRDS(file.path(OUT_DIR, "epv3_finalise_ship.rds"))
CAL <- list(sub_scale = c(cont_aerial = 1, cont_stop = 1),
            points_scale = FIN$points_scale)
PG  <- FIN$prior_games
pg_of <- function(ch) PG[channel == ch, prior_games]

say("=== Final gate: finished v3 against production v2 ===")
say("run at ", format(Sys.time()), " | test seasons ", paste(TEST_SEASONS, collapse = "-"))
say("EPV3_SUB_SCALE    ", paste(names(CAL$sub_scale), round(CAL$sub_scale, 4), sep = "=", collapse = ", "))
say("EPV3_POINTS_SCALE ", paste(names(CAL$points_scale), round(CAL$points_scale, 4), sep = "=", collapse = ", "))
say("EPR_PRIOR_GAMES   ", paste(PG$channel, PG$prior_games, sep = "=", collapse = ", "))

pbp    <- load_pbp(TRUE); stats_ <- load_player_stats(TRUE)
teams  <- load_teams(TRUE); chains <- load_chains(TRUE)
shared_stat_ratings <- get_player_stat_ratings(current = FALSE)
shared_fixtures     <- load_fixtures(TRUE)
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                   error = function(e) NULL)

build_pgd <- function(tag, engine, neg_mult) {
  # The v3 arm's frame is the one epv3_finalise.R already built and fitted the
  # constants against. Reuse it rather than rebuilding: a rebuilt frame would be
  # equivalent but the constants were fitted on THIS one, and "the arm you
  # scored is the arm you shipped" is the check this repo keeps failing.
  f <- file.path(OUT_DIR, paste0("epv3_fin_pgd_", sub("^fin_", "", tag), ".parquet"))
  if (!file.exists(f)) f <- file.path(OUT_DIR, paste0("epv3_cal_pgd_", tag, ".parquet"))
  if (file.exists(f)) { cli::cli_alert_info("Reusing pgd {basename(f)}")
    d <- as.data.table(read_parquet(f))
  } else {
    p <- default_epv_params(); p$recv_neg_mult <- neg_mult
    d <- as.data.table(create_player_game_data(pbp, stats_, teams, chains,
                                               epv_params = p, epv_engine = engine))
    write_parquet(d, f)
  }
  setattr(d, "epv_engine", engine); d
}

build_ratings <- function(pgd, tag, engine) {
  f <- file.path(OUT_DIR, paste0("epv3_cal_rt_", tag, ".parquet"))
  if (file.exists(f)) { cli::cli_alert_info("Reusing ratings {tag}")
    return(as.data.table(read_parquet(f))) }
  d <- adjust_epv_for_opponents(as.data.table(copy(pgd)))
  setattr(d, "epv_engine", engine)
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
  out <- as.data.table(out); write_parquet(out, f); out
}

V2_CONST <- list(EPV_ENGINE = "v2", EPV3_CHANNELS = 3L,
                 EPV3_SUB_SCALE = c(cont_aerial = 1, cont_stop = 1),
                 EPV3_STOP_ZERO_SUM = FALSE,
                 EPV_STANDARDISE_CHANNELS = c("recv", "disp", "spoil"),
                 EPV3_POINTS_SCALE = c(recv = 1, disp = 1, cont_aerial = 1, cont_stop = 1),
                 EPR_PRIOR_RATE_RECV = -0.7 * 0.919, EPR_PRIOR_RATE_DISP = -0.7 * 0.919,
                 EPR_PRIOR_RATE_SPOIL = -0.3 * 0.919, EPR_PRIOR_RATE_HITOUT = -0.3 * 0.919,
                 EPR_PRIOR_GAMES_RECV = 3, EPR_PRIOR_GAMES_DISP = 3,
                 EPR_PRIOR_GAMES_SPOIL = 3, EPR_PRIOR_GAMES_HITOUT = 3)

v3_const <- function(prior_games) c(
  list(EPV_ENGINE = "v3", EPV3_CHANNELS = 3L,
       EPV3_SUB_SCALE = CAL$sub_scale, EPV3_POINTS_SCALE = CAL$points_scale,
       # The two structural decisions of 2026-08-04: the contest channel is not
       # standardised (standardising a partly ruck-exclusive channel normalises
       # away what it measures), and the stoppage term is a win/loss ledger
       # rather than an attendance count.
       EPV_STANDARDISE_CHANNELS = c("recv", "disp"),
       EPV3_STOP_ZERO_SUM = TRUE,
       EPR_PRIOR_RATE_RECV  = -0.7 * CAL$points_scale[["recv"]],
       EPR_PRIOR_RATE_DISP  = -0.7 * CAL$points_scale[["disp"]],
       EPR_PRIOR_RATE_SPOIL = -0.3 * CAL$points_scale[["cont_aerial"]],
       EPR_PRIOR_RATE_HITOUT = 0),
  if (prior_games) list(EPR_PRIOR_GAMES_RECV = pg_of("recv"),
                        EPR_PRIOR_GAMES_DISP = pg_of("disp"),
                        EPR_PRIOR_GAMES_SPOIL = pg_of("spoil"),
                        EPR_PRIOR_GAMES_HITOUT = 3)
  else list(EPR_PRIOR_GAMES_RECV = 3, EPR_PRIOR_GAMES_DISP = 3,
            EPR_PRIOR_GAMES_SPOIL = 3, EPR_PRIOR_GAMES_HITOUT = 3))

rts <- list()
set_const(V2_CONST)
rts[["v2 production"]] <- build_ratings(build_pgd("v2prod", "v2", 1.0), "v2prod", "v2")

set_const(v3_const(FALSE))
rts[["v3 final, prior_games 3"]] <-
  build_ratings(build_pgd("fin_ship", "v3", 1.0), "gate_v3_pg3", "v3")

set_const(v3_const(TRUE))
rts[["v3 final"]] <-
  build_ratings(build_pgd("fin_ship", "v3", 1.0), "gate_v3_final", "v3")

say("")
say("--- ARMS GUARD: no two arms may be identical ---")
k <- c("player_id", "season", "round")
nms <- names(rts)
for (i in 1:(length(nms) - 1)) for (j in (i + 1):length(nms)) {
  m <- merge(rts[[i]][, c(k, "epr"), with = FALSE], rts[[j]][, c(k, "epr"), with = FALSE],
             by = k, suffixes = c("_a", "_b"))
  d <- mean(abs(m$epr_a - m$epr_b), na.rm = TRUE)
  say(sprintf("  %-26s vs %-26s mean|diff| %.5f", nms[i], nms[j], d))
  if (d < 1e-9) say("    !! IDENTICAL -- an arm is not live")
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
  p[, arm := label]
  # Checkpoint after every arm -- a 20-minute arm lost to a crash in the next
  # one is 20 minutes that has to be paid again.
  write_parquet(p, file.path(OUT_DIR, paste0("epv3_final_gate_", gsub("[^a-z0-9]+", "_", tolower(label)), ".parquet")))
  p
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
  data.table(n = nrow(d), MAE = round(mean(abs(d$pred_margin - d$margin)), 4),
             RMSE = round(sqrt(mean((d$pred_margin - d$margin)^2)), 4),
             bits = round(.bits(pmin(pmax(d$pred_win, 1e-6), 1 - 1e-6), hw), 4),
             Brier = round(mean((d$pred_win - hw)^2), 4),
             tips = sum((d$pred_margin > 0) == (d$margin > 0), na.rm = TRUE))
}

say("")
say("=== GATE: pooled 2025-26 (the window the decision is made on) ===")
say_dt(preds[, metrics(.SD), by = arm], 12)
say("")
say("--- by season, reported not decided on ---")
say_dt(preds[, metrics(.SD, 2025), by = arm], 12)
say_dt(preds[, metrics(.SD, 2026), by = arm], 12)

say("")
say("--- paired against v2 production (negative = v3 BETTER) ---")
base <- preds[arm == "v2 production", .(match_id, e0 = abs(pred_margin - margin))]
for (a in setdiff(unique(preds$arm), "v2 production")) {
  x <- preds[arm == a, .(match_id, e1 = abs(pred_margin - margin))]
  m <- merge(base, x, by = "match_id"); d <- m$e1 - m$e0
  ci <- t.test(d)$conf.int
  say(sprintf("  %-26s dMAE %+.4f  95%% CI [%+.4f, %+.4f]  P(better) %.3f",
              a, mean(d), ci[1], ci[2], mean(d < 0)))
}
say("")
say("--- shrinkage alone (v3 final vs v3 final with prior_games 3) ---")
b <- preds[arm == "v3 final, prior_games 3", .(match_id, e0 = abs(pred_margin - margin))]
x <- preds[arm == "v3 final", .(match_id, e1 = abs(pred_margin - margin))]
m <- merge(b, x, by = "match_id"); d <- m$e1 - m$e0
ci <- t.test(d)$conf.int
say(sprintf("  dMAE %+.4f  95%% CI [%+.4f, %+.4f]", mean(d), ci[1], ci[2]))

write_parquet(preds, file.path(OUT_DIR, "epv3_final_gate_preds.parquet"))
say("")
say("done ", format(Sys.time()))
close(con)
cat("\nDone\n")
