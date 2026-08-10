# What does v3 actually cost against production? The number nobody has.
#
# WHY THIS HAS TO BE RE-RUN. ws22 reported v3 costing **+0.477 dMAE** against v2
# and that has been the headline ever since. It was measured off
# `epv3_fin_pgd_ship.parquet`, written 2026-08-04 23:22 -- three commits before
# the contest population was fixed. Rebuilding v3 from current code changes six
# columns across all 56,576 player-games, and the fitted constants in
# `epv3_finalise_ship.rds` were fitted against that same stale frame.
#
# Three things have since moved in v3's favour:
#   ~0.030   the contest-population fixes (the same ship arm read 25.8902 in
#            ws23 off the stale frame and 25.8598 in ws24 off current code)
#    0.077   the difficulty split (ws24, clean 3-arm)
#      --    a refitted contest scale that HALVED, 0.5226 -> 0.2634
#
# **Those do not compose.** Subtracting them from +0.477 would be adding dMAEs
# measured on three differently-constituted gates against three different
# baselines, which is not arithmetic that holds. The only way to know v3's price
# is to measure it, against production, with everything current.
#
# THREE ARMS:
#   1  v2 production          exactly what is live -- engine v2, the global 0.919
#                             points scale, prior_games 3.0 everywhere
#   2  v3 final               engine v3, difficulty split ON, points scale
#                             refitted and verified, prior_games INHERITED from
#                             the stale finalise run
#   3  v3 final + shrinkage   same, with EPR_PRIOR_GAMES_* refitted on this
#                             frame. Arm 2 vs arm 3 isolates it.
#
# Arm 3 exists so v3 gets its best shot. Losing to production with shrinkage
# tuned for channels that no longer exist would be a bad input to a ship
# decision, and refitting only for the winner would be worse.
#
# Every arm builds through cached_frame(on_stale = "abort") and the run pins its
# code fingerprint on the first call, so an arm built by different code aborts
# rather than being quietly compared.
#
# PERFORMANCE: 2 pgd builds (~7 min each), 5 rating builds (~2 min), 3 rolling
# evals (~20 min). ~100 min. Run detached, and do not edit torp/R/ while it runs.

suppressMessages({
  library(dplyr); library(data.table); library(arrow)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
source("C:/dev/torpverse/torp/data-raw/04-analysis/cache_guard.R")
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "epv3_v2_vs_v3_final.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con); flush(con) }
say_dt <- function(x, n = 40) for (l in capture.output(print(utils::head(x, n)))) say(l)
set_const <- function(l) for (nm in names(l)) assignInNamespace(nm, l[[nm]], ns = "torp")

# Set constants, run, RESTORE. The first version of this script did not restore,
# and it cost a full 100-minute run: the v2 builder set EPV3_STOP_ZERO_SUM =
# FALSE and never put it back, so the v3 player-game frame was built with v2's
# stoppage formula. Nothing errored -- it surfaced only because the fitted
# contest scale came out 0.3929 where ws24's identical arm read 0.4350.
#
# This is the same leak ws22 documented and fixed inside its own arm loop. It
# came back because the fix lived in one script instead of in a helper, so the
# helper is the fix this time.
with_const <- function(l, expr) {
  old <- lapply(names(l), function(nm) get(nm, envir = asNamespace("torp")))
  names(old) <- names(l)
  set_const(l)
  on.exit(set_const(old), add = TRUE)
  force(expr)
}

say("=== v2 production vs v3 final ===")
say("run at ", format(Sys.time()), " | test seasons ", paste(TEST_SEASONS, collapse = "-"))
say("code fingerprint: ", code_fingerprint())

pbp <- load_pbp(TRUE); stats_ <- load_player_stats(TRUE)
teams <- load_teams(TRUE); chains <- load_chains(TRUE)
shared_stat_ratings <- get_player_stat_ratings(current = FALSE)
shared_fixtures <- load_fixtures(TRUE)
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                   error = function(e) NULL)
res <- as.data.table(load_results(TRUE)); xg <- as.data.table(load_xg(TRUE))
tgt <- merge(res[, .(match_id = as.character(match_id), margin = home_score - away_score)],
             xg[, .(match_id = as.character(match_id), xmargin = xscore_diff)],
             by = "match_id")[is.finite(margin) & is.finite(xmargin)]

STALE_PG <- readRDS(file.path(OUT_DIR, "epv3_finalise_ship.rds"))$prior_games
say("inherited (stale-frame) prior_games: ",
    paste(STALE_PG$channel, round(STALE_PG$prior_games, 2), sep = "=", collapse = ", "))

V2_CONST <- list(EPV_ENGINE = "v2", EPV3_CHANNELS = 3L,
                 EPV3_SUB_SCALE = c(cont_aerial = 1, cont_stop = 1),
                 EPV3_STOP_ZERO_SUM = FALSE,
                 EPV_STANDARDISE_CHANNELS = c("recv", "disp", "spoil"),
                 EPV3_POINTS_SCALE = c(recv = 1, disp = 1, cont_aerial = 1, cont_stop = 1),
                 EPR_PRIOR_RATE_RECV = -0.7 * 0.919, EPR_PRIOR_RATE_DISP = -0.7 * 0.919,
                 EPR_PRIOR_RATE_SPOIL = -0.3 * 0.919, EPR_PRIOR_RATE_HITOUT = -0.3 * 0.919,
                 EPR_PRIOR_GAMES_RECV = 3, EPR_PRIOR_GAMES_DISP = 3,
                 EPR_PRIOR_GAMES_SPOIL = 3, EPR_PRIOR_GAMES_HITOUT = 3)
V3_BASE <- list(EPV_ENGINE = "v3", EPV3_CHANNELS = 3L,
                EPV3_SUB_SCALE = c(cont_aerial = 1, cont_stop = 1),
                EPV_STANDARDISE_CHANNELS = c("recv", "disp"),
                EPV3_STOP_ZERO_SUM = TRUE)
scale_const <- function(pts) list(
  EPV3_POINTS_SCALE = pts,
  EPR_PRIOR_RATE_RECV = -0.7 * pts[["recv"]], EPR_PRIOR_RATE_DISP = -0.7 * pts[["disp"]],
  EPR_PRIOR_RATE_SPOIL = -0.3 * pts[["cont_aerial"]], EPR_PRIOR_RATE_HITOUT = 0)
pg_const <- function(pg) list(
  EPR_PRIOR_GAMES_RECV = pg[["recv"]], EPR_PRIOR_GAMES_DISP = pg[["disp"]],
  EPR_PRIOR_GAMES_SPOIL = pg[["spoil"]], EPR_PRIOR_GAMES_HITOUT = 3)
UNIT <- c(recv = 1, disp = 1, cont_aerial = 1, cont_stop = 1)

build_ratings <- function(pgd, tag, engine) {
  f <- file.path(OUT_DIR, paste0("epv3_v2v3_rt_", tag, ".parquet"))
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
CH3 <- c("epr_recv", "epr_disp", "epr_spoil")
fit3 <- function(rt) {
  tr <- as.data.table(.build_team_ratings_df(teams, as.data.frame(rt), psr_df))
  h <- tr[team_type == "home"]; a <- tr[team_type == "away"]
  m <- merge(h[, c("match_id", CH3), with = FALSE], a[, c("match_id", CH3), with = FALSE],
             by = "match_id", suffixes = c("_h", "_a"))
  for (v in CH3) m[, (paste0("d_", v)) := get(paste0(v, "_h")) - get(paste0(v, "_a"))]
  m <- merge(m, tgt, by = "match_id")
  co <- summary(lm(as.formula(paste("xmargin ~ 0 +", paste0("d_", CH3, collapse = " + "))),
                   data = m))$coefficients
  setNames(co[, 1], CH3)
}

# Shrinkage from a variance ratio: within-player variance over between-player
# variance, on the frame the arm will actually use. Same procedure as
# epv3_finalise.R -- no model fitting, so it is cheap enough to do per arm.
fit_prior_games <- function(pgd) {
  d <- adjust_epv_for_opponents(as.data.table(copy(pgd)))
  setattr(d, "epv_engine", "v3")
  d <- centre_epv_by_position(d)
  d[, tog_safe := pmax(fifelse(is.na(time_on_ground_percentage), 100,
                               time_on_ground_percentage) / 100, 0.1)]
  sfx <- if (all(paste0("epv_", c("recv", "disp", "spoil"), "_oadj") %in% names(d))) "_oadj" else "_adj"
  hi <- d[time_on_ground_percentage > 50]
  out <- vapply(c("recv", "disp", "spoil"), function(c) {
    x <- hi[[paste0("epv_", c, sfx)]] / hi$tog_safe
    ok <- is.finite(x)
    agg <- data.table(pid = hi$player_id[ok], x = x[ok])[
      , .(n = .N, m = mean(x), ss = sum((x - mean(x))^2)), by = pid][n >= 3]
    s2w <- sum(agg$ss) / sum(agg$n - 1)
    tau2 <- var(agg$m) - s2w * mean(1 / agg$n)
    if (is.finite(tau2) && tau2 > 0) round(s2w / tau2, 2) else NA_real_
  }, numeric(1))
  if (anyNA(out)) {
    cli::cli_warn("prior_games unfittable for {names(out)[is.na(out)]}; falling back to 3.")
    out[is.na(out)] <- 3
  }
  out
}

# ------------------------------------------------------------------- build
# Each frame is built under its OWN full constant set, restored afterwards, so
# no arm can inherit another's. The constants that reach
# create_player_game_data() are EPV3_STOP_ZERO_SUM, EPV3_CHANNELS,
# EPV3_SUB_SCALE and EPV_DIFFICULTY_SPLIT -- the points scale and prior_games
# act later, at the rating build.
pgd_v2 <- cached_frame("v2v3_pgd_v2", function() {
  with_const(V2_CONST,
    as.data.table(create_player_game_data(pbp, stats_, teams, chains, epv_engine = "v2")))
}, on_stale = "abort")
pgd_v3 <- cached_frame("v2v3_pgd_v3diff", function() {
  with_const(c(V3_BASE, list(EPV_DIFFICULTY_SPLIT = TRUE,
                             EPV_DIFFICULTY_SURPRISE_BY_TYPE = FALSE)),
    as.data.table(create_player_game_data(pbp, stats_, teams, chains, epv_engine = "v3")))
}, on_stale = "abort")

# Assert the frames actually differ where the engines differ. A silent constant
# leak makes two arms converge, and the arms guard downstream only checks
# RATINGS -- by then the damage is upstream and harder to read.
.cs <- function(d) round(sd(d$epv_spoil, na.rm = TRUE), 4)
say(sprintf("build check: sd(epv_spoil) v2 %.4f | v3+difficulty %.4f",
            .cs(pgd_v2), .cs(pgd_v3)))
if (abs(.cs(pgd_v2) - .cs(pgd_v3)) < 1e-6) {
  cli::cli_abort("v2 and v3 contest channels are identical -- a constant leaked between the builds.")
}
pgd_v2 <- as.data.table(pgd_v2); pgd_v3 <- as.data.table(pgd_v3)
setattr(pgd_v2, "epv_engine", "v2"); setattr(pgd_v3, "epv_engine", "v3")

say(""); say("--- v2 production ---")
rt_v2 <- with_const(V2_CONST, build_ratings(pgd_v2, "v2prod", "v2"))

say(""); say("--- v3 final (points scale refitted, prior_games inherited) ---")
STALE <- c(recv = STALE_PG[channel == "recv", prior_games],
           disp = STALE_PG[channel == "disp", prior_games],
           spoil = STALE_PG[channel == "spoil", prior_games])
set_const(c(V3_BASE, scale_const(UNIT), pg_const(STALE)))
pts <- (function(co) c(recv = unname(co[["epr_recv"]]), disp = unname(co[["epr_disp"]]),
                       cont_aerial = unname(co[["epr_spoil"]]), cont_stop = 1))(
  fit3(build_ratings(pgd_v3, "v3_unscaled", "v3")))
say(sprintf("  fitted scale: recv %.4f  disp %.4f  cont_aerial %.4f",
            pts[["recv"]], pts[["disp"]], pts[["cont_aerial"]]))
set_const(c(V3_BASE, scale_const(pts), pg_const(STALE)))
rt_v3 <- build_ratings(pgd_v3, "v3_stalepg", "v3")
v <- fit3(rt_v3)
say(sprintf("  VERIFY (target 1.000): recv %.4f  disp %.4f  cont_aerial %.4f",
            v[["epr_recv"]], v[["epr_disp"]], v[["epr_spoil"]]))

say(""); say("--- v3 final + refitted shrinkage ---")
FRESH <- fit_prior_games(pgd_v3)
say(sprintf("  prior_games  stale %.2f/%.2f/%.2f  ->  refitted %.2f/%.2f/%.2f",
            STALE[["recv"]], STALE[["disp"]], STALE[["spoil"]],
            FRESH[["recv"]], FRESH[["disp"]], FRESH[["spoil"]]))
set_const(c(V3_BASE, scale_const(pts), pg_const(FRESH)))
rt_v3b <- build_ratings(pgd_v3, "v3_freshpg", "v3")

ARMS <- list(
  list(label = "v2 production", rt = rt_v2,   const = V2_CONST),
  list(label = "v3 final",      rt = rt_v3,   const = c(V3_BASE, scale_const(pts), pg_const(STALE))),
  list(label = "v3 + shrinkage", rt = rt_v3b, const = c(V3_BASE, scale_const(pts), pg_const(FRESH)))
)

say(""); say("--- ARMS GUARD ---")
k <- c("player_id", "season", "round")
for (i in 1:2) for (j in (i + 1):3) {
  m <- merge(ARMS[[i]]$rt[, c(k, "epr"), with = FALSE],
             ARMS[[j]]$rt[, c(k, "epr"), with = FALSE], by = k, suffixes = c("_a", "_b"))
  dd <- mean(abs(m$epr_a - m$epr_b), na.rm = TRUE)
  say(sprintf("  %-15s vs %-15s mean|diff| %.5f", ARMS[[i]]$label, ARMS[[j]]$label, dd))
  if (dd < 1e-9) say("    !! IDENTICAL -- an arm is not live")
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
                              ifelse(hw == 0, 1 + log2(1 - pw), 1 + 0.5 * log2(pw * (1 - pw)))))

preds <- list()
for (a in ARMS) {
  say(""); say("=== evaluating: ", a$label, " ===")
  set_const(a$const)
  tm <- build_with_ratings(a$rt)
  ft <- grep("^(epr|psr|torp|elo|xelo).*_diff$|^(epr|psr|torp)\\.[xy]$", names(tm), value = TRUE)
  keep <- stats::complete.cases(tm[, ft, drop = FALSE]); if (any(!keep)) tm <- tm[keep, , drop = FALSE]
  degen <- ft[vapply(ft, function(v) length(unique(tm[[v]][is.finite(tm[[v]])])) < 50, logical(1))]
  if (length(degen)) say("  degenerate (reported, not removed): ", paste(degen, collapse = ", "))
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff", cv_extra_feature_cols = "xelo_diff")
  p <- unique(as.data.table(roll$input_blend_preds), by = "match_id")[, arm := a$label]
  write_parquet(p, file.path(OUT_DIR,
    paste0("epv3_v2v3_", gsub("[^a-z0-9]+", "_", tolower(a$label)), ".parquet")))
  preds[[a$label]] <- p
}

say(""); say("=== RESULTS ===")
allp <- rbindlist(preds, use.names = TRUE, fill = TRUE)[is.finite(pred_margin) & is.finite(margin)]
say_dt(allp[, .(n = .N, MAE = round(mean(abs(pred_margin - margin)), 4),
                RMSE = round(sqrt(mean((pred_margin - margin)^2)), 4),
                bits = round(.bits(pmin(pmax(pred_win, 1e-6), 1 - 1e-6), home_win), 4),
                tips = sum((pred_margin > 0) == (margin > 0))), by = arm], 5)

common <- Reduce(intersect, lapply(preds, function(p) p$match_id))
ba <- preds[["v2 production"]][match_id %chin% common][order(match_id)]
say(""); say("paired against v2 production on ", length(common), " matches")
say("(POSITIVE = v3 costs MAE; this is the number the ship decision needs)")
for (nm in c("v3 final", "v3 + shrinkage")) {
  q <- preds[[nm]][match_id %chin% common][order(match_id)]
  dd <- abs(q$pred_margin - q$margin) - abs(ba$pred_margin - ba$margin)
  dd <- dd[is.finite(dd)]
  tt <- t.test(dd)
  say(sprintf("  %-16s dMAE %+.4f  95%% CI [%+.4f, %+.4f]", nm, mean(dd),
              tt$conf.int[1], tt$conf.int[2]))
}
q1 <- preds[["v3 final"]][match_id %chin% common][order(match_id)]
q2 <- preds[["v3 + shrinkage"]][match_id %chin% common][order(match_id)]
dd <- abs(q2$pred_margin - q2$margin) - abs(q1$pred_margin - q1$margin); dd <- dd[is.finite(dd)]
tt <- t.test(dd)
say(""); say(sprintf("refitted shrinkage vs stale: dMAE %+.4f  95%% CI [%+.4f, %+.4f]",
                     mean(dd), tt$conf.int[1], tt$conf.int[2]))

say(""); say("=== READING IT ===")
say("ws22's +0.477 is superseded by whatever the v3 arms read here, and this is")
say("the first time the comparison has been made with both sides on current")
say("code. Decide on the POOLED window; the per-season split has over-promised")
say("before. A CI spanning zero is an interval with a point estimate in it --")
say("quote both.")
say("")
say("Whatever the number, it is a PRICE, not a verdict. The rating-quality case")
say("for v3 is made elsewhere (conservation, repeatability, count-dependence,")
say("position balance) and Pete's stated tolerance is 'somewhat predictive,")
say("slightly worse is fine'.")

say(""); say("per-season:")
for (nm in names(preds)) {
  d <- preds[[nm]][is.finite(pred_margin) & is.finite(margin)]
  say_dt(d[, .(arm = nm, n = .N, MAE = round(mean(abs(pred_margin - margin)), 3),
               tips = sum((pred_margin > 0) == (margin > 0))), by = season][order(season)], 4)
}

saveRDS(list(pts = pts, stale_pg = STALE, fresh_pg = FRESH),
        file.path(OUT_DIR, "epv3_v2_vs_v3_final.rds"))
say(""); say("done ", format(Sys.time())); close(con); cat("\nDone\n")
