# The difficulty gate again, with every arm built by the SAME code.
#
# WHY ws23 HAD TO BE REDONE. Its "v3 ship" arm read
# `epv3_fin_pgd_ship.parquet`, written 2026-08-04 23:22. Three commits since then
# changed the contest population -- the `Kick Inside 50 Result` removal, the duel
# flag, the mirror allocation. Rebuilding v3 from current code and diffing all 75
# columns gives six that differ across all 56,576 player-games
# (`epv_cont_aerial`, `epv_spoil`, `contests_lost`, plus two derived).
#
# So ws23 compared stale-contest code against fresh-contest code and would have
# called the difference a difficulty effect. Its arm-2-vs-arm-3 comparison (flat
# share vs measured share) is still valid -- both were built fresh -- but its
# arm-1 comparisons are not, and arm 1 is the one that answers "should this
# ship".
#
# Nothing failed to make this visible. The parquet loaded, every column was
# present, the numbers were plausible. It surfaced only because a check written
# for an unrelated purpose -- proving the difficulty wiring inert -- rebuilt the
# frame and diffed it.
#
# WHAT IS DIFFERENT HERE:
#   * every arm builds its player-game frame from current code, no exceptions,
#     through cached_frame() so a stale cache aborts rather than loads
#   * every arm refits and verifies its own EPV3_POINTS_SCALE, including ship.
#     The scale in epv3_finalise_ship.rds was fitted against the stale frame and
#     is not v3's calibration any more.
#   * EPR_PRIOR_GAMES_* still inherited, still stated -- see below
#
# STILL HELD FIXED: EPR_PRIOR_GAMES_* at 14.38 / 24.33 / 11.09. Those were also
# fitted on the stale frame, so they are wrong for every arm here -- equally
# wrong, which keeps the comparison fair but means none of the three is running
# its own best shrinkage. Refitting them per arm is next-steps item 7 and would
# add ~40 min. If the arms finish close, this is the first thing to remove.
#
# PERFORMANCE: 3 pgd builds (~7 min), 6 rating builds (~2 min), 3 rolling evals
# (~20 min). ~100 min. Run detached.

suppressMessages({
  library(dplyr); library(data.table); library(arrow)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
source("C:/dev/torpverse/torp/data-raw/04-analysis/cache_guard.R")
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "epv3_difficulty_gate_clean.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con); flush(con) }
say_dt <- function(x, n = 40) for (l in capture.output(print(utils::head(x, n)))) say(l)
set_const <- function(l) for (nm in names(l)) assignInNamespace(nm, l[[nm]], ns = "torp")

say("=== Difficulty gate, all arms on current code ===")
say("run at ", format(Sys.time()), " | test seasons ", paste(TEST_SEASONS, collapse = "-"))
say("code fingerprint: ", code_fingerprint())

FIN <- readRDS(file.path(OUT_DIR, "epv3_finalise_ship.rds"))
PG <- FIN$prior_games
pg_of <- function(ch) PG[channel == ch, prior_games]
say("EPR_PRIOR_GAMES inherited (fitted on the STALE frame, equally wrong for all arms): ",
    paste(PG$channel, round(PG$prior_games, 2), sep = "=", collapse = ", "))

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

BASE <- list(EPV_ENGINE = "v3", EPV3_CHANNELS = 3L,
             EPV3_SUB_SCALE = c(cont_aerial = 1, cont_stop = 1),
             EPV_STANDARDISE_CHANNELS = c("recv", "disp"),
             EPV3_STOP_ZERO_SUM = TRUE,
             EPR_PRIOR_GAMES_RECV = pg_of("recv"), EPR_PRIOR_GAMES_DISP = pg_of("disp"),
             EPR_PRIOR_GAMES_SPOIL = pg_of("spoil"), EPR_PRIOR_GAMES_HITOUT = 3)
scale_const <- function(pts) list(
  EPV3_POINTS_SCALE = pts,
  EPR_PRIOR_RATE_RECV = -0.7 * pts[["recv"]], EPR_PRIOR_RATE_DISP = -0.7 * pts[["disp"]],
  EPR_PRIOR_RATE_SPOIL = -0.3 * pts[["cont_aerial"]], EPR_PRIOR_RATE_HITOUT = 0)
UNIT <- c(recv = 1, disp = 1, cont_aerial = 1, cont_stop = 1)

build_ratings <- function(pgd, tag) {
  f <- file.path(OUT_DIR, paste0("epv3_dgc_rt_", tag, ".parquet"))
  if (file.exists(f)) { cli::cli_alert_info("Reusing ratings {tag}")
    return(as.data.table(read_parquet(f))) }
  d <- adjust_epv_for_opponents(as.data.table(copy(pgd)))
  setattr(d, "epv_engine", "v3")
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
  list(coef = setNames(co[, 1], CH3), n = nrow(m))
}

ARMS <- list(
  list(label = "v3 ship",         tag = "dgc_ship",  flag = FALSE, by_type = FALSE),
  list(label = "v3 + difficulty", tag = "dgc_flat",  flag = TRUE,  by_type = FALSE),
  list(label = "v3 + measured",   tag = "dgc_table", flag = TRUE,  by_type = TRUE)
)

rts <- list(); pts_used <- list()
for (a in ARMS) {
  say(""); say("--- ", a$label, " ---")
  pgd <- cached_frame(a$tag, function() {
    set_const(list(EPV_DIFFICULTY_SPLIT = a$flag, EPV_DIFFICULTY_SURPRISE_BY_TYPE = a$by_type))
    on.exit(set_const(list(EPV_DIFFICULTY_SPLIT = FALSE,
                           EPV_DIFFICULTY_SURPRISE_BY_TYPE = FALSE)), add = TRUE)
    as.data.table(create_player_game_data(pbp, stats_, teams, chains, epv_engine = "v3"))
  }, on_stale = "abort")
  pgd <- as.data.table(pgd); setattr(pgd, "epv_engine", "v3")

  set_const(c(BASE, scale_const(UNIT)))
  f0 <- fit3(build_ratings(pgd, paste0(a$tag, "_unscaled")))
  pts <- c(recv = unname(f0$coef[["epr_recv"]]), disp = unname(f0$coef[["epr_disp"]]),
           cont_aerial = unname(f0$coef[["epr_spoil"]]), cont_stop = 1)
  say(sprintf("  fitted scale: recv %.4f  disp %.4f  cont_aerial %.4f  (n %d)",
              pts[["recv"]], pts[["disp"]], pts[["cont_aerial"]], f0$n))
  set_const(c(BASE, scale_const(pts)))
  rt <- build_ratings(pgd, paste0(a$tag, "_scaled"))
  f1 <- fit3(rt)
  say(sprintf("  VERIFY (target 1.000): recv %.4f  disp %.4f  cont_aerial %.4f",
              f1$coef[["epr_recv"]], f1$coef[["epr_disp"]], f1$coef[["epr_spoil"]]))
  if (max(abs(f1$coef - 1)) > 5e-3) {
    say(sprintf("  NOTE: worst channel off by %.4f. One fit iteration does not reach a",
                max(abs(f1$coef - 1))))
    say("  fixed point because the EPR prior rate itself scales with the constant.")
    say("  Reported for every arm so they are comparably treated, not silently.")
  }
  pts_used[[a$label]] <- pts; rts[[a$label]] <- rt
}

say(""); say("--- ARMS GUARD ---")
k <- c("player_id", "season", "round"); nms <- names(rts)
for (i in 1:(length(nms) - 1)) for (j in (i + 1):length(nms)) {
  m <- merge(rts[[i]][, c(k, "epr"), with = FALSE], rts[[j]][, c(k, "epr"), with = FALSE],
             by = k, suffixes = c("_a", "_b"))
  dd <- mean(abs(m$epr_a - m$epr_b), na.rm = TRUE)
  say(sprintf("  %-18s vs %-18s mean|diff| %.5f", nms[i], nms[j], dd))
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

run_arm <- function(label, torp_df, const) {
  set_const(const)
  tm <- build_with_ratings(torp_df)
  ft <- grep("^(epr|psr|torp|elo|xelo).*_diff$|^(epr|psr|torp)\\.[xy]$", names(tm), value = TRUE)
  keep <- stats::complete.cases(tm[, ft, drop = FALSE]); if (any(!keep)) tm <- tm[keep, , drop = FALSE]
  degen <- ft[vapply(ft, function(v) length(unique(tm[[v]][is.finite(tm[[v]])])) < 50, logical(1))]
  if (length(degen) > 0) say("  [", label, "] degenerate (reported, not removed): ",
                             paste(degen, collapse = ", "))
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff", cv_extra_feature_cols = "xelo_diff")
  p <- unique(as.data.table(roll$input_blend_preds), by = "match_id")
  p[, arm := label]
  write_parquet(p, file.path(OUT_DIR,
    paste0("epv3_dgc_", gsub("[^a-z0-9]+", "_", tolower(label)), ".parquet")))
  p
}

preds <- list()
for (a in ARMS) {
  say(""); say("=== evaluating: ", a$label, " ===")
  preds[[a$label]] <- run_arm(a$label, rts[[a$label]], c(BASE, scale_const(pts_used[[a$label]])))
}

say(""); say("=== RESULTS ===")
# The rolling eval returns `pred_win`, NOT `pred_home_win_prob` -- the latter
# does not exist and crashed ws23's summary after all three arms had already
# run. `margin` and `home_win` come back joined on, so no merge is needed here.
allp <- rbindlist(preds, use.names = TRUE, fill = TRUE)[
  is.finite(pred_margin) & is.finite(margin)]
say_dt(allp[, .(n = .N, MAE = round(mean(abs(pred_margin - margin)), 4),
                RMSE = round(sqrt(mean((pred_margin - margin)^2)), 4),
                bits = round(.bits(pmin(pmax(pred_win, 1e-6), 1 - 1e-6), home_win), 4),
                tips = sum((pred_margin > 0) == (margin > 0))), by = arm], 5)

ref <- "v3 ship"
common <- Reduce(intersect, lapply(preds, function(p) p$match_id))
say(""); say("paired against '", ref, "' on ", length(common), " common matches:")
ba <- preds[[ref]][match_id %chin% common][order(match_id)]
for (nm in setdiff(names(preds), ref)) {
  q <- preds[[nm]][match_id %chin% common][order(match_id)]
  dd <- abs(q$pred_margin - q$margin) - abs(ba$pred_margin - ba$margin)
  dd <- dd[is.finite(dd)]
  tt <- t.test(dd)
  say(sprintf("  %-18s dMAE %+.4f  95%% CI [%+.4f, %+.4f]  (negative = better than ship)",
              nm, mean(dd), tt$conf.int[1], tt$conf.int[2]))
}
# The two difficulty arms differ only in the surprise share, so this pair
# isolates it -- and it was already valid in ws23, where both were built fresh.
q1 <- preds[["v3 + difficulty"]][match_id %chin% common][order(match_id)]
q2 <- preds[["v3 + measured"]][match_id %chin% common][order(match_id)]
dd <- abs(q2$pred_margin - q2$margin) - abs(q1$pred_margin - q1$margin)
dd <- dd[is.finite(dd)]
tt <- t.test(dd)
say(""); say(sprintf("measured share vs flat 0.5: dMAE %+.4f  95%% CI [%+.4f, %+.4f]",
                     mean(dd), tt$conf.int[1], tt$conf.int[2]))
say("(negative = the measured share beats the assumed 0.5)")

saveRDS(list(pts = pts_used), file.path(OUT_DIR, "epv3_difficulty_gate_clean.rds"))
say(""); say("done ", format(Sys.time())); close(con); cat("\nDone\n")
