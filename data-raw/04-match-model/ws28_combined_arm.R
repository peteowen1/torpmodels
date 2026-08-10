# The arm Pete actually approved: centring fixes AND the v2 per-channel scale.
#
# WHY THIS RUN EXISTS. Both halves have been gated, and neither was gated with
# the other on:
#   ws26  per-channel scale, fitted and gated on the UNCENTRED frame  (+0.0582)
#   ws27  the three centring fixes, gated at the GLOBAL 0.919 scale   (+0.0534)
# So the combination has never been measured, and it is not a formality: ws26
# FITS the scale from the frame in hand (`fit_ch` below), and the centring work
# changes what the hitout channel contains for rucks. `cont_stop` was fitted at
# 4.0332 against a channel that celled bench-starting rucks with benchwarmers.
# Re-fitting it on the centred frame is the whole point of this script.
#
# ARMS
#   1  v2 production   uncentred frame, one global EPV_POINTS_SCALE = 0.919
#   2  v2 combined     centred frame + per-channel scale FITTED ON THAT FRAME
#
# Reports the gate AND the top-40 leaderboard, because the gate cannot see the
# thing the centring was for and the leaderboard cannot see the thing the scale
# was for. Expect a null on the gate; the leaderboard is where the case lives.
#
# ~50 min. Run detached.

suppressMessages({
  library(dplyr); library(data.table); library(arrow)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "combined_arm.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con); flush(con) }
say_dt <- function(x, n = 45) for (l in capture.output(print(utils::head(x, n), nrows = n + 5))) say(l)
set_const <- function(l) for (nm in names(l)) assignInNamespace(nm, l[[nm]], ns = "torp")
with_const <- function(l, expr) {
  old <- lapply(names(l), function(nm) get(nm, envir = asNamespace("torp")))
  names(old) <- names(l); set_const(l); on.exit(set_const(old), add = TRUE); force(expr)
}

say("=== Combined arm: centring + v2 per-channel scale ===")
say("run at ", format(Sys.time()), " | test seasons ", paste(TEST_SEASONS, collapse = "-"))

teams <- load_teams(TRUE); shared_fixtures <- load_fixtures(TRUE)
shared_stat_ratings <- get_player_stat_ratings(current = FALSE)
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                   error = function(e) NULL)
res <- as.data.table(load_results(TRUE)); xg <- as.data.table(load_xg(TRUE))
tgt <- merge(res[, .(match_id = as.character(match_id), margin = home_score - away_score)],
             xg[, .(match_id = as.character(match_id), xscore_diff)],
             by = "match_id")[is.finite(margin) & is.finite(xscore_diff)]
setnames(tgt, "xscore_diff", "xmargin")

V2_BASE <- list(EPV_ENGINE = "v2", EPV3_CHANNELS = 3L,
                EPV3_SUB_SCALE = c(cont_aerial = 1, cont_stop = 1),
                EPV3_STOP_ZERO_SUM = FALSE,
                EPV_STANDARDISE_CHANNELS = c("recv", "disp", "spoil"),
                EPV_DIFFICULTY_SPLIT = FALSE,
                EPR_PRIOR_GAMES_RECV = 3, EPR_PRIOR_GAMES_DISP = 3,
                EPR_PRIOR_GAMES_SPOIL = 3, EPR_PRIOR_GAMES_HITOUT = 3)
UNIT <- c(recv = 1, disp = 1, cont_aerial = 1, cont_stop = 1)
GLOBAL <- c(V2_BASE, list(
  EPV_PER_CHANNEL_POINTS_SCALE = FALSE, EPV_POINTS_SCALE = 0.919,
  EPV3_POINTS_SCALE = UNIT,
  EPR_PRIOR_RATE_RECV = -0.7 * 0.919, EPR_PRIOR_RATE_DISP = -0.7 * 0.919,
  EPR_PRIOR_RATE_SPOIL = -0.3 * 0.919, EPR_PRIOR_RATE_HITOUT = -0.3 * 0.919))
per_channel <- function(pts) c(V2_BASE, list(
  EPV_PER_CHANNEL_POINTS_SCALE = TRUE, EPV_POINTS_SCALE = 1,
  EPV3_POINTS_SCALE = pts,
  EPR_PRIOR_RATE_RECV = -0.7 * pts[["recv"]], EPR_PRIOR_RATE_DISP = -0.7 * pts[["disp"]],
  EPR_PRIOR_RATE_SPOIL = -0.3 * pts[["cont_aerial"]],
  EPR_PRIOR_RATE_HITOUT = -0.3 * pts[["cont_stop"]]))

# The two player-game frames, both built earlier and both v2.
#   v2v3_pgd_v2   production centring (lineup slot as the cell, INT included)
#   v2_blend_pgd  bench remap + hitout celled on ruck involvement + blended ref
load_pgd <- function(f) {
  p <- file.path(OUT_DIR, paste0(f, ".parquet"))
  if (!file.exists(p)) cli::cli_abort("Missing frame {.file {p}} -- build it before this gate.")
  d <- as.data.table(read_parquet(p)); setattr(d, "epv_engine", "v2"); d
}
PGD <- list(prod = load_pgd("v2v3_pgd_v2"), comb = load_pgd("v2_blend_pgd"))

say(""); say("--- FRAMES GUARD ---")
kk <- c("match_id", "player_id")
mm <- merge(PGD$prod[, c(kk, "epv_hitout_adj"), with = FALSE],
            PGD$comb[, c(kk, "epv_hitout_adj"), with = FALSE],
            by = kk, suffixes = c("_p", "_c"))
say(sprintf("  rows matched %s | mean|diff| in epv_hitout_adj: %.5f",
            format(nrow(mm), big.mark = ","),
            mean(abs(mm$epv_hitout_adj_p - mm$epv_hitout_adj_c), na.rm = TRUE)))
if (mean(abs(mm$epv_hitout_adj_p - mm$epv_hitout_adj_c), na.rm = TRUE) < 1e-9) {
  say("  !! FRAMES IDENTICAL -- the centring frame is not what it claims; aborting")
  close(con); quit(status = 1)
}

build_ratings <- function(pgd, tag) {
  f <- file.path(OUT_DIR, paste0("comb_rt_", tag, ".parquet"))
  if (file.exists(f)) { cli::cli_alert_info("Reusing ratings {tag}")
    return(as.data.table(read_parquet(f))) }
  d <- adjust_epv_for_opponents(as.data.table(copy(pgd)))
  setattr(d, "epv_engine", "v2")
  if (isTRUE(EPV_LEVEL_CENTRE)) d <- centre_epv_by_position(d)
  out <- rbindlist(lapply(sort(unique(d$season)), function(s) {
    sr <- if (s >= 2024) 0 else 1
    mr <- if (s == get_afl_season()) get_afl_week(type = "next") else 28
    torp:::.build_epr_season(s, sr:mr, d, shared_stat_ratings, shared_fixtures)
  }), use.names = TRUE, fill = TRUE)
  if (isTRUE(EPR_POSITION_CENTRE)) out <- centre_epr_by_position(out)
  if (!is.null(psr_df) && nrow(psr_df) > 0 && "psr" %in% names(psr_df)) out <- calculate_torp(out, psr_df)
  out <- as.data.table(out); write_parquet(out, f); out
}

CH <- c("epr_recv", "epr_disp", "epr_spoil", "epr_hitout")
fit_ch <- function(rt) {
  tr <- as.data.table(.build_team_ratings_df(teams, as.data.frame(rt), psr_df))
  h <- tr[team_type == "home"]; a <- tr[team_type == "away"]
  m <- merge(h[, c("match_id", CH), with = FALSE], a[, c("match_id", CH), with = FALSE],
             by = "match_id", suffixes = c("_h", "_a"))
  for (v in CH) m[, (paste0("d_", v)) := get(paste0(v, "_h")) - get(paste0(v, "_a"))]
  m <- merge(m, tgt, by = "match_id")
  setNames(stats::coef(stats::lm(
    as.formula(paste("xmargin ~ 0 +", paste0("d_", CH, collapse = " + "))), data = m)), CH)
}

say(""); say("--- fitting the per-channel scale ON THE CENTRED FRAME ---")
f0 <- with_const(c(V2_BASE, list(EPV_PER_CHANNEL_POINTS_SCALE = TRUE, EPV_POINTS_SCALE = 1,
                                 EPV3_POINTS_SCALE = UNIT,
                                 EPR_PRIOR_RATE_RECV = -0.7, EPR_PRIOR_RATE_DISP = -0.7,
                                 EPR_PRIOR_RATE_SPOIL = -0.3, EPR_PRIOR_RATE_HITOUT = -0.3)),
                 fit_ch(build_ratings(PGD$comb, "comb_unscaled")))
PTS <- c(recv = unname(f0[["epr_recv"]]), disp = unname(f0[["epr_disp"]]),
         cont_aerial = unname(f0[["epr_spoil"]]), cont_stop = unname(f0[["epr_hitout"]]))
say(sprintf("  fitted on CENTRED : recv %.4f  disp %.4f  spoil %.4f  hitout %.4f",
            PTS[["recv"]], PTS[["disp"]], PTS[["cont_aerial"]], PTS[["cont_stop"]]))
say(  "  ws26 on UNCENTRED : recv 0.8701  disp 0.5021  spoil 2.8922  hitout 4.0332")
say("  The hitout entry is the one to watch -- a big move there is the interaction")
say("  this run exists to catch, and it means ws26's constant was not reusable.")

f1 <- with_const(per_channel(PTS), fit_ch(build_ratings(PGD$comb, "comb_percal")))
say(sprintf("  VERIFY (target 1.000): %s", paste(CH, round(f1, 4), sep = " ", collapse = "  ")))
if (max(abs(f1 - 1)) > 5e-3)
  say(sprintf("  NOTE: worst channel off by %.4f -- one fit iteration is not a fixed point,",
              max(abs(f1 - 1))), " because the EPR prior rate scales with the constant.")

ARMS <- list(
  list(label = "v2 production", pgd = PGD$prod, tag = "prod_global", const = GLOBAL),
  list(label = "v2 combined",   pgd = PGD$comb, tag = "comb_percal", const = per_channel(PTS)))
rts <- lapply(ARMS, function(a) with_const(a$const, build_ratings(a$pgd, a$tag)))
names(rts) <- vapply(ARMS, `[[`, "", "label")

say(""); say("--- ARMS GUARD ---")
k <- c("player_id", "season", "round")
m <- merge(rts[[1]][, c(k, "epr"), with = FALSE], rts[[2]][, c(k, "epr"), with = FALSE],
           by = k, suffixes = c("_a", "_b"))
dd <- mean(abs(m$epr_a - m$epr_b), na.rm = TRUE)
say(sprintf("  mean|diff| in epr: %.5f over %s rows", dd, format(nrow(m), big.mark = ",")))
if (dd < 1e-9) { say("  !! IDENTICAL -- aborting"); close(con); quit(status = 1) }

# ---- leaderboard, where the centring case actually lives -------------------
say(""); say("########## TOP 40 UNDER THE COMBINED ARM ##########")
latest <- function(x) { s <- max(x$season, na.rm = TRUE)
  y <- x[season == s][, .SD[which.max(round)], by = player_id]; y[is.finite(epr)] }
b <- latest(rts[["v2 production"]]); a <- latest(rts[["v2 combined"]])
lb <- merge(b[, .(player_id, player_name, position_group, epr_b = epr, hb = epr_hitout)],
            a[, .(player_id, epr_a = epr, ha = epr_hitout)], by = "player_id")
lb[, `:=`(rk_b = frank(-epr_b), rk_a = frank(-epr_a))]
say(sprintf("players rated: %d | season %d", nrow(lb), max(a$season, na.rm = TRUE)))
setorder(lb, rk_a)
say_dt(lb[1:40, .(rk = as.integer(rk_a), player = player_name,
                  pos = substr(position_group, 1, 14), epr = round(epr_a, 2),
                  was = as.integer(rk_b))], 40)
say(""); say("--- position mix in the top 40 ---")
pm <- merge(lb[rk_b <= 40, .(before = .N), by = position_group],
            lb[rk_a <= 40, .(after = .N), by = position_group], by = "position_group", all = TRUE)
pm[is.na(before), before := 0L][is.na(after), after := 0L]
say_dt(pm[order(-after)], 10)
say(""); say("--- the rucks (by hitout channel, after) ---")
setorder(lb, -ha)
say_dt(lb[1:10, .(player = player_name, pos = substr(position_group, 1, 14),
                  hitout_before = round(hb, 2), hitout_after = round(ha, 2),
                  epr_rank = as.integer(rk_a))], 10)
say(sprintf("\nSpearman %.4f | mean |rank change| %.1f",
            cor(lb$rk_b, lb$rk_a, method = "spearman"), mean(abs(lb$rk_b - lb$rk_a))))

# ---- production match gate -------------------------------------------------
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
  tm <- build_with_ratings(as.data.frame(rts[[a$label]]))
  ft <- grep("^(epr|psr|torp|elo|xelo).*_diff$|^(epr|psr|torp)\\.[xy]$", names(tm), value = TRUE)
  keep <- stats::complete.cases(tm[, ft, drop = FALSE]); if (any(!keep)) tm <- tm[keep, , drop = FALSE]
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff", cv_extra_feature_cols = "xelo_diff")
  p <- unique(as.data.table(roll$input_blend_preds), by = "match_id")[, arm := a$label]
  write_parquet(p, file.path(OUT_DIR, paste0("combined_arm_", a$tag, ".parquet")))
  preds[[a$label]] <- p
}
set_const(GLOBAL)

say(""); say("=== RESULTS ===")
allp <- rbindlist(preds, use.names = TRUE, fill = TRUE)[is.finite(pred_margin) & is.finite(margin)]
say_dt(allp[, .(n = .N, MAE = round(mean(abs(pred_margin - margin)), 4),
                RMSE = round(sqrt(mean((pred_margin - margin)^2)), 4),
                bits = round(.bits(pmin(pmax(pred_win, 1e-6), 1 - 1e-6), home_win), 4),
                tips = sum((pred_margin > 0) == (margin > 0))), by = arm], 5)

common <- Reduce(intersect, lapply(preds, function(p) p$match_id))
ba <- preds[["v2 production"]][match_id %chin% common][order(match_id)]
q  <- preds[["v2 combined"]][match_id %chin% common][order(match_id)]
d <- abs(q$pred_margin - q$margin) - abs(ba$pred_margin - ba$margin); d <- d[is.finite(d)]
tt <- t.test(d)
say(""); say(sprintf("paired on %d matches: dMAE %+.4f  95%% CI [%+.4f, %+.4f]",
                     length(common), mean(d), tt$conf.int[1], tt$conf.int[2]))
say("(negative = the combined arm is better; this is a guardrail, so ~0 is the pass)")
say("")
say("Reference, each half measured alone: ws26 per-channel +0.0582, ws27 centring")
say("+0.0534. If this run lands near their sum the two are additive and independent;")
say("well outside it means they interact and neither earlier number described this arm.")

saveRDS(list(dMAE = mean(d), ci = tt$conf.int, pts = PTS, verify = f1, leaderboard = lb),
        file.path(OUT_DIR, "combined_arm.rds"))
say(""); say("done ", format(Sys.time())); close(con); cat("\nDone\n")
