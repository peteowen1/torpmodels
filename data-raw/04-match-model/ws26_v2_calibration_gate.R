# Gate the v2 per-channel calibration through the production match model.
#
# WHAT IS BEING GATED. v2 applies ONE global points scale, EPV_POINTS_SCALE =
# 0.919, to all four channels. They convert to margin at 0.893 / 1.556 / 0.344,
# so one factor cannot make each read a point per unit -- and raw v2 `epv`
# conserves to margin at only 0.4778 in consequence. Per-channel scaling takes
# it to 1.0000, lifts the benchmark skill score 0.3688 -> 0.3809 and
# within-position repeatability 0.6762 -> 0.6809, and loses on nothing.
#
# EXPECT A SMALL OR NULL RESULT, AND DO NOT READ THAT AS FAILURE. The match
# model consumes each channel diff separately and a GAM smooth of k*x spans the
# same function space as a smooth of x, so the per-channel terms are INVARIANT
# to a constant rescale. The change reaches the model through exactly two
# routes:
#
#   1. `epr_diff`, the SUM of the channels -- rescaling changes the mix, so this
#      really is a different feature
#   2. Bayesian shrinkage, which pulls toward EPR_PRIOR_RATE_*; those are scaled
#      to match here, so this route is intentionally closed
#
# So route 1 is the whole of it. This is the same reason five earlier rating
# changes gated neutral, and it is a structural property of the pipeline rather
# than a fact about the change. It is being run as a GUARDRAIL -- "does the
# descriptive fix cost prediction" -- and the answer wanted is "no".
#
# The rating-quality case is made elsewhere and does not depend on this.
#
# TWO ARMS. Both v2, both current code, both built under their own constants:
#   1  v2 production           global EPV_POINTS_SCALE = 0.919
#   2  v2 + per-channel        EPV3_POINTS_SCALE fitted on THIS frame, verified
#
# ~50 min. Run detached; do not edit torp/R/ while it runs.

suppressMessages({
  library(dplyr); library(data.table); library(arrow)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
source("C:/dev/torpverse/torp/data-raw/04-analysis/cache_guard.R")
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "v2_calibration_gate.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con); flush(con) }
say_dt <- function(x, n = 40) for (l in capture.output(print(utils::head(x, n)))) say(l)
set_const <- function(l) for (nm in names(l)) assignInNamespace(nm, l[[nm]], ns = "torp")
with_const <- function(l, expr) {
  old <- lapply(names(l), function(nm) get(nm, envir = asNamespace("torp")))
  names(old) <- names(l); set_const(l); on.exit(set_const(old), add = TRUE); force(expr)
}

say("=== v2 per-channel calibration: production match gate ===")
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

V2_BASE <- list(EPV_ENGINE = "v2", EPV3_CHANNELS = 3L,
                EPV3_SUB_SCALE = c(cont_aerial = 1, cont_stop = 1),
                EPV3_STOP_ZERO_SUM = FALSE,
                EPV_STANDARDISE_CHANNELS = c("recv", "disp", "spoil"),
                EPV_DIFFICULTY_SPLIT = FALSE,
                EPR_PRIOR_GAMES_RECV = 3, EPR_PRIOR_GAMES_DISP = 3,
                EPR_PRIOR_GAMES_SPOIL = 3, EPR_PRIOR_GAMES_HITOUT = 3)
GLOBAL <- c(V2_BASE, list(
  EPV_PER_CHANNEL_POINTS_SCALE = FALSE, EPV_POINTS_SCALE = 0.919,
  EPV3_POINTS_SCALE = c(recv = 1, disp = 1, cont_aerial = 1, cont_stop = 1),
  EPR_PRIOR_RATE_RECV = -0.7 * 0.919, EPR_PRIOR_RATE_DISP = -0.7 * 0.919,
  EPR_PRIOR_RATE_SPOIL = -0.3 * 0.919, EPR_PRIOR_RATE_HITOUT = -0.3 * 0.919))
per_channel <- function(pts) c(V2_BASE, list(
  EPV_PER_CHANNEL_POINTS_SCALE = TRUE, EPV_POINTS_SCALE = 1,
  EPV3_POINTS_SCALE = pts,
  # Scaled to match, so shrinkage changes units and not its strength.
  EPR_PRIOR_RATE_RECV = -0.7 * pts[["recv"]], EPR_PRIOR_RATE_DISP = -0.7 * pts[["disp"]],
  EPR_PRIOR_RATE_SPOIL = -0.3 * pts[["cont_aerial"]],
  EPR_PRIOR_RATE_HITOUT = -0.3 * pts[["cont_stop"]]))
UNIT <- c(recv = 1, disp = 1, cont_aerial = 1, cont_stop = 1)

pgd <- cached_frame("v2v3_pgd_v2", function() {
  with_const(GLOBAL,
    as.data.table(create_player_game_data(pbp, stats_, teams, chains, epv_engine = "v2")))
}, on_stale = "rebuild")
pgd <- as.data.table(pgd); setattr(pgd, "epv_engine", "v2")

build_ratings <- function(tag) {
  f <- file.path(OUT_DIR, paste0("v2cal_rt_", tag, ".parquet"))
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
  co <- stats::coef(stats::lm(as.formula(paste("xmargin ~ 0 +", paste0("d_", CH, collapse = " + "))), data = m))
  setNames(co, CH)
}

say(""); say("--- fitting the v2 per-channel scale ---")
f0 <- with_const(c(V2_BASE, list(EPV_PER_CHANNEL_POINTS_SCALE = TRUE, EPV_POINTS_SCALE = 1,
                                 EPV3_POINTS_SCALE = UNIT,
                                 EPR_PRIOR_RATE_RECV = -0.7, EPR_PRIOR_RATE_DISP = -0.7,
                                 EPR_PRIOR_RATE_SPOIL = -0.3, EPR_PRIOR_RATE_HITOUT = -0.3)),
                 fit_ch(build_ratings("unscaled")))
PTS <- c(recv = unname(f0[["epr_recv"]]), disp = unname(f0[["epr_disp"]]),
         cont_aerial = unname(f0[["epr_spoil"]]), cont_stop = unname(f0[["epr_hitout"]]))
say(sprintf("  fitted: recv %.4f  disp %.4f  spoil %.4f  hitout %.4f",
            PTS[["recv"]], PTS[["disp"]], PTS[["cont_aerial"]], PTS[["cont_stop"]]))
f1 <- with_const(per_channel(PTS), fit_ch(build_ratings("percal")))
say(sprintf("  VERIFY (target 1.000): %s",
            paste(CH, round(f1, 4), sep = " ", collapse = "  ")))
if (max(abs(f1 - 1)) > 5e-3)
  say(sprintf("  NOTE: worst channel off by %.4f -- one fit iteration is not a fixed point,",
              max(abs(f1 - 1))), " because the EPR prior rate scales with the constant.")

ARMS <- list(list(label = "v2 production", tag = "global", const = GLOBAL),
             list(label = "v2 per-channel", tag = "percal", const = per_channel(PTS)))
rts <- lapply(ARMS, function(a) with_const(a$const, build_ratings(a$tag)))
names(rts) <- vapply(ARMS, `[[`, "", "label")

say(""); say("--- ARMS GUARD ---")
k <- c("player_id", "season", "round")
m <- merge(rts[[1]][, c(k, "epr"), with = FALSE], rts[[2]][, c(k, "epr"), with = FALSE],
           by = k, suffixes = c("_a", "_b"))
dd <- mean(abs(m$epr_a - m$epr_b), na.rm = TRUE)
say(sprintf("  mean|diff| in epr: %.5f", dd))
if (dd < 1e-9) { say("  !! IDENTICAL -- the flag did not take; aborting"); close(con); quit(status = 1) }

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
  tm <- build_with_ratings(rts[[a$label]])
  ft <- grep("^(epr|psr|torp|elo|xelo).*_diff$|^(epr|psr|torp)\\.[xy]$", names(tm), value = TRUE)
  keep <- stats::complete.cases(tm[, ft, drop = FALSE]); if (any(!keep)) tm <- tm[keep, , drop = FALSE]
  degen <- ft[vapply(ft, function(v) length(unique(tm[[v]][is.finite(tm[[v]])])) < 50, logical(1))]
  if (length(degen)) say("  degenerate (reported, not removed): ", paste(degen, collapse = ", "))
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff", cv_extra_feature_cols = "xelo_diff")
  p <- unique(as.data.table(roll$input_blend_preds), by = "match_id")[, arm := a$label]
  write_parquet(p, file.path(OUT_DIR, paste0("v2cal_", a$tag, ".parquet")))
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
q <- preds[["v2 per-channel"]][match_id %chin% common][order(match_id)]
d <- abs(q$pred_margin - q$margin) - abs(ba$pred_margin - ba$margin); d <- d[is.finite(d)]
tt <- t.test(d)
say(""); say(sprintf("paired on %d matches: dMAE %+.4f  95%% CI [%+.4f, %+.4f]",
                     length(common), mean(d), tt$conf.int[1], tt$conf.int[2]))
say("(negative = per-channel is BETTER; this is a guardrail, so ~0 is the pass)")
say("")
say("A null here is the expected and wanted result. The channel smooths are")
say("invariant to a constant rescale, so only `epr_diff` carries the change --")
say("while conservation goes 0.4778 -> 1.0000 on the descriptive side, which no")
say("match-model metric can see.")

saveRDS(list(pts = PTS, verify = f1), file.path(OUT_DIR, "v2_calibration_gate.rds"))
say(""); say("done ", format(Sys.time())); close(con); cat("\nDone\n")
