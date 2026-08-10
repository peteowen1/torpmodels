# Reprice the ruck box weights: you must WIN contests to gain points.
#
# THE STRUCTURE IS PETE'S CALL, made 2026-08-06 and reaffirmed after the
# measurement below was put to him. Attending a contest costs, winning one pays,
# winning one to a teammate pays well:
#
#     epv_hitout = hitouts * (+)  +  hta * (++)  +  ruck_contests * (-)
#
# Today the third term is POSITIVE, so a ruck banks ~0.70 a game for turning up
# and does not have to win anything. That is most of the "over-pays rucks ~11x"
# finding, and flipping its sign is the change.
#
# WHERE THE MEASUREMENT AGREES, AND WHERE IT DOES NOT. Recorded because a future
# session will otherwise re-derive it (epv3_ruck_three_way.R, 1,242 matches,
# per-ruck = half the differential):
#
#   to advantage  paid +0.1748  measured +0.1013 (t 6.4)   AGREES on sign, 1.7x
#                 bigger than measured. LEFT AT 0.1748. Dropping it to the
#                 measured value costs a third of the channel's remaining spread
#                 (0.49x -> 0.34x of shipped) on top of what the attendance flip
#                 already costs, and direction is the term that should dominate:
#                 at 0.1748 it carries 70.5% of channel variance, at 0.1015 only
#                 55.6%. If the channel needs to be quieter that is a points
#                 SCALE decision, not a reason to underweight the skill.
#   won tap       paid +0.0510  measured -0.0209 (t -3.5)  DISAGREES. The fit
#                 says an undirected tap is worth slightly less than nothing,
#                 stable across halves (-0.0586 / -0.0299). Pete's judgement is
#                 that a won tap should not be priced negative. Set to +0.0615,
#                 which puts break-even at the LEAGUE AVERAGE win rate of 37.7%
#                 -- an average ruck's contest work nets zero, better rucks
#                 positive. Flagged, not silently reconciled.
#   attendance    paid +0.0232  NOT MEASURABLE. `ruck_contests` averages 92.1 a
#                 team-match with a differential sd of 0.59 -- both teams attend
#                 the same contests, so there is almost no variation to fit
#                 (t 0.65, sign flips between halves). The margin data cannot
#                 price this term either way, so its sign IS a design choice and
#                 Pete's is as admissible as zero.
#
# A NOTE ON WHAT WAS ALMOST SHIPPED. An earlier version of this script set the
# won-tap weight to -0.0209 on the argument that the two-way fit had confounded
# winning with attending. That argument was wrong: cor(rc, h) = 0.059, VIF 1.00,
# and adding attendance moved the coefficient from -0.0447 to -0.0450.
#
# ORDER OF CHECKS, and it matters. Face validity runs FIRST and the match gate
# only runs if it passes. On 2026-08-06 three changes passed every numeric test
# and failed on inspection of the top 40, twice after a 50-minute run. Face
# validity costs a rating build; the gate costs an hour. Cheap test first.
#
# Constants are set via assignInNamespace rather than edited in the repo, so a
# failed arm leaves nothing to revert.

suppressMessages({
  library(dplyr); library(data.table); library(arrow)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})
source("C:/dev/torpverse/torp/data-raw/04-analysis/benchmark_suite.R")

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "ruck_reprice.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con); flush(con) }
say_o <- function(x) for (l in capture.output(print(x))) say(l)
set_const <- function(l) for (nm in names(l)) assignInNamespace(nm, l[[nm]], ns = "torp")
with_const <- function(l, expr) {
  old <- lapply(names(l), function(nm) get(nm, envir = asNamespace("torp")))
  names(old) <- names(l); set_const(l); on.exit(set_const(old), add = TRUE); force(expr)
}

say("=== Ruck repricing: measured weights vs what is paid ===")
say("run at ", format(Sys.time()))
say("")
say("             paid      proposed   basis")
say(sprintf("  hitout     %+.4f   %+.4f    par set at the LEAGUE win rate 37.7%%", 0.0510, 0.0615))
say(sprintf("            (measurement says %+.4f, t -3.5; overruled deliberately)", -0.0209))
say(sprintf("  advantage  %+.4f   %+.4f    UNCHANGED -- kept above measured (0.1015) to", 0.1748, 0.1748))
say(         "            hold spread; direction is 70.5%% of channel variance")
say(sprintf("  attendance %+.4f   %+.4f    SIGN FLIPPED -- Pete: you must win to gain", 0.0232, -0.0232))
say(         "            (not measurable either way: d_ruck_contests sd 0.59 on a level of 92.1)")
say("")
say("Effect on a 30-contest ruck winning 15, 5 to advantage:")
say(sprintf("  before %.3f   after %.3f   (%.1fx)",
            15 * 0.0510 + 5 * 0.1748 + 30 * 0.0232,
            15 * 0.0615 + 5 * 0.1748 - 30 * 0.0232,
            (15 * 0.0510 + 5 * 0.1748 + 30 * 0.0232) /
              (15 * 0.0615 + 5 * 0.1748 - 30 * 0.0232)))
say("A ruck who attends 30 and wins only 10 without direction now goes NEGATIVE,")
say("which is the intent: turning up is not worth points.")

pbp <- load_pbp(TRUE); stats_ <- load_player_stats(TRUE)
teams <- load_teams(TRUE); chains <- load_chains(TRUE)
shared_stat_ratings <- get_player_stat_ratings(current = FALSE)
shared_fixtures <- load_fixtures(TRUE)
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                   error = function(e) NULL)

# The SHIPPED configuration as of torp 6a27aba1: centring on, one global scale.
SHIPPED <- list(EPV_ENGINE = "v2", EPV3_CHANNELS = 3L,
                EPV3_SUB_SCALE = c(cont_aerial = 1, cont_stop = 1),
                EPV3_STOP_ZERO_SUM = FALSE,
                EPV_STANDARDISE_CHANNELS = c("recv", "disp", "spoil"),
                EPV_DIFFICULTY_SPLIT = FALSE,
                EPV_PER_CHANNEL_POINTS_SCALE = FALSE, EPV_POINTS_SCALE = 0.919,
                EPV3_POINTS_SCALE = c(recv = 1, disp = 1, cont_aerial = 1, cont_stop = 1),
                ROLE_REMAP_BENCH = TRUE, EPV_HITOUT_CENTRE_ON_RUCK = TRUE,
                EPV_RUCK_BLEND_WIDTH = 10,
                EPR_PRIOR_RATE_RECV = -0.7 * 0.919, EPR_PRIOR_RATE_DISP = -0.7 * 0.919,
                EPR_PRIOR_RATE_SPOIL = -0.3 * 0.919, EPR_PRIOR_RATE_HITOUT = -0.3 * 0.919,
                EPR_PRIOR_GAMES_RECV = 3, EPR_PRIOR_GAMES_DISP = 3,
                EPR_PRIOR_GAMES_SPOIL = 3, EPR_PRIOR_GAMES_HITOUT = 3)
REPRICED <- c(SHIPPED, list(EPV_HITOUT_WT = 0.0615, EPV_HITOUT_ADV_WT = 0.1748,
                            EPV_RUCK_CONTEST_WT = -0.0232))

build_pgd <- function(const, tag) {
  f <- file.path(OUT_DIR, paste0("ruck_pgd_", tag, ".parquet"))
  if (file.exists(f)) { cli::cli_alert_info("Reusing pgd {tag}");
    d <- as.data.table(read_parquet(f)); setattr(d, "epv_engine", "v2"); return(d) }
  d <- with_const(const, as.data.table(
    create_player_game_data(pbp, stats_, teams, chains, epv_engine = "v2")))
  setattr(d, "epv_engine", "v2"); write_parquet(d, f); d
}
build_rt <- function(pgd, const, tag) {
  f <- file.path(OUT_DIR, paste0("ruck_rt_", tag, ".parquet"))
  if (file.exists(f)) { cli::cli_alert_info("Reusing ratings {tag}")
    return(as.data.table(read_parquet(f))) }
  with_const(const, {
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
  })
}

say(""); say("--- building both arms ---")
pgd_a <- build_pgd(SHIPPED,  "shipped")
pgd_b <- build_pgd(REPRICED, "repriced")

say(""); say("--- FRAMES GUARD ---")
kk <- c("match_id", "player_id")
mm <- merge(pgd_a[, c(kk, "epv_hitout"), with = FALSE],
            pgd_b[, c(kk, "epv_hitout"), with = FALSE], by = kk, suffixes = c("_a", "_b"))
dd <- mean(abs(mm$epv_hitout_a - mm$epv_hitout_b), na.rm = TRUE)
say(sprintf("  mean|diff| in raw epv_hitout: %.5f over %s rows", dd, format(nrow(mm), big.mark = ",")))
if (dd < 1e-9) { say("  !! IDENTICAL -- the weights did not take; aborting"); close(con); quit(status = 1) }

# How much the channel actually shrinks, which is the "over-pays ~11x" claim.
rk <- pgd_a[!is.na(ruck_contests) & ruck_contests >= 10]
rkb <- pgd_b[!is.na(ruck_contests) & ruck_contests >= 10]
say(sprintf("  ruck-games (>=10 contests): mean raw epv_hitout %.3f -> %.3f  (%.1fx)",
            mean(rk$epv_hitout, na.rm = TRUE), mean(rkb$epv_hitout, na.rm = TRUE),
            mean(rk$epv_hitout, na.rm = TRUE) / max(abs(mean(rkb$epv_hitout, na.rm = TRUE)), 1e-9)))

rt_a <- build_rt(pgd_a, SHIPPED,  "shipped")
rt_b <- build_rt(pgd_b, REPRICED, "repriced")

latest <- function(x) { s <- max(x$season, na.rm = TRUE)
  y <- x[season == s][, .SD[which.max(round)], by = player_id]; y[is.finite(epr)] }

say(""); say("########## FACE VALIDITY (runs first, gates the rest) ##########")
fv <- face_validity(latest(rt_a), latest(rt_b))
say_o(fv)
say(""); say("  OVERALL: ", attr(fv, "overall"))
detail <- attr(fv, "detail")
say(""); say("  position mix:"); say_o(detail$mix[order(-after)])
say(""); say("  the rucks specifically:")
a1 <- latest(rt_a); b1 <- latest(rt_b)
rr <- merge(a1[position_group == "RUCK", .(player_id, player_name, h_a = epr_hitout, e_a = epr)],
            b1[, .(player_id, h_b = epr_hitout, e_b = epr)], by = "player_id")
setorder(rr, -h_a)
say_o(head(rr[, .(player_name, hitout_before = round(h_a, 2), hitout_after = round(h_b, 2),
                  epr_before = round(e_a, 2), epr_after = round(e_b, 2))], 10))

if (!identical(attr(fv, "overall"), "pass")) {
  say(""); say("  FACE VALIDITY FAILED -- not running the match gate.")
  say("  The whole point of building that check was to stop here rather than")
  say("  spend an hour and find out afterwards. Nothing is committed.")
  say(""); say("done ", format(Sys.time())); close(con); cat("\nDone\n"); quit(status = 0)
}

say(""); say("  passed -- proceeding to the production match gate.")

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

ARMS <- list(list(label = "shipped",  rt = rt_a, const = SHIPPED),
             list(label = "repriced", rt = rt_b, const = REPRICED))
preds <- list()
for (a in ARMS) {
  say(""); say("=== evaluating: ", a$label, " ===")
  set_const(a$const)
  tm <- build_with_ratings(as.data.frame(a$rt))
  ft <- grep("^(epr|psr|torp|elo|xelo).*_diff$|^(epr|psr|torp)\\.[xy]$", names(tm), value = TRUE)
  keep <- stats::complete.cases(tm[, ft, drop = FALSE]); if (any(!keep)) tm <- tm[keep, , drop = FALSE]
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff", cv_extra_feature_cols = "xelo_diff")
  preds[[a$label]] <- unique(as.data.table(roll$input_blend_preds), by = "match_id")[, arm := a$label]
}
set_const(SHIPPED)

say(""); say("=== RESULTS ===")
allp <- rbindlist(preds, use.names = TRUE, fill = TRUE)[is.finite(pred_margin) & is.finite(margin)]
say_o(allp[, .(n = .N, MAE = round(mean(abs(pred_margin - margin)), 4),
               bits = round(.bits(pmin(pmax(pred_win, 1e-6), 1 - 1e-6), home_win), 4),
               tips = sum((pred_margin > 0) == (margin > 0))), by = arm])
common <- Reduce(intersect, lapply(preds, function(p) p$match_id))
ba <- preds[["shipped"]][match_id %chin% common][order(match_id)]
q  <- preds[["repriced"]][match_id %chin% common][order(match_id)]
d <- abs(q$pred_margin - q$margin) - abs(ba$pred_margin - ba$margin); d <- d[is.finite(d)]
tt <- t.test(d)
say(""); say(sprintf("paired on %d matches: dMAE %+.4f  95%% CI [%+.4f, %+.4f]",
                     length(common), mean(d), tt$conf.int[1], tt$conf.int[2]))
say("(negative = repriced is better; guardrail, so ~0 is the pass)")
say("")
say("NOTE: compare this only against the arm built in THIS run. The v2 production")
say("baseline has read 25.3305 and 25.4159 in different runs, so cross-run point")
say("estimates are not comparable.")

saveRDS(list(face_validity = fv, dMAE = mean(d), ci = tt$conf.int),
        file.path(OUT_DIR, "ruck_reprice.rds"))
say(""); say("done ", format(Sys.time())); close(con); cat("\nDone\n")
