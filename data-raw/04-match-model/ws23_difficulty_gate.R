# The match gate for the difficulty split.
#
# WHAT CHANGED SINCE THE LAST GATE. Disposals are no longer split 50/50 between
# disposer and receiver. Each one is priced by P(turnover) given the situation:
#
#   disp = (V_pre - exp_pts) + (1 - ss) * (V_after - V_pre)
#   recv =                        +/- ss * (V_after - V_pre)
#
# so an easy handball pays its receiver almost nothing and a 50m pass hit under
# pressure pays him a lot. The row closes exactly, which the first build did not
# -- it measured the surprise against the fitted branch value and left
# `V_after - V_branch` (64.2% of gross |delta_epv|) paid to nobody.
#
# Measured on the player-game frame (epv3_difficulty_wired.txt), against the
# ship build:
#   conservation      0.9879 -> 0.9936 total; channel shares 21/74/5 -> 50/44/6
#   repeatability     epv year-over-year 0.5898 -> 0.6546
#   count-dependence  cor(epv_disp, disposals) 0.052 -> -0.047, i.e. it went the
#                     RIGHT way while repeatability rose, which is the specific
#                     degenerate outcome this was meant to avoid
#
# None of that is a match-model result, which is what this measures.
#
# THREE ARMS:
#   1  v3 ship         the finished v3, cached ratings, unchanged reference
#   2  v3 + difficulty flat surprise share of 0.5
#   3  v3 + measured   the share by branch and disposal type, from
#                      epv3_surprise_share_v2.txt
#
# THE POINTS SCALE IS REFITTED PER ARM, and it has to be. EPV3_POINTS_SCALE
# makes one unit of each channel equal one point, and it was fitted against the
# SHIP channels. The difficulty split changes what recv and disp mean, so
# inheriting the ship scale would score arms 2 and 3 with the wrong units and
# call the result a difficulty effect. Each arm therefore builds ratings twice
# -- unscaled to fit, scaled to verify -- and the verify must read 1.000.
#
# HELD FIXED, AND STATED BECAUSE IT IS A REAL LIMITATION: EPR_PRIOR_GAMES_* keeps
# the ship values. They were fitted on the ship channels too, and refitting them
# per arm as well would be better. They are held rather than silently inherited,
# so if arms 2/3 lose narrowly, shrinkage tuned for the wrong channels is a live
# explanation and not one to reason past.
#
# PERFORMANCE: 1 pgd build (~7 min), 4 rating builds (~2 min each), 3 rolling
# evals (~20 min each, no ship-gate-safe speedup). ~80 min. Run detached.

suppressMessages({
  library(dplyr); library(data.table); library(arrow)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "epv3_difficulty_gate.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con); flush(con) }
say_dt <- function(x, n = 40) for (l in capture.output(print(utils::head(x, n)))) say(l)
set_const <- function(l) for (nm in names(l)) assignInNamespace(nm, l[[nm]], ns = "torp")

FIN <- readRDS(file.path(OUT_DIR, "epv3_finalise_ship.rds"))
PG  <- FIN$prior_games
pg_of <- function(ch) PG[channel == ch, prior_games]

say("=== Difficulty split: match gate ===")
say("run at ", format(Sys.time()), " | test seasons ", paste(TEST_SEASONS, collapse = "-"))

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

build_pgd <- function(tag, diff_flag, by_type) {
  f <- file.path(OUT_DIR, paste0("epv3_diffgate_pgd_", tag, ".parquet"))
  if (tag == "ship") f <- file.path(OUT_DIR, "epv3_fin_pgd_ship.parquet")
  if (tag == "diff_flat") f <- file.path(OUT_DIR, "epv3_difficulty_wired_pgd.parquet")
  if (file.exists(f)) { cli::cli_alert_info("Reusing pgd {basename(f)}")
    d <- as.data.table(read_parquet(f))
  } else {
    set_const(list(EPV_DIFFICULTY_SPLIT = diff_flag,
                   EPV_DIFFICULTY_SURPRISE_BY_TYPE = by_type))
    d <- as.data.table(create_player_game_data(pbp, stats_, teams, chains,
                                               epv_engine = "v3"))
    write_parquet(d, f)
    set_const(list(EPV_DIFFICULTY_SPLIT = FALSE, EPV_DIFFICULTY_SURPRISE_BY_TYPE = FALSE))
  }
  setattr(d, "epv_engine", "v3"); d
}

build_ratings <- function(pgd, tag) {
  f <- file.path(OUT_DIR, paste0("epv3_diffgate_rt_", tag, ".parquet"))
  if (tag == "ship_final") f <- file.path(OUT_DIR, "epv3_cal_rt_gate_v3_final.parquet")
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
  list(coef = setNames(co[, 1], CH3), t = setNames(co[, 3], CH3), n = nrow(m))
}

# ---------------------------------------------------------------- build arms
ARMS <- list(
  list(label = "v3 ship",        pgd = "ship",      rt = "ship",      flag = FALSE, by_type = FALSE),
  list(label = "v3 + difficulty", pgd = "diff_flat", rt = "diff_flat", flag = TRUE,  by_type = FALSE),
  list(label = "v3 + measured",   pgd = "diff_type", rt = "diff_type", flag = TRUE,  by_type = TRUE)
)

rts <- list(); pts_used <- list()
for (a in ARMS) {
  say(""); say("--- ", a$label, " ---")
  pgd <- build_pgd(a$pgd, a$flag, a$by_type)
  if (a$label == "v3 ship") {
    set_const(c(BASE, scale_const(FIN$points_scale)))
    pts_used[[a$label]] <- FIN$points_scale
    rts[[a$label]] <- build_ratings(pgd, "ship_final")
    say("  points scale: inherited from epv3_finalise_ship.rds (this IS that arm)")
  } else {
    set_const(c(BASE, scale_const(UNIT)))
    f0 <- fit3(build_ratings(pgd, paste0(a$rt, "_unscaled")))
    pts <- c(recv = unname(f0$coef[["epr_recv"]]), disp = unname(f0$coef[["epr_disp"]]),
             cont_aerial = unname(f0$coef[["epr_spoil"]]), cont_stop = 1)
    say(sprintf("  fitted points scale: recv %.4f  disp %.4f  cont_aerial %.4f  (n %d)",
                pts[["recv"]], pts[["disp"]], pts[["cont_aerial"]], f0$n))
    set_const(c(BASE, scale_const(pts)))
    rt <- build_ratings(pgd, paste0(a$rt, "_scaled"))
    f1 <- fit3(rt)
    say(sprintf("  VERIFY (each must read 1.000): recv %.4f  disp %.4f  cont_aerial %.4f",
                f1$coef[["epr_recv"]], f1$coef[["epr_disp"]], f1$coef[["epr_spoil"]]))
    if (max(abs(f1$coef - 1)) > 5e-3) {
      say("  !! SCALE NOT MET (worst ", round(max(abs(f1$coef - 1)), 4),
          ") -- this arm is being scored in the wrong units")
    }
    pts_used[[a$label]] <- pts
    rts[[a$label]] <- rt
  }
  a$pts <- pts_used[[a$label]]
}

say(""); say("--- ARMS GUARD: no two arms may be identical ---")
k <- c("player_id", "season", "round"); nms <- names(rts)
for (i in 1:(length(nms) - 1)) for (j in (i + 1):length(nms)) {
  m <- merge(rts[[i]][, c(k, "epr"), with = FALSE], rts[[j]][, c(k, "epr"), with = FALSE],
             by = k, suffixes = c("_a", "_b"))
  dd <- mean(abs(m$epr_a - m$epr_b), na.rm = TRUE)
  say(sprintf("  %-18s vs %-18s mean|diff| %.5f", nms[i], nms[j], dd))
  if (dd < 1e-9) say("    !! IDENTICAL -- an arm is not live")
}

# ------------------------------------------------------------------ evaluate
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
  keep <- stats::complete.cases(tm[, ft, drop = FALSE])
  if (any(!keep)) tm <- tm[keep, , drop = FALSE]
  # Report degenerate features, never remove them: epr_hitout_diff is constant
  # under 3 channels, the GAMs drop it themselves, and deleting the column
  # breaks XGBoost's fixed feature list.
  degen <- ft[vapply(ft, function(v) length(unique(tm[[v]][is.finite(tm[[v]])])) < 50, logical(1))]
  if (length(degen) > 0) say("  [", label, "] degenerate: ", paste(degen, collapse = ", "))
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff", cv_extra_feature_cols = "xelo_diff")
  p <- unique(as.data.table(roll$input_blend_preds), by = "match_id")
  p[, arm := label]
  write_parquet(p, file.path(OUT_DIR,
    paste0("epv3_diffgate_", gsub("[^a-z0-9]+", "_", tolower(label)), ".parquet")))
  p
}

preds <- list()
for (a in ARMS) {
  say(""); say("=== evaluating: ", a$label, " ===")
  preds[[a$label]] <- run_arm(a$label, rts[[a$label]],
                              c(BASE, scale_const(pts_used[[a$label]])))
}

say(""); say("=== RESULTS ===")
allp <- rbindlist(preds, use.names = TRUE, fill = TRUE)
allp <- merge(allp, res[, .(match_id = as.character(match_id),
                            actual = home_score - away_score)], by = "match_id")
allp[, home_win := as.integer(actual > 0)]
summ <- allp[, .(n = .N,
                 MAE = round(mean(abs(pred_margin - actual)), 4),
                 bits = round(.bits(pmin(pmax(pred_home_win_prob, 1e-6), 1 - 1e-6), home_win), 4),
                 tips = sum((pred_margin > 0) == (actual > 0))), by = arm]
say_dt(summ, 5)

# Paired difference against the reference arm, on the matches all arms scored.
ref <- "v3 ship"
common <- Reduce(intersect, lapply(preds, function(p) p$match_id))
say(""); say("paired against '", ref, "' on ", length(common), " common matches:")
base_p <- preds[[ref]][match_id %chin% common][order(match_id)]
base_a <- merge(base_p, res[, .(match_id = as.character(match_id),
                                actual = home_score - away_score)], by = "match_id")
for (nm in setdiff(names(preds), ref)) {
  q <- preds[[nm]][match_id %chin% common][order(match_id)]
  q <- merge(q, res[, .(match_id = as.character(match_id),
                        actual = home_score - away_score)], by = "match_id")
  d <- abs(q$pred_margin - q$actual) - abs(base_a$pred_margin - base_a$actual)
  tt <- t.test(d)
  say(sprintf("  %-18s dMAE %+.4f  95%% CI [%+.4f, %+.4f]  (negative = better than ship)",
              nm, mean(d), tt$conf.int[1], tt$conf.int[2]))
}
say("")
say("A CI spanning zero is not 'no evidence' -- it is an interval, and the point")
say("estimate is still the best guess. Quote both.")
say("")
say("points scales actually used:")
for (nm in names(pts_used)) say(sprintf("  %-18s recv %.4f  disp %.4f  cont_aerial %.4f",
  nm, pts_used[[nm]][["recv"]], pts_used[[nm]][["disp"]], pts_used[[nm]][["cont_aerial"]]))

saveRDS(list(summary = summ, pts = pts_used), file.path(OUT_DIR, "epv3_difficulty_gate.rds"))
say(""); say("done ", format(Sys.time())); close(con); cat("\nDone\n")
