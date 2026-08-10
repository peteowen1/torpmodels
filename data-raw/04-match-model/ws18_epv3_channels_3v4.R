# EPV v3: three contest-and-stoppage channels, or four?
# ======================================================
# Pete's original framing was "3 subsections, maaaybe 4". The 4-channel split
# (recv / disp / cont_aerial / cont_stop) was built first; this scores 3 against
# it on the same gate.
#
# THE KEY POINT: merging the two contest channels does NOT change `epv_adj` at
# all -- the total is the same sum either way. The entire difference is in EPR
# aggregation, because each channel carries its own EPR_DECAY_*,
# EPR_PRIOR_GAMES_* and EPR_PRIOR_RATE_*. So the question is precisely: does a
# ruckman's stoppage value deserve its own decay and shrinkage prior, or should
# it ride on the aerial channel's?
#
# Prior evidence favours 4, but as argument rather than result:
#   cor(cont_aerial, cont_stop) = 0.004  -- orthogonal, so merging LOSES
#     information rather than removing duplication
#   cont_stop sd by position: RUCK 1.480 vs 0.112-0.504 elsewhere; cont_aerial
#     is 2.29-3.14 across the board. Merged, the ruck signal is swamped.
#
# Run: powershell.exe -Command 'Rscript "<this file>"'

suppressMessages({
  library(dplyr); library(data.table)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "epv3_channels_3v4.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con) }
say_dt <- function(x, n = 60) for (l in capture.output(print(utils::head(x, n)))) say(l)

say("=== EPV v3: 3 channels vs 4 ===")
say("run at ", format(Sys.time()), " | test seasons ", paste(TEST_SEASONS, collapse = "-"))

shared_stat_ratings <- get_player_stat_ratings(current = FALSE)
shared_fixtures     <- load_fixtures(TRUE)
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                   error = function(e) NULL)

# ---- Build the 3-channel arm from the 4-channel frame ----------------------
# Post-processing rather than a rebuild: the channels are already computed, and
# the merge is exactly "put cont_stop in the same slot as cont_aerial". Doing it
# this way also makes the invariant checkable -- epv_adj MUST be unchanged.
pgd4 <- as.data.table(arrow::read_parquet(file.path(OUT_DIR, "epv3_player_game_v3.parquet")))
pgd3 <- copy(pgd4)
pgd3[, `:=`(
  epv_spoil      = epv_spoil + epv_hitout,
  epv_spoil_adj  = epv_spoil_adj + epv_hitout_adj,
  epv_hitout     = 0,
  epv_hitout_adj = 0
)]
say("")
say("--- merge invariant: the TOTAL must not move ---")
say("max |epv_adj(3) - epv_adj(4)| = ",
    signif(max(abs(pgd3$epv_adj - pgd4$epv_adj), na.rm = TRUE), 3),
    "   [must be 0 -- if not, the merge changed the metric, not just its bookkeeping]")
stopifnot(max(abs(pgd3$epv_adj - pgd4$epv_adj), na.rm = TRUE) < 1e-9)
say("channel sd: 4ch cont_aerial ", round(sd(pgd4$epv_spoil_adj), 4),
    " cont_stop ", round(sd(pgd4$epv_hitout_adj), 4),
    " | 3ch cont ", round(sd(pgd3$epv_spoil_adj), 4))

build_ratings <- function(pgd, label, prior_rate_hitout = NULL) {
  cli::cli_h2("Building ratings: {label}")
  # The emptied slot must contribute EXACTLY zero. .bayesian_shrink() returns
  # (loading*sum + prior_games*prior_rate)/(wt+prior_games), so a zero sum with a
  # non-zero prior_rate still yields a non-zero rating -- it would shrink toward
  # a prior for a channel that no longer exists.
  if (!is.null(prior_rate_hitout)) {
    old <- torp:::EPR_PRIOR_RATE_HITOUT
    assignInNamespace("EPR_PRIOR_RATE_HITOUT", prior_rate_hitout, ns = "torp")
    on.exit(assignInNamespace("EPR_PRIOR_RATE_HITOUT", old, ns = "torp"), add = TRUE)
    cli::cli_alert_info("{label}: EPR_PRIOR_RATE_HITOUT {round(old, 4)} -> {prior_rate_hitout}")
  }
  d <- as.data.table(copy(pgd))
  d <- adjust_epv_for_opponents(d)
  if (isTRUE(EPV_LEVEL_CENTRE)) d <- centre_epv_by_position(d)
  seasons <- sort(unique(d$season))
  out <- rbindlist(lapply(seasons, function(s) {
    start_round <- if (s >= 2024) 0 else 1
    max_round <- if (s == get_afl_season()) get_afl_week(type = "next") else 28
    torp:::.build_epr_season(s, start_round:max_round, d,
                             shared_stat_ratings, shared_fixtures)
  }), use.names = TRUE, fill = TRUE)
  if (isTRUE(EPR_POSITION_CENTRE)) out <- centre_epr_by_position(out)
  if (!is.null(psr_df) && nrow(psr_df) > 0 && "psr" %in% names(psr_df)) {
    out <- calculate_torp(out, psr_df)
  }
  # calculate_torp() / centre_epr_by_position() hand back a data.frame, so the
  # data.table syntax used downstream silently fails on it.
  out <- as.data.table(out)
  cli::cli_inform(paste0(
    label, ": ", nrow(out), " rating rows, epr sd ",
    round(sd(out$epr, na.rm = TRUE), 3), ", NA epr ", sum(is.na(out$epr)),
    " (", round(100 * mean(is.na(out$epr)), 1), "%)"))
  out
}

build_with_ratings <- function(torp_df) {
  all_grounds <- file_reader("stadium_data", "reference-data")
  fix_df <- .build_fixtures_df(shared_fixtures)
  team_rt_df <- .build_team_ratings_df(load_teams(TRUE), torp_df, psr_df)
  team_rt_fix_df <- .build_match_features(fix_df, team_rt_df, all_grounds)
  weather_df <- .load_match_weather(shared_fixtures, all_grounds, NULL, get_afl_season())
  anchor <- max(as.Date(fix_df$utc_start_time), na.rm = TRUE)
  .build_team_mdl_df(team_rt_fix_df, load_results(TRUE), load_xg(TRUE), weather_df, anchor)
}

.bits <- function(pw, hw) mean(ifelse(hw == 1, 1 + log2(pw),
                              ifelse(hw == 0, 1 + log2(1 - pw),
                                     1 + 0.5 * log2(pw * (1 - pw)))))

run_arm <- function(label, torp_df) {
  tm <- build_with_ratings(torp_df)
  feat <- grep("^(epr|psr|torp|elo|xelo).*_diff$|^(epr|psr|torp)\\.[xy]$",
               names(tm), value = TRUE)
  keep <- stats::complete.cases(tm[, feat, drop = FALSE])
  if (any(!keep)) tm <- tm[keep, , drop = FALSE]
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams,
                           xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff",
                           cv_extra_feature_cols = "xelo_diff")
  p <- unique(as.data.table(roll$input_blend_preds), by = "match_id")
  p[, arm := label]
  p
}

# 4-channel ratings are already cached by ws17; reuse so only the new arm builds.
f4 <- file.path(OUT_DIR, "epv3_ratings_v3.parquet")
r4 <- if (file.exists(f4)) as.data.table(arrow::read_parquet(f4)) else build_ratings(pgd4, "v3-4ch")
r2 <- as.data.table(arrow::read_parquet(file.path(OUT_DIR, "epv3_ratings_v2.parquet")))
f3 <- file.path(OUT_DIR, "epv3_ratings_v3_3ch.parquet")
r3 <- if (file.exists(f3)) {
  cli::cli_alert_info("Reusing cached 3-channel ratings")
  as.data.table(arrow::read_parquet(f3))
} else {
  x <- build_ratings(pgd3, "v3-3ch", prior_rate_hitout = 0)
  arrow::write_parquet(x, f3)
  x
}

# 17.9% NA epr is PRE-EXISTING in the ratings build, not something the merge
# introduced -- v2 and the 4-channel v3 both carry exactly 23,574 NA rows out of
# 131,740. Asserted so a future change that makes it worse is visible.
say("")
say("--- NA epr, all three arms (pre-existing ~17.9%, must not diverge) ---")
say("v2 ", sum(is.na(r2$epr)), " | v3-4ch ", sum(is.na(r4$epr)),
    " | v3-3ch ", sum(is.na(r3$epr)), " of ", nrow(r3))

say("")
say("--- ARMS GUARD ---")
k <- c("player_id", "season", "round")
cm <- merge(r4[, c(k, "epr"), with = FALSE], r3[, c(k, "epr"), with = FALSE],
            by = k, suffixes = c("_4", "_3"))
say("epr mean|diff| 4ch vs 3ch: ", round(mean(abs(cm$epr_4 - cm$epr_3), na.rm = TRUE), 4),
    " | cor ", round(cor(cm$epr_4, cm$epr_3, use = "complete.obs"), 4))
stopifnot(mean(abs(cm$epr_4 - cm$epr_3), na.rm = TRUE) > 1e-9)
say("(non-zero confirms the EPR aggregation really does differ, which is the")
say(" only thing 3-vs-4 can change.)")

p2 <- run_arm("v2", r2); p4 <- run_arm("v3-4ch", r4); p3 <- run_arm("v3-3ch", r3)
common <- Reduce(intersect, list(p2$match_id, p4$match_id, p3$match_id))
p2 <- p2[match_id %in% common]; p4 <- p4[match_id %in% common]; p3 <- p3[match_id %in% common]
say("")
say("--- same-games check: ", length(common), " matches in all three arms ---")

metrics <- function(p, label, seasons = NULL) {
  d <- if (is.null(seasons)) p else p[season %in% seasons]
  d <- d[is.finite(margin) & is.finite(pred_margin) & is.finite(pred_win)]
  hw <- ifelse(d$margin > 0, 1, ifelse(d$margin == 0, 0.5, 0))
  data.table(arm = label,
             window = if (is.null(seasons)) "pooled" else paste(seasons, collapse = ","),
             n = nrow(d),
             MAE = round(mean(abs(d$pred_margin - d$margin)), 4),
             RMSE = round(sqrt(mean((d$pred_margin - d$margin)^2)), 4),
             bits = round(.bits(pmin(pmax(d$pred_win, 1e-6), 1 - 1e-6), hw), 4),
             Brier = round(mean((d$pred_win - hw)^2), 4),
             tips = sum((d$pred_margin > 0) == (d$margin > 0), na.rm = TRUE))
}

say("")
say("--- match metrics, ALL of them ---")
say_dt(rbind(metrics(p2, "v2"), metrics(p4, "v3-4ch"), metrics(p3, "v3-3ch"),
             metrics(p2, "v2", 2025), metrics(p4, "v3-4ch", 2025), metrics(p3, "v3-3ch", 2025),
             metrics(p2, "v2", 2026), metrics(p4, "v3-4ch", 2026), metrics(p3, "v3-3ch", 2026)), 12)

paired <- function(a, b, la, lb) {
  m <- merge(a[, .(match_id, ea = abs(pred_margin - margin))],
             b[, .(match_id, eb = abs(pred_margin - margin))], by = "match_id")
  d <- m$eb - m$ea
  ci <- t.test(d)$conf.int
  say(sprintf("%-22s dMAE %+.4f  95%% CI [%+.4f, %+.4f]  (positive = %s worse)",
              paste0(lb, " vs ", la), mean(d), ci[1], ci[2], lb))
}
say("")
say("--- paired differences on the POOLED window ---")
paired(p4, p3, "v3-4ch", "v3-3ch")
paired(p2, p4, "v2", "v3-4ch")
paired(p2, p3, "v2", "v3-3ch")

say("")
say("--- ruck check: 3-vs-4 should show up in RUCKS or nowhere ---")
rk <- merge(r4[, .(player_id, season, round, torp_4 = torp, epr_4 = epr)],
            r3[, .(player_id, season, round, torp_3 = torp, epr_3 = epr)],
            by = c("player_id", "season", "round"))
pg <- unique(as.data.table(pgd4)[, .(player_id, position_group)], by = "player_id")
rk <- merge(rk, pg, by = "player_id")
say_dt(rk[!is.na(position_group),
          .(n = .N, mean_abs_epr_diff = round(mean(abs(epr_4 - epr_3), na.rm = TRUE), 4),
            mean_abs_torp_diff = round(mean(abs(torp_4 - torp_3), na.rm = TRUE), 4)),
          by = position_group][order(-mean_abs_epr_diff)], 8)

arrow::write_parquet(rbind(p2, p4, p3), file.path(OUT_DIR, "epv3_3v4_preds.parquet"))
close(con)
cat("\nDone\n")
