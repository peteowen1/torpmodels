# ws_defender_position_split.R --------------------------------------------
# THE production gate for the defender-value program.
# docs/plans/FABLE-DEFENDER-VALUE-PLAN.md §6.7 -> §1.1
#
# HYPOTHESIS
# ----------
# torp_diff is pooled over all 22 players before the match model sees it, so
# the model cannot express a position-aware weighting -- unlike the EPR/PSR
# component reweighting it already owns (2026-PSR-EPR-DIAGNOSIS §3). The
# defender-value diagnosis found key defenders carry ~1.65x the margin per
# rating point of key forwards, stable across four independent walk-forward
# windows (plan §6.7: 1.50 / 1.62 / 1.57 / 1.72). A standalone rating-only
# model improved by dMAE -0.265 [-0.562, +0.030] under position-aware
# weighting. This script asks the only question that can authorise a change:
# does it survive the real rolling OOS harness on the C6-era production chain?
#
# LEAK SAFETY (plan §1.3, FABLE-MATCH-MAE-PLAN G6)
# ------------------------------------------------
# * Buckets come from lineup_position (the team sheet, known pre-match), NOT
#   position_group (derived from PBP, i.e. post-match). The diagnosis used
#   position_group because it was descriptive; a predictive feature cannot.
#   This also resolves plan open question O1 in favour of the lineup taxonomy.
# * TOG weighting uses POSITION_AVG_TOG, exactly as production's
#   .build_team_ratings_df() does -- a static per-position constant. Actual
#   time-on-ground is unknown pre-match and would leak.
# * EMERG/SUB filtered exactly as production filters them.
# * P1's fixed weights are fitted on <= 2025 only and scored on 2026.
# * The match model retrains inside the rolling loop, so P2's raw features
#   are leak-safe by construction.
#
# VARIANTS
#   P1 -- one extra feature: position-reweighted torp diff, weights fixed
#         a priori from the pre-2026 walk-forward fit.
#   P2 -- seven extra features: the raw per-bucket torp diffs; let the model
#         learn the weighting itself.
#
# Both are added as XGB extra_feature_cols on top of the C6 production
# baseline (elo_diff). GAMs are left at production, so the comparison
# isolates the new information rather than confounding it with formula
# changes. Per plan G5, torp/R/*.R is never modified.
#
# Stages:
#   Rscript ws_defender_position_split.R data      # attach position features
#   Rscript ws_defender_position_split.R baseline  # C6 baseline, 2026 screen
#   Rscript ws_defender_position_split.R screen_p1 # (and screen_p2)
#   Rscript ws_defender_position_split.R summary
# -------------------------------------------------------------------------

stage <- {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) >= 1) a[1] else "all"
}
cat("=== ws_defender_position_split.R stage:", stage, "===\n")

suppressPackageStartupMessages({
  library(tidyverse); library(xgboost); library(mgcv); library(MLmetrics)
  library(geosphere); library(cli); library(data.table); library(arrow)
})
devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)

EXPERIMENTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RESULTS_DIR <- file.path(EXPERIMENTS_DIR, "results")
.rds <- function(name) file.path(RESULTS_DIR, name)
source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))

DATA_DIR <- "C:/dev/torpverse/torpdata/data/"
TEST_SEASONS <- 2026
SEASONS <- 2021:2026

# lineup_position (team sheet) -> position bucket. INT is kept as its own
# bucket rather than force-assigned: 4 players per team sit there and their
# on-field role is genuinely unknown pre-match.
BUCKET_MAP <- c(
  FB = "KEY_DEF", CHB = "KEY_DEF",
  BPL = "MED_DEF", BPR = "MED_DEF", HBFL = "MED_DEF", HBFR = "MED_DEF",
  C = "MID", R = "MID", RR = "MID", WL = "MID", WR = "MID",
  RK = "RUCK",
  HFFL = "MED_FWD", HFFR = "MED_FWD", FPL = "MED_FWD", FPR = "MED_FWD",
  CHF = "KEY_FWD", FF = "KEY_FWD",
  INT = "INT"
)
BUCKETS <- c("KEY_DEF", "MED_DEF", "MID", "RUCK", "MED_FWD", "KEY_FWD", "INT")

# P1 weights: relative per-bucket margin value, fitted on <= 2025 ONLY
# (plan §6.7's 2026 walk-forward column, which trained on 2021-2025).
# Normalised so the mean weight is 1, i.e. P1 is a pure REWEIGHTING of
# torp_diff and not a rescaling of it.
P1_RAW <- c(KEY_DEF = 2.08, MED_DEF = 0.81, MID = 1.65, RUCK = 2.17,
            MED_FWD = 1.38, KEY_FWD = 1.21, INT = 1.00)
P1_W <- P1_RAW / mean(P1_RAW)

# =========================================================================
if (stage %in% c("data", "all")) {
  cli::cli_h1("Attaching position-split rating features")

  # lineup_position comes from player_game_*.parquet, NOT teams_*.parquet:
  # the local teams parquets have lineup_position populated for 2025 only
  # (every other season is NA), which silently reduced coverage to one
  # season on the first run. player_game carries the same team-sheet field,
  # 100% populated 2021-2026. It is still pre-match information -- team
  # sheets are published before the game and production reads exactly this
  # field via .build_team_ratings_df() -- so no leak is introduced. Actual
  # time-on-ground from the same table is deliberately NOT used.
  teams <- rbindlist(lapply(SEASONS, function(s)
    as.data.table(read_parquet(file.path(DATA_DIR, sprintf("player_game_%d.parquet", s))))[
      , .(player_id, match_id, season, round_number = as.numeric(round),
          team_id, lineup_position)]),
    use.names = TRUE, fill = TRUE)
  tr <- as.data.table(read_parquet(file.path(DATA_DIR, "torp_ratings.parquet")))
  tr[, round := as.numeric(round)]

  cov_chk <- teams[, .(pct = 100 * mean(!is.na(lineup_position))), by = season][order(season)]
  print(cov_chk)
  if (any(cov_chk$pct < 90)) {
    cli::cli_abort("lineup_position coverage below 90% in some season -- see cov_chk.")
  }

  # Mirror .build_team_ratings_df(): drop EMERG/SUB, join as-of ratings,
  # impute missing with the component priors, weight by POSITION_AVG_TOG.
  tl <- merge(teams[!lineup_position %in% c("EMERG", "SUB")],
              tr[, .(player_id, season, round, epr, psr)],
              by.x = c("player_id", "season", "round_number"),
              by.y = c("player_id", "season", "round"),
              all.x = TRUE)
  prior_epr <- EPR_PRIOR_RATE_RECV + EPR_PRIOR_RATE_DISP +
               EPR_PRIOR_RATE_SPOIL + EPR_PRIOR_RATE_HITOUT
  na_pct <- 100 * mean(is.na(tl$epr))
  cli::cli_inform("Missing EPR on lineup rows: {round(na_pct, 1)}%")
  if (na_pct > 25) cli::cli_warn("High missing-rating rate ({round(na_pct,1)}%)")
  tl[is.na(epr), epr := prior_epr]
  tl[is.na(psr), psr := 0]

  tl[, ltog := fifelse(is.na(POSITION_AVG_TOG[lineup_position]),
                       POSITION_AVG_TOG_DEFAULT, POSITION_AVG_TOG[lineup_position])]
  tl[, bucket := BUCKET_MAP[lineup_position]]
  n_unbucketed <- sum(is.na(tl$bucket))
  if (n_unbucketed > 0) {
    cli::cli_warn("{n_unbucketed} lineup rows have no bucket ({paste(unique(tl$lineup_position[is.na(tl$bucket)]), collapse=', ')})")
  }
  tl <- tl[!is.na(bucket)]
  cli::cli_inform("bucketed lineup rows: {nrow(tl)} across {length(unique(tl$match_id))} matches")
  tl[, torp_w := (TORP_EPR_WEIGHT * epr + (1 - TORP_EPR_WEIGHT) * psr) * ltog]

  bs <- tl[, .(v = sum(torp_w)), by = .(match_id, team_id, bucket)]
  wide <- dcast(bs, match_id + team_id ~ bucket, value.var = "v", fill = 0)
  setnames(wide, BUCKETS, paste0("pos_", BUCKETS), skip_absent = TRUE)

  team_mdl_df <- copy(as.data.table(readRDS(.rds("ws0_team_mdl_df.rds"))))
  # Opponent id: the other team_id within the same match.
  team_mdl_df[, opp_team_id := {
    ids <- team_id
    if (length(ids) == 2L) rev(ids) else NA_character_
  }, by = match_id]

  pc <- paste0("pos_", BUCKETS)
  pc <- pc[pc %in% names(wide)]
  own <- copy(wide); setnames(own, pc, paste0(pc, "_own"))
  opp <- copy(wide); setnames(opp, pc, paste0(pc, "_opp"))

  n0 <- nrow(team_mdl_df)
  team_mdl_df <- merge(team_mdl_df, own, by = c("match_id", "team_id"), all.x = TRUE)
  team_mdl_df <- merge(team_mdl_df, opp, by.x = c("match_id", "opp_team_id"),
                       by.y = c("match_id", "team_id"), all.x = TRUE)
  stopifnot(nrow(team_mdl_df) == n0)

  for (b in pc) {
    team_mdl_df[[paste0(b, "_diff")]] <-
      team_mdl_df[[paste0(b, "_own")]] - team_mdl_df[[paste0(b, "_opp")]]
  }
  diff_cols <- paste0(pc, "_diff")
  # Rows with no lineup coverage (e.g. future fixtures) -> 0 differential.
  for (v in diff_cols) set(team_mdl_df, which(is.na(team_mdl_df[[v]])), v, 0)

  # P1: single reweighted feature.
  team_mdl_df[, pos_reweighted_diff := 0]
  for (b in BUCKETS) {
    cc <- paste0("pos_", b, "_diff")
    if (cc %in% names(team_mdl_df)) {
      team_mdl_df[, pos_reweighted_diff := pos_reweighted_diff + P1_W[[b]] * get(cc)]
    }
  }

  cat("\n--- coverage ---\n")
  recon <- rowSums(as.matrix(team_mdl_df[, ..diff_cols]))
  cov_rows <- sum(recon != 0)
  cat(sprintf("  rows with position features: %d of %d (%.1f%%)\n",
              cov_rows, nrow(team_mdl_df), 100 * cov_rows / nrow(team_mdl_df)))
  print(team_mdl_df[, .(rows = .N, covered = sum(rowSums(as.matrix(.SD)) != 0)),
                    by = season.x, .SDcols = diff_cols][order(season.x)])
  if (cov_rows / sum(!is.na(team_mdl_df$win)) < 0.9) {
    cli::cli_abort("Position-feature coverage below 90% of completed rows -- join is broken.")
  }

  cat("\n--- sanity: reconstructed vs production torp_diff ---\n")
  ok <- !is.na(team_mdl_df$torp_diff) & recon != 0
  cat(sprintf("  cor(sum of bucket diffs, production torp_diff) = %.4f  (n = %d)\n",
              cor(recon[ok], team_mdl_df$torp_diff[ok]), sum(ok)))
  cat("  (Not expected to be 1.000 -- production weights by lineup_tog over a\n")
  cat("   slightly different lineup filter -- but a low value means the\n")
  cat("   reconstruction is wrong, not merely different.)\n")
  cat(sprintf("  cor(pos_reweighted_diff, torp_diff) = %.4f\n",
              cor(team_mdl_df$pos_reweighted_diff[ok], team_mdl_df$torp_diff[ok])))
  print(summary(team_mdl_df[, ..diff_cols]))

  saveRDS(as.data.frame(team_mdl_df), .rds("wsdef_team_mdl_df.rds"))
  cli::cli_alert_success("Saved wsdef_team_mdl_df.rds")
}

# =========================================================================
.print_metrics <- function(m, label) {
  cat(sprintf(
    "%-46s MAE=%.3f RMSE=%.3f Brier=%.4f Slope=%.3f Cor=%.3f\n",
    label, m$mae, m$rmse, m$brier, m$slope, m$cor))
}

run_variant <- function(extra_cols, label, key) {
  team_mdl_df <- readRDS(.rds("wsdef_team_mdl_df.rds"))
  cli::cli_h1(label)
  t0 <- Sys.time()
  roll <- run_rolling_eval(
    team_mdl_df,
    test_seasons = TEST_SEASONS,
    extra_feature_cols = extra_cols,
    cv_extra_feature_cols = extra_cols,
    verbose = TRUE)
  cli::cli_inform("elapsed {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  preds <- roll$input_blend
  m <- .compute_metrics(preds)
  .print_metrics(m, label)
  saveRDS(list(roll = roll, preds = preds, metrics = m, extra_cols = extra_cols),
          .rds(paste0("wsdef_", key, ".rds")))
  invisible(list(preds = preds, metrics = m))
}

if (stage %in% c("baseline", "all")) {
  run_variant("elo_diff", "BASELINE C6 (elo_diff only), 2026 screen", "baseline")
}
if (stage %in% c("screen_p1", "all")) {
  run_variant(c("elo_diff", "pos_reweighted_diff"),
              "P1 position-reweighted torp_diff, 2026 screen", "p1")
}
if (stage %in% c("screen_p2", "all")) {
  pcd <- paste0("pos_", BUCKETS, "_diff")
  run_variant(c("elo_diff", pcd), "P2 raw per-bucket diffs, 2026 screen", "p2")
}

if (stage %in% c("summary", "all")) {
  cli::cli_h1("Summary")
  b <- readRDS(.rds("wsdef_baseline.rds"))
  for (k in c("p1", "p2")) {
    f <- .rds(paste0("wsdef_", k, ".rds"))
    if (!file.exists(f)) next
    v <- readRDS(f)
    .print_metrics(b$metrics, "baseline C6")
    .print_metrics(v$metrics, toupper(k))
    bt <- boot_mae_diff(v$preds, b$preds)
    cat(sprintf("  %s - baseline  dMAE = %+.3f  95%% CI [%+.3f, %+.3f]  P(better) %.2f\n\n",
                toupper(k), bt$mean_diff, bt$ci_lower, bt$ci_upper, bt$p_better))
  }
}
