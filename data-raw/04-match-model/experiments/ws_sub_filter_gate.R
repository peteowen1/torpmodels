# ws_sub_filter_gate.R ------------------------------------------------------
# Should the team-rating build include the medical sub?
# torpverse/docs/plans/FABLE-DEFENDER-VALUE-PLAN.md §7.30
#
# CONTEXT (§7.29). The AFL recoded the medical sub from `SUB` to `INT` in 2026.
# The filter `(lineup_position != "EMERG" & != "SUB")` in
# match_data_prep.R:101 / match_model.R:138 therefore keeps 23 players per team
# in 2026 and 22 in earlier seasons. Squad size never changed (~23 named
# throughout); only the coding did. And the player being dropped was never
# nothing -- the SUB-coded player averaged 32.5-32.7% time on ground.
#
# IMPORTANT SCOPE. This filter lives in TEAM-RATING CONSTRUCTION for the match
# model, not in the player ratings. Changing it needs no ratings regeneration:
# EPR and PSR are per-player and untouched. That makes this a cheap, bounded
# experiment rather than a full-history rewrite.
#
# ARMS (published v2 ratings throughout -- current canonical):
#   A  current      : exclude EMERG and SUB   -> 22 players pre-2026, 23 in 2026
#   B  include-all  : exclude EMERG only      -> 23 players in EVERY season
#   C  strict-22    : exclude EMERG/SUB, and in 2026 also drop each team's
#                     lowest-expected-TOG interchange player -> 22 everywhere
#
# B and C are the two coherent options: consistent-at-23 or consistent-at-22.
# A is the status quo and is consistent at neither.
suppressPackageStartupMessages({
  library(tidyverse); library(xgboost); library(mgcv); library(MLmetrics)
  library(geosphere); library(cli); library(data.table); library(arrow)
})
devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
try(clear_skip_markers(), silent = TRUE)

EXPERIMENTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RESULTS_DIR <- file.path(EXPERIMENTS_DIR, "results")
.rds <- function(n) file.path(RESULTS_DIR, n)
source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))
DD <- "C:/dev/torpverse/torpdata/data/"
SEASONS <- 2021:2026
TEST_SEASONS <- 2026

r <- as.data.table(load_torp_ratings()); r[, round := as.numeric(round)]
pg <- rbindlist(lapply(SEASONS, function(s) {
  f <- file.path(DD, sprintf("player_game_%d.parquet", s)); if (!file.exists(f)) return(NULL)
  as.data.table(read_parquet(f))[, .(player_id, match_id, season,
    round = as.numeric(round), team_id, lineup_position,
    tog = time_on_ground_percentage)] }), use.names = TRUE, fill = TRUE)

build <- function(mode) {
  x <- copy(pg)
  if (mode == "A") {
    x <- x[!lineup_position %in% c("EMERG", "SUB") | is.na(lineup_position)]
  } else if (mode == "B") {
    x <- x[lineup_position != "EMERG" | is.na(lineup_position)]
  } else if (mode == "C") {
    x <- x[!lineup_position %in% c("EMERG", "SUB") | is.na(lineup_position)]
    # in seasons with no SUB coding, drop the lowest-TOG interchange player so
    # every season contributes the same 22. TOG is post-match, so this arm is
    # DIAGNOSTIC ONLY -- it cannot be a production rule.
    x[, .n := .N, by = .(match_id, team_id)]
    x[, .rk := fifelse(lineup_position == "INT",
                       frank(tog, ties.method = "first"), NA_real_),
      by = .(match_id, team_id)]
    x <- x[!(.n > 22 & !is.na(.rk) & .rk == 1)]
    x[, c(".n", ".rk") := NULL]
  }
  x[, ltog := POSITION_AVG_TOG[lineup_position]]
  x[is.na(ltog), ltog := POSITION_AVG_TOG_DEFAULT]
  y <- merge(x, r[, .(player_id, season, round, epr, psr)],
             by = c("player_id","season","round"), all.x = TRUE)
  y[, `:=`(e = epr * ltog, p = fifelse(is.na(psr), PSR_PRIOR_RATE, psr) * ltog)]
  out <- y[, .(e = sum(e, na.rm = TRUE), p = sum(p, na.rm = TRUE), np = .N),
           by = .(match_id, team_id)]
  cli::cli_inform("{mode}: mean players/team = {round(mean(out$np),2)}")
  out
}
TM <- lapply(c(A="A", B="B", C="C"), build)

base <- as.data.table(readRDS(.rds("ws0_team_mdl_df.rds")))
mk_arm <- function(tm) {
  d <- copy(base)
  d[, opp_team_id := { tt <- team_id; ifelse(seq_len(.N) == 1L, tt[2], tt[1]) }, by = match_id]
  d <- merge(d, tm[, .(match_id, team_id, e, p)], by = c("match_id","team_id"), all.x = TRUE)
  setnames(d, c("e","p"), c("eo","po"))
  d <- merge(d, tm[, .(match_id, team_id, e, p)],
             by.x = c("match_id","opp_team_id"), by.y = c("match_id","team_id"), all.x = TRUE)
  setnames(d, c("e","p"), c("ep","pp"))
  ok <- !is.na(d$eo) & !is.na(d$ep)
  d[ok, `:=`(epr.x = eo, epr.y = ep, psr.x = po, psr.y = pp,
             epr_diff = eo - ep, psr_diff = po - pp,
             torp.x = TORP_EPR_WEIGHT*eo + (1-TORP_EPR_WEIGHT)*po,
             torp.y = TORP_EPR_WEIGHT*ep + (1-TORP_EPR_WEIGHT)*pp,
             torp_diff = TORP_EPR_WEIGHT*(eo-ep) + (1-TORP_EPR_WEIGHT)*(po-pp))]
  d[, c("eo","po","ep","pp","opp_team_id") := NULL]
  as.data.frame(d)
}
arms <- lapply(TM, mk_arm)

run_arm <- function(df, label) {
  cli::cli_h1(label); t0 <- Sys.time()
  roll <- run_rolling_eval(df, test_seasons = TEST_SEASONS, verbose = FALSE)
  cli::cli_inform("elapsed {round(difftime(Sys.time(), t0, units='mins'),2)} min")
  p <- roll$input_blend; m <- .compute_metrics(p)
  cli::cli_inform("MAE {round(m$mae,3)} | Brier {round(m$brier,5)} | acc {round(m$accuracy,2)}%")
  list(preds = p, metrics = m)
}
res <- list(A = run_arm(arms$A, "A current (22 pre-2026, 23 in 2026)"),
            B = run_arm(arms$B, "B include-all (23 every season)"),
            C = run_arm(arms$C, "C strict-22 (diagnostic only)"))

cli::cli_h1("PAIRED BOOTSTRAP vs A (current)")
for (k in c("B","C")) {
  bd <- boot_mae_diff(res[[k]]$preds, res$A$preds, B = 2000)
  cli::cli_inform("{k} vs A: dMAE {round(bd$mae_diff,3)} [{round(bd$mae_ci[1],3)}, {round(bd$mae_ci[2],3)}] | dBrier {round(bd$brier_diff,5)} [{round(bd$brier_ci[1],5)}, {round(bd$brier_ci[2],5)}]")
}
saveRDS(res, .rds("ws_sub_filter_gate.rds"))
cli::cli_alert_success("Saved ws_sub_filter_gate.rds")
