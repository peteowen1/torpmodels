# ws_v2_feature_improve.R ---------------------------------------------------
# Can the match model be improved to USE v2 better?
# torpverse/docs/plans/FABLE-DEFENDER-VALUE-PLAN.md §7.34
#
# §7.33 diagnosed v2's match-model cost as a GAIN problem, not a signal
# problem: sd(epr_diff) rose 9.97 -> 16.80 (+68%) while cor(epr_diff, margin)
# IMPROVED 0.5145 -> 0.5348. The model reads a better signal at the wrong
# scale. Two consequences point at fixes:
#
#   * XGBoost is scale-invariant (tree splits on order, not magnitude), so a
#     pure scale change cannot hurt it. The GAM half can: match_train.R uses
#     s(epr_diff, bs="ts", k=5) -- a 5-basis smooth whose knot placement was
#     effectively tuned against v1's narrower range. Over a 68% wider range,
#     k=5 has to spend its budget covering tails, under-resolving the middle
#     where most matches live.
#
#   * §6.8 closed position-split features as null-to-negative -- but §6.10 then
#     found those features were UNCENTRED, so they partly encoded roster shape
#     rather than rating quality, and explicitly flagged "worth one re-run with
#     position-centred inputs before treating that branch as fully closed".
#     That re-run was never done. v2's ratings are centred AND standardised, so
#     this is the first clean test of that hypothesis.
#
# ARMS (all on published v2 ratings):
#   A  baseline           : v2 as production consumes it
#   B  rescaled epr        : epr_* diffs scaled to v1's spread, so the k=5
#                           smooths see the range they were tuned for. Pure
#                           reparameterisation -- no information added.
#   C  + position splits   : per-bucket torp diffs as extra XGB features, the
#                           §6.8 re-run on clean inputs.
#
# LEAK SAFETY (§1.3): buckets come from lineup_position (team sheet, known
# pre-match) via LINEUP_POSITION_GROUP_MAP, and TOG weighting uses the static
# POSITION_AVG_TOG -- never actual TOG, which is post-match.
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
BK <- c("KEY_DEFENDER","MEDIUM_DEFENDER","MIDFIELDER","RUCK","MEDIUM_FORWARD","KEY_FORWARD")

r2 <- as.data.table(load_torp_ratings());              r2[, round := as.numeric(round)]
r1 <- as.data.table(load_torp_ratings(version = "v1")); r1[, round := as.numeric(round)]
pg <- rbindlist(lapply(SEASONS, function(s) {
  f <- file.path(DD, sprintf("player_game_%d.parquet", s)); if (!file.exists(f)) return(NULL)
  as.data.table(read_parquet(f))[, .(player_id, match_id, season,
    round = as.numeric(round), team_id, lineup_position)] }), use.names=TRUE, fill=TRUE)
pg <- pg[lineup_position != "EMERG" | is.na(lineup_position)]   # matches production now
pg[, ltog := POSITION_AVG_TOG[lineup_position]]
pg[is.na(ltog), ltog := POSITION_AVG_TOG_DEFAULT]
pg[, bucket := unname(LINEUP_POSITION_GROUP_MAP[lineup_position])]

agg <- function(rt) {
  x <- merge(pg, rt[, .(player_id, season, round, epr, psr)],
             by = c("player_id","season","round"), all.x = TRUE)
  x[, `:=`(e = epr*ltog, p = fifelse(is.na(psr), PSR_PRIOR_RATE, psr)*ltog)]
  x[, .(e = sum(e, na.rm=TRUE), p = sum(p, na.rm=TRUE)), by = .(match_id, team_id)]
}
TM2 <- agg(r2); TM1 <- agg(r1)

base <- as.data.table(readRDS(.rds("ws0_team_mdl_df.rds")))
attach_ratings <- function(d, TM) {
  d[, opp_team_id := { tt <- team_id; ifelse(seq_len(.N)==1L, tt[2], tt[1]) }, by=match_id]
  d <- merge(d, TM, by=c("match_id","team_id"), all.x=TRUE); setnames(d, c("e","p"), c("eo","po"))
  d <- merge(d, TM, by.x=c("match_id","opp_team_id"), by.y=c("match_id","team_id"), all.x=TRUE)
  setnames(d, c("e","p"), c("ep","pp"))
  ok <- !is.na(d$eo) & !is.na(d$ep)
  d[ok, `:=`(epr.x=eo, epr.y=ep, psr.x=po, psr.y=pp,
             epr_diff=eo-ep, psr_diff=po-pp,
             torp.x=TORP_EPR_WEIGHT*eo+(1-TORP_EPR_WEIGHT)*po,
             torp.y=TORP_EPR_WEIGHT*ep+(1-TORP_EPR_WEIGHT)*pp,
             torp_diff=TORP_EPR_WEIGHT*(eo-ep)+(1-TORP_EPR_WEIGHT)*(po-pp))]
  d[, c("eo","po","ep","pp") := NULL]
  d
}
armA <- attach_ratings(copy(base), TM2)

# --- B: rescale the epr family to v1's spread -------------------------------
armB <- copy(armA)
d1 <- attach_ratings(copy(base), TM1)
k_scale <- sd(d1$epr_diff, na.rm=TRUE) / sd(armA$epr_diff, na.rm=TRUE)
cli::cli_inform("epr rescale factor (v1 sd / v2 sd) = {round(k_scale, 4)}")
for (cc in c("epr_diff","epr_recv_diff","epr_disp_diff","epr_spoil_diff","epr_hitout_diff")) {
  if (cc %in% names(armB)) armB[[cc]] <- armB[[cc]] * k_scale
}
armB[, torp_diff := TORP_EPR_WEIGHT*epr_diff + (1-TORP_EPR_WEIGHT)*psr_diff]

# --- C: position-split features on clean v2 ratings -------------------------
pos <- merge(pg[!is.na(bucket)], r2[, .(player_id, season, round, torp)],
             by=c("player_id","season","round"), all.x=TRUE)
pos[, tw := fifelse(is.na(torp), 0, torp) * ltog]
pb <- dcast(pos[, .(v = sum(tw)), by=.(match_id, team_id, bucket)],
            match_id + team_id ~ bucket, value.var="v", fill=0)
bcols <- intersect(BK, names(pb))
armC <- copy(armA)
armC <- merge(armC, pb, by=c("match_id","team_id"), all.x=TRUE)
opp <- copy(pb); setnames(opp, bcols, paste0("o_", bcols)); setnames(opp, "team_id","opp_team_id")
armC <- merge(armC, opp, by=c("match_id","opp_team_id"), all.x=TRUE)
split_cols <- paste0("pd_", bcols)
for (b in bcols) {
  armC[[paste0("pd_", b)]] <- armC[[b]] - armC[[paste0("o_", b)]]
  armC[is.na(get(paste0("pd_", b))), (paste0("pd_", b)) := 0]
}
armC[, c(bcols, paste0("o_", bcols)) := NULL]
cli::cli_inform("position-split features: {paste(split_cols, collapse=', ')}")

for (nm in c("armA","armB","armC")) assign(nm, as.data.frame(get(nm)))
run_arm <- function(df, label, extra = NULL) {
  cli::cli_h1(label); t0 <- Sys.time()
  roll <- run_rolling_eval(df, test_seasons = TEST_SEASONS,
                           extra_feature_cols = extra,
                           cv_extra_feature_cols = extra, verbose = FALSE)
  cli::cli_inform("elapsed {round(difftime(Sys.time(), t0, units='mins'),2)} min")
  p <- roll$input_blend; m <- .compute_metrics(p)
  cli::cli_inform("MAE {round(m$mae,3)} | Brier {round(m$brier,5)} | acc {round(m$accuracy,2)}% | slope {round(m$slope,3)}")
  list(preds = p, metrics = m)
}
res <- list(A = run_arm(armA, "A v2 baseline"),
            B = run_arm(armB, "B epr rescaled to v1 spread"),
            C = run_arm(armC, "C + position splits (clean v2)", extra = split_cols))
cli::cli_h1("PAIRED BOOTSTRAP vs A")
for (k in c("B","C")) {
  bd <- boot_mae_diff(res[[k]]$preds, res$A$preds, B = 2000)
  cli::cli_inform("{k} vs A: dMAE {round(bd$mae_diff,3)} [{round(bd$mae_ci[1],3)}, {round(bd$mae_ci[2],3)}] | dBrier {round(bd$brier_diff,5)} [{round(bd$brier_ci[1],5)}, {round(bd$brier_ci[2],5)}]")
}
saveRDS(res, .rds("ws_v2_feature_improve.rds"))
cli::cli_alert_success("Saved")
