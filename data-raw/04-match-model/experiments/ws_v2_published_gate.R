# ws_v2_published_gate.R ---------------------------------------------------
# The CLEAN §1.1 gate: published v1 vs published v2, no rebuild anywhere.
# torpverse/docs/plans/FABLE-DEFENDER-VALUE-PLAN.md §7.27
#
# WHY THIS SUPERSEDES ws_v2_ratings_gate.R. That script rebuilt player ratings
# from raw data because v2 did not exist as a published artifact. The rebuild
# validated at r = 0.95 against cached features -- good enough to detect a
# regression, and (as §7.26 found the hard way) NOT good enough to describe the
# published article. Both vintages are now published, so the ratings can simply
# be READ. Nothing here recomputes EPR or PSR.
#
# The only reconstruction step left is the team-level aggregation, which uses
# production's own weighting (match_data_prep.R:119-125):
#     team rating = sum over the named lineup of
#                   player_rating * POSITION_AVG_TOG[lineup_position]
# and that is validated against the cached C6 features before anything is
# compared -- the published-v1 arm should reproduce them closely, because the
# cache was built from those very ratings.
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

# ---- 1. read both published vintages ---------------------------------------
cli::cli_h1("Reading published ratings (no rebuild)")
r2 <- as.data.table(load_torp_ratings());              r2[, round := as.numeric(round)]
r1 <- as.data.table(load_torp_ratings(version = "v1")); r1[, round := as.numeric(round)]
cli::cli_inform("v2 (canonical): {nrow(r2)} rows | v1 (preserved): {nrow(r1)} rows")
stopifnot(all(c("epr","psr") %in% names(r2)), all(c("epr","psr") %in% names(r1)))

pg <- rbindlist(lapply(SEASONS, function(s) {
  f <- file.path(DD, sprintf("player_game_%d.parquet", s)); if (!file.exists(f)) return(NULL)
  as.data.table(read_parquet(f))[, .(player_id, match_id, season,
                                     round = as.numeric(round), team_id, lineup_position)]
}), use.names = TRUE, fill = TRUE)
pg <- pg[!lineup_position %in% c("EMERG","SUB")]
pg[, ltog := POSITION_AVG_TOG[lineup_position]]
pg[is.na(ltog), ltog := POSITION_AVG_TOG_DEFAULT]

agg <- function(rt, tag) {
  x <- merge(pg, rt[, .(player_id, season, round, epr, psr)],
             by = c("player_id","season","round"), all.x = TRUE)
  x[, `:=`(e = epr * ltog,
           p = fifelse(is.na(psr), PSR_PRIOR_RATE, psr) * ltog)]
  out <- x[, .(e = sum(e, na.rm = TRUE), p = sum(p, na.rm = TRUE)),
           by = .(match_id, team_id)]
  cli::cli_inform("{tag}: {nrow(out)} team-match rows aggregated")
  out
}
TM1 <- agg(r1, "v1"); TM2 <- agg(r2, "v2")

# ---- 2. validate the published-v1 aggregation against the cached features ---
cli::cli_h1("Validating published-v1 aggregation against cached C6 features")
base <- as.data.table(readRDS(.rds("ws0_team_mdl_df.rds")))
chk <- merge(base[, .(match_id, team_id, season = season.x,
                      epr_cached = epr.x, psr_cached = psr.x)],
             TM1, by = c("match_id","team_id"))

# Validate on PRE-TEST seasons only. The cache is a snapshot built partway
# through 2026 by an earlier experiment, so its 2026 ratings reflect a
# mid-season state while the published v1 file carries the completed season.
# Diagnosed rather than assumed: per-season r is 0.998 for 2021-2025 and 0.863
# for 2026, with mean |diff| ~0.3 vs 7.17 -- i.e. the join is exact and the
# reference is stale. PSR corroborates (slope 1.005, R2 0.995 throughout), and
# PSR travels the IDENTICAL join/weight/sum path, so a mechanical error would
# have shown there too.
# This affects the CHECK only. Both arms overwrite the rating columns with
# freshly aggregated values, so no stale number enters either model.
val <- chk[season < min(TEST_SEASONS)]
r_epr <- cor(val$epr_cached, val$e, use = "complete.obs")
r_psr <- cor(val$psr_cached, val$p, use = "complete.obs")
cli::cli_inform("matched rows: {nrow(chk)} of {nrow(base)} ({nrow(val)} pre-{min(TEST_SEASONS)} used for validation)")
cli::cli_inform("cor(published-v1 EPR, cached EPR) = {round(r_epr, 4)}  [pre-test seasons]")
cli::cli_inform("cor(published-v1 PSR, cached PSR) = {round(r_psr, 4)}  [pre-test seasons]")
cli::cli_inform("2026 (stale cache, excluded): EPR r = {round(cor(chk[season >= min(TEST_SEASONS)]$epr_cached, chk[season >= min(TEST_SEASONS)]$e, use='complete.obs'), 4)}")
if (is.na(r_epr) || r_epr < 0.99) {
  cli::cli_abort(c("Published-v1 aggregation does not reproduce the cached feature (r = {round(r_epr,4)}).",
    "i" = "Reading published ratings through production's own weighting should be near-exact on settled seasons.",
    "x" = "Aborting -- the aggregation is wrong, not the ratings."))
}
cli::cli_alert_success("Aggregation validated on settled seasons")

# ---- 3. build the two arms -------------------------------------------------
mk_arm <- function(TM, label) {
  d <- copy(base)
  d[, opp_team_id := { tt <- team_id; ifelse(seq_len(.N) == 1L, tt[2], tt[1]) }, by = match_id]
  d <- merge(d, TM, by = c("match_id","team_id"), all.x = TRUE)
  setnames(d, c("e","p"), c("e_own","p_own"))
  d <- merge(d, TM, by.x = c("match_id","opp_team_id"), by.y = c("match_id","team_id"), all.x = TRUE)
  setnames(d, c("e","p"), c("e_opp","p_opp"))
  ok <- !is.na(d$e_own) & !is.na(d$e_opp)
  d[ok, `:=`(epr.x = e_own, epr.y = e_opp, psr.x = p_own, psr.y = p_opp,
             epr_diff = e_own - e_opp, psr_diff = p_own - p_opp,
             torp.x = TORP_EPR_WEIGHT*e_own + (1-TORP_EPR_WEIGHT)*p_own,
             torp.y = TORP_EPR_WEIGHT*e_opp + (1-TORP_EPR_WEIGHT)*p_opp,
             torp_diff = TORP_EPR_WEIGHT*(e_own-e_opp) + (1-TORP_EPR_WEIGHT)*(p_own-p_opp))]
  d[, c("e_own","p_own","e_opp","p_opp","opp_team_id") := NULL]
  cli::cli_inform("{label}: {sum(ok)} of {nrow(d)} rows re-rated")
  as.data.frame(d)
}
arm1 <- mk_arm(TM1, "arm v1 (published)")
arm2 <- mk_arm(TM2, "arm v2 (published)")

# ---- 4. rolling OOS --------------------------------------------------------
run_arm <- function(df, label) {
  cli::cli_h1(label); t0 <- Sys.time()
  roll <- run_rolling_eval(df, test_seasons = TEST_SEASONS, verbose = FALSE)
  cli::cli_inform("elapsed {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  preds <- roll$input_blend; m <- .compute_metrics(preds); print(m)
  list(preds = preds, metrics = m)
}
res1 <- run_arm(arm1, "PUBLISHED v1")
res2 <- run_arm(arm2, "PUBLISHED v2")

cli::cli_h1("PAIRED BOOTSTRAP: published v2 vs published v1")
bd <- boot_mae_diff(res2$preds, res1$preds, B = 2000)
print(bd)
saveRDS(list(v1 = res1, v2 = res2, boot = bd, r_epr = r_epr, r_psr = r_psr),
        .rds("ws_v2_published_gate.rds"))
cli::cli_alert_success("Saved ws_v2_published_gate.rds")
