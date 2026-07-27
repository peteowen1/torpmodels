# ws_v2_ratings_gate.R -----------------------------------------------------
# THE §1.1 SHIP GATE for the v2 rating vintage.
# torpverse/docs/plans/FABLE-DEFENDER-VALUE-PLAN.md §7.25
#
# QUESTION. The adopted v2 ratings (EPV position standardisation, weekly PSR
# centring, PSR standardisation, corrected lineup taxonomy) change every
# historical rating. Does feeding them to the C6-era production match model
# help, hurt, or do nothing?
#
# EXPECTATION, SET BEFORE RUNNING. §7.11's standing rule: this gate is
# structurally near-blind to positional calibration. §7.8c showed that
# conditioning on team strength destroys the between-team variation that
# identifies positional coefficients, and the C6 chain contains elo_diff.
# A NEUTRAL RESULT IS THE EXPECTED OUTCOME AND IS NOT EVIDENCE AGAINST v2.
# What this gate can genuinely do is catch a REGRESSION, which is its purpose.
#
# METHOD. Reuse the cached C6 team_mdl_df (ws0_team_mdl_df.rds) and swap the
# rating columns for v2-derived equivalents, rebuilt with production's own
# aggregation: team rating = sum over the named lineup of
#   player_rating * POSITION_AVG_TOG[lineup_position]
# (match_data_prep.R:119-125).
#
# VALIDATION FIRST, per the §7.24 rule. Before any v2 comparison is trusted,
# the v1 reconstruction must reproduce the cached epr.x / psr.x columns. If it
# does not, the v2 numbers are measuring my rebuild, not the change.
suppressPackageStartupMessages({
  library(tidyverse); library(xgboost); library(mgcv); library(MLmetrics)
  library(geosphere); library(cli); library(data.table); library(arrow)
})
devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)

EXPERIMENTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RESULTS_DIR <- file.path(EXPERIMENTS_DIR, "results")
.rds <- function(n) file.path(RESULTS_DIR, n)
source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))
DD <- "C:/dev/torpverse/torpdata/data/"
SEASONS <- 2021:2026
TEST_SEASONS <- 2026
CH <- c("recv","disp","spoil","hitout")
DECAY <- c(recv=EPR_DECAY_RECV, disp=EPR_DECAY_DISP, spoil=EPR_DECAY_SPOIL, hitout=EPR_DECAY_HITOUT)
PGAMES <- c(recv=EPR_PRIOR_GAMES_RECV, disp=EPR_PRIOR_GAMES_DISP,
            spoil=EPR_PRIOR_GAMES_SPOIL, hitout=EPR_PRIOR_GAMES_HITOUT)
PRATE <- c(recv=EPR_PRIOR_RATE_RECV, disp=EPR_PRIOR_RATE_DISP,
           spoil=EPR_PRIOR_RATE_SPOIL, hitout=EPR_PRIOR_RATE_HITOUT)
rdp <- function(pat, ss) rbindlist(lapply(ss, function(s) {
  f <- file.path(DD, sprintf(pat, s)); if (!file.exists(f)) return(NULL)
  as.data.table(read_parquet(f)) }), use.names=TRUE, fill=TRUE)
wmean <- function(x,w) sum(x*w, na.rm=TRUE)/sum(w[!is.na(x)], na.rm=TRUE)
wsd <- function(x,w){ m <- wmean(x,w); sqrt(sum(w*(x-m)^2, na.rm=TRUE)/sum(w[!is.na(x)], na.rm=TRUE)) }

# ---- 1. player-level v1 / v2 ratings (point-in-time, pre-match) ------------
cli::cli_h1("Rebuilding player ratings under both vintages")
pg <- rdp("player_game_%d.parquet", SEASONS); pg[, round := as.numeric(round)]
pg[, tog_safe := pmax(fifelse(is.na(time_on_ground_percentage),100,
                              time_on_ground_percentage)/100, 0.1)]
for (ch in CH) {
  raw <- paste0(ch,"_epv"); p80 <- paste0(".p80_",ch); pg[, (p80) := get(raw)/tog_safe]
  pg[, .m := wmean(get(p80), tog_safe), by=lineup_position]
  pg[, .s := wsd(get(p80), tog_safe),   by=lineup_position]
  S <- wsd(pg[[p80]], pg$tog_safe)
  pg[, (paste0("v1_",ch)) := (get(p80) - .m) * tog_safe]
  pg[, (paste0("v2_",ch)) := if (ch %in% EPV_STANDARDISE_CHANNELS)
       (get(p80) - .m) * pmin(S/pmax(.s,1e-6),1e9) * tog_safe else (get(p80) - .m) * tog_safe]
  pg[, c(".m",".s") := NULL]
}
setorder(pg, player_id, utc_start_time); pg[, .date := as.Date(utc_start_time)]
# accumulates BEFORE adding the current game, so each row carries the player's
# PRE-MATCH rating -- which is what a predictive feature must use
run_decay <- function(x, dates, lam) { n <- length(x); out <- numeric(n); s <- 0; prev <- dates[1]
  for (i in seq_len(n)) { s <- s*exp(-as.numeric(dates[i]-prev)/lam); prev <- dates[i]
    out[i] <- s; s <- s + x[i] }; out }
for (ch in CH) { lam <- DECAY[[ch]]
  pg[, .den := run_decay(tog_safe, .date, lam), by=player_id]
  for (v in c("v1","v2")) { pg[, .S := run_decay(get(paste0(v,"_",ch))*tog_safe, .date, lam), by=player_id]
    pg[, (paste0("epr_",v,"_",ch)) := (EPR_LOADING_DEFAULT*.S + PGAMES[[ch]]*PRATE[[ch]])/(.den+PGAMES[[ch]])] }
  pg[, c(".den",".S") := NULL] }
for (v in c("v1","v2")) pg[, (paste0("epr_",v)) := rowSums(.SD), .SDcols=paste0("epr_",v,"_",CH)]

coefs <- fread("C:/dev/torpverse/torp/inst/extdata/psr_coefficients.csv")[beta != 0]
sr <- rdp("player_stat_ratings_%d.parquet", SEASONS); sr[, round := as.numeric(round)]
vv <- numeric(nrow(sr))
for (i in seq_len(nrow(coefs))) { cc <- paste0(coefs$stat_name[i],"_rating")
  if (!cc %in% names(sr)) next; sdv <- coefs$sd[i]; if (is.na(sdv)||sdv==0) sdv <- 1
  x <- sr[[cc]]; x[is.na(x)] <- 0; vv <- vv + coefs$beta[i]*(x/sdv) }
sr[, psr_raw := vv]
lp <- unique(pg[, .(player_id, season, round, lineup_position)], by=c("player_id","season","round"))
sr <- merge(sr, lp, by=c("player_id","season","round"), all.x=TRUE)
sr[, lpg := unname(LINEUP_POSITION_GROUP_MAP[lineup_position])]
sr[, psr_v1 := psr_raw - wmean(psr_raw, wt_80s), by=pos_group]
sr[, psr_v2 := NA_real_]
sr[!is.na(lpg), psr_v2 := psr_raw - wmean(psr_raw, wt_80s), by=lpg]
sr[is.na(lpg),  psr_v2 := psr_raw - wmean(psr_raw, wt_80s), by=pos_group]
if (isTRUE(PSR_POSITION_STANDARDISE)) {
  pooled <- wsd(sr$psr_v2, sr$wt_80s)
  sr[!is.na(lpg), .gsd := wsd(psr_v2, wt_80s), by=lpg]
  sr[!is.na(lpg) & !is.na(.gsd) & .gsd > 1e-6, psr_v2 := psr_v2/.gsd*pooled]
  sr[, .gsd := NULL]
}

# ---- 2. aggregate to team-match, exactly as production does ----------------
cli::cli_h1("Aggregating to team ratings (production weighting)")
P <- merge(pg[, .(player_id, match_id, season, round, team_id, lineup_position,
                  epr_v1, epr_v2)],
           sr[, .(player_id, season, round, psr_v1, psr_v2)],
           by = c("player_id","season","round"), all.x = TRUE)
P <- P[!lineup_position %in% c("EMERG","SUB")]
P[, ltog := POSITION_AVG_TOG[lineup_position]]
P[is.na(ltog), ltog := POSITION_AVG_TOG_DEFAULT]
for (v in c("v1","v2")) {
  P[[paste0("epr_w_", v)]] <- P[[paste0("epr_", v)]] * P$ltog
  P[[paste0("psr_w_", v)]] <- fifelse(is.na(P[[paste0("psr_", v)]]),
                                      PSR_PRIOR_RATE, P[[paste0("psr_", v)]]) * P$ltog
}
TM <- P[, .(epr_v1 = sum(epr_w_v1, na.rm=TRUE), epr_v2 = sum(epr_w_v2, na.rm=TRUE),
            psr_v1 = sum(psr_w_v1, na.rm=TRUE), psr_v2 = sum(psr_w_v2, na.rm=TRUE)),
        by = .(match_id, team_id)]

# ---- 3. VALIDATE the v1 reconstruction before trusting anything ------------
cli::cli_h1("Validating the v1 reconstruction against the cached C6 features")
base <- as.data.table(readRDS(.rds("ws0_team_mdl_df.rds")))
chk <- merge(base[, .(match_id, team_id, epr_cached = epr.x, psr_cached = psr.x)],
             TM, by = c("match_id","team_id"))
r_epr <- cor(chk$epr_cached, chk$epr_v1, use="complete.obs")
r_psr <- cor(chk$psr_cached, chk$psr_v1, use="complete.obs")
cli::cli_inform("matched team-match rows: {nrow(chk)} of {nrow(base)}")
cli::cli_inform("cor(rebuilt v1 EPR, cached EPR) = {round(r_epr, 4)}")
cli::cli_inform("cor(rebuilt v1 PSR, cached PSR) = {round(r_psr, 4)}")
if (is.na(r_epr) || r_epr < 0.95) {
  cli::cli_abort(c("v1 EPR reconstruction does not reproduce the cached feature (r = {round(r_epr,4)}).",
    "x" = "Any v2 comparison from here would measure the rebuild, not the change."))
}
if (is.na(r_psr) || r_psr < 0.90) {
  cli::cli_abort("v1 PSR reconstruction does not reproduce the cached feature (r = {round(r_psr,4)}).")
}
cli::cli_alert_success("Reconstruction validated -- proceeding")

# ---- 4. build the two arms -------------------------------------------------
mk_arm <- function(vint) {
  d <- copy(base)
  own <- TM[, .(match_id, team_id,
                e = get(paste0("epr_", vint)), p = get(paste0("psr_", vint)))]
  d[, opp_team_id := {
    m <- match(match_id, match_id)   # placeholder, replaced below
    NA_character_ }]
  # opponent id: the other team_id sharing this match_id
  d[, opp_team_id := {
    tt <- team_id; ifelse(seq_len(.N) == 1L, tt[2], tt[1]) }, by = match_id]
  d <- merge(d, own, by = c("match_id","team_id"), all.x = TRUE)
  setnames(d, c("e","p"), c("e_own","p_own"))
  d <- merge(d, own, by.x = c("match_id","opp_team_id"), by.y = c("match_id","team_id"),
             all.x = TRUE)
  setnames(d, c("e","p"), c("e_opp","p_opp"))
  ok <- !is.na(d$e_own) & !is.na(d$e_opp)
  d[ok, `:=`(epr.x = e_own, epr.y = e_opp, psr.x = p_own, psr.y = p_opp,
             epr_diff = e_own - e_opp, psr_diff = p_own - p_opp,
             torp.x = TORP_EPR_WEIGHT*e_own + (1-TORP_EPR_WEIGHT)*p_own,
             torp.y = TORP_EPR_WEIGHT*e_opp + (1-TORP_EPR_WEIGHT)*p_opp,
             torp_diff = TORP_EPR_WEIGHT*(e_own-e_opp) + (1-TORP_EPR_WEIGHT)*(p_own-p_opp))]
  d[, c("e_own","p_own","e_opp","p_opp","opp_team_id") := NULL]
  cli::cli_inform("{vint} arm: {sum(ok)} of {nrow(d)} rows re-rated")
  as.data.frame(d)
}
arm_v1 <- mk_arm("v1")
arm_v2 <- mk_arm("v2")

# ---- 5. rolling OOS, both arms --------------------------------------------
run_arm <- function(df, label) {
  cli::cli_h1(label)
  t0 <- Sys.time()
  roll <- run_rolling_eval(df, test_seasons = TEST_SEASONS, verbose = FALSE)
  cli::cli_inform("elapsed {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  preds <- roll$input_blend
  m <- .compute_metrics(preds)
  print(m)
  list(preds = preds, metrics = m)
}
res_v1 <- run_arm(arm_v1, "ARM v1 (rebuilt, sanity baseline)")
res_v2 <- run_arm(arm_v2, "ARM v2 (adopted ratings)")

cli::cli_h1("PAIRED BOOTSTRAP: v2 vs v1")
bd <- boot_mae_diff(res_v2$preds, res_v1$preds, B = 2000)
print(bd)
saveRDS(list(v1 = res_v1, v2 = res_v2, boot = bd,
             recon = list(r_epr = r_epr, r_psr = r_psr)),
        .rds("ws_v2_ratings_gate.rds"))
cli::cli_alert_success("Saved ws_v2_ratings_gate.rds")
