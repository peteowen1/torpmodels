# Can the centred ratings BUY something, rather than just cost 0.12 MAE?
# =====================================================================
# WS4 showed position-centring is free-ish but nominally -0.12 MAE. The likely
# cause: the team-level shift is sum(count_p * shift_p), and teams field 2-9
# listed midfielders, so the shift VARIES between teams (sd 1.57) while carrying
# no information about strength. That is added noise on epr_diff.
#
# But centring also unlocks something. The model currently receives CHANNEL
# splits (epr_recv_diff, epr_disp_diff, ...) and no POSITION splits. We know
# positions convert to points at different rates (med_def 0.46, midfield 1.12),
# so position-split features should carry real signal -- and the reason they
# were not usable before is precisely the roster-shape artefact that centring
# removes ("uncentred bucket sums encode roster shape; teams differ in bucket
# counts in 40-76% of matches").
#
# So: centring alone is a cost, centring PLUS position splits may be a gain.
# The raw_pos arm is the control that tests whether centring is what makes the
# difference, rather than the splits alone.
#
# Arms:
#   current      - baseline
#   centred      - centred ratings, totals only            (WS4's result)
#   centred_pos  - centred ratings + 6 position-bucket diffs
#   raw_pos      - UNCENTRED ratings + the same 6 diffs    (control)
#
# All arms scored through run_rolling_eval_parallel() with identical settings,
# so absolute MAEs are not comparable with sequentially-scored numbers -- only
# arm-vs-arm. Re-confirm any winner sequentially before shipping.

suppressMessages({
  library(dplyr); library(data.table)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})
options(torp.local_data_dir = NA)

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RES <- file.path(EXP, "results")
source(file.path(EXP, "rolling_lib.R")); source(file.path(EXP, "signal_gate.R"))
source(file.path(EXP, "arm_lib.R"));     source(file.path(EXP, "scorecard_lib.R"))

TEST_SEASONS <- 2025:2026
N_WORKERS    <- 5L
GRP  <- c("key_def", "med_def", "midfield", "med_fwd", "key_fwd", "rucks")
PDIF <- paste0(GRP, "_pdiff")

#' Add opponent-relative position-bucket differentials
#'
#' team_mdl_df carries each team's own position sums but no opponent version, so
#' the differential has to be built by joining each row to its opposite number.
add_pos_diffs <- function(tm) {
  d <- as.data.table(copy(tm))
  d[, .tt := as.character(team_type)]
  opp <- d[, c("match_id", ".tt", GRP), with = FALSE]
  opp[, .tt := fifelse(.tt == "home", "away", "home")]   # flip so it lands on the opponent
  setnames(opp, GRP, paste0(GRP, "_opp"))
  d <- merge(d, opp, by = c("match_id", ".tt"), all.x = TRUE)
  for (g in GRP) d[[paste0(g, "_pdiff")]] <- d[[g]] - d[[paste0(g, "_opp")]]

  # A differential must be antisymmetric within a match, else the join is wrong.
  chk <- d[!is.na(get(PDIF[1])), .(s = sum(get(PDIF[1]))), by = match_id]
  stopifnot(max(abs(chk$s)) < 1e-8)

  # Upcoming fixtures have no published team list, so the position sums are NA
  # (72 rows of 2314, ALL of them incomplete matches -- verified). The existing
  # base features never hit this because the rating join prior-imputes upstream.
  #
  # Left as NA these rows are silently DROPPED by model.matrix(), which does not
  # error -- it returns fewer predictions than rows and the failure surfaces much
  # later as a row-count mismatch on assignment. Impute 0: with no lineups, the
  # honest positional differential is "no known advantage either way", which is
  # also what the rest of the pipeline does when lineups are missing.
  n_imp <- sum(is.na(d[[PDIF[1]]]))
  if (n_imp > 0) {
    bad <- sum(is.na(d[[PDIF[1]]]) & !is.na(d$score_diff))
    if (bad > 0) cli::cli_abort("{bad} COMPLETED match{?es} have NA position sums -- not a lineup gap, investigate")
    cli::cli_inform("position diffs: imputed 0 for {n_imp} row{?s} with no published team list (all incomplete matches)")
    for (cc in PDIF) d[is.na(get(cc)), (cc) := 0]
  }
  stopifnot(!anyNA(d[, ..PDIF]))
  d[, .tt := NULL]
  as.data.frame(d)
}

cached <- readRDS(file.path(RES, "ws4_arms_team_mdl.rds"))
arms <- list(
  current     = cached$current,
  centred     = cached$centred,
  centred_pos = add_pos_diffs(cached$centred),
  raw_pos     = add_pos_diffs(cached$current)
)
for (nm in names(arms)) cli::cli_inform("{nm}: {nrow(arms[[nm]])} rows, {ncol(arms[[nm]])} cols")

feats <- list(current = "xelo_diff", centred = "xelo_diff",
              centred_pos = c("xelo_diff", PDIF), raw_pos = c("xelo_diff", PDIF))

t0 <- Sys.time(); res <- list()
for (nm in names(arms)) {
  cli::cli_h2("scoring: {nm} ({length(feats[[nm]])} extra feature{?s})")
  res[[nm]] <- run_rolling_eval_parallel(
    arms[[nm]], test_seasons = TEST_SEASONS,
    extra_feature_cols = feats[[nm]], cv_extra_feature_cols = feats[[nm]],
    n_workers = N_WORKERS, verbose = FALSE)
}
cli::cli_alert_success("4 arms in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

sc <- scorecard(lapply(res, function(r) r$input_blend_preds),
                squiggle_sources = c("Aggregate", "Wheelo Ratings"))
print_scorecard(sc, "WS5: centred position features, 2025-26 pooled")
sc26 <- scorecard(lapply(res, function(r) as.data.table(r$input_blend_preds)[season == 2026]),
                  squiggle_sources = c("Aggregate", "Wheelo Ratings"))
print_scorecard(sc26, "WS5: 2026 only")

cat("\n=== bootstrap vs current (pooled, raw predictions) ===\n")
for (nm in setdiff(names(res), "current")) {
  b <- boot_mae_diff(res[[nm]]$input_blend_preds, res$current$input_blend_preds)
  cat(sprintf("%-12s dMAE %+.3f  95%% CI [%+.3f, %+.3f] | dBrier %+.5f\n",
              nm, b$mae_diff, b$mae_ci[1], b$mae_ci[2], b$brier_diff))
}
cat("(negative dMAE = better than current)\n")

saveRDS(list(res = res, scorecard = sc, scorecard_2026 = sc26, parallel = TRUE),
        file.path(RES, "ws5_centred_position.rds"))
cli::cli_alert_success("Saved ws5_centred_position.rds")
