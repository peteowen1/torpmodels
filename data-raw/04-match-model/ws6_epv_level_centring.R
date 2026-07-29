# Does fixing the positional level at EPV change match MAE?
# =========================================================
# The first arm scored through build_ratings_history() -- the offline ratings
# builder added 2026-07-29 so a RATING-DEFINITION change can be measured
# without publishing one.
#
# Arms differ in exactly one toggle:
#   base : epv_level_centre = FALSE  (what production publishes today)
#   epv  : epv_level_centre = TRUE   (the proposed fix)
# Both keep epr_position_centre = TRUE, which is the shipped backstop.
#
# Everything else is shared BY CONSTRUCTION, not by convention: one player-game
# load, one opponent adjustment, one PSR frame, one set of match inputs. The
# opponent adjustment is done once up front and both arms are handed the result
# with opponent_adjust = FALSE, so the only thing that can differ downstream is
# the centring itself.
#
# Read check_ratings_build_fidelity.R before trusting an ABSOLUTE number from
# this: the builder is PAIRED-safe (epr cor 0.9997, no per-round offset) but
# does NOT reproduce production player-for-player. The delta is the result; the
# levels are not comparable to a live leaderboard.

suppressMessages({
  library(data.table); library(dplyr)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
  devtools::load_all("C:/dev/torpverse/torpmodels", quiet = TRUE)
})
options(torp.local_data_dir = NA)   # release, never the stale local mirror

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
source(file.path(EXP, "rolling_lib.R"))
source(file.path(EXP, "arm_lib.R"))
source(file.path(EXP, "scorecard_lib.R"))

RATING_SEASONS <- 2021:2026          # decay needs the history
TEST_SEASONS   <- 2025:2026          # the pooled window; 2026-only over-promised 3x
OUT <- file.path(EXP, "results", "ws6_epv_level_centring.rds")
t_start <- Sys.time()

# ---- 0. is the builder even deterministic? ----------------------------------
# Cheapest possible explanation for the unexplained STRICT fidelity gap. If two
# identical calls disagree, the gap is ours and nothing below is worth running.
# If they agree bit-for-bit, the difference lives in production's inputs and the
# paired comparison is sound.
cli::cli_h1("0. determinism of build_ratings_history()")
d1 <- as.data.table(build_ratings_history(2026L, epv_level_centre = FALSE))
d2 <- as.data.table(build_ratings_history(2026L, epv_level_centre = FALSE))
setorder(d1, player_id, round); setorder(d2, player_id, round)
det_ok <- identical(nrow(d1), nrow(d2)) &&
  isTRUE(all.equal(d1$epr, d2$epr)) && isTRUE(all.equal(d1$epr_recv, d2$epr_recv))
cli::cli_alert_info("deterministic: {det_ok}")
if (!det_ok) {
  cli::cli_alert_danger(
    "Builder is NOT deterministic -- the STRICT fidelity gap is ours, not production's input drift.")
  cli::cli_alert_info("max |epr diff| between identical calls: {max(abs(d1$epr - d2$epr), na.rm = TRUE)}")
} else {
  cli::cli_alert_success("Identical calls agree exactly; the STRICT gap is input drift, not our arithmetic.")
}
rm(d1, d2)

# Stages 1-3 take ~15 min and do not change between scoring attempts, so cache
# them. The eval stage is what gets iterated on; rebuilding ratings each time
# just to fix a harness bug wastes a quarter of an hour per try.
CACHE <- file.path(EXP, "results", "ws6_arms_cache.rds")

if (file.exists(CACHE)) {
  cli::cli_alert_info("Reusing cached arms from {CACHE}")
  cached <- readRDS(CACHE)
  arms_tm <- cached$arms_tm
} else {

# ---- 1. shared inputs, loaded once ------------------------------------------
cli::cli_h1("1. shared inputs")
pgd <- as.data.table(load_player_game_data(TRUE))
pgd <- adjust_epv_for_opponents(pgd)      # ONCE, so both arms share it exactly
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                   error = function(e) { cli::cli_warn("PSR: {conditionMessage(e)}"); NULL })
stat_ratings <- tryCatch(get_player_stat_ratings(current = FALSE), error = function(e) NULL)
fixtures <- load_fixtures(TRUE)
src <- load_match_inputs()

# ---- 2. the two ratings histories -------------------------------------------
cli::cli_h1("2. ratings histories")
mk <- function(centre) build_ratings_history(
  seasons = RATING_SEASONS, pgd = pgd, stat_ratings = stat_ratings,
  fixtures = fixtures, psr_df = psr_df,
  epv_level_centre = centre, epr_position_centre = TRUE,
  opponent_adjust = FALSE)          # already applied above, to both alike

r_base <- as.data.frame(mk(FALSE))
r_epv  <- as.data.frame(mk(TRUE))

# The arms must actually differ, and differ in the way claimed. A silent no-op
# here would produce a clean 0.000 delta that reads as "no effect".
cli::cli_h2("sanity: the arms differ, and in the right direction")
chk <- merge(as.data.table(r_base)[, .(player_id, season, round, e_base = epr)],
             as.data.table(r_epv)[,  .(player_id, season, round, e_epv  = epr)],
             by = c("player_id", "season", "round"))
cli::cli_inform("rows compared: {nrow(chk)} | identical: {isTRUE(all.equal(chk$e_base, chk$e_epv))}")
cli::cli_inform("mean |diff|: {round(mean(abs(chk$e_base - chk$e_epv), na.rm = TRUE), 4)}")
if (isTRUE(all.equal(chk$e_base, chk$e_epv))) {
  cli::cli_abort("The two arms are identical -- the toggle did nothing. Refusing to report a delta.")
}

# ---- 3. team model frames ----------------------------------------------------
cli::cli_h1("3. team_mdl_df per arm")
arms_tm <- list(base = build_team_mdl_with(src, r_base),
                epv  = build_team_mdl_with(src, r_epv))

# Drop rows with no opponent after the self-join. These are placeholder FINALS
# fixtures (2026 rounds 25-26, teams still TBD) that appeared when the AFL
# published the finals schedule. They carry NA features, and predict_all() ->
# model.matrix() DROPS NA rows silently, so the prediction vector comes back
# shorter than the frame and the assignment fails with a recycling error that
# names neither NAs nor finals. Left in, they would either crash the run (as
# they did) or, worse, silently misalign predictions with rows.
#
# Filtering here rather than inside the arms keeps it paired: the same rows go
# from both, and the assertion below makes that a fact rather than an intention.
arms_tm <- lapply(arms_tm, function(tm) {
  d <- as.data.table(tm)
  opp <- intersect(c("epr.y", "psr.y", "torp.y"), names(d))
  if (length(opp) == 0) return(d)
  keep <- Reduce(`&`, lapply(opp, function(cc) !is.na(d[[cc]])))
  if (sum(!keep) > 0) {
    cli::cli_alert_warning(
      "Dropping {sum(!keep)} row{?s} with no opponent data (placeholder finals fixtures).")
  }
  # as.data.frame, NOT the data.table: the rolling harness subsets with
  # df[, feature_cols, drop = FALSE], which is data.frame semantics. On a
  # data.table that is NSE and errors with "j is a single symbol but column
  # name 'feature_cols' is not found" -- a message that points at the harness,
  # not at the class change made here. build_team_mdl_with() returns a
  # data.frame, so this restores what the caller had.
  as.data.frame(d[keep])
})
ids <- lapply(arms_tm, function(d) sort(as.character(d$match_id)))
if (!identical(ids$base, ids$epv)) {
  cli::cli_abort("Arms cover different matches after filtering -- the comparison would not be paired.")
}
for (nm in names(arms_tm)) cli::cli_inform("{nm}: {nrow(arms_tm[[nm]])} team-rows")

saveRDS(list(arms_tm = arms_tm), CACHE)
cli::cli_alert_success("cached arms to {CACHE}")
}   # end cache-miss branch

# Belt and braces, and it also repairs a cache written before the class fix:
# whatever route arms_tm arrived by, the harness needs data.frames.
arms_tm <- lapply(arms_tm, as.data.frame)

# ---- 4. rolling OOS ----------------------------------------------------------
# parallel = TRUE is safe here and only here: both arms run the same path with
# the same thread count, so xgboost's hist nondeterminism shifts them together
# and largely cancels in the delta. Never compare these to a sequentially-scored
# champion.
cli::cli_h1("4. rolling eval")
EVAL_CACHE <- file.path(EXP, "results", "ws6_eval_cache.rds")
if (file.exists(EVAL_CACHE)) {
  cli::cli_alert_info("Reusing cached eval from {EVAL_CACHE}")
  res <- readRDS(EVAL_CACHE)
} else {
  res <- score_arms(arms_tm, test_seasons = TEST_SEASONS, parallel = TRUE)
  # Save BEFORE the scorecard. Run 3 completed both evals and then died in
  # scorecard formatting, throwing away four minutes of correct work for a
  # column-name mismatch.
  saveRDS(res, EVAL_CACHE)
  cli::cli_alert_success("cached eval to {EVAL_CACHE}")
}

# run_rolling_eval*() returns list(gam_preds, xgb_preds, blend_preds,
# input_blend_preds, ...). Production serves the 50/50 INPUT BLEND, so that is
# the frame to score -- picking blend_preds instead would measure a model we do
# not run.
preds <- lapply(res, function(r) {
  if (is.null(r$input_blend_preds)) {
    cli::cli_abort("Arm has no {.field input_blend_preds}; got {.val {names(r)}}")
  }
  r$input_blend_preds
})

# ---- 5. scorecard + paired bootstrap ----------------------------------------
cli::cli_h1("5. scorecard")
sc <- scorecard(preds, squiggle_sources = "Aggregate")
print_scorecard(sc)

cli::cli_h1("6. paired delta")
a <- as.data.table(preds$base); b <- as.data.table(preds$epv)
key <- intersect(c("season", "round", "match_id", "home_team", "away_team"), names(a))
j <- merge(a[, c(key, "pred_margin", "margin"), with = FALSE],
           b[, c(key, "pred_margin", "margin"), with = FALSE],
           by = key, suffixes = c(".base", ".epv"))
j[, `:=`(ae_base = abs(pred_margin.base - margin.base),
         ae_epv  = abs(pred_margin.epv  - margin.epv))]
d <- j$ae_epv - j$ae_base
set.seed(20260729)
bs <- replicate(10000, mean(sample(d, length(d), replace = TRUE)))
ci <- quantile(bs, c(0.025, 0.975))
cli::cli_inform("n matches: {length(d)}")
cli::cli_inform("MAE base: {round(mean(j$ae_base), 4)} | epv: {round(mean(j$ae_epv), 4)}")
cli::cli_inform("dMAE (epv - base): {round(mean(d), 4)}  95% CI [{round(ci[1],4)}, {round(ci[2],4)}]")
cli::cli_inform("SHIP gate (CI excludes zero AND dMAE < 0): {ci[2] < 0}")

saveRDS(list(scorecard = sc, res = res, delta = list(d = d, ci = ci, n = length(d)),
             deterministic = det_ok, rating_seasons = RATING_SEASONS,
             test_seasons = TEST_SEASONS), OUT)
cli::cli_alert_success("saved {OUT}")
cli::cli_alert_info("total {round(difftime(Sys.time(), t_start, units = 'mins'), 1)} min")
