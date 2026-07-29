# Does per-round PSR centring change match MAE?
# =============================================
# PSR's position centring was grouped by position ALONE, pooled over all
# history. Every position therefore averaged zero across the dataset while any
# individual round stayed skewed -- on the served round, rucks +0.451 vs key
# forwards -0.030, a 0.481 spread TORP inherited half of. It is also a backtest
# leak: 2021 round-1 ratings centred with 2026 games.
#
# Arms differ in exactly one flag, and the OLD arm comes from the production
# code path via centre_by_round = FALSE rather than a hand-rolled reconstruction
# of the old behaviour. Verified: FALSE reproduces the 0.4810 spread measured
# before the change, TRUE gives 0.0000.
#
# NOTE this is the PSR arm, not the PSV one. PSR is fed by stat ratings, NOT by
# PSV -- they are parallel products of the same coefficient vector, not a
# per-game -> rating chain like EPV -> EPR. `psv` appears nowhere in the match
# pipeline, so the PSV level fix cannot move match MAE and is not scoreable here.
#
# Absolute levels are harness-relative (check_ratings_build_fidelity.R is
# PAIRED-safe, not STRICT-clean). The delta is the result.

suppressMessages({
  library(data.table); library(dplyr)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
  devtools::load_all("C:/dev/torpverse/torpmodels", quiet = TRUE)
})
options(torp.local_data_dir = NA)

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
source(file.path(EXP, "rolling_lib.R"))
source(file.path(EXP, "arm_lib.R"))
source(file.path(EXP, "scorecard_lib.R"))

RATING_SEASONS <- 2021:2026
TEST_SEASONS   <- 2025:2026
CACHE <- file.path(EXP, "results", "ws7_arms_cache.rds")
EVAL_CACHE <- file.path(EXP, "results", "ws7_eval_cache.rds")
OUT <- file.path(EXP, "results", "ws7_psr_round_centring.rds")
t_start <- Sys.time()

if (file.exists(CACHE)) {
  cli::cli_alert_info("Reusing cached arms")
  arms_tm <- readRDS(CACHE)$arms_tm
} else {
  cli::cli_h1("1. shared inputs")
  pgd <- as.data.table(load_player_game_data(TRUE))
  pgd <- adjust_epv_for_opponents(pgd)     # ONCE, shared by both arms exactly
  sk <- load_player_stat_ratings(TRUE)
  stat_ratings <- tryCatch(get_player_stat_ratings(current = FALSE), error = function(e) NULL)
  fixtures <- load_fixtures(TRUE)
  src <- load_match_inputs()

  cli::cli_h1("2. PSR frames + ratings histories")
  mk_psr <- function(by_round) .compute_psr_from_stat_ratings(sk, centre_by_round = by_round)
  psr_old <- mk_psr(FALSE); psr_new <- mk_psr(TRUE)

  # The flag must actually change PSR, or the arms are a no-op dressed up as a
  # comparison and the delta reads as "no effect" instead of "not tested".
  po <- as.data.table(psr_old)[, .(player_id, season, round, p_old = psr)]
  pn <- as.data.table(psr_new)[, .(player_id, season, round, p_new = psr)]
  j <- merge(po, pn, by = c("player_id", "season", "round"))
  if (isTRUE(all.equal(j$p_old, j$p_new))) {
    cli::cli_abort("centre_by_round did nothing -- refusing to report a delta.")
  }
  cli::cli_inform("PSR mean |diff| between arms: {round(mean(abs(j$p_old - j$p_new), na.rm = TRUE), 4)}")

  mk_ratings <- function(psr_df) build_ratings_history(
    seasons = RATING_SEASONS, pgd = pgd, stat_ratings = stat_ratings,
    fixtures = fixtures, psr_df = psr_df, opponent_adjust = FALSE)

  cli::cli_h1("3. team_mdl_df per arm")
  mk_tm <- function(psr_df) {
    s <- src; s$psr_df <- psr_df          # the arm's PSR must reach team ratings too
    build_team_mdl_with(s, as.data.frame(mk_ratings(psr_df)))
  }
  arms_tm <- list(pooled = mk_tm(psr_old), per_round = mk_tm(psr_new))

  # Placeholder finals fixtures (2026 r25-26, teams TBD) carry NA features, and
  # model.matrix() drops NA rows silently -> predictions shorter than the frame.
  arms_tm <- lapply(arms_tm, function(tm) {
    d <- as.data.table(tm)
    opp <- intersect(c("epr.y", "psr.y", "torp.y"), names(d))
    keep <- if (length(opp)) Reduce(`&`, lapply(opp, function(cc) !is.na(d[[cc]]))) else rep(TRUE, nrow(d))
    if (sum(!keep)) cli::cli_alert_warning("Dropping {sum(!keep)} row{?s} with no opponent (finals placeholders).")
    as.data.frame(d[keep])   # data.frame: the harness subsets with df[, cols, drop = FALSE]
  })
  if (!identical(sort(as.character(arms_tm$pooled$match_id)),
                 sort(as.character(arms_tm$per_round$match_id)))) {
    cli::cli_abort("Arms cover different matches -- not paired.")
  }
  saveRDS(list(arms_tm = arms_tm), CACHE)
}

cli::cli_h1("4. rolling eval")
if (file.exists(EVAL_CACHE)) {
  res <- readRDS(EVAL_CACHE)
} else {
  res <- score_arms(arms_tm, test_seasons = TEST_SEASONS, parallel = TRUE)
  saveRDS(res, EVAL_CACHE)   # save BEFORE the scorecard; run 3 of ws6 lost 4 min this way
}

# Production serves the 50/50 INPUT BLEND.
preds <- lapply(res, function(r) {
  if (is.null(r$input_blend_preds)) cli::cli_abort("No input_blend_preds; got {.val {names(r)}}")
  r$input_blend_preds
})

cli::cli_h1("5. scorecard")
sc <- scorecard(preds, squiggle_sources = "Aggregate")
print_scorecard(sc)

cli::cli_h1("6. paired delta")
a <- as.data.table(preds$pooled); b <- as.data.table(preds$per_round)
key <- intersect(c("season", "round", "home_team"), names(a))
j <- merge(a[, c(key, "pred_margin", "margin"), with = FALSE],
           b[, c(key, "pred_margin", "margin"), with = FALSE],
           by = key, suffixes = c(".old", ".new"))
d <- abs(j$pred_margin.new - j$margin.new) - abs(j$pred_margin.old - j$margin.old)
set.seed(20260729)
ci <- quantile(replicate(10000, mean(sample(d, length(d), replace = TRUE))), c(0.025, 0.975))
cli::cli_inform("n matches: {length(d)}")
cli::cli_inform("MAE pooled: {round(mean(abs(j$pred_margin.old - j$margin.old)), 4)} | per_round: {round(mean(abs(j$pred_margin.new - j$margin.new)), 4)}")
cli::cli_inform("dMAE (per_round - pooled): {round(mean(d), 4)}  95% CI [{round(ci[1],4)}, {round(ci[2],4)}]")
cli::cli_inform("SHIP gate (CI excludes zero AND dMAE < 0): {ci[2] < 0}")

saveRDS(list(scorecard = sc, delta = list(d = d, ci = ci)), OUT)
cli::cli_alert_success("saved {OUT}")
cli::cli_alert_info("total {round(difftime(Sys.time(), t_start, units = 'mins'), 1)} min")
