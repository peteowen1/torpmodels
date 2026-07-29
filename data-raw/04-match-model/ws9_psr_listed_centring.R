# Does centring PSR on the LISTED position change match MAE?
# ==========================================================
# PSR centred on `pos_group` -- a PBP-derived PLAYSTYLE label -- while EPV, EPR
# and PSV all centre on the LISTED taxonomy, and torp_ratings then presents PSR
# under its own listed `position_group`. The two labels disagree for 13.2% of
# player-rounds (20.3% in 2026). Centring on one key and reading on the other
# put ~0.30 of positional level back into TORP, undoing half of what the
# 2026-07-29 program removed at EPV/EPR.
#
# Not a mapping bug: .collapse_listed_position() and .map_position_group() were
# verified to produce an IDENTICAL partition of the same seven raw labels. The
# disagreement is entirely in the source data -- two different facts about a
# player, not two vintages of one. 2021, where the labels agree 100%, shows
# exactly 0.000 spread: the control that identified the mechanism.
#
# The OLD arm comes from the production code path via centre_on_listed = FALSE,
# NOT a hand-rolled reconstruction -- same contract as ws7's centre_by_round.
#
# As in ws7: this is the PSR arm. `psv` appears nowhere in the match pipeline,
# so the PSV level fix is not scoreable here.
#
# Absolute levels are harness-relative (fidelity is PAIRED-safe, not STRICT).
# The delta is the result.

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
CACHE      <- file.path(EXP, "results", "ws9_arms_cache.rds")
EVAL_CACHE <- file.path(EXP, "results", "ws9_eval_cache.rds")
OUT        <- file.path(EXP, "results", "ws9_psr_listed_centring.rds")
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
  # Load listed positions ONCE and hand the same frame to both arms, so the new
  # arm cannot differ from the old by which roster vintage it happened to fetch.
  listed <- torp:::.load_listed_positions(RATING_SEASONS)
  if (is.null(listed) || nrow(listed) == 0) {
    cli::cli_abort("No listed positions loaded -- the new arm would be untestable.")
  }
  cli::cli_inform("listed positions: {nrow(listed)} players")

  psr_old <- .compute_psr_from_stat_ratings(sk, centre_on_listed = FALSE)
  psr_new <- .compute_psr_from_stat_ratings(sk, centre_on_listed = TRUE, listed_pos = listed)

  # The flag must actually change PSR, or the arms are a no-op dressed up as a
  # comparison and the delta reads as "no effect" instead of "not tested".
  po <- as.data.table(psr_old)[, .(player_id, season, round, p_old = psr)]
  pn <- as.data.table(psr_new)[, .(player_id, season, round, p_new = psr)]
  j <- merge(po, pn, by = c("player_id", "season", "round"))
  if (isTRUE(all.equal(j$p_old, j$p_new))) {
    cli::cli_abort("centre_on_listed did nothing -- refusing to report a delta.")
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
  arms_tm <- list(playstyle = mk_tm(psr_old), listed = mk_tm(psr_new))

  # Placeholder finals fixtures (2026 r25-26, teams TBD) carry NA features, and
  # model.matrix() drops NA rows silently -> predictions shorter than the frame.
  arms_tm <- lapply(arms_tm, function(tm) {
    d <- as.data.table(tm)
    opp <- intersect(c("epr.y", "psr.y", "torp.y"), names(d))
    keep <- if (length(opp)) Reduce(`&`, lapply(opp, function(cc) !is.na(d[[cc]]))) else rep(TRUE, nrow(d))
    if (sum(!keep)) cli::cli_alert_warning("Dropping {sum(!keep)} row{?s} with no opponent (finals placeholders).")
    as.data.frame(d[keep])   # data.frame: the harness subsets with df[, cols, drop = FALSE]
  })
  if (!identical(sort(as.character(arms_tm$playstyle$match_id)),
                 sort(as.character(arms_tm$listed$match_id)))) {
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
a <- as.data.table(preds$playstyle); b <- as.data.table(preds$listed)
key <- intersect(c("season", "round", "home_team"), names(a))
j <- merge(a[, c(key, "pred_margin", "margin"), with = FALSE],
           b[, c(key, "pred_margin", "margin"), with = FALSE],
           by = key, suffixes = c(".old", ".new"))
d <- abs(j$pred_margin.new - j$margin.new) - abs(j$pred_margin.old - j$margin.old)
set.seed(20260729)
ci <- quantile(replicate(10000, mean(sample(d, length(d), replace = TRUE))), c(0.025, 0.975))
cli::cli_inform("n matches: {length(d)}")
cli::cli_inform("MAE playstyle: {round(mean(abs(j$pred_margin.old - j$margin.old)), 4)} | listed: {round(mean(abs(j$pred_margin.new - j$margin.new)), 4)}")
cli::cli_inform("dMAE (listed - playstyle): {round(mean(d), 4)}  95% CI [{round(ci[1],4)}, {round(ci[2],4)}]")
cli::cli_inform("SHIP gate (CI excludes zero AND dMAE < 0): {ci[2] < 0}")
cli::cli_inform("NOTE: a null here still ships on correctness, exactly as #134 did -- the")
cli::cli_inform("      change makes TORP's two halves agree on what a position IS.")

saveRDS(list(scorecard = sc, delta = list(d = d, ci = ci)), OUT)
cli::cli_alert_success("saved {OUT}")
cli::cli_alert_info("total {round(difftime(Sys.time(), t_start, units = 'mins'), 1)} min")
