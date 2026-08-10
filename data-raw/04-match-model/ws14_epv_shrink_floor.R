# EPV shrinkage, FLOOR rule: does confining it to thin cells cost anything?
# =========================================================================
# Third attempt at the same idea, and the first two failures are why this one is
# shaped the way it is.
#
#   ws11  shrank the EPR layer toward zero. The EPR correction is ~0.002 (a
#         backstop), so it did nothing: TORP moved 0.0044, top 20 identical.
#   ws12  shrank the EPV layer toward the bucket's earlier mean -- the right layer
#         and the right target -- but with a SMOOTH prior, which at prior 2 moved
#         55,758 of 56,162 rows. dMAE +0.232 / -0.051 / +0.090 at priors 1/2/5:
#         non-monotonic, inside the ~0.157 noise floor, every arm costing bits.
#         It diluted every normal cell by ~4.7% to reach the 22 that needed it,
#         so the gate was measuring the dilution, not the fix.
#
# The FLOOR rule keeps ws12's target and drops its collateral: lambda =
# min(1, wt / floor), so cells at or above the floor are BIT-IDENTICAL to
# production and only thinner ones ramp toward the bucket's earlier mean.
# Verified on real data at floor 8: 306 of 56,162 rows move, 0.544% of total TOG
# weight, rounds 0 and 24-28 only, 2026 untouched entirely.
#
# What it is for: a Grand Final key forward currently has up to 7 points
# subtracted from him because the other three key forwards on the ground that day
# happened to average high (2025 r28 key_fwd, cell weight 3.19, correction -7.05).
# Those are the most-read ratings of the year.
#
# Sweeping the floor at 5 / 8 / 15 -- 8 is the measured 5th percentile of cell
# weight, so 5 and 15 bracket it. See the verdict block for how this gate is read;
# it is NOT "does MAE improve".

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
# 1 barely touches a normal cell (lambda 0.976 at the median weight) and still
# halves the thinnest; 5 is what the EPR version used. Three points is enough to
# see whether the curve is smooth -- ws11's was a scatter, which is what told us
# its deltas were noise.
FLOORS <- c(5, 8, 15)
CACHE      <- file.path(EXP, "results", "ws14_arms_cache.rds")
EVAL_CACHE <- file.path(EXP, "results", "ws14_eval_cache.rds")
OUT        <- file.path(EXP, "results", "ws14_epv_shrink_floor.rds")
t_start <- Sys.time()

set_flags <- function(shrink, floor) {
  assignInNamespace("EPV_POSITION_SHRINK", shrink, ns = "torp")
  assignInNamespace("EPV_POSITION_SHRINK_RULE", "floor", ns = "torp")
  assignInNamespace("EPV_POSITION_SHRINK_FLOOR", floor, ns = "torp")
}

# REACHABILITY. ws10 could not score the ROLE flags here because the role stage
# runs during pgd CONSTRUCTION, upstream of this harness's starting point. This
# flag is different: it lives inside centre_epv_by_position(), which
# build_ratings_history() calls, so it IS reachable -- the same place
# EPV_LEVEL_CENTRE was scored from. The identical-arms guard below is what
# actually proves it rather than this comment.
if (file.exists(CACHE)) {
  cli::cli_alert_info("Reusing cached arms")
  arms_tm <- readRDS(CACHE)$arms_tm
} else {
  cli::cli_h1("1. shared inputs")
  pgd_raw <- as.data.table(load_player_game_data(TRUE))
  sk <- load_player_stat_ratings(TRUE)
  stat_ratings <- tryCatch(get_player_stat_ratings(current = FALSE), error = function(e) NULL)
  fixtures <- load_fixtures(TRUE)
  src <- load_match_inputs()
  psr_df <- .compute_psr_from_stat_ratings(sk)

  # Arm-invariant, so done once and shared. The centring flags act after it.
  pgd_oadj <- adjust_epv_for_opponents(copy(pgd_raw))

  cli::cli_h1("2. ratings history per arm")
  mk_tm <- function(shrink, floor) {
    set_flags(shrink, floor)
    r <- build_ratings_history(
      seasons = RATING_SEASONS, pgd = copy(pgd_oadj), stat_ratings = stat_ratings,
      fixtures = fixtures, psr_df = psr_df, opponent_adjust = FALSE)
    build_team_mdl_with(src, as.data.frame(r))
  }

  arms_tm <- list(production = { cli::cli_alert_info("production"); mk_tm(FALSE, 8) })
  for (p in FLOORS) {
    nm <- paste0("floor_", p)
    cli::cli_alert_info(nm)
    arms_tm[[nm]] <- mk_tm(TRUE, p)
  }
  set_flags(FALSE, 8)   # never leave the namespace mutated

  # A flag that silently does nothing reads as "no effect" instead of "not
  # tested" -- which is exactly how ws10 nearly reported two unreachable arms.
  for (nm in setdiff(names(arms_tm), "production")) {
    a <- arms_tm$production$epr.x; b <- arms_tm[[nm]]$epr.x
    if (isTRUE(all.equal(a, b))) {
      cli::cli_abort("Arm {.val {nm}} is identical to production -- refusing to report a delta.")
    }
    cli::cli_inform("{nm}: mean |d epr| vs production = {round(mean(abs(a - b), na.rm = TRUE), 4)}")
  }
  if (length(FLOORS) > 1) {
    a <- arms_tm[[paste0("floor_", FLOORS[1])]]$epr.x
    b <- arms_tm[[paste0("floor_", FLOORS[length(FLOORS)])]]$epr.x
    if (isTRUE(all.equal(a, b))) cli::cli_abort("Smallest and largest floor give identical ratings.")
  }

  # Placeholder finals fixtures carry NA features; model.matrix() drops NA rows
  # silently, so predictions come back shorter than the frame.
  arms_tm <- lapply(arms_tm, function(tm) {
    d <- as.data.table(tm)
    opp <- intersect(c("epr.y", "psr.y", "torp.y"), names(d))
    keep <- if (length(opp)) Reduce(`&`, lapply(opp, function(cc) !is.na(d[[cc]]))) else rep(TRUE, nrow(d))
    if (sum(!keep)) cli::cli_alert_warning("Dropping {sum(!keep)} row{?s} with no opponent.")
    as.data.frame(d[keep])
  })
  ids <- lapply(arms_tm, function(x) sort(as.character(x$match_id)))
  if (length(unique(ids)) != 1L) cli::cli_abort("Arms cover different matches -- not paired.")
  saveRDS(list(arms_tm = arms_tm), CACHE)
}

cli::cli_h1("3. rolling eval")
if (file.exists(EVAL_CACHE)) {
  res <- readRDS(EVAL_CACHE)
} else {
  res <- score_arms(arms_tm, test_seasons = TEST_SEASONS, parallel = TRUE)
  saveRDS(res, EVAL_CACHE)
}
preds <- lapply(res, function(r) {
  if (is.null(r$input_blend_preds)) cli::cli_abort("No input_blend_preds")
  r$input_blend_preds
})

cli::cli_h1("4. scorecard")
sc <- scorecard(preds, squiggle_sources = "Aggregate")
print_scorecard(sc)

cli::cli_h1("5. paired deltas + the probability condition")
base <- as.data.table(preds$production)
key <- intersect(c("season", "round", "home_team"), names(base))
scdt <- as.data.table(sc)
bits_base <- scdt[source == "production"]$bits
set.seed(20260730)
tab <- rbindlist(lapply(setdiff(names(preds), "production"), function(nm) {
  b <- as.data.table(preds[[nm]])
  j <- merge(base[, c(key, "pred_margin", "margin"), with = FALSE],
             b[,    c(key, "pred_margin", "margin"), with = FALSE],
             by = key, suffixes = c(".base", ".arm"))
  d <- abs(j$pred_margin.arm - j$margin.arm) - abs(j$pred_margin.base - j$margin.base)
  ci <- quantile(replicate(10000, mean(sample(d, length(d), replace = TRUE))), c(0.025, 0.975))
  bits_arm <- scdt[source == nm]$bits
  data.table(arm = nm,
             mae = round(mean(abs(j$pred_margin.arm - j$margin.arm)), 4),
             dMAE = round(mean(d), 4), lo = round(ci[[1]], 4), hi = round(ci[[2]], 4),
             bits = round(bits_arm, 4), d_bits = round(bits_arm - bits_base, 4),
             no_trade = mean(d) < 0 && bits_arm >= bits_base)
}))
print(tab, row.names = FALSE)

# THE GATE IS READ DIFFERENTLY HERE, and that has to be stated before the numbers
# rather than after, or it is just moving the goalposts.
#
# ws12 asked "does shrinkage BUY MAE?" because the smooth prior rule changed
# almost every rating, so MAE was a fair test of it and it failed. The floor rule
# makes a different claim: it changes ONLY cells below the floor -- 306 of 56,162
# player-games, 0.54% of total TOG weight, all of it finals and round 0, with 2026
# untouched entirely. The claim is therefore "this fixes September without
# disturbing anything else", and the corresponding test is NOT that MAE improves.
# It is that MAE does not DEGRADE, with the correctness argument carrying the
# decision.
#
# So: a dMAE indistinguishable from zero is a PASS. A clear degradation is a fail.
# A clear improvement would be suspicious -- 0.5% of weight should not be able to
# buy real MAE, and if it appears to, suspect the arms are not paired.
NOISE <- 0.157   # measured XGBoost noise floor on this harness
best <- tab[which.min(dMAE)]
cli::cli_inform("Best MAE: {best$arm} ({best$dMAE}), d_bits {best$d_bits}")
worst <- tab[which.max(dMAE)]
if (worst$dMAE > NOISE) {
  cli::cli_alert_danger("Arm {.val {worst$arm}} degrades MAE by {worst$dMAE}, beyond the {NOISE} noise floor -- investigate before shipping.")
} else if (all(abs(tab$dMAE) < NOISE)) {
  cli::cli_alert_success("Every arm is inside the noise floor on MAE -- no degradation. Ships on the correctness argument.")
  if (any(tab$d_bits < -0.005)) {
    cli::cli_alert_warning("But at least one arm costs >0.005 bits. 0.5% of weight should not do that -- check pairing.")
  }
} else {
  cli::cli_alert_info("Mixed: read the table rather than a verdict line.")
}

saveRDS(list(scorecard = sc, deltas = tab), OUT)
cli::cli_alert_success("saved {OUT}")
cli::cli_alert_info("total {round(difftime(Sys.time(), t_start, units = 'mins'), 1)} min")
