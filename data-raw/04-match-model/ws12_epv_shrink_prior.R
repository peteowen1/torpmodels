# EPV-layer position shrinkage: does it pass the match-MAE gate, and at what prior?
# ================================================================================
# The EPR-layer version of this (ws11) was shipped and reverted the same night:
# the EPR correction is ~0.002, a backstop, so shrinking it moved published TORP
# by 0.0044 and left the top 20 in identical order. The EPV layer is where the
# correction is actually large -- median 1.065 across the four channels, up to
# 8.178, on cells whose weight goes as low as 1.45.
#
# The shape is DIFFERENT from ws11's and that difference is the point. ws11 shrank
# toward ZERO, which withholds correction, and every point withheld is a point of
# positional level handed back (measured: 0.477 of restored spread at prior 5,
# against the 2.94 the v2 fix removed, and mostly from NORMAL cells). This shrinks
# toward the bucket's mean over strictly EARLIER rounds, so a full position level
# is still subtracted -- it only changes which games measure that level when the
# round's own cell is too thin to measure it. Residual level restored: 0.03-0.12.
#
# Unlike ws11's arm this moves ratings materially (mean |d epv| 0.040 per
# player-game, max 2.765), so it needs the real gate, not a correctness argument.
#
# Baseline to beat: 25.4335. Two conditions, per the D-M1 two-tier rule:
#   1. dMAE < 0
#   2. bits not worse -- ws11 died because every MAE-helping prior cost bits.

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
PRIORS <- c(1, 2, 5)
CACHE      <- file.path(EXP, "results", "ws12_arms_cache.rds")
EVAL_CACHE <- file.path(EXP, "results", "ws12_eval_cache.rds")
OUT        <- file.path(EXP, "results", "ws12_epv_shrink_prior.rds")
t_start <- Sys.time()

set_flags <- function(shrink, prior) {
  assignInNamespace("EPV_POSITION_SHRINK", shrink, ns = "torp")
  assignInNamespace("EPV_POSITION_SHRINK_PRIOR", prior, ns = "torp")
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
  mk_tm <- function(shrink, prior) {
    set_flags(shrink, prior)
    r <- build_ratings_history(
      seasons = RATING_SEASONS, pgd = copy(pgd_oadj), stat_ratings = stat_ratings,
      fixtures = fixtures, psr_df = psr_df, opponent_adjust = FALSE)
    build_team_mdl_with(src, as.data.frame(r))
  }

  arms_tm <- list(production = { cli::cli_alert_info("production"); mk_tm(FALSE, 2) })
  for (p in PRIORS) {
    nm <- paste0("prior_", p)
    cli::cli_alert_info(nm)
    arms_tm[[nm]] <- mk_tm(TRUE, p)
  }
  set_flags(FALSE, 2)   # never leave the namespace mutated

  # A flag that silently does nothing reads as "no effect" instead of "not
  # tested" -- which is exactly how ws10 nearly reported two unreachable arms.
  for (nm in setdiff(names(arms_tm), "production")) {
    a <- arms_tm$production$epr.x; b <- arms_tm[[nm]]$epr.x
    if (isTRUE(all.equal(a, b))) {
      cli::cli_abort("Arm {.val {nm}} is identical to production -- refusing to report a delta.")
    }
    cli::cli_inform("{nm}: mean |d epr| vs production = {round(mean(abs(a - b), na.rm = TRUE), 4)}")
  }
  if (length(PRIORS) > 1) {
    a <- arms_tm[[paste0("prior_", PRIORS[1])]]$epr.x
    b <- arms_tm[[paste0("prior_", PRIORS[length(PRIORS)])]]$epr.x
    if (isTRUE(all.equal(a, b))) cli::cli_abort("Smallest and largest prior give identical ratings.")
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

best <- tab[which.min(dMAE)]
cli::cli_inform("Best MAE: {best$arm} ({best$dMAE}), d_bits {best$d_bits}")
if (any(tab$no_trade)) {
  cli::cli_alert_success("A prior improves MAE without costing bits -- gate passed.")
} else {
  cli::cli_alert_danger("Every prior that helps MAE costs bits, or none help. Not a shippable gain on match MAE.")
  cli::cli_alert_info("NOTE this does not by itself sink the change: the thin-cell defect is a")
  cli::cli_alert_info("correctness argument about September ratings, and 22 of 966 cells carry")
  cli::cli_alert_info("~0.5% of total weight, so match MAE has little power to see it either way.")
}

saveRDS(list(scorecard = sc, deltas = tab), OUT)
cli::cli_alert_success("saved {OUT}")
cli::cli_alert_info("total {round(difftime(Sys.time(), t_start, units = 'mins'), 1)} min")
