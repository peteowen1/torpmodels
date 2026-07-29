# The value/rating split: does it beat production on match MAE?
# ==============================================================
# Pete's design (2026-07-29). The VALUE layer (EPV, PSV) describes one game, so
# it should centre on the WEEKLY role -- he played CHF that day, judge him
# against the other CHFs that day. The RATING layer (EPR, PSR) accumulates
# across a season, so it should take a small SHRUNK adjustment on the STABLE
# listed position -- a rating must not lurch because he was moved for one week.
#
# Motivated by the ws9 result: centring PSR on the season listing REGRESSED
# (+0.382 [+0.081, +0.684]). The likely reason is that it made the key LESS
# stable at the layer that wants stability most. This is the other direction.
#
# Arms differ in one flag each and all come from PRODUCTION code paths, never a
# replica. Every flag was verified to change the numbers before this ran -- a
# flag that silently does nothing reads as "no effect" instead of "not tested".
#
# NOTE the corrected premise: PSV was NOT missing a weekly stage. It has had one
# all along, keyed on the same raw lineup_position as EPV. The two value layers
# were already symmetric; the open question is only whether the arbitrary
# left/right mirrors should be merged.
#
# Absolute levels are harness-relative (PAIRED-safe, not STRICT). Deltas are the
# result. Baseline to beat: 25.4335.

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
CACHE      <- file.path(EXP, "results", "ws10_arms_cache.rds")
EVAL_CACHE <- file.path(EXP, "results", "ws10_eval_cache.rds")
OUT        <- file.path(EXP, "results", "ws10_value_rating_split.rds")
t_start <- Sys.time()

# Flags are package constants read at call time, so an arm is defined by the
# values it sets. assignInNamespace, not a copied implementation: the arm must
# run the SAME code production runs.
FLAGS <- c("ROLE_USE_LINEUP_GROUP", "PSV_ROLE_CENTRE_BY_ROUND",
           "EPR_POSITION_SHRINK", "EPV_LEVEL_CENTRE")
set_flags <- function(v) {
  for (n in names(v)) assignInNamespace(n, v[[n]], ns = "torp")
}
BASE <- list(ROLE_USE_LINEUP_GROUP = FALSE, PSV_ROLE_CENTRE_BY_ROUND = FALSE,
             EPR_POSITION_SHRINK = FALSE, EPV_LEVEL_CENTRE = TRUE)

ARMS <- list(
  production    = BASE,
  # value layer: merge the arbitrary left/right mirrors
  mirrors       = modifyList(BASE, list(ROLE_USE_LINEUP_GROUP = TRUE)),
  # rating layer: shrink the correction instead of applying it in full
  shrunk        = modifyList(BASE, list(EPR_POSITION_SHRINK = TRUE)),
  # the full design: value layer does the role job, rating layer does the
  # stable job (shrunk), and EPV stops doing the stable job as well
  split         = modifyList(BASE, list(ROLE_USE_LINEUP_GROUP = TRUE,
                                        EPV_LEVEL_CENTRE = FALSE,
                                        EPR_POSITION_SHRINK = TRUE))
)

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

  cli::cli_h1("2. ratings history per arm")
  # The opponent adjustment is arm-INVARIANT, so do it once and share it. The
  # centring flags act after it.
  pgd_oadj <- adjust_epv_for_opponents(copy(pgd_raw))

  mk_tm <- function(flags) {
    set_flags(flags)
    r <- build_ratings_history(
      seasons = RATING_SEASONS, pgd = copy(pgd_oadj), stat_ratings = stat_ratings,
      fixtures = fixtures, psr_df = .compute_psr_from_stat_ratings(sk),
      opponent_adjust = FALSE,
      epv_level_centre = isTRUE(flags$EPV_LEVEL_CENTRE))
    build_team_mdl_with(src, as.data.frame(r))
  }

  arms_tm <- lapply(names(ARMS), function(nm) {
    cli::cli_alert_info("building arm {.val {nm}}")
    mk_tm(ARMS[[nm]])
  })
  names(arms_tm) <- names(ARMS)
  set_flags(BASE)   # never leave the namespace mutated

  # Every arm must DIFFER from production, or it is a no-op dressed as a test.
  for (nm in setdiff(names(arms_tm), "production")) {
    a <- arms_tm$production$epr.x; b <- arms_tm[[nm]]$epr.x
    if (isTRUE(all.equal(a, b))) {
      cli::cli_abort("Arm {.val {nm}} is identical to production -- refusing to report a delta.")
    }
    cli::cli_inform("{nm}: mean |d epr| vs production = {round(mean(abs(a - b), na.rm = TRUE), 4)}")
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
  saveRDS(res, EVAL_CACHE)   # save BEFORE the scorecard
}

preds <- lapply(res, function(r) {
  if (is.null(r$input_blend_preds)) cli::cli_abort("No input_blend_preds; got {.val {names(r)}}")
  r$input_blend_preds
})

cli::cli_h1("4. scorecard")
sc <- scorecard(preds, squiggle_sources = "Aggregate")
print_scorecard(sc)

cli::cli_h1("5. paired deltas vs production")
base <- as.data.table(preds$production)
key <- intersect(c("season", "round", "home_team"), names(base))
set.seed(20260729)
tab <- rbindlist(lapply(setdiff(names(preds), "production"), function(nm) {
  b <- as.data.table(preds[[nm]])
  j <- merge(base[, c(key, "pred_margin", "margin"), with = FALSE],
             b[,    c(key, "pred_margin", "margin"), with = FALSE],
             by = key, suffixes = c(".base", ".arm"))
  d <- abs(j$pred_margin.arm - j$margin.arm) - abs(j$pred_margin.base - j$margin.base)
  ci <- quantile(replicate(10000, mean(sample(d, length(d), replace = TRUE))), c(0.025, 0.975))
  data.table(arm = nm, n = length(d),
             mae_base = round(mean(abs(j$pred_margin.base - j$margin.base)), 4),
             mae_arm  = round(mean(abs(j$pred_margin.arm  - j$margin.arm)),  4),
             dMAE = round(mean(d), 4),
             lo = round(ci[[1]], 4), hi = round(ci[[2]], 4),
             ship = ci[[2]] < 0)
}))
print(tab, row.names = FALSE)
cli::cli_inform("SHIP gate: dMAE < 0 AND the CI excludes zero.")
cli::cli_inform("Baseline to beat: 25.4335. ws9 (PSR->listed) scored +0.382 and was rejected.")

saveRDS(list(scorecard = sc, deltas = tab), OUT)
cli::cli_alert_success("saved {OUT}")
cli::cli_alert_info("total {round(difftime(Sys.time(), t_start, units = 'mins'), 1)} min")
