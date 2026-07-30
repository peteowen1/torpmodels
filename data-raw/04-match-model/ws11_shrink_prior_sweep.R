# Tune EPR_POSITION_SHRINK_PRIOR -- and check the probability cost is not inherent
# ================================================================================
# ws10 found the shrunk rating-layer correction had the best MAE of four arms
# (25.284 vs production 25.434, dMAE -0.150 [-0.401, +0.099]) but degraded EVERY
# probability metric: bits 0.2468 -> 0.2403, logloss 0.5221 -> 0.5266, Brier
# 0.1748 -> 0.1765, accuracy 73.6% -> 73.4%.
#
# That is a TRADE, not an improvement, and it fails the two-tier D-M1 rule --
# WS1b earned EXPLORE by improving all six metrics with a CI that could never
# exclude zero. This improves one and degrades four.
#
# The prior was never tuned: 25 was picked off a rough cell-size estimate. Two
# questions, and the second is the one that decides it:
#   1. Where is the MAE optimum?
#   2. Is the probability degradation INHERENT to shrinking, or an artefact of
#      a badly-chosen prior? If some prior improves MAE *and* bits, it ships as
#      EXPLORE. If every prior that helps MAE hurts bits, it is a metric trade
#      and does not ship at all.
#
# Note -0.150 sits right on the measured XGBoost noise floor (~0.157), so treat
# any single arm's MAE gain as unproven regardless of where the optimum lands.

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
PRIORS <- c(5, 10, 25, 50, 100, 250)   # 25 is ws10's untuned value
CACHE      <- file.path(EXP, "results", "ws11_arms_cache.rds")
EVAL_CACHE <- file.path(EXP, "results", "ws11_eval_cache.rds")
OUT        <- file.path(EXP, "results", "ws11_shrink_prior_sweep.rds")
t_start <- Sys.time()

set_flags <- function(shrink, prior) {
  assignInNamespace("EPR_POSITION_SHRINK", shrink, ns = "torp")
  assignInNamespace("EPR_POSITION_SHRINK_PRIOR", prior, ns = "torp")
}

if (file.exists(CACHE)) {
  cli::cli_alert_info("Reusing cached arms")
  arms_tm <- readRDS(CACHE)$arms_tm
} else {
  cli::cli_h1("1. shared inputs")
  pgd <- adjust_epv_for_opponents(as.data.table(load_player_game_data(TRUE)))
  sk <- load_player_stat_ratings(TRUE)
  stat_ratings <- tryCatch(get_player_stat_ratings(current = FALSE), error = function(e) NULL)
  fixtures <- load_fixtures(TRUE)
  src <- load_match_inputs()
  psr_df <- .compute_psr_from_stat_ratings(sk)

  mk_tm <- function(shrink, prior) {
    set_flags(shrink, prior)
    r <- build_ratings_history(
      seasons = RATING_SEASONS, pgd = copy(pgd), stat_ratings = stat_ratings,
      fixtures = fixtures, psr_df = psr_df, opponent_adjust = FALSE)
    build_team_mdl_with(src, as.data.frame(r))
  }

  cli::cli_h1("2. arms")
  arms_tm <- list(production = { cli::cli_alert_info("production"); mk_tm(FALSE, 25) })
  for (p in PRIORS) {
    nm <- paste0("prior_", p)
    cli::cli_alert_info(nm)
    arms_tm[[nm]] <- mk_tm(TRUE, p)
  }
  set_flags(FALSE, 25)   # never leave the namespace mutated

  # Every arm must differ from production AND from its neighbours -- if two
  # priors give identical ratings the sweep is not sweeping anything.
  for (nm in setdiff(names(arms_tm), "production")) {
    if (isTRUE(all.equal(arms_tm$production$epr.x, arms_tm[[nm]]$epr.x))) {
      cli::cli_abort("Arm {.val {nm}} is identical to production.")
    }
  }
  if (length(PRIORS) > 1) {
    a <- arms_tm[[paste0("prior_", PRIORS[1])]]$epr.x
    b <- arms_tm[[paste0("prior_", PRIORS[length(PRIORS)])]]$epr.x
    if (isTRUE(all.equal(a, b))) cli::cli_abort("Smallest and largest prior give identical ratings.")
  }

  arms_tm <- lapply(arms_tm, function(tm) {
    d <- as.data.table(tm)
    opp <- intersect(c("epr.y", "psr.y", "torp.y"), names(d))
    keep <- if (length(opp)) Reduce(`&`, lapply(opp, function(cc) !is.na(d[[cc]]))) else rep(TRUE, nrow(d))
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
set.seed(20260729)
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
             # The decisive column: MAE better AND probability not worse.
             no_trade = mean(d) < 0 && bits_arm >= bits_base)
}))
print(tab, row.names = FALSE)

best <- tab[which.min(dMAE)]
cli::cli_inform("Best MAE: {best$arm} ({best$dMAE}), d_bits {best$d_bits}")
if (any(tab$no_trade)) {
  cli::cli_alert_success("Some prior improves MAE WITHOUT costing bits -- EXPLORE candidate.")
} else {
  cli::cli_alert_danger("EVERY prior that helps MAE costs bits -- a metric trade, not an improvement. Do not ship.")
}

saveRDS(list(scorecard = sc, deltas = tab), OUT)
cli::cli_alert_success("saved {OUT}")
cli::cli_alert_info("total {round(difftime(Sys.time(), t_start, units = 'mins'), 1)} min")
