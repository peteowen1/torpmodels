# Fix 1, stage 2: what does folding spoil_epv_ctx in do to the RATINGS and to MAE?
# ==============================================================================
# Stage 1 (`torp/data-raw/04-analysis/preview_spoil_ctx_fold.R`) passed decisively:
# R^2 of the new spoil term on spoil COUNT is only 0.285, so 71.5% of its variance is
# information a flat weight cannot carry; 14.9% of player-games gain negative spoil
# credit (was 0% by construction); and KEY_DEFENDER spread rises x1.467 against
# MIDFIELDER x1.062 -- the intended ordering.
#
# Stage 2 pushes it through the real rating path. Reachable WITHOUT a pgd rebuild from
# PBP, because everything needed is already published in player_game: the box-score
# stats, `spoil_epv_ctx`, `spoils_priced` and `lineup_position`. What has to be redone
# by hand is the position adjustment, since `epv_spoil_adj` is what EPR actually
# consumes:
#     p80  = epv_spoil / tog_safe
#     adj  = .position_adjust(p80, tog, pooled_sd, standardise) grouped by lineup_position
# `spoil` IS in EPV_STANDARDISE_CHANNELS, so standardise = TRUE here.
#
# Arms: production (published channel, rebuilt and verified identical) vs folded.

suppressMessages({
  library(data.table); library(dplyr)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
  devtools::load_all("C:/dev/torpverse/torpmodels", quiet = TRUE)
})
options(torp.local_data_dir = NA)

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
source(file.path(EXP, "rolling_lib.R")); source(file.path(EXP, "arm_lib.R"))
source(file.path(EXP, "scorecard_lib.R"))

p <- torp:::default_epv_params()
RATING_SEASONS <- 2021:2026
TEST_SEASONS   <- 2025:2026
SHARE_SCALE <- 1.5   # contest_share 1/3 -> 1/2, matching disp_scale

# THE FALLBACK BASE IS NOT (spoils - spoils_priced). An earlier version of this script
# assumed it was and DOUBLE-PAID 28% of all spoils. Measured breakdown
# (audit_spoil_pricing_dropouts.R, 36,721 spoils 2024-2026):
#   priced                                          56.9%
#   contest triple, ALREADY paid via contest_epv    28.0%   <- in the RECV channel
#   same-team kick (chain-logging artifact)         15.0%
#   no kick within 5 rows                            0.0%   (14 spoils in 3 seasons)
# Only the last two are a genuine gap: 34.9% of the unpriced group.
FALLBACK_FRACTION <- 0.349
CACHE      <- file.path(EXP, "results", "ws15_arms_cache.rds")
EVAL_CACHE <- file.path(EXP, "results", "ws15_eval_cache.rds")
OUT        <- file.path(EXP, "results", "ws15_spoil_ctx_fold.rds")
t0 <- Sys.time()

OTHER <- c(tackles = "tackle_wt", pressure_acts = "pressure_wt",
           def_half_pressure_acts = "def_pressure_wt", intercepts = "intercepts_wt",
           one_percenters = "one_percenters_wt", rebound50s = "rebound50s_wt",
           frees_against = "frees_against_wt")

if (file.exists(CACHE)) {
  cli::cli_alert_info("Reusing cached arms")
  arms_tm <- readRDS(CACHE)$arms_tm
} else {
  pgd0 <- as.data.table(load_player_game_data(TRUE))
  pgd0[, tog_safe := pmax(dplyr::coalesce(time_on_ground_percentage / 100, 0.1), 0.1)]
  stopifnot(all(c("spoil_epv_ctx", "spoils_priced", "lineup_position", "epv_spoil_adj") %in% names(pgd0)))

  cli::cli_h1("1. rebuild the spoil channel, and verify the OLD rebuild is exact")
  pgd0[, other_box := Reduce(`+`, lapply(names(OTHER), function(s) get(s) * p[[OTHER[[s]]]]))]
  pgd0[, spoil_old := other_box + spoils * p$spoil_wt]
  gap <- max(abs(pgd0$spoil_old - pgd0$epv_spoil), na.rm = TRUE)
  cli::cli_alert_info("max |rebuilt - published| epv_spoil = {signif(gap, 3)}")
  if (gap > 1e-8) cli::cli_abort("Cannot reproduce published epv_spoil; refusing to score a swap on it.")

  cov <- pgd0[, .(pr = sum(spoils_priced, na.rm = TRUE), ctx = sum(spoil_epv_ctx, na.rm = TRUE))]
  FALLBACK <- (cov$ctx / cov$pr) * SHARE_SCALE
  cli::cli_alert_info("fallback for unpriced spoils = {round(FALLBACK, 4)} (production flat {p$spoil_wt})")
  pgd0[, spoil_new := other_box + spoil_epv_ctx * SHARE_SCALE +
         pmax(spoils - spoils_priced, 0) * FALLBACK_FRACTION * FALLBACK]

  cli::cli_h1("2. redo the position adjustment for each arm")
  # Mirrors create_player_game_data(): per-lineup_position centring, standardised,
  # against the channel's pooled weighted SD.
  mk_adj <- function(dt, col) {
    x <- copy(dt)
    x[, p80 := get(col) / tog_safe]
    pooled <- torp:::.wtd_sd(x$p80, x$tog_safe)
    x[, adj := torp:::.position_adjust(p80, tog_safe, pooled, TRUE), by = lineup_position]
    # x$adj, not bare `adj` -- it is a COLUMN, not a variable in this scope, and a bare
    # reference inside the cli string is what crashed the first run.
    adj_sd <- sd(x$adj, na.rm = TRUE)
    cli::cli_alert_info("  {col}: pooled_sd {round(pooled, 4)}, adj sd {round(adj_sd, 4)}")
    x$adj
  }
  pgd0[, adj_old := mk_adj(pgd0, "spoil_old")]
  pgd0[, adj_new := mk_adj(pgd0, "spoil_new")]
  # Sanity: the OLD recomputed adjustment should match the published one closely. It
  # will not be bit-identical (pooled_sd is recomputed on a possibly different row set)
  # so this is a correlation check, not an equality one.
  cc <- cor(pgd0$adj_old, pgd0$epv_spoil_adj, use = "complete.obs")
  cli::cli_alert_info("cor(recomputed old adj, published epv_spoil_adj) = {round(cc, 4)}")
  if (cc < 0.99) cli::cli_alert_danger("Recomputed adjustment does not track the published one -- treat results with caution.")

  cli::cli_h1("3. build both arms")
  sk <- load_player_stat_ratings(TRUE)
  stat_ratings <- tryCatch(get_player_stat_ratings(current = FALSE), error = function(e) NULL)
  fixtures <- load_fixtures(TRUE); src <- load_match_inputs()
  psr_df <- .compute_psr_from_stat_ratings(sk)

  mk_arm <- function(adj_col, label) {
    cli::cli_alert_info("arm {label}")
    x <- copy(pgd0)
    x[, epv_spoil_adj := get(adj_col)]
    x[, epv_adj := epv_recv_adj + epv_disp_adj + epv_spoil_adj + epv_hitout_adj]
    r <- build_ratings_history(seasons = RATING_SEASONS, pgd = x, stat_ratings = stat_ratings,
                              fixtures = fixtures, psr_df = psr_df)
    build_team_mdl_with(src, as.data.frame(r))
  }
  arms_tm <- list(production = mk_arm("adj_old", "production"),
                  folded     = mk_arm("adj_new", "folded"))

  a <- arms_tm$production$epr.x; b <- arms_tm$folded$epr.x
  if (isTRUE(all.equal(a, b))) cli::cli_abort("Arms identical -- the fold did not reach the ratings.")
  cli::cli_inform("mean |d epr| vs production = {round(mean(abs(a - b), na.rm = TRUE), 4)}")

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

cli::cli_h1("4. rolling eval")
res <- if (file.exists(EVAL_CACHE)) readRDS(EVAL_CACHE) else {
  r <- score_arms(arms_tm, test_seasons = TEST_SEASONS, parallel = TRUE); saveRDS(r, EVAL_CACHE); r
}
preds <- lapply(res, function(r) r$input_blend_preds)

cli::cli_h1("5. scorecard + paired delta")
sc <- scorecard(preds, squiggle_sources = "Aggregate"); print_scorecard(sc)
base <- as.data.table(preds$production); key <- intersect(c("season", "round", "home_team"), names(base))
b <- as.data.table(preds$folded)
j <- merge(base[, c(key, "pred_margin", "margin"), with = FALSE],
           b[,    c(key, "pred_margin", "margin"), with = FALSE], by = key, suffixes = c(".base", ".arm"))
d <- abs(j$pred_margin.arm - j$margin.arm) - abs(j$pred_margin.base - j$margin.base)
set.seed(20260730)
ci <- quantile(replicate(10000, mean(sample(d, length(d), replace = TRUE))), c(0.025, 0.975))
scdt <- as.data.table(sc)
cli::cli_alert_info("dMAE {round(mean(d), 4)} [{round(ci[[1]], 4)}, {round(ci[[2]], 4)}]")
cli::cli_alert_info("bits {round(scdt[source=='folded']$bits, 4)} vs {round(scdt[source=='production']$bits, 4)} (d {round(scdt[source=='folded']$bits - scdt[source=='production']$bits, 4)})")
cli::cli_alert_info("READ THIS AS A NO-DEGRADATION TEST, not a gain test: the change is about WHICH")
cli::cli_alert_info("defender gets credit, and match MAE has little power to see a redistribution")
cli::cli_alert_info("inside one team's rating sum. A dMAE inside the ~0.157 noise floor is a pass.")

saveRDS(list(scorecard = sc, dMAE = mean(d), ci = ci), OUT)
cli::cli_alert_info("total {round(difftime(Sys.time(), t0, units='mins'), 1)} min")
cli::cli_alert_success("done")
