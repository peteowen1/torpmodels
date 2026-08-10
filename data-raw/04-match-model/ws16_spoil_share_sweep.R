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
  library(data.table); library(dplyr); library(arrow)
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
# ws16 SWEEPS THE SHARE. ws15 scored one arm and found MAE +0.0803 (inside the
# noise floor, CI spanning zero) with bits -0.0035. A single arm cannot say whether
# that bits cost is real, and the difference matters: it is the dose-response across
# floors that made ws14's bits cost credible, and the NON-monotonic scatter across
# priors that showed ws11/ws12's MAE deltas were noise.
#
# So: three shares. If |d bits| scales with the size of the change, the cost is real
# and this is a genuine metric trade. If it does not, it is noise and the
# rating-quality argument (84.5% of the term's variance is signal the spoil COUNT
# cannot carry; key defenders +38% spread) carries the decision.
SHARES <- c(1.0, 1.5, 2.0)   # contest_share 1/3, 1/2, 2/3
SHARE_SCALE <- 1.5           # retained for the fallback-value derivation below

# THE FALLBACK BASE IS NOT (spoils - spoils_priced). An earlier version of this script
# assumed it was and DOUBLE-PAID 28% of all spoils. Measured breakdown
# (audit_spoil_pricing_dropouts.R, 36,721 spoils 2024-2026):
#   priced                                          56.9%
#   contest triple, ALREADY paid via contest_epv    28.0%   <- in the RECV channel
#   same-team kick (chain-logging artifact)         15.0%
#   no kick within 5 rows                            0.0%   (14 spoils in 3 seasons)
# Only the last two are a genuine gap: 34.9% of the unpriced group.
# EXACT per-player-match gap counts, not a global fraction. The 0.349 constant used
# in the first run is position-biased by up to 24x (KEY_DEFENDER 0.039 vs KEY_FORWARD
# 0.934), because a key defender's spoil is usually a contest triple already paid via
# contest_epv while a key forward's is a same-team artifact. Built and validated by
# build_exact_spoil_gap_counts.R (reproduces production spoils_priced on 100% of
# player-matches, cor 0.9999).
GAPS <- file.path(EXP, "results", "spoil_gap_counts.parquet")  # built by build_exact_spoil_gap_counts.R; was a dead Claude-session temp path
CACHE      <- file.path(EXP, "results", "ws16_arms_cache.rds")
EVAL_CACHE <- file.path(EXP, "results", "ws16_eval_cache.rds")
OUT        <- file.path(EXP, "results", "ws16_spoil_share_sweep.rds")
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
  # One rebuilt channel per share. The fallback scales with the share too, so each
  # arm is internally consistent rather than mixing a 1/2-share fallback into a
  # 1/3-share channel.
  per_priced <- cov$ctx / cov$pr
  if (!file.exists(GAPS)) cli::cli_abort("Run build_exact_spoil_gap_counts.R first -- refusing a biased global fraction.")
  gp <- as.data.table(arrow::read_parquet(GAPS))
  pgd0[, `:=`(.pid = as.character(player_id), .mid = as.character(match_id))]
  pgd0[gp, spoils_gap := i.spoils_gap, on = c(.pid = "player_id", .mid = "match_id")]
  pgd0[is.na(spoils_gap), spoils_gap := 0L]
  cli::cli_alert_info("exact gaps joined: {sum(pgd0$spoils_gap)} genuine vs {sum(pmax(pgd0$spoils - pgd0$spoils_priced, 0))} unpriced total")
  for (sh in SHARES) {
    col <- paste0("spoil_new_", sub("\\.", "", format(sh, nsmall = 1)))
    pgd0[, (col) := other_box + spoil_epv_ctx * sh + spoils_gap * (per_priced * sh)]
  }

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
  share_cols <- character(0)
  for (sh in SHARES) {
    src_col <- paste0("spoil_new_", sub("\\.", "", format(sh, nsmall = 1)))
    dst_col <- paste0("adj_", sub("\\.", "", format(sh, nsmall = 1)))
    pgd0[, (dst_col) := mk_adj(pgd0, src_col)]
    share_cols <- c(share_cols, dst_col)
  }
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
  arms_tm <- list(production = mk_arm("adj_old", "production"))
  for (i in seq_along(SHARES)) {
    nm <- paste0("share_", format(SHARES[i], nsmall = 1))
    arms_tm[[nm]] <- mk_arm(share_cols[i], nm)
  }

  a <- arms_tm$production$epr.x
  for (nm in setdiff(names(arms_tm), "production")) {
    b <- arms_tm[[nm]]$epr.x
    if (isTRUE(all.equal(a, b))) cli::cli_abort("Arm {.val {nm}} is identical to production.")
    cli::cli_inform("{nm}: mean |d epr| vs production = {round(mean(abs(a - b), na.rm = TRUE), 4)}")
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

cli::cli_h1("4. rolling eval")
res <- if (file.exists(EVAL_CACHE)) readRDS(EVAL_CACHE) else {
  r <- score_arms(arms_tm, test_seasons = TEST_SEASONS, parallel = TRUE); saveRDS(r, EVAL_CACHE); r
}
preds <- lapply(res, function(r) r$input_blend_preds)

cli::cli_h1("5. scorecard + paired delta")
sc <- scorecard(preds, squiggle_sources = "Aggregate"); print_scorecard(sc)
base <- as.data.table(preds$production); key <- intersect(c("season", "round", "home_team"), names(base))
scdt <- as.data.table(sc)
bits_base <- scdt[source == "production"]$bits
set.seed(20260730)
tab <- rbindlist(lapply(setdiff(names(preds), "production"), function(nm) {
  b <- as.data.table(preds[[nm]])
  j <- merge(base[, c(key, "pred_margin", "margin"), with = FALSE],
             b[,    c(key, "pred_margin", "margin"), with = FALSE], by = key, suffixes = c(".base", ".arm"))
  d <- abs(j$pred_margin.arm - j$margin.arm) - abs(j$pred_margin.base - j$margin.base)
  ci <- quantile(replicate(10000, mean(sample(d, length(d), replace = TRUE))), c(0.025, 0.975))
  bits_arm <- scdt[source == nm]$bits
  data.table(arm = nm, mae = round(mean(abs(j$pred_margin.arm - j$margin.arm)), 4),
             dMAE = round(mean(d), 4), lo = round(ci[[1]], 4), hi = round(ci[[2]], 4),
             bits = round(bits_arm, 4), d_bits = round(bits_arm - bits_base, 4))
}))
print(tab, row.names = FALSE)

cli::cli_h1("6. THE DECIDING TEST -- does the bits cost scale with the change?")
# A real cost rises monotonically with the size of the intervention (as ws14's floor
# sweep did). Noise scatters (as ws11's and ws12's prior sweeps did).
mono <- all(diff(tab$d_bits) <= 0) || all(diff(tab$d_bits) >= 0)
cli::cli_alert_info("d_bits across shares {paste(SHARES, collapse=' / ')}: {paste(tab$d_bits, collapse=' / ')}")
cli::cli_alert_info("monotonic in share: {mono}")
if (mono && abs(tab$d_bits[nrow(tab)]) > abs(tab$d_bits[1]) * 1.3) {
  cli::cli_alert_danger("DOSE-RESPONSE: the bits cost is REAL and scales with the change. Treat as a genuine trade.")
} else if (!mono) {
  cli::cli_alert_success("NON-MONOTONIC: the bits cost does not track the change -- consistent with noise.")
  cli::cli_alert_info("The rating-quality argument then carries: 84.5% of the term's variance is signal")
  cli::cli_alert_info("the spoil COUNT cannot carry, and key defenders gain +38% spread.")
} else {
  cli::cli_alert_warning("Monotonic but flat -- weak evidence either way; read the table.")
}
cli::cli_alert_info("READ THIS AS A NO-DEGRADATION TEST, not a gain test: the change is about WHICH")
cli::cli_alert_info("defender gets credit, and match MAE has little power to see a redistribution")
cli::cli_alert_info("inside one team's rating sum. A dMAE inside the ~0.157 noise floor is a pass.")

saveRDS(list(scorecard = sc, deltas = tab), OUT)
cli::cli_alert_info("total {round(difftime(Sys.time(), t0, units='mins'), 1)} min")
cli::cli_alert_success("done")
