# Does centring PSR on the WEEKLY team-sheet role change match MAE?
# ================================================================
# Thread G (torpverse docs/NEXT-STEPS.md). Production PSR centres on `pos_group`,
# the CAREER-MODAL registered position, while torp_ratings presents the result
# under the season listing. 136 of 669 2026 players differ between the two
# (walked with Pete 2026-10-11: Silvagni, a career key forward now listed key
# defender, is centred against key forwards). ws9 scored the season listing and
# it REGRESSED (+0.382). Plan §7.15 found weekly resolution is where the whole
# calibration gain sits, so this scores the weekly arm on its own.
#
# NEW arm: `lineup_pos_group` = the 6-way group of the slot he was NAMED in that
# round (LINEUP_POSITION_GROUP_MAP). Since torp#251 calculate_psr() centres on
# this column only when asked with psr_centre = "weekly", so both arms name
# their key explicitly below. Rows with no on-field
# slot (bench, did not play) carry forward his last named slot, then fall back
# to pos_group -- never to calculate_psr()'s global-mean branch, which would turn
# every non-playing round into a different arm by accident.
#
# OLD arm: production code path, unmodified frame.
# Absolute levels are harness-relative (PAIRED-safe). The delta is the result.

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
CACHE      <- file.path(EXP, "results", "ws31_arms_cache.rds")
EVAL_CACHE <- file.path(EXP, "results", "ws31_eval_cache.rds")
OUT        <- file.path(EXP, "results", "ws31_psr_weekly_centring.rds")
t_start <- Sys.time()

# Last-observation-carried-forward for a character vector.
nafill_chr <- function(x) {
  idx <- cummax(ifelse(is.na(x), 0L, seq_along(x)))
  out <- x[pmax(idx, 1L)]; out[idx == 0L] <- NA_character_; out
}

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
  # Weekly named slot per player-round, from the same pgd both arms share.
  if (!"lineup_position" %in% names(pgd)) cli::cli_abort("pgd has no lineup_position")
  lp <- pgd[, .(player_id, season = as.integer(season), round = as.numeric(round),
                lpg = unname(LINEUP_POSITION_GROUP_MAP[lineup_position]))][!is.na(lpg)]
  lp <- unique(lp, by = c("player_id", "season", "round"))
  cli::cli_inform("named on-field slots: {nrow(lp)} player-rounds")

  sk_new <- copy(as.data.table(sk))
  sk_new[, `:=`(.s = as.integer(season), .r = as.numeric(round))]
  sk_new[lp, on = c(player_id = "player_id", .s = "season", .r = "round"), lineup_pos_group := i.lpg]
  n_direct <- sum(!is.na(sk_new$lineup_pos_group))
  setorder(sk_new, player_id, .s, .r)
  sk_new[, lineup_pos_group := nafill_chr(lineup_pos_group), by = player_id]
  n_locf <- sum(!is.na(sk_new$lineup_pos_group)) - n_direct
  sk_new[is.na(lineup_pos_group), lineup_pos_group := pos_group]
  n_na <- sum(is.na(sk_new$lineup_pos_group))
  cli::cli_inform("weekly key: {n_direct} named that round, {n_locf} carried forward, {nrow(sk_new) - n_direct - n_locf - n_na} pos_group fallback, {n_na} NA (of {nrow(sk_new)})")
  if (n_direct / nrow(sk_new) < 0.2) cli::cli_abort("Under 20% of rows matched a team sheet -- join is broken.")
  sk_new[, c(".s", ".r") := NULL]

  psr_old <- .compute_psr_from_stat_ratings(sk, psr_centre = "career")
  psr_new <- .compute_psr_from_stat_ratings(as.data.frame(sk_new), psr_centre = "weekly")

  po <- as.data.table(psr_old)[, .(player_id, season, round, p_old = psr)]
  pn <- as.data.table(psr_new)[, .(player_id, season, round, p_new = psr)]
  j <- merge(po, pn, by = c("player_id", "season", "round"))
  if (nrow(j) != nrow(po)) cli::cli_abort("Arms have different rows: {nrow(po)} vs {nrow(j)} matched")
  if (isTRUE(all.equal(j$p_old, j$p_new))) {
    cli::cli_abort("weekly key did nothing -- refusing to report a delta.")
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
  arms_tm <- list(career = mk_tm(psr_old), weekly = mk_tm(psr_new))

  # Placeholder finals fixtures (2026 r25-26, teams TBD) carry NA features, and
  # model.matrix() drops NA rows silently -> predictions shorter than the frame.
  arms_tm <- lapply(arms_tm, function(tm) {
    d <- as.data.table(tm)
    opp <- intersect(c("epr.y", "psr.y", "torp.y"), names(d))
    keep <- if (length(opp)) Reduce(`&`, lapply(opp, function(cc) !is.na(d[[cc]]))) else rep(TRUE, nrow(d))
    if (sum(!keep)) cli::cli_alert_warning("Dropping {sum(!keep)} row{?s} with no opponent (finals placeholders).")
    as.data.frame(d[keep])   # data.frame: the harness subsets with df[, cols, drop = FALSE]
  })
  if (!identical(sort(as.character(arms_tm$career$match_id)),
                 sort(as.character(arms_tm$weekly$match_id)))) {
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
a <- as.data.table(preds$career); b <- as.data.table(preds$weekly)
key <- intersect(c("season", "round", "home_team"), names(a))
j <- merge(a[, c(key, "pred_margin", "margin"), with = FALSE],
           b[, c(key, "pred_margin", "margin"), with = FALSE],
           by = key, suffixes = c(".old", ".new"))
d <- abs(j$pred_margin.new - j$margin.new) - abs(j$pred_margin.old - j$margin.old)
set.seed(20261011)
ci <- quantile(replicate(10000, mean(sample(d, length(d), replace = TRUE))), c(0.025, 0.975))
cli::cli_inform("n matches: {length(d)}")
cli::cli_inform("MAE career: {round(mean(abs(j$pred_margin.old - j$margin.old)), 4)} | weekly: {round(mean(abs(j$pred_margin.new - j$margin.new)), 4)}")
cli::cli_inform("dMAE (weekly - career): {round(mean(d), 4)}  95% CI [{round(ci[1],4)}, {round(ci[2],4)}]")
cli::cli_inform("SHIP gate (CI excludes zero AND dMAE < 0): {ci[2] < 0}")
cli::cli_inform("NOTE: a null here still ships on correctness, exactly as #134 did -- the")
cli::cli_inform("      change makes TORP's two halves agree on what a position IS.")

saveRDS(list(scorecard = sc, delta = list(d = d, ci = ci)), OUT)
cli::cli_alert_success("saved {OUT}")
cli::cli_alert_info("total {round(difftime(Sys.time(), t_start, units = 'mins'), 1)} min")
