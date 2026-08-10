# Do the optimised EPR parameters survive a real, held-out gate?
# ==============================================================
# epv3_optimise_epr.R tunes 12 (or 9) parameters against TRAIN xscore margin,
# 2021-2024. This scores the winners on 2025-26 margin/RMSE/bits/Brier/tips --
# a window theta has never seen, a target it was not optimised against, and the
# genuine production code path rather than the optimiser's fast approximation.
#
# THE THREE SEPARATIONS THAT MAKE THIS A REAL TEST, all deliberate:
#   window  theta fitted on 2021-2024; scored on 2025-26
#   target  optimised on xscore margin; scored on actual margin
#   code    optimised through a fast approximation of the EPR batch stage;
#           scored through .build_epr_season(epr_params = ...), which is
#           production. The approximation never reaches a reported number.
#
# Optimising and gating on the same quantity is how a metric-forcing fix gets
# through, so none of these three is a nicety.
#
# ANCHORS, pre-registered:
#   * every arm evaluated on the SAME matches
#   * arms must differ numerically from each other and from baseline
#   * any parameter that landed ON a bound is reported here again, because a
#     bound-pinned arm winning is a bug signal rather than a result
#   * elite players stay top-30 by TORP; key defenders do not collapse
#
# Run: powershell.exe -Command 'Rscript "<this file>"'

suppressMessages({
  library(dplyr); library(data.table)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "epv3_optimised_gate.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con) }
say_dt <- function(x, n = 40) for (l in capture.output(print(utils::head(x, n)))) say(l)

say("=== EPV v3: optimised EPR parameters, held-out gate ===")
say("run at ", format(Sys.time()))

PARAM_FILE <- file.path(OUT_DIR, "epv3_optimised_params.rds")
if (!file.exists(PARAM_FILE)) {
  say("epv3_optimised_params.rds not found -- run epv3_optimise_epr.R first.")
  close(con); quit(status = 1)
}
opt <- readRDS(PARAM_FILE)
say("optimised arms available: ", paste(names(opt), collapse = ", "))

say("")
say("--- bound check carried forward from the optimiser ---")
for (nm in names(opt)) {
  nb <- sum(opt[[nm]]$tab$on_bound)
  say("  ", nm, ": ", nb, " parameter(s) on a bound",
      if (nb > 0) "   <- BUG SIGNAL, a win here is not a result" else "")
}

shared_stat_ratings <- get_player_stat_ratings(current = FALSE)
shared_fixtures     <- load_fixtures(TRUE)
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                   error = function(e) NULL)
teams <- load_teams(TRUE)

pgd4 <- as.data.table(arrow::read_parquet(file.path(OUT_DIR, "epv3_player_game_v3.parquet")))
pgd3 <- copy(pgd4)[, `:=`(epv_spoil = epv_spoil + epv_hitout,
                          epv_spoil_adj = epv_spoil_adj + epv_hitout_adj,
                          epv_hitout = 0, epv_hitout_adj = 0)]

#' Build ratings through the PRODUCTION path with a given parameter set.
#' epr_params goes straight to calculate_epr_stats_batch(), so there is no second
#' implementation of the aggregation to drift.
build_ratings <- function(pgd, label, epr_params = NULL) {
  cli::cli_h2("Ratings: {label}")
  d <- as.data.table(copy(pgd))
  d <- adjust_epv_for_opponents(d)
  if (isTRUE(EPV_LEVEL_CENTRE)) d <- centre_epv_by_position(d)
  seasons <- sort(unique(d$season))
  out <- rbindlist(lapply(seasons, function(s) {
    start_round <- if (s >= 2024) 0 else 1
    max_round <- if (s == get_afl_season()) get_afl_week(type = "next") else 28
    torp:::.build_epr_season(s, start_round:max_round, d,
                             shared_stat_ratings, shared_fixtures,
                             epr_params = epr_params)
  }), use.names = TRUE, fill = TRUE)
  if (isTRUE(EPR_POSITION_CENTRE)) out <- centre_epr_by_position(out)
  if (!is.null(psr_df) && nrow(psr_df) > 0 && "psr" %in% names(psr_df)) {
    out <- calculate_torp(out, psr_df)
  }
  as.data.table(out)
}

build_with_ratings <- function(torp_df) {
  all_grounds <- file_reader("stadium_data", "reference-data")
  fix_df <- .build_fixtures_df(shared_fixtures)
  team_rt_df <- .build_team_ratings_df(teams, torp_df, psr_df)
  team_rt_fix_df <- .build_match_features(fix_df, team_rt_df, all_grounds)
  weather_df <- .load_match_weather(shared_fixtures, all_grounds, NULL, get_afl_season())
  anchor <- max(as.Date(fix_df$utc_start_time), na.rm = TRUE)
  .build_team_mdl_df(team_rt_fix_df, load_results(TRUE), load_xg(TRUE), weather_df, anchor)
}

.bits <- function(pw, hw) mean(ifelse(hw == 1, 1 + log2(pw),
                              ifelse(hw == 0, 1 + log2(1 - pw),
                                     1 + 0.5 * log2(pw * (1 - pw)))))

run_arm <- function(label, torp_df) {
  tm <- build_with_ratings(torp_df)
  feat <- grep("^(epr|psr|torp|elo|xelo).*_diff$|^(epr|psr|torp)\\.[xy]$",
               names(tm), value = TRUE)
  keep <- stats::complete.cases(tm[, feat, drop = FALSE])
  if (any(!keep)) tm <- tm[keep, , drop = FALSE]
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams,
                           xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff",
                           cv_extra_feature_cols = "xelo_diff")
  p <- unique(as.data.table(roll$input_blend_preds), by = "match_id")
  p[, arm := label]
  p
}

# ---- Arms ------------------------------------------------------------------
# Baselines are cached from ws17/ws18; only the optimised arms are new.
arms <- list()
arms[["v2"]] <- as.data.table(arrow::read_parquet(file.path(OUT_DIR, "epv3_ratings_v2.parquet")))
arms[["v3-4ch-default"]] <- as.data.table(arrow::read_parquet(file.path(OUT_DIR, "epv3_ratings_v3.parquet")))
for (nm in names(opt)) {
  merge3 <- grepl("^3ch", nm)
  arms[[paste0("v3-", nm)]] <- build_ratings(if (merge3) pgd3 else pgd4, nm,
                                             epr_params = opt[[nm]]$params)
}
say("")
say("arms built: ", paste(names(arms), collapse = ", "))

say("")
say("--- ARMS GUARD: every arm must differ from the default ---")
base <- arms[["v3-4ch-default"]]
k <- c("player_id", "season", "round")
for (nm in setdiff(names(arms), "v3-4ch-default")) {
  cm <- merge(base[, c(k, "epr"), with = FALSE], arms[[nm]][, c(k, "epr"), with = FALSE],
              by = k, suffixes = c("_a", "_b"))
  d <- mean(abs(cm$epr_a - cm$epr_b), na.rm = TRUE)
  say(sprintf("  %-22s mean|diff| epr %.5f", nm, d))
  if (d < 1e-9) say("    !! identical to default -- this arm is not live")
}

preds <- rbindlist(lapply(names(arms), function(nm) run_arm(nm, arms[[nm]])),
                   use.names = TRUE, fill = TRUE)
common <- Reduce(intersect, split(preds$match_id, preds$arm))
preds <- preds[match_id %in% common]
say("")
say("--- same-games check: ", length(common), " matches in every arm ---")

metrics <- function(p, seasons = NULL) {
  d <- if (is.null(seasons)) p else p[season %in% seasons]
  d <- d[is.finite(margin) & is.finite(pred_margin) & is.finite(pred_win)]
  hw <- ifelse(d$margin > 0, 1, ifelse(d$margin == 0, 0.5, 0))
  data.table(window = if (is.null(seasons)) "pooled" else paste(seasons, collapse = ","),
             n = nrow(d),
             MAE = round(mean(abs(d$pred_margin - d$margin)), 4),
             RMSE = round(sqrt(mean((d$pred_margin - d$margin)^2)), 4),
             bits = round(.bits(pmin(pmax(d$pred_win, 1e-6), 1 - 1e-6), hw), 4),
             Brier = round(mean((d$pred_win - hw)^2), 4),
             tips = sum((d$pred_margin > 0) == (d$margin > 0), na.rm = TRUE))
}
say("")
say("--- GATE: all metrics, pooled 2025-26 ---")
say_dt(preds[, metrics(.SD), by = arm], 12)
say("")
say("--- by season (reported, NOT decided on) ---")
say_dt(preds[, metrics(.SD, 2025), by = arm], 12)
say_dt(preds[, metrics(.SD, 2026), by = arm], 12)

say("")
say("--- paired against v2, pooled ---")
ref <- preds[arm == "v2", .(match_id, e_ref = abs(pred_margin - margin))]
for (nm in setdiff(names(arms), "v2")) {
  m <- merge(ref, preds[arm == nm, .(match_id, e = abs(pred_margin - margin))], by = "match_id")
  d <- m$e - m$e_ref
  ci <- t.test(d)$conf.int
  say(sprintf("  %-22s dMAE %+.4f  95%% CI [%+.4f, %+.4f]  (positive = worse than v2)",
              nm, mean(d), ci[1], ci[2]))
}

arrow::write_parquet(preds, file.path(OUT_DIR, "epv3_optimised_gate_preds.parquet"))
say("")
say("Nothing here is decided on the by-season rows. The pooled window is the")
say("one that counts -- 2026-only over-promised threefold in a single session")
say("before. And an arm that won while a parameter sat on a bound is a bug")
say("signal, not a result.")
close(con)
cat("\nDone\n")
