# Does the chain-native EPV v3 rebuild help or hurt MATCH prediction?
# ===================================================================
# v3 strips ~30 box-score weights out of EPV and prices everything from
# delta_epv, adding a surprise-weighted aerial contest channel. Design and the
# rest of the gates: torpverse/docs/plans/EPV-V3-CHAIN-NATIVE.md
#
# This is the gate that matters most for shipping, because it is the one v3 is
# most likely to FAIL. Measured 2026-07-30: the box-score block is near-orthogonal
# to the chain part (R2 0.072) and adds forward predictive value in 5 of 5
# seasons, while the chain part contributes essentially nothing forward
# (coef 0.0118, p = 0.92). v3 deletes exactly that block. The argument for doing
# it anyway is that TORP is 0.5*EPV + 0.5*PSV and PSV is already the stat-based
# half -- the signal moves rather than disappears -- but that is an argument, not
# a measurement, and this script is the measurement.
#
# Same pipeline, same games, same trainers. ONLY the EPV engine differs.
#
# Method discipline carried in from prior burns:
#   * decide on the POOLED window, not 2026-only (2026-only over-promised 3x in
#     one session)
#   * quote MAE, RMSE, bits, Brier and tips -- all of them, not the flattering ones
#   * "CI spans zero" is not "no evidence"
#   * assert the two arms actually differ numerically before believing either
#
# Run: powershell.exe -Command 'Rscript "<this file>"'

suppressMessages({
  library(dplyr); library(data.table)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RES <- file.path(EXP, "results")
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "epv3_match_gate.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con) }
say_dt <- function(x, n = 60) for (l in capture.output(print(utils::head(x, n)))) say(l)

say("=== EPV v3 match-model gate ===")
say("run at ", format(Sys.time()), " | test seasons ", paste(TEST_SEASONS, collapse = "-"))

# ---- Shared inputs ---------------------------------------------------------
shared_stat_ratings <- get_player_stat_ratings(current = FALSE)
shared_fixtures     <- load_fixtures(TRUE)
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                   error = function(e) { cli::cli_warn("PSR: {conditionMessage(e)}"); NULL })

#' Turn a player_game_data frame into a torp_ratings frame
#'
#' Mirrors run_ratings_pipeline.R stages 3-5 WITHOUT publishing: opponent
#' adjustment, positional level centring, per-season EPR, EPR centring, then the
#' PSR blend. Anything skipped here would make the two arms differ by more than
#' the engine.
build_ratings <- function(pgd, label) {
  cli::cli_h2("Building ratings: {label}")
  d <- data.table::as.data.table(data.table::copy(pgd))
  d <- adjust_epv_for_opponents(d)
  if (isTRUE(EPV_LEVEL_CENTRE)) d <- centre_epv_by_position(d)

  seasons <- sort(unique(d$season))
  out <- data.table::rbindlist(lapply(seasons, function(s) {
    start_round <- if (s >= 2024) 0 else 1
    max_round <- if (s == get_afl_season()) get_afl_week(type = "next") else 28
    torp:::.build_epr_season(s, start_round:max_round, d,
                             shared_stat_ratings, shared_fixtures)
  }), use.names = TRUE, fill = TRUE)

  if (isTRUE(EPR_POSITION_CENTRE)) out <- centre_epr_by_position(out)
  if (!is.null(psr_df) && nrow(psr_df) > 0 && "psr" %in% names(psr_df)) {
    out <- calculate_torp(out, psr_df)
  }
  cli::cli_inform("{label}: {nrow(out)} rating rows, epr sd {round(sd(out$epr, na.rm = TRUE), 3)}")
  out
}

build_with_ratings <- function(torp_df) {
  all_grounds <- file_reader("stadium_data", "reference-data")
  xg_df <- load_xg(TRUE); fixtures <- shared_fixtures
  results <- load_results(TRUE); teams <- load_teams(TRUE)
  fix_df <- .build_fixtures_df(fixtures)
  team_rt_df <- .build_team_ratings_df(teams, torp_df, psr_df)
  team_rt_fix_df <- .build_match_features(fix_df, team_rt_df, all_grounds)
  weather_df <- .load_match_weather(fixtures, all_grounds, NULL, get_afl_season())
  anchor <- max(as.Date(fix_df$utc_start_time), na.rm = TRUE)
  .build_team_mdl_df(team_rt_fix_df, results, xg_df, weather_df, anchor)
}

.bits <- function(pw, hw) mean(ifelse(hw == 1, 1 + log2(pw),
                              ifelse(hw == 0, 1 + log2(1 - pw),
                                     1 + 0.5 * log2(pw * (1 - pw)))))

# ---- Arms ------------------------------------------------------------------
pgd_v2 <- arrow::read_parquet(file.path(OUT_DIR, "epv3_player_game_v2.parquet"))
pgd_v3 <- arrow::read_parquet(file.path(OUT_DIR, "epv3_player_game_v3.parquet"))
say("pgd rows: v2 ", nrow(pgd_v2), " | v3 ", nrow(pgd_v3))

# Cache the rating frames. Rebuilding them is ~25 minutes of the run and they do
# not change while the scoring code is being fixed; delete the parquets to force
# a rebuild after any change to the engine or the ratings pipeline.
cached <- function(label, pgd) {
  f <- file.path(OUT_DIR, sprintf("epv3_ratings_%s.parquet", label))
  if (file.exists(f)) {
    cli::cli_alert_info("Reusing cached {label} ratings ({f})")
    return(as.data.table(arrow::read_parquet(f)))
  }
  r <- build_ratings(pgd, label)
  arrow::write_parquet(r, f)
  r
}
r_v2 <- cached("v2", pgd_v2)
r_v3 <- cached("v3", pgd_v3)

# ---- ARMS GUARD ------------------------------------------------------------
say("")
say("--- ARMS GUARD: the two rating frames must differ numerically ---")
k <- c("player_id", "season", "round")
cm <- merge(as.data.table(r_v2)[, c(k, "epr", "torp"), with = FALSE],
            as.data.table(r_v3)[, c(k, "epr", "torp"), with = FALSE],
            by = k, suffixes = c("_v2", "_v3"))
say("rows in both: ", format(nrow(cm), big.mark = ","))
say("epr  mean|diff| ", round(mean(abs(cm$epr_v3 - cm$epr_v2), na.rm = TRUE), 4),
    " | cor ", round(cor(cm$epr_v2, cm$epr_v3, use = "complete.obs"), 4))
say("torp mean|diff| ", round(mean(abs(cm$torp_v3 - cm$torp_v2), na.rm = TRUE), 4),
    " | cor ", round(cor(cm$torp_v2, cm$torp_v3, use = "complete.obs"), 4))
stopifnot(mean(abs(cm$epr_v3 - cm$epr_v2), na.rm = TRUE) > 1e-6)

run_arm <- function(label, torp_df) {
  tm <- build_with_ratings(torp_df)
  # .build_team_mdl_df() emits rows for fixtures with no opponent yet (2026
  # finals), whose rating features are NA. The harness warns about exactly this
  # -- "model.matrix() drops NA rows silently -- filter these before predicting,
  # or the prediction vector will not line up with the frame" -- and then dies in
  # vctrs with a recycle error rather than a legible one. Filter, and say how
  # many, because a silent row drop is how arms stop being comparable.
  feat <- grep("^(epr|psr|torp|elo|xelo).*_diff$|^(epr|psr|torp)\\.[xy]$",
               names(tm), value = TRUE)
  keep <- stats::complete.cases(tm[, feat, drop = FALSE])
  if (any(!keep)) {
    cli::cli_alert_warning(
      "{label}: dropping {sum(!keep)} row{?s} with incomplete rating features (unplayed fixtures).")
    tm <- tm[keep, , drop = FALSE]
  }
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams,
                           xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff",
                           cv_extra_feature_cols = "xelo_diff")
  # run_rolling_eval() returns gam_preds / xgb_preds / blend_preds /
  # input_blend_preds -- there is no `preds`. Production serves the 50/50 Input
  # Blend (torp/CLAUDE.md), so that is the arm to score.
  p <- as.data.table(roll$input_blend_preds)
  # .format_match_preds() emits BOTH orientations of every match (home row and a
  # mirrored away row). Absolute error, bits and Brier are all symmetric under
  # that mirror, so metrics are unaffected -- but n would be double and the
  # paired CI correspondingly half as wide as it should be. Dedupe.
  before <- nrow(p)
  p <- unique(p, by = "match_id")
  cli::cli_inform("{label}: {before} prediction rows -> {nrow(p)} unique matches")
  p[, arm := label]
  p
}

p2 <- run_arm("v2", r_v2)
p3 <- run_arm("v3", r_v3)

# Both arms must have been evaluated on the SAME games, or the comparison is
# between different questions rather than different engines.
common <- intersect(p2$match_id, p3$match_id)
say("")
say("--- same-games check: v2 ", nrow(p2), " rows | v3 ", nrow(p3),
    " rows | common ", length(common), " ---")
p2 <- p2[match_id %in% common]; p3 <- p3[match_id %in% common]

metrics <- function(p, label, seasons = NULL) {
  d <- if (is.null(seasons)) p else p[season %in% seasons]
  d <- d[is.finite(margin) & is.finite(pred_margin) & is.finite(pred_win)]
  hw <- ifelse(d$margin > 0, 1, ifelse(d$margin == 0, 0.5, 0))
  data.table(
    arm = label, window = if (is.null(seasons)) "pooled" else paste(seasons, collapse = ","),
    n = nrow(d),
    MAE   = round(mean(abs(d$pred_margin - d$margin)), 4),
    RMSE  = round(sqrt(mean((d$pred_margin - d$margin)^2)), 4),
    bits  = round(.bits(pmin(pmax(d$pred_win, 1e-6), 1 - 1e-6), hw), 4),
    Brier = round(mean((d$pred_win - hw)^2), 4),
    tips  = sum((d$pred_margin > 0) == (d$margin > 0), na.rm = TRUE)
  )
}

say("")
say("--- GATE 6: match metrics, ALL of them ---")
tbl <- rbind(
  metrics(p2, "v2"), metrics(p3, "v3"),
  metrics(p2, "v2", 2025), metrics(p3, "v3", 2025),
  metrics(p2, "v2", 2026), metrics(p3, "v3", 2026)
)
say_dt(tbl, 12)

say("")
say("--- paired difference on the POOLED window (v3 - v2) ---")
pp <- merge(p2[, .(match_id, m = margin, e2 = abs(pred_margin - margin), w2 = pred_win)],
            p3[, .(match_id, e3 = abs(pred_margin - margin), w3 = pred_win)],
            by = "match_id")
d <- pp$e3 - pp$e2
ci <- t.test(d)$conf.int
say("n paired ", nrow(pp))
say("dMAE ", round(mean(d), 4), "  95% CI [", round(ci[1], 4), ", ", round(ci[2], 4), "]",
    "   (positive = v3 WORSE)")
say("A CI spanning zero is not 'no evidence' -- read the point estimate and the")
say("width together, and weigh them against what v3 buys structurally.")

set.seed(1)
bs <- replicate(2000, mean(sample(d, length(d), TRUE)))
say("bootstrap P(v3 better) = ", round(mean(bs < 0), 3))

arrow::write_parquet(rbind(p2, p3), file.path(OUT_DIR, "epv3_match_preds.parquet"))
arrow::write_parquet(tbl, file.path(OUT_DIR, "epv3_match_metrics.parquet"))
say("")
say("wrote predictions and metrics to ", OUT_DIR)
close(con)
cat("\nDone\n")
