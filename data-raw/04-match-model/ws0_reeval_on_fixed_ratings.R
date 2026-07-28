# Re-measure the match model on CORRECTED ratings
# ===============================================
# Tonight's headline (MAE 24.895 / bits 0.2413 on 2026, clearing Aggregate's
# 24.93 / 0.2421) was measured on a team_mdl_df built from published ratings in
# which 2022 and 2025 were corrupted -- every player deflated as 1/wt_gms by the
# stale-vintage bug (docs/NEXT-STEPS.md). 216 of the 387 pooled matches, 56%,
# sat on those compressed features, and every arm trained on history including
# 2022.
#
# Arm-vs-arm comparisons are unaffected (all arms shared one team_mdl_df), so
# C6, the xScore swap and the Elo study all stand. The ABSOLUTE numbers do not,
# and the direction of the error is informative: the model was fed a degraded
# signal, so corrected ratings should help rather than hurt.
#
# Run AFTER the full-history regenerate has published corrected 2021-2025
# ratings. Rebuilds team_mdl_df from scratch (do NOT reuse the cached one --
# it carries the corrupted features) and re-runs the same rolling OOS
# evaluation, then prints the before/after against the Squiggle bars.
#
# Run: powershell.exe -Command 'Rscript "<this file>"'

suppressMessages({
  library(dplyr); library(data.table)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RES <- file.path(EXP, "results")
source(file.path(EXP, "rolling_lib.R"))
source(file.path(EXP, "signal_gate.R"))

TEST_SEASONS <- 2025:2026
DF  <- file.path(RES, "fixed_team_mdl_df.rds")
PR  <- file.path(RES, "fixed_roll_pooled.rds")

# --- sanity gate: refuse to run on still-corrupted ratings --------------------
cli::cli_h1("Checking the published ratings are actually fixed")
tr <- as.data.table(load_torp_ratings())
traj <- tr[!is.na(epr), .(sd = sd(epr)), by = .(season, round)][order(season, round)]
bad <- traj[, .(first = sd[1], last = sd[.N],
                ratio = sd[.N] / sd[1]), by = season][order(season)]
print(bad, row.names = FALSE)
# The corruption signature is a season whose dispersion COLLAPSES: healthy
# seasons rise gently as evidence accumulates and shrinkage weakens.
still_bad <- bad[ratio < 0.8]
if (nrow(still_bad) > 0) {
  cli::cli_abort(c(
    "Season{?s} {.val {as.character(still_bad$season)}} still collapse{?s/} through the year (end/start < 0.8).",
    "x" = "The regenerate has not landed, or did not fix them -- re-measuring now would just reproduce the old numbers.",
    "i" = "Expected healthy behaviour: dispersion rises gently within a season."
  ))
}
cli::cli_alert_success("All seasons show healthy within-season dispersion")

# --- rebuild + evaluate -------------------------------------------------------
cli::cli_h1("Rebuilding team_mdl_df on corrected ratings")
t0 <- Sys.time()
team_mdl_df <- build_team_mdl_df()
cli::cli_inform("built in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
saveRDS(team_mdl_df, DF)

cat("\nteam-level epr SD by season (corrupted run was 9.06 in 2022, 10.03 in 2025;\n")
cat("clean seasons ran 13.6-17.1):\n")
print(as.data.table(team_mdl_df)[!is.na(epr.x),
      .(n = .N, epr_sd = round(sd(epr.x), 3)), by = season.x][order(season.x)],
      row.names = FALSE)

cli::cli_h1("Rolling OOS evaluation, {TEST_SEASONS}")
t0 <- Sys.time()
roll <- run_rolling_eval(team_mdl_df, test_seasons = TEST_SEASONS,
                         gam_trainer = .train_match_gams,
                         xgb_trainer = .train_xgb_fixed,
                         extra_feature_cols = "xelo_diff",
                         cv_extra_feature_cols = "xelo_diff")
cli::cli_inform("eval took {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
saveRDS(roll, PR)

ib <- roll$input_blend_preds
calib <- tryCatch(fit_match_margin_calibration(team_mdl_df), error = function(e) NULL)
if (!is.null(calib)) ib$pred_recal <- apply_match_margin_calibration(ib$pred_margin, calib)

.bits <- function(pw, hw) mean(ifelse(hw == 1, 1 + log2(pw),
                              ifelse(hw == 0, 1 + log2(1 - pw),
                                     1 + 0.5 * log2(pw * (1 - pw)))))
report <- function(p, lab) {
  cat(sprintf("%-22s MAE %.3f | RMSE %.3f | bits %.4f | Brier %.4f | cor %.3f | slope %.3f\n",
              lab, mean(abs(p$pm - p$margin)), sqrt(mean((p$pm - p$margin)^2)),
              .bits(p$pred_win, p$home_win), mean((p$pred_win - p$home_win)^2),
              cor(p$pm, p$margin), unname(coef(lm(p$margin ~ p$pm))[2])))
}

cli::cli_h1("2026 -- the number that must be re-measured")
p26 <- ib[ib$season == 2026, ]
report(within(p26, pm <- pred_margin), "no recal")
if (!is.null(calib)) report(within(p26, pm <- pred_recal), "with recal")

cat("\n--- reference ---\n")
cat("BEFORE (corrupted ratings) : MAE 24.895 | bits 0.2413\n")
cat("Aggregate (the bar)        : MAE 24.93  | bits 0.2421\n")
cat("Wheelo (leaderboard #1)    : MAE 24.43  | bits 0.2418\n")
cat("live submitted (pre-C6)    : MAE 26.49  | bits 0.2299\n")

cli::cli_h1("Pooled {TEST_SEASONS}")
report(within(ib, pm <- pred_margin), "pooled, no recal")
if (!is.null(calib)) report(within(ib, pm <- pred_recal), "pooled, with recal")
cat("\nBEFORE (corrupted): pooled MAE 25.586 / bits 0.2479\n")

cli::cli_alert_success("Saved {basename(DF)} and {basename(PR)}")
