# Re-validates the C6 production port (V4b GAM formula + elo_diff feature +
# match_margin_calibration) against the CURRENT state of torp/torpmodels dev.
# Originally run 2026-07-14 (reference numbers below), producing MAE=25.586 /
# slope=1.004 -- but that run used team_elo.R's build_team_elo() BEFORE the
# Elo MOV multiplier fix (abs() -> winner-relative, commit ab2a548 in torp),
# which changes every elo_diff value team_mdl_df carries. MUST be re-run
# (this script rebuilds team_mdl_df from scratch, so it picks up the fix
# automatically) before treating any MAE/slope number as current. See
# docs/plans/FABLE-C6-SHIP-PLAN.md for the full ship checklist this feeds.
suppressPackageStartupMessages({
  devtools::load_all("C:/dev/torpverse/torpmodels", quiet = TRUE)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
  library(dplyr); library(xgboost); library(mgcv); library(cli); library(MLmetrics)
})
source("C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/rolling_lib.R")

cli::cli_h1("Rebuilding team_mdl_df fresh (need elo_diff, not in the old cache)")
team_mdl_df <- build_team_mdl_df()
cat("elo_diff present:", "elo_diff" %in% names(team_mdl_df), "\n")
cat("elo_diff summary:\n")
print(summary(team_mdl_df$elo_diff))
saveRDS(team_mdl_df, "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/results/team_mdl_df_cache_with_elo.rds")

cli::cli_h1("Rolling eval: PRODUCTION .train_match_gams (V4b+elo formula) + .train_xgb_fixed(extra_feature_cols='elo_diff')")
cli::cli_inform("NOTE: production's .train_match_xgb() (used for the nrounds CV pre-opt step) now")
cli::cli_inform("always includes elo_diff -- this differs from round-1 C6's validated elo-BLIND CV.")

roll <- run_rolling_eval(
  team_mdl_df, test_seasons = 2025:2026,
  gam_trainer = .train_match_gams,   # production, unqualified -- picks up my edits
  xgb_trainer = .train_xgb_fixed,    # rolling_lib.R's own (unedited)
  extra_feature_cols = "elo_diff"
)

m_ib <- .compute_metrics(roll$input_blend_preds)
cat(sprintf("\nProduction port (V4b+elo, NO recal), pooled 2025:2026 (n=%d):\n", nrow(roll$input_blend_preds)))
cat(sprintf("MAE=%.3f RMSE=%.3f Brier=%.4f Bits=%.4f Slope=%.3f\n",
            m_ib$mae, m_ib$rmse, m_ib$brier, m_ib$bits, m_ib$slope))

cat("\nReference points:\n")
cat("Original champion (no changes):     MAE=26.026 Brier=0.1766 Slope=0.920\n")
cat("Round-1 C6 (V4b+elo+V1a recal):     MAE=25.545 Brier=0.1741 Bits=0.2471 Slope=0.955\n")
cat("Round-2 EloRetune (elo-aware CV,\n  V4b+elo+V1a recal):                MAE=25.726 Brier=0.1744 Bits=0.2449 Slope=0.960\n")

# Now apply a production-style single-holdout recalibration on top, using the
# EARLIEST test round's pooled predictions as a stand-in for "what a
# temporal-holdout fit would produce applied forward" (approximate check --
# full fidelity requires actually calling fit_match_margin_calibration() on
# real team_mdl_df, done separately below).
cat("\n=== Applying production fit_match_margin_calibration() for real ===\n")
calib <- fit_match_margin_calibration(team_mdl_df)
cat(sprintf("Fitted: b=%.4f, raw OOS slope=%.4f, n_oos=%d, holdout_season=%s\n",
            calib$b, calib$slope_raw, calib$n_oos, calib$holdout_season))

# Apply that SAME b to the whole pooled rolling predictions (approximation --
# production would only apply it going forward from when it was fit, but this
# gives a directional read on the calibration lever's magnitude)
recal_preds <- roll$input_blend_preds
recal_preds$pred_margin <- apply_match_margin_calibration(recal_preds$pred_margin, calib)
m_recal <- .compute_metrics(recal_preds)
cat(sprintf("\nWith production single-holdout recal applied (approx, same b throughout): MAE=%.3f Slope=%.3f\n",
            m_recal$mae, m_recal$slope))

cat("\nDONE\n")
