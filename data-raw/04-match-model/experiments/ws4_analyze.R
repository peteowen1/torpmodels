# ws4_analyze.R — Recovery/analysis script for WS4, reads whichever
# ws4_roll_*.rds files exist under experiments/results/ and reports metrics.
# No torp load, no build_team_mdl_df() -- just reads saved run_rolling_eval()
# outputs. Safe to run repeatedly while ws4_formula_variants.R is (or was)
# running in the background.

library(dplyr)
library(MLmetrics)
library(cli)

RESULTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/results"
source("C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/rolling_lib.R")

variant_files <- c(
  Baseline = "ws4_roll_baseline.rds",
  V4a      = "ws4_roll_v4a.rds",
  V4b      = "ws4_roll_v4b.rds",
  V4c      = "ws4_roll_v4c.rds"
)

variants <- list()
for (label in names(variant_files)) {
  fp <- file.path(RESULTS_DIR, variant_files[[label]])
  if (file.exists(fp)) {
    variants[[label]] <- readRDS(fp)
    cli::cli_alert_success("Loaded {label} ({nrow(variants[[label]]$gam_preds)} matches)")
  } else {
    cli::cli_alert_warning("Missing {label} ({fp}) -- not yet completed")
  }
}

if (length(variants) == 0) {
  cli::cli_abort("No ws4_roll_*.rds files found yet in {RESULTS_DIR}")
}

metrics_table <- purrr::imap_dfr(variants, function(roll, label) {
  gam_m <- .compute_metrics(roll$gam_preds)
  ib_m  <- .compute_metrics(roll$input_blend_preds)
  dplyr::bind_rows(
    data.frame(Variant = label, Model = "GAM-only", N = nrow(roll$gam_preds),
               MAE = gam_m$mae, RMSE = gam_m$rmse, Brier = gam_m$brier,
               Slope = gam_m$slope, Cor = gam_m$cor, SDRatio = gam_m$sd_ratio,
               CloseMAE = gam_m$close_mae, CloseN = gam_m$close_n),
    data.frame(Variant = label, Model = "Input Blend", N = nrow(roll$input_blend_preds),
               MAE = ib_m$mae, RMSE = ib_m$rmse, Brier = ib_m$brier,
               Slope = ib_m$slope, Cor = ib_m$cor, SDRatio = ib_m$sd_ratio,
               CloseMAE = ib_m$close_mae, CloseN = ib_m$close_n)
  )
})
metrics_table[, -(1:2)] <- round(metrics_table[, -(1:2)], 4)

cat("\n=== WS4 Screening Results (2026 rolling OOS) -- as of this read ===\n")
print(metrics_table, row.names = FALSE)
write.csv(metrics_table, file.path(RESULTS_DIR, "ws4_metrics_2026.csv"), row.names = FALSE)

if ("Baseline" %in% names(variants)) {
  base_gam_m <- .compute_metrics(variants[["Baseline"]]$gam_preds)
  base_ib_m  <- .compute_metrics(variants[["Baseline"]]$input_blend_preds)
  cat(sprintf("\nBaseline GAM-only slope=%.3f cor=%.3f sd_ratio=%.3f MAE=%.2f | Input Blend slope=%.3f MAE=%.2f\n",
              base_gam_m$slope, base_gam_m$cor, base_gam_m$sd_ratio, base_gam_m$mae,
              base_ib_m$slope, base_ib_m$mae))

  cat("\n=== boot_mae_diff() vs Baseline ===\n")
  for (v in setdiff(names(variants), "Baseline")) {
    b_gam <- boot_mae_diff(variants[[v]]$gam_preds, variants[["Baseline"]]$gam_preds, B = 2000)
    b_ib  <- boot_mae_diff(variants[[v]]$input_blend_preds, variants[["Baseline"]]$input_blend_preds, B = 2000)
    v_gam_m <- .compute_metrics(variants[[v]]$gam_preds)
    v_ib_m  <- .compute_metrics(variants[[v]]$input_blend_preds)
    cat(sprintf(
      "%s  GAM-only slope=%.3f (d=%+.3f)  dMAE=%+.3f 95%%CI[%+.3f,%+.3f]  | InputBlend slope=%.3f (d=%+.3f) dMAE=%+.3f 95%%CI[%+.3f,%+.3f]\n",
      v, v_gam_m$slope, v_gam_m$slope - base_gam_m$slope, b_gam$mae_diff, b_gam$mae_ci[1], b_gam$mae_ci[2],
      v_ib_m$slope, v_ib_m$slope - base_ib_m$slope, b_ib$mae_diff, b_ib$mae_ci[1], b_ib$mae_ci[2]
    ))
  }
} else {
  cli::cli_alert_warning("No baseline yet -- cannot compute deltas/bootstrap CIs")
}

cat("\n=== Margin Calibration by Predicted-Margin Bucket (GAM-only) ===\n")
bucket_table <- purrr::imap_dfr(variants, function(roll, label) {
  margin_calibration_by_pred_bucket(roll$gam_preds) |> dplyr::mutate(Variant = label, .before = 1)
})
print(as.data.frame(bucket_table), row.names = FALSE)
write.csv(bucket_table, file.path(RESULTS_DIR, "ws4_bucket_2026.csv"), row.names = FALSE)
