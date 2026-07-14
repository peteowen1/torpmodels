# ws3_run.R — WS3 driver: build data, run baseline + cv_stack, report.
# See ws3_cv_stack.R for the trainer + hypothesis. Run via PowerShell
# (arrow segfaults under Git Bash R):
#   Rscript "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/ws3_run.R"

library(tidyverse)
library(xgboost)
library(mgcv)
library(MLmetrics)
library(geosphere)
library(cli)
library(withr)

torp_paths <- c("../torp", "../../torp", "../../../torp", "C:/dev/torpverse/torp")
torp_loaded <- FALSE
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p)
    torp_loaded <- TRUE
    break
  }
}
if (!torp_loaded) stop("Cannot find torp package (dev). Run from torpverse workspace.")

this_dir_candidates <- c(".", "experiments", "04-match-model/experiments",
                          "data-raw/04-match-model/experiments",
                          "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments")
find_file <- function(fname) {
  hits <- file.path(this_dir_candidates, fname)
  hits <- hits[file.exists(hits)]
  if (length(hits) == 0) stop("Cannot find ", fname)
  hits[1]
}
source(find_file("rolling_lib.R"))
source(find_file("ws3_cv_stack.R"))

cli::cli_h1("WS3: Building match model dataset")
tictoc::tic("build_data")
team_mdl_df <- build_team_mdl_df()
tictoc::toc()
cli::cli_inform("Seasons: {paste(sort(unique(team_mdl_df$season.x)), collapse = ', ')}")

# ---- Step 0: sanity-check the cross-fit wiring is not a silent no-op ----
# Fit stage 1 in-sample (as .train_match_gams() does today) vs cross-fitted
# (K=3) on the SAME snapshot of completed matches, and confirm the two
# differ materially (cor < 1, not a byte-identical pass-through).
cli::cli_h1("Step 0: cross-fit wiring sanity check (stage 1, all completed matches)")
chk_mask <- !is.na(team_mdl_df$win)
chk_df <- team_mdl_df[chk_mask, ]
m1_formula_chk <- stats::as.formula(
  paste(
    "total_xpoints_adj ~",
    "s(team_type_fac, bs = \"re\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    "+ s(abs(epr_diff), bs = \"ts\", k = 5)",
    "+ s(abs(torp_diff), bs = \"ts\", k = 5)",
    "+ s(log_dist_diff, bs = \"ts\", k = 5)",
    "+ s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
)
fit_insample_chk <- mgcv::bam(
  m1_formula_chk, data = chk_df, weights = chk_df$weightz, family = gaussian(),
  nthreads = 4L, select = TRUE, discrete = TRUE, drop.unused.levels = FALSE, gamma = 1.4
)
insample_pred_chk <- predict(fit_insample_chk, newdata = chk_df, type = "response")

match_ids_chk <- unique(chk_df$match_id)
set.seed(1234L)
fold_lookup_chk <- stats::setNames(
  sample(rep(seq_len(3L), length.out = length(match_ids_chk))),
  as.character(match_ids_chk)
)
chk_df$.fold <- unname(fold_lookup_chk[as.character(chk_df$match_id)])
cv_pred_chk <- .cross_fit_stage(
  m1_formula_chk, chk_df, chk_df$weightz, gaussian(), chk_df$.fold, 3L, 4L, 1.4
)
cor_check <- stats::cor(insample_pred_chk, cv_pred_chk)
cli::cli_inform("In-sample vs cross-fitted stage-1 pred: cor = {round(cor_check, 4)}, mean|diff| = {round(mean(abs(insample_pred_chk - cv_pred_chk)), 3)}")
if (cor_check > 0.9999) {
  cli::cli_abort("Cross-fit wiring looks like a no-op (cor > 0.9999 vs in-sample) — STOP, do not trust downstream results.")
} else {
  cli::cli_alert_success("Cross-fit wiring confirmed non-trivial (cor = {round(cor_check, 4)} < 1).")
}
rm(chk_df, chk_mask, fit_insample_chk, insample_pred_chk, fold_lookup_chk, cv_pred_chk, match_ids_chk)

# ---- Step 1: 2026 screen — baseline vs cv_stack ----
cli::cli_h1("Step 1: 2026 screen")
TEST_SEASONS <- 2026

cli::cli_h2("Baseline (default .train_match_gams)")
t_base <- system.time(
  roll_base <- run_rolling_eval(team_mdl_df, TEST_SEASONS)
)
cli::cli_inform("Baseline 2026 screen wall time: {round(t_base[['elapsed']], 1)}s")

cli::cli_h2("cv_stack (K=3, stages 1-3 cross-fit)")
t_cv <- system.time(
  roll_cv <- run_rolling_eval(team_mdl_df, TEST_SEASONS, gam_trainer = .train_match_gams_cv_stack)
)
cli::cli_inform("cv_stack 2026 screen wall time: {round(t_cv[['elapsed']], 1)}s")
runtime_ratio <- t_cv[["elapsed"]] / t_base[["elapsed"]]
cli::cli_inform("Runtime ratio (cv_stack / baseline): {round(runtime_ratio, 2)}x")

# ---- Metrics: GAM-only (mechanism) and Input Blend (ship comparison) ----
m_base_gam <- .compute_metrics(roll_base$gam_preds)
m_cv_gam   <- .compute_metrics(roll_cv$gam_preds)
m_base_ib  <- .compute_metrics(roll_base$input_blend_preds)
m_cv_ib    <- .compute_metrics(roll_cv$input_blend_preds)

metric_names <- c("mae", "rmse", "brier", "slope", "intercept", "cor", "sd_ratio", "close_mae")
screen_table <- data.frame(
  Variant = c("Baseline GAM", "cv_stack GAM", "Baseline Input Blend", "cv_stack Input Blend"),
  N = c(nrow(roll_base$gam_preds), nrow(roll_cv$gam_preds), nrow(roll_base$input_blend_preds), nrow(roll_cv$input_blend_preds)),
  do.call(rbind, lapply(list(m_base_gam, m_cv_gam, m_base_ib, m_cv_ib), function(m) {
    as.data.frame(m[metric_names])
  })),
  stringsAsFactors = FALSE
)
screen_table[metric_names] <- round(screen_table[metric_names], 4)

cat("\n=== WS3 2026 Screen: Baseline vs cv_stack ===\n")
print(screen_table, row.names = FALSE)

b_base_gam <- .pooled_through_origin_b(roll_base$gam_preds)
b_cv_gam   <- .pooled_through_origin_b(roll_cv$gam_preds)
b_base_ib  <- .pooled_through_origin_b(roll_base$input_blend_preds)
b_cv_ib    <- .pooled_through_origin_b(roll_cv$input_blend_preds)

cat("\n=== WS1-style pooled through-origin b (lm(margin ~ pred_margin + 0)) — 2026 ===\n")
cat(sprintf("Baseline GAM:          b = %.3f\n", b_base_gam))
cat(sprintf("cv_stack GAM:          b = %.3f\n", b_cv_gam))
cat(sprintf("Baseline Input Blend:  b = %.3f\n", b_base_ib))
cat(sprintf("cv_stack Input Blend:  b = %.3f\n", b_cv_ib))

boot_ib_2026 <- boot_mae_diff(roll_cv$input_blend_preds, roll_base$input_blend_preds, B = 2000)
cat("\n=== boot_mae_diff (Input Blend): cv_stack - baseline, 2026 ===\n")
cat(sprintf("N=%d  MAE diff=%.3f  95%% CI [%.3f, %.3f]  Brier diff=%.5f  95%% CI [%.5f, %.5f]\n",
            boot_ib_2026$n_matches, boot_ib_2026$mae_diff, boot_ib_2026$mae_ci[1], boot_ib_2026$mae_ci[2],
            boot_ib_2026$brier_diff, boot_ib_2026$brier_ci[1], boot_ib_2026$brier_ci[2]))

# Persist screen results
results_dir <- file.path(dirname(find_file("ws3_cv_stack.R")), "results")
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)
saveRDS(list(
  screen_table = screen_table,
  through_origin_b = c(base_gam = b_base_gam, cv_gam = b_cv_gam, base_ib = b_base_ib, cv_ib = b_cv_ib),
  boot_ib_2026 = boot_ib_2026,
  t_base = t_base[["elapsed"]], t_cv = t_cv[["elapsed"]], runtime_ratio = runtime_ratio
), file.path(results_dir, "ws3_screen_2026.rds"))
cli::cli_alert_success("Screen results saved to {file.path(results_dir, 'ws3_screen_2026.rds')}")

# ---- Step 2: decide on confirmation ----
# Plan: confirm any real candidate on 2025:2026. "Real candidate" here means
# cv_stack's slope moved meaningfully toward 1 and/or MAE improved without
# regression. Always attempt the confirmation run regardless of the 2026
# verdict, per the task instruction ("confirm on full 2025:2026 regardless"),
# subject to a runtime guard.
cli::cli_h1("Step 2: 2025:2026 confirmation")
if (runtime_ratio > 8) {
  cli::cli_alert_warning("Runtime ratio {round(runtime_ratio,1)}x is very high — skipping full 2025:2026 confirmation to stay within budget. Screen-only result stands.")
} else {
  TEST_SEASONS <- 2025:2026
  cli::cli_h2("Baseline (2025:2026)")
  t_base_full <- system.time(
    roll_base_full <- run_rolling_eval(team_mdl_df, TEST_SEASONS)
  )
  cli::cli_inform("Baseline 2025:2026 wall time: {round(t_base_full[['elapsed']], 1)}s")

  cli::cli_h2("cv_stack (2025:2026)")
  t_cv_full <- system.time(
    roll_cv_full <- run_rolling_eval(team_mdl_df, TEST_SEASONS, gam_trainer = .train_match_gams_cv_stack)
  )
  cli::cli_inform("cv_stack 2025:2026 wall time: {round(t_cv_full[['elapsed']], 1)}s")

  m_base_gam_full <- .compute_metrics(roll_base_full$gam_preds)
  m_cv_gam_full   <- .compute_metrics(roll_cv_full$gam_preds)
  m_base_ib_full  <- .compute_metrics(roll_base_full$input_blend_preds)
  m_cv_ib_full    <- .compute_metrics(roll_cv_full$input_blend_preds)

  confirm_table <- data.frame(
    Variant = c("Baseline GAM", "cv_stack GAM", "Baseline Input Blend", "cv_stack Input Blend"),
    N = c(nrow(roll_base_full$gam_preds), nrow(roll_cv_full$gam_preds), nrow(roll_base_full$input_blend_preds), nrow(roll_cv_full$input_blend_preds)),
    do.call(rbind, lapply(list(m_base_gam_full, m_cv_gam_full, m_base_ib_full, m_cv_ib_full), function(m) {
      as.data.frame(m[metric_names])
    })),
    stringsAsFactors = FALSE
  )
  confirm_table[metric_names] <- round(confirm_table[metric_names], 4)

  cat("\n=== WS3 2025:2026 Confirmation: Baseline vs cv_stack ===\n")
  print(confirm_table, row.names = FALSE)

  b_base_gam_full <- .pooled_through_origin_b(roll_base_full$gam_preds)
  b_cv_gam_full   <- .pooled_through_origin_b(roll_cv_full$gam_preds)
  b_base_ib_full  <- .pooled_through_origin_b(roll_base_full$input_blend_preds)
  b_cv_ib_full    <- .pooled_through_origin_b(roll_cv_full$input_blend_preds)

  cat("\n=== WS1-style pooled through-origin b — 2025:2026 ===\n")
  cat(sprintf("Baseline GAM:          b = %.3f\n", b_base_gam_full))
  cat(sprintf("cv_stack GAM:          b = %.3f\n", b_cv_gam_full))
  cat(sprintf("Baseline Input Blend:  b = %.3f\n", b_base_ib_full))
  cat(sprintf("cv_stack Input Blend:  b = %.3f\n", b_cv_ib_full))

  boot_ib_full <- boot_mae_diff(roll_cv_full$input_blend_preds, roll_base_full$input_blend_preds, B = 2000)
  cat("\n=== boot_mae_diff (Input Blend): cv_stack - baseline, 2025:2026 (G3 ship gate) ===\n")
  cat(sprintf("N=%d  MAE diff=%.3f  95%% CI [%.3f, %.3f]  Brier diff=%.5f  95%% CI [%.5f, %.5f]\n",
              boot_ib_full$n_matches, boot_ib_full$mae_diff, boot_ib_full$mae_ci[1], boot_ib_full$mae_ci[2],
              boot_ib_full$brier_diff, boot_ib_full$brier_ci[1], boot_ib_full$brier_ci[2]))
  ci_excludes_zero <- boot_ib_full$mae_ci[1] > 0 || boot_ib_full$mae_ci[2] < 0
  brier_ok <- boot_ib_full$brier_diff <= 0.002
  cat(sprintf("\nG3 ship gate: CI excludes 0 = %s | Brier does not worsen by >0.002 = %s\n",
              ci_excludes_zero, brier_ok))

  saveRDS(list(
    confirm_table = confirm_table,
    through_origin_b = c(base_gam = b_base_gam_full, cv_gam = b_cv_gam_full, base_ib = b_base_ib_full, cv_ib = b_cv_ib_full),
    boot_ib_full = boot_ib_full,
    t_base = t_base_full[["elapsed"]], t_cv = t_cv_full[["elapsed"]],
    ci_excludes_zero = ci_excludes_zero, brier_ok = brier_ok
  ), file.path(results_dir, "ws3_confirm_2025_2026.rds"))
  cli::cli_alert_success("Confirmation results saved to {file.path(results_dir, 'ws3_confirm_2025_2026.rds')}")
}

cli::cli_alert_success("WS3 run complete.")
