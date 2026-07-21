# run_all_variants.R -- FABLE-WP-SHARPNESS-PLAN.md S1-S5 batch, step 1
# =====================================================================
# Runs baseline + every S1-S5 variant sequentially, using the base
# EP/WP frames cached by prep_base_data.R (must be run first). Each
# variant's full result (preds/labels/meta + scorecard) is cached to
# results/<variant>.rds as it completes, and the running results table
# is written to results/RESULTS_TABLE.csv after every variant so nothing
# is lost if the batch is interrupted.
#
# Run via PowerShell (arrow segfaults under Git Bash R):
#   powershell.exe -Command 'Rscript "C:/dev/torpverse/torpmodels/data-raw/experiments-wp-sharpness/run_all_variants.R"'

library(devtools)
library(tidyverse)
library(zoo)
library(janitor)
library(lubridate)
library(xgboost)
library(mgcv)
library(cli)

here_dir <- "C:/dev/torpverse/torpmodels/data-raw/experiments-wp-sharpness"
results_dir <- file.path(here_dir, "results")

torp_path <- "C:/dev/torpverse/torp"
devtools::load_all(torp_path)
torpmodels_path <- "C:/dev/torpverse/torpmodels"
devtools::load_all(torpmodels_path)

source(file.path(torpmodels_path, "data-raw", "lib", "train_lib.R"))
source(file.path(here_dir, "wp_sharpness_lib.R"))

log_file <- file.path(results_dir, "run_all_variants_log.txt")
sink_con <- file(log_file, open = "wt")
sink(sink_con, split = TRUE)
sink(sink_con, type = "message", append = TRUE)

cli::cli_h1("WP sharpness batch: S1-S5 variants")
cli::cli_inform("Started: {Sys.time()}")

stopifnot(file.exists(file.path(results_dir, "model_data_wp_train_base.rds")))
model_data_wp_train_base <- readRDS(file.path(results_dir, "model_data_wp_train_base.rds"))
gate_wp_data_base <- readRDS(file.path(results_dir, "gate_wp_data_base.rds"))
pace_rows_per_min <- readRDS(file.path(results_dir, "pace_rows_per_min.rds"))
prep_meta <- readRDS(file.path(results_dir, "prep_meta.rds"))
cli::cli_inform("Base frames loaded: train {nrow(model_data_wp_train_base)} rows, gate {nrow(gate_wp_data_base)} rows")

# S3 frames (features added once, reused by the S3 variant + monotonicity checks)
train_s3 <- add_decisiveness_features(model_data_wp_train_base, pace_rows_per_min)
gate_s3  <- add_decisiveness_features(gate_wp_data_base, pace_rows_per_min)

results_table <- list()
run_log <- list()

save_and_record <- function(res, group, notes = "") {
  saveRDS(res, file.path(results_dir, paste0(res$variant, ".rds")))
  sc <- wp_scorecard(res$preds_gate, res$labels_gate, res$meta_gate)
  row <- data.frame(
    group = group, variant = res$variant,
    slope_all = round(sc$slope_all, 4), slope_q4close = round(sc$slope_q4close, 4),
    se_q4close = round(sc$se_q4close, 4), n_q4close = sc$n_q4close,
    logloss = round(sc$logloss, 6), brier = round(sc$brier, 6),
    n_train = res$n_train, optimal_nrounds = res$optimal_nrounds,
    cv_logloss = round(res$cv_logloss, 6), elapsed_min = round(res$elapsed_min, 2),
    monotone_constraints = res$monotone_constraints, notes = notes,
    stringsAsFactors = FALSE
  )
  results_table[[res$variant]] <<- row
  utils::write.csv(dplyr::bind_rows(results_table), file.path(results_dir, "RESULTS_TABLE.csv"), row.names = FALSE)
  cli::cli_alert_success(
    "[{res$variant}] slope_all={round(sc$slope_all,3)} slope_q4close={round(sc$slope_q4close,3)} (n={sc$n_q4close}) logloss={round(sc$logloss,5)} brier={round(sc$brier,5)}"
  )
  row
}

# ---- Baseline ---------------------------------------------------------

res_baseline <- run_wp_variant(
  "baseline", model_data_wp_train_base, gate_wp_data_base
)
save_and_record(res_baseline, "baseline")

mono_baseline <- monotonicity_violation_rate(gate_wp_data_base, res_baseline$model, torp:::WP_MODEL_FEATURES)
cli::cli_inform("[baseline] monotonicity violation rate (Q4/close, points_diff perturbation): {round(mono_baseline$violation_rate, 4)} ({mono_baseline$n_pairs} pairs, {mono_baseline$n_rows} rows)")
saveRDS(mono_baseline, file.path(results_dir, "mono_baseline.rds"))

# ---- S1: leverage weights, binary form, k in {1,3,7} -------------------

for (k in c(1, 3, 7)) {
  variant <- paste0("S1_leverage_binary_k", k)
  res <- run_wp_variant(
    variant, model_data_wp_train_base, gate_wp_data_base,
    weight_fn = wp_weight_leverage_binary(k)
  )
  save_and_record(res, "S1", notes = paste0("weight = 1 + ", k, "*is_q4close"))
}

# S1 bonus: continuous urgency form (cheap given the EP-cache reuse) at k=3
variant <- "S1_leverage_continuous_k3"
res <- run_wp_variant(
  variant, model_data_wp_train_base, gate_wp_data_base,
  weight_fn = wp_weight_leverage_continuous(3)
)
save_and_record(res, "S1", notes = "weight = 1 + 3*|score_urgency| normalized to [0,1] via p99 cap (continuous form, bonus)")

# ---- S2: monotone-constraint ablation (a) drop entirely -----------------

variant <- "S2_drop_constraints"
res_s2 <- run_wp_variant(
  variant, model_data_wp_train_base, gate_wp_data_base,
  drop_constraints = TRUE
)
save_and_record(res_s2, "S2", notes = "all 5 monotone constraints dropped (unconstrained)")

mono_s2 <- monotonicity_violation_rate(gate_wp_data_base, res_s2$model, torp:::WP_MODEL_FEATURES)
cli::cli_inform("[S2_drop_constraints] monotonicity violation rate (Q4/close, points_diff perturbation): {round(mono_s2$violation_rate, 4)} ({mono_s2$n_pairs} pairs, {mono_s2$n_rows} rows) -- vs baseline {round(mono_baseline$violation_rate, 4)}")
saveRDS(mono_s2, file.path(results_dir, "mono_s2.rds"))

# ---- S3: engineered decisiveness features --------------------------------

variant <- "S3_decisiveness_features"
res_s3 <- run_wp_variant(
  variant, train_s3, gate_s3,
  extra_features = S3_EXTRA_FEATURES, extra_monotone = S3_EXTRA_MONOTONE
)
save_and_record(res_s3, "S3", notes = paste0("pace_rows_per_min=", round(pace_rows_per_min, 3), "; features: ", paste(S3_EXTRA_FEATURES, collapse = ", ")))

# ---- S4: recency weighting, half-life in {1, 2, 4} seasons ---------------

for (hl in c(1, 2, 4)) {
  variant <- paste0("S4_recency_hl", hl)
  res <- run_wp_variant(
    variant, model_data_wp_train_base, gate_wp_data_base,
    weight_fn = wp_weight_recency(hl)
  )
  save_and_record(res, "S4", notes = paste0("half-life = ", hl, " season(s), anchored at max training season"))
}

# ---- S5: draw-row handling ------------------------------------------------

variant <- "S5_exclude_draws"
res_s5a <- run_wp_variant(
  variant, model_data_wp_train_base, gate_wp_data_base,
  exclude_draws_train = TRUE
)
save_and_record(res_s5a, "S5", notes = "draw-labeled (0.5) rows dropped from training entirely")

variant <- "S5_downweight_draws_0.5"
res_s5b <- run_wp_variant(
  variant, model_data_wp_train_base, gate_wp_data_base,
  weight_fn = wp_weight_downweight_draws(0.5)
)
save_and_record(res_s5b, "S5", notes = "draw-labeled (0.5) rows kept but down-weighted to 0.5")

cli::cli_h1("Batch complete")
cli::cli_inform("Finished: {Sys.time()}")
print(dplyr::bind_rows(results_table))

sink(type = "message")
sink()
