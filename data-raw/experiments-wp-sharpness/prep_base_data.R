# prep_base_data.R -- FABLE-WP-SHARPNESS-PLAN.md S1-S5 batch, step 0
# =====================================================================
# Fits EP ONCE (identical across every S1-S5 WP variant -- none of them
# touch EP) on seasons < gate_season, builds the base WP train frame
# (build_wp_data() output, pre feature/weight/filter hooks) and the base
# WP gate-season frame (scored through the SAME temporal EP model), and
# caches everything to results/*.rds so run_all_variants.R never repeats
# the expensive chain-load + EP-CV + EP-OOS-refit step.
#
# Run via PowerShell (arrow segfaults under Git Bash R):
#   powershell.exe -Command 'Rscript "C:/dev/torpverse/torpmodels/data-raw/experiments-wp-sharpness/prep_base_data.R"'

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
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)

torp_path <- "C:/dev/torpverse/torp"
stopifnot(file.exists(file.path(torp_path, "DESCRIPTION")))
cli::cli_inform("Loading dev torp from {torp_path}...")
devtools::load_all(torp_path)

torpmodels_path <- "C:/dev/torpverse/torpmodels"
cli::cli_inform("Loading dev torpmodels from {torpmodels_path}...")
devtools::load_all(torpmodels_path)

source(file.path(torpmodels_path, "data-raw", "lib", "train_lib.R"))
source(file.path(here_dir, "wp_sharpness_lib.R"))

log_file <- file.path(results_dir, "prep_log.txt")
sink_con <- file(log_file, open = "wt")
sink(sink_con, split = TRUE)
sink(sink_con, type = "message", append = TRUE)

cli::cli_h1("WP sharpness batch: prep base EP+WP data (S1-S5)")
cli::cli_inform("Started: {Sys.time()}")

seasons <- default_training_seasons()
gate_season <- 2025L
cli::cli_inform("default_training_seasons(): {min(seasons)}-{max(seasons)} | gate_season fixed at {gate_season} per the plan")
stopifnot(gate_season %in% seasons)

# ---- Load + split -----------------------------------------------------

model_data_epv <- load_training_pbp(seasons)
cli::cli_inform("Loaded model_data_epv: {nrow(model_data_epv)} rows, {length(unique(model_data_epv$torp_match_id))} matches")
saveRDS(model_data_epv, file.path(results_dir, "model_data_epv.rds"))

data <- model_data_epv
if (!"season" %in% names(data)) {
  data$season <- as.numeric(substr(data$match_id, 5, 8))
}
train_data <- data |> dplyr::filter(.data$season < gate_season)
gate_data  <- data |> dplyr::filter(.data$season == gate_season)
cli::cli_inform(
  "train_data: {nrow(train_data)} rows, seasons {min(train_data$season)}-{max(train_data$season)} | gate_data: {nrow(gate_data)} rows, season {gate_season}"
)
stopifnot(nrow(train_data) > 0, nrow(gate_data) > 0)

# ---- EP: fit once on train_data, 5-fold OOS preds ---------------------

cli::cli_h2("Fitting EP (train seasons only) -- this is the expensive, ONE-TIME step")
t_ep0 <- Sys.time()
ep_fit <- fit_ep(train_data)
cli::cli_inform("EP fit done in {round(as.numeric(difftime(Sys.time(), t_ep0, units = 'mins')), 1)} min")

row_folds_ep <- make_match_folds(train_data$torp_match_id)
t_oos0 <- Sys.time()
ep_oos <- cv_ep_oos_preds(ep_fit$X, ep_fit$y, ep_fit$folds, row_folds_ep, ep_params(), ep_fit$optimal_nrounds)
cli::cli_inform("EP 5-fold OOS preds done in {round(as.numeric(difftime(Sys.time(), t_oos0, units = 'mins')), 1)} min")

# ---- Base WP train frame -----------------------------------------------

model_data_wp_train_base <- build_wp_data(train_data, ep_oos)
cli::cli_inform("model_data_wp_train_base: {nrow(model_data_wp_train_base)} rows")

# ---- Base WP gate frame: score gate_data through the SAME temporal EP model

epv_vars_gate <- gate_data |> torp:::select_epv_model_vars()
abort_on_feature_na(epv_vars_gate, "EP (gate scoring)")
X_ep_gate <- stats::model.matrix(~ . + 0, data = epv_vars_gate, na.action = na.pass)
stopifnot(nrow(X_ep_gate) == nrow(epv_vars_gate))
ep_preds_gate <- predict(ep_fit$model, X_ep_gate)
if (!is.matrix(ep_preds_gate)) ep_preds_gate <- matrix(ep_preds_gate, ncol = 5, byrow = TRUE)
colnames(ep_preds_gate) <- c("opp_goal", "opp_behind", "behind", "goal", "no_score")

gate_wp_data_base <- build_wp_data(gate_data, ep_preds_gate)
cli::cli_inform("gate_wp_data_base: {nrow(gate_wp_data_base)} rows")

pace_rows_per_min <- compute_pace_rows_per_min(model_data_wp_train_base)
cli::cli_inform("Empirical pace constant (WP rows / match-minute, training data): {round(pace_rows_per_min, 3)}")

# ---- Cache everything ---------------------------------------------------

saveRDS(ep_fit$model, file.path(results_dir, "ep_model_temporal.rds"))
saveRDS(model_data_wp_train_base, file.path(results_dir, "model_data_wp_train_base.rds"))
saveRDS(gate_wp_data_base, file.path(results_dir, "gate_wp_data_base.rds"))
saveRDS(pace_rows_per_min, file.path(results_dir, "pace_rows_per_min.rds"))
saveRDS(list(seasons = seasons, gate_season = gate_season,
            ep_optimal_nrounds = ep_fit$optimal_nrounds, ep_cv_logloss = ep_fit$cv_logloss),
       file.path(results_dir, "prep_meta.rds"))

cli::cli_alert_success("prep_base_data.R complete: {Sys.time()}")
sink(type = "message")
sink()
