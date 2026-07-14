# Canonical Model Training CLI
# =============================
# THE entry point for training EP/WP/shot models. Replaces the *_run.R
# wrappers (their only value -- load dev torp first -- is now the default)
# and the standalone train_ep_model.R / train_wp_model*.R / train_shot_model.R
# scripts, all deleted in this commit.
#
# Usage (from torpmodels root):
#   Rscript data-raw/train_models.R ep wp shot                 # canonical full retrain + upload
#   Rscript data-raw/train_models.R wp --no-upload              # local-only
#   Rscript data-raw/train_models.R ep wp --seasons 2021 2024   # explicit window
#   Rscript data-raw/train_models.R wp --insample-ep            # legacy comparison; never uploads
#   Rscript data-raw/train_models.R ep --torp ../torp           # explicit torp path
#   Rscript data-raw/train_models.R wp --skip-slope-gate         # disable the temporal Q4/close release gate (emergencies only)
#   Rscript data-raw/train_models.R wp --no-calibrate            # skip the WP recalibration layer entirely; forces --no-upload
#
# With no model names given, trains ep + wp + shot (the full set).
#
# --skip-slope-gate and --no-calibrate are the WP recalibration + temporal
# slope release gate flags (torpverse/docs/plans/FABLE-RECAL-PLAN.md D5/Step 4): the gate is ON
# and calibration fits by default for every WP training run.
#
# torpverse/docs/plans/TRAINING-CONSOLIDATION-PLAN.md Step 3.

library(devtools)
library(tidyverse)
library(zoo)
library(janitor)
library(lubridate)
library(xgboost)
library(mgcv)
library(cli)

# --- Locate + load dev torp (reuse the relative-path search convention) ----

.locate_torp <- function(explicit = NULL) {
  candidates <- c(explicit, "../torp", "../../torp", "../../../torp", "torp")
  candidates <- candidates[!is.na(candidates) & nzchar(candidates)]
  for (p in candidates) {
    if (file.exists(file.path(p, "DESCRIPTION"))) return(p)
  }
  NULL
}

# --- Parse args --------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

torp_override <- NULL
torp_idx <- which(args == "--torp")
if (length(torp_idx) > 0) {
  if (torp_idx[1] == length(args)) stop("--torp requires a path argument", call. = FALSE)
  torp_override <- args[torp_idx[1] + 1]
  args <- args[-c(torp_idx[1], torp_idx[1] + 1)]
}

seasons_override <- NULL
seasons_idx <- which(args == "--seasons")
if (length(seasons_idx) > 0) {
  flag_pos <- seasons_idx[1]
  val_positions <- (flag_pos + 1):(flag_pos + 2)
  # F16's lesson: validate exactly two values immediately after the flag,
  # never silently swallow a missing/extra value.
  if (max(val_positions) > length(args) || any(grepl("^--", args[val_positions]))) {
    stop("--seasons requires exactly two integer values immediately after the flag (start end)", call. = FALSE)
  }
  season_vals <- args[val_positions]
  season_start <- suppressWarnings(as.integer(season_vals[1]))
  season_end <- suppressWarnings(as.integer(season_vals[2]))
  if (is.na(season_start) || is.na(season_end)) {
    stop("--seasons values must be integers, got: ", paste(season_vals, collapse = " "), call. = FALSE)
  }
  seasons_override <- season_start:season_end
  args <- args[-c(flag_pos, val_positions)]
}

no_upload <- "--no-upload" %in% args
insample_ep <- "--insample-ep" %in% args
skip_slope_gate <- "--skip-slope-gate" %in% args
no_calibrate <- "--no-calibrate" %in% args
args <- setdiff(args, c("--no-upload", "--insample-ep", "--skip-slope-gate", "--no-calibrate"))

unknown_flags <- args[grepl("^--", args)]
if (length(unknown_flags) > 0) {
  stop("Unknown flag(s): ", paste(unknown_flags, collapse = ", "), call. = FALSE)
}

valid_models <- c("ep", "wp", "shot")
model_names <- args
if (length(model_names) == 0) {
  model_names <- valid_models
} else if (!all(model_names %in% valid_models)) {
  stop("Unknown model(s): ", paste(setdiff(model_names, valid_models), collapse = ", "),
       ". Valid: ", paste(valid_models, collapse = ", "), call. = FALSE)
}

# --- Load dev torp + dev torpmodels ------------------------------------

torp_path <- .locate_torp(torp_override)
if (is.null(torp_path)) {
  stop("Cannot find torp package (tried --torp, ../torp, ../../torp, ../../../torp, torp). ",
       "Run from torpverse/ or torpmodels/, or pass --torp <path>.", call. = FALSE)
}
cli::cli_inform("Loading dev torp from {torp_path}...")
devtools::load_all(torp_path)

cli::cli_inform("Loading dev torpmodels from '.'...")
devtools::load_all(".")

source("data-raw/lib/train_lib.R")

# --- Run -----------------------------------------------------------------

seasons <- seasons_override %||% default_training_seasons()
upload <- !no_upload
wp_ep_source <- if (insample_ep) "insample" else "cv"
slope_gate <- !skip_slope_gate
calibrate <- !no_calibrate

cli::cli_h1("Training: {paste(model_names, collapse = ', ')}")
cli::cli_inform("Seasons: {min(seasons)}-{max(seasons)} | upload: {upload} | wp_ep_source: {wp_ep_source} | calibrate: {calibrate} | slope_gate: {slope_gate}")

results <- train_core_models(
  models = model_names,
  seasons = seasons,
  upload = upload,
  wp_ep_source = wp_ep_source,
  slope_gate = slope_gate,
  calibrate = calibrate
)

cli::cli_h1("Training complete")
for (meta in results) describe_model_meta(meta)
