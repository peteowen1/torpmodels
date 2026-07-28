# WS0 step 1 — rebuild team_mdl_df through the current round, then re-run the gate
# ================================================================================
# FABLE-MATCH-FEATURES-PLAN.md §6.3 action 1.
#
# Why this has to happen before anything else: every cached artifact in
# results/ stops at 2026 R18 and predates C6 serving in production (C6 merged
# to main 2026-07-14, calibration sidecar first published 2026-07-20). So no
# number anywhere yet measures the model that is actually running.
#
# Stages (checkpointed, run individually or "all"):
#   Rscript ws0_rebuild_and_regate.R data     # rebuild team_mdl_df (slow)
#   Rscript ws0_rebuild_and_regate.R roll     # rolling OOS eval, 2025:2026 (slow)
#   Rscript ws0_rebuild_and_regate.R gate     # signal gate on the fresh preds (fast)

suppressMessages({
  library(dplyr)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
  devtools::load_all("C:/dev/torpverse/torpmodels", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RES <- file.path(EXP, "results")
source(file.path(EXP, "rolling_lib.R"))
source(file.path(EXP, "signal_gate.R"))

stage <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(stage)) stage <- "all"
TEST_SEASONS <- 2025:2026
DF <- file.path(RES, "ws0r3_team_mdl_df.rds")
PR <- file.path(RES, "ws0r3_roll_pooled.rds")

if (stage %in% c("data", "all")) {
  cli::cli_h1("Rebuilding team_mdl_df through the current round")
  t0 <- Sys.time()
  team_mdl_df <- build_team_mdl_df()
  cli::cli_inform("build_team_mdl_df() took {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

  n_done <- sum(!is.na(team_mdl_df$win)) / 2
  cli::cli_inform("rows {nrow(team_mdl_df)} | seasons {paste(sort(unique(team_mdl_df$season.x)), collapse=', ')}")
  cli::cli_inform("completed matches: {n_done} | elo_diff present: {'elo_diff' %in% names(team_mdl_df)}")
  cur <- team_mdl_df |> filter(!is.na(win), season.x == max(season.x))
  cli::cli_inform("2026: {nrow(cur)/2} completed, max round {max(cur$round_number.x)}")
  # The whole point of the rebuild — fail loudly if it did not actually extend.
  if (nrow(cur) / 2 <= 153) {
    cli::cli_alert_danger("2026 completed matches did not exceed the 153 in the old cache — rebuild added nothing")
  }
  saveRDS(team_mdl_df, DF)
  cli::cli_alert_success("Saved {basename(DF)}")
}

if (stage %in% c("roll", "all")) {
  cli::cli_h1("Rolling OOS eval, {TEST_SEASONS} (production trainers + elo_diff)")
  team_mdl_df <- readRDS(DF)
  t0 <- Sys.time()
  roll <- run_rolling_eval(
    team_mdl_df, test_seasons = TEST_SEASONS,
    gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
    extra_feature_cols = "elo_diff", cv_extra_feature_cols = "elo_diff"
  )
  cli::cli_inform("rolling eval took {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  saveRDS(roll, PR)
  cli::cli_alert_success("Saved {basename(PR)}")
}

if (stage %in% c("gate", "all")) {
  cli::cli_h1("Signal gate on the freshly-built champion")
  team_mdl_df <- readRDS(DF)
  roll <- readRDS(PR)
  ib <- roll$input_blend_preds

  cli::cli_h2("Headline metrics (no recal)")
  print(unlist(.compute_metrics(ib)), digits = 4)

  # Production's own recalibration, as production fits it (single temporal
  # holdout) — NOT the research walk-forward reconstruction. Plan §6.2 flagged
  # these disagreeing; this is the reconciliation.
  calib <- tryCatch(fit_match_margin_calibration(team_mdl_df), error = function(e) NULL)
  if (!is.null(calib)) {
    cli::cli_inform("production calibration: b={round(calib$b, 4)}, raw OOS slope={round(calib$slope_raw, 4)}, n_oos={calib$n_oos}")
    ib_recal <- ib
    ib_recal$pred_margin <- apply_match_margin_calibration(ib$pred_margin, calib)
    cli::cli_h2("With production recal applied")
    print(unlist(.compute_metrics(ib_recal)), digits = 4)
    cli::cli_text("delta MAE from recal: {round(mean(abs(ib_recal$pred_margin - ib_recal$margin)) - mean(abs(ib$pred_margin - ib$margin)), 4)}")
  } else {
    cli::cli_alert_warning("fit_match_margin_calibration() unavailable/failed — recal reconciliation skipped")
  }

  base <- elo_baseline_preds(team_mdl_df, test_seasons = TEST_SEASONS)
  signal_gate_report(ib, base, label = "fresh champion (Input Blend, no recal)")

  saveRDS(list(roll = roll, calib = calib), file.path(RES, "ws0r3_gate.rds"))
  cli::cli_alert_success("Saved ws0r3_gate.rds")
}
