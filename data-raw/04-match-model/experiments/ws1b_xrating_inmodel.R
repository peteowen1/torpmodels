# WS1b — in-model swap: does an xScore rating beat Elo AS A FEATURE?
# ==================================================================
# FABLE-MATCH-FEATURES-PLAN.md §6.3 action 3.
#
# WS1 (§6.1) showed the xScore power rating beats production's Elo decisively as
# a STANDALONE predictor (MAE 26.38 vs 27.15, cor 0.559 vs 0.524, and it renders
# Elo redundant at p 9e-10). It also showed it adds almost nothing on top of C6
# post-hoc (delta r2 0.0015, p 0.21). Those two facts do not settle the question
# this file asks: a feature the GAM/XGB chain CONSUMES is different from a
# prediction blended onto its output — the chain can learn different smooths
# from a cleaner input.
#
# METHOD — why the arms are built by column substitution
# ------------------------------------------------------
# Production's .train_match_gams() hardcodes `s(elo_diff, ...)` in models 2 and
# 4, and .train_xgb_fixed() takes the feature by name. Hand-transcribing those
# formulas to reference a differently-named column is exactly the fidelity
# hazard the round-1 plan flagged (its C6 trainer was a ~150-line hand copy that
# needed a separate verification pass). So instead each arm OVERWRITES the
# `elo_diff` column with a different rating and calls the *unmodified*
# production trainers. Every arm is then guaranteed to differ in the rating and
# in nothing else.
#
#   A  elo_diff = production Elo          (the champion — reuses the cached roll)
#   B  elo_diff = xScore rating, slow k   (WS1's W1b)
#   C  elo_diff = xScore rating, fast k   (does the timescale matter in-model?)
#
# Run: powershell.exe -Command 'Rscript "<this file>"'

suppressMessages({
  library(dplyr)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RES <- file.path(EXP, "results")
source(file.path(EXP, "rolling_lib.R"))
source(file.path(EXP, "signal_gate.R"))
source(file.path(EXP, "xrating_lib.R"))

TEST_SEASONS <- 2025:2026
team_mdl_df <- readRDS(file.path(RES, "ws0r3_team_mdl_df.rds"))
team_mdl_df <- add_xrating_diff(team_mdl_df, pair = TRUE)

.run_arm <- function(df, label) {
  cli::cli_h1("Arm: {label}")
  t0 <- Sys.time()
  r <- run_rolling_eval(df, test_seasons = TEST_SEASONS,
                        gam_trainer = .train_match_gams,
                        xgb_trainer = .train_xgb_fixed,
                        extra_feature_cols = "elo_diff",
                        cv_extra_feature_cols = "elo_diff",
                        verbose = FALSE)
  cli::cli_inform("{label} took {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  r
}

arms <- list()
arms$A_elo <- readRDS(file.path(RES, "ws0r3_roll_pooled.rds"))   # already run

dfB <- team_mdl_df; dfB$elo_diff <- dfB$xelo_slow_diff
arms$B_xelo_slow <- .run_arm(dfB, "B: xScore rating (slow k)")

dfC <- team_mdl_df; dfC$elo_diff <- dfC$xelo_fast_diff
arms$C_xelo_fast <- .run_arm(dfC, "C: xScore rating (fast k)")

saveRDS(arms, file.path(RES, "ws1b_xrating_arms.rds"))

cli::cli_h1("Input Blend, pooled {min(TEST_SEASONS)}:{max(TEST_SEASONS)}")
tab <- do.call(rbind, lapply(names(arms), function(n) {
  m <- .compute_metrics(arms[[n]]$input_blend_preds)
  data.frame(arm = n, MAE = m$mae, RMSE = m$rmse, Brier = m$brier, bits = m$bits,
             slope = m$slope, cor = m$cor)
}))
print(tab, row.names = FALSE, digits = 5)

cli::cli_h1("Ship gate vs arm A (bootstrap CI on delta MAE)")
for (n in c("B_xelo_slow", "C_xelo_fast")) {
  bt <- boot_mae_diff(arms[[n]]$input_blend_preds, arms$A_elo$input_blend_preds)
  cli::cli_text("{n}: dMAE {round(bt$mae_diff, 3)} [{round(bt$mae_ci[1], 3)}, {round(bt$mae_ci[2], 3)}] | dBrier {round(bt$brier_diff, 5)}")
}

cli::cli_h1("Signal gate vs the Elo baseline")
base <- elo_baseline_preds(team_mdl_df, test_seasons = TEST_SEASONS)
for (n in names(arms)) signal_gate_report(arms[[n]]$input_blend_preds, base, label = n)
cli::cli_alert_success("Saved results/ws1b_xrating_arms.rds")
