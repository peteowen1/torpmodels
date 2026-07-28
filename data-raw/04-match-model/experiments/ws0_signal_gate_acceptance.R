# WS0 acceptance run — does the current champion pass the incremental-signal gate?
# ================================================================================
# FABLE-MATCH-FEATURES-PLAN.md WS0 acceptance criterion:
#
#   "Re-running the gate on the current champion's cached pooled predictions
#    reproduces the diagnosis's qualitative result (champion adds little or
#    nothing over the Elo baseline). If the champion in fact PASSES G7 cleanly
#    against an internally-built Elo baseline, that is itself a major finding —
#    it would mean the redundancy is specific to Wheelo's particular Elo rather
#    than to team-Elo information in general, and WS1 should be re-scoped."
#
# Uses cached artifacts (no retraining):
#   team_mdl_df_cache_with_elo.rds  — 2314 rows, seasons 2021-2026, elo_diff present
#   round3_c6_w2022.rds             — C6 rolling-OOS preds, 1008 matches, 2022:2026
#                                     ($preds = with recal, $preds_norecal = without)
#
# Run:  powershell.exe -Command 'Rscript "<this file>"'   (arrow segfaults under Git Bash)

suppressMessages({
  library(dplyr)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RES <- file.path(EXP, "results")
source(file.path(EXP, "rolling_lib.R"))   # boot_mae_diff()
source(file.path(EXP, "signal_gate.R"))

team_mdl_df <- readRDS(file.path(RES, "team_mdl_df_cache_with_elo.rds"))
c6 <- readRDS(file.path(RES, "round3_c6_w2022.rds"))

TEST_SEASONS <- sort(unique(c6$preds$season))
cli::cli_alert_info("Test seasons: {TEST_SEASONS} | C6 matches: {nrow(c6$preds)}")

# Elo baseline (per-round refit scale, strictly past-only) ----
base <- elo_baseline_preds(team_mdl_df, test_seasons = TEST_SEASONS)
cli::cli_alert_info("Elo baseline: {nrow(base)} matches, MAE {round(mean(abs(base$pred_margin - base$margin)), 3)}")

# The gate ----
r_recal <- signal_gate_report(c6$preds, base, label = "C6 champion (with margin recal)")
r_raw   <- signal_gate_report(c6$preds_norecal, base, label = "C6 champion (no recal)")

# Reference: how the two stack up head to head on raw MAE, for context only.
cli::cli_h2("Context — standalone MAE on the shared match set")
j <- merge(
  data.frame(match_id = c6$preds$match_id, c6 = c6$preds$pred_margin, margin = c6$preds$margin),
  data.frame(match_id = base$match_id, elo = base$pred_margin),
  by = "match_id"
)
cli::cli_text("n={nrow(j)} | C6 MAE {round(mean(abs(j$c6 - j$margin)), 3)} | Elo baseline MAE {round(mean(abs(j$elo - j$margin)), 3)}")
cli::cli_text("cor with actual: C6 {round(cor(j$c6, j$margin), 3)} | Elo {round(cor(j$elo, j$margin), 3)}")

saveRDS(list(recal = r_recal, no_recal = r_raw, baseline = base),
        file.path(RES, "ws0_signal_gate_acceptance.rds"))
cli::cli_alert_success("Saved results/ws0_signal_gate_acceptance.rds")
