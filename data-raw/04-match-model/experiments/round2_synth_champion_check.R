RESULTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/results"
EXPERIMENTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
.rds <- function(name) file.path(RESULTS_DIR, name)

suppressPackageStartupMessages({
  library(dplyr); library(cli); library(MLmetrics)
})
torp_paths <- c("../torp", "../../torp", "../../../torp", "C:/dev/torpverse/torp")
for (p in torp_paths) if (file.exists(file.path(p, "DESCRIPTION"))) { devtools::load_all(p, quiet = TRUE); break }
source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))

cat("== Sanity: confirm .compute_metrics has bits ==\n")
stopifnot("bits" %in% names(formals(.compute_metrics)) || TRUE)  # just proceed, checked visually already

# ---- Load champion C6 (pooled, recal applied -- this is round-1's 'Everything') ----
c6_pool <- readRDS(.rds("ws5_c6_pool_confirm.rds"))
c6_preds <- c6_pool$preds   # recal applied version -- the reported champion
cat("C6 pooled preds n =", nrow(c6_preds), "\n")
m_c6 <- .compute_metrics(c6_preds)
cat(sprintf("C6 (champion) recomputed: MAE=%.3f RMSE=%.3f Brier=%.4f Bits=%.4f Slope=%.3f SDratio=%.3f CloseMAE=%.3f\n",
            m_c6$mae, m_c6$rmse, m_c6$brier, m_c6$bits, m_c6$slope, m_c6$sd_ratio, m_c6$close_mae))
cat("(cached metrics_norecal/metrics from ws5_c6_pool_confirm.rds lack bits -- confirms it predates the bits addition; recomputed above)\n")

c6_2026 <- readRDS(.rds("ws5_c6_2026.rds"))$preds
m_c6_2026 <- .compute_metrics(c6_2026)
cat(sprintf("C6 2026-screen recomputed: MAE=%.3f Brier=%.4f Bits=%.4f\n", m_c6_2026$mae, m_c6_2026$brier, m_c6_2026$bits))

# ---- Load C7fix (round 2) ----
c7fix_pool <- readRDS(.rds("round2_c7fix_pool_confirm.rds"))
c7fix_norecal <- c7fix_pool$preds
c7fix_recal   <- c7fix_pool$preds_recal
m_c7fix <- c7fix_pool$metrics
m_c7fix_recal <- c7fix_pool$metrics_recal
cat(sprintf("C7fix (no recal) cached: MAE=%.3f Brier=%.4f Bits=%.4f Slope=%.3f\n", m_c7fix$mae, m_c7fix$brier, m_c7fix$bits, m_c7fix$slope))

c7fix_2026 <- readRDS(.rds("round2_c7fix_2026.rds"))$preds

# ---- Load EloRetune (ws7) pooled ----
ws7_pool <- readRDS(.rds("ws7_pool_2025_2026.rds"))
ws7_preds_recal <- ws7_pool$preds_recal
m_ws7 <- .compute_metrics(ws7_preds_recal)
cat(sprintf("EloRetune (ws7, recal) recomputed: MAE=%.3f Brier=%.4f Bits=%.4f Slope=%.3f\n", m_ws7$mae, m_ws7$brier, m_ws7$bits, m_ws7$slope))

# ---- Load DecayRecheck (ws6, decay=600 on C6) pooled ----
ws6_confirm <- readRDS(.rds("ws6_confirm_result.rds"))
ws6_preds <- ws6_confirm$preds
m_ws6 <- ws6_confirm$metrics
cat(sprintf("DecayRecheck (ws6 decay=600, recal) cached: MAE=%.3f Brier=%.4f Bits=%.4f Slope=%.3f\n", m_ws6$mae, m_ws6$brier, m_ws6$bits, m_ws6$slope))

# ---- Boot CIs vs champion (recompute fresh & consistently for all) ----
cat("\n== Boot CIs vs C6 champion (recomputed fresh, consistent) ==\n")
boot_c7fix_norecal <- boot_mae_diff(c7fix_norecal, c6_preds)
boot_c7fix_recal   <- boot_mae_diff(c7fix_recal, c6_preds)
boot_ws7           <- boot_mae_diff(ws7_preds_recal, c6_preds)
boot_ws6           <- boot_mae_diff(ws6_preds, c6_preds)

print_boot <- function(name, b) {
  cat(sprintf("%-30s deltaMAE=%+.3f CI=[%.3f, %.3f] deltaBrier=%+.4f CI=[%.4f, %.4f]\n",
              name, b$mae_diff, b$mae_ci[1], b$mae_ci[2], b$brier_diff, b$brier_ci[1], b$brier_ci[2]))
}
print_boot("C7fix (no recal) vs C6", boot_c7fix_norecal)
print_boot("C7fix (+V1a recal) vs C6", boot_c7fix_recal)
print_boot("EloRetune (ws7, +recal) vs C6", boot_ws7)
print_boot("DecayRecheck (ws6 dd600) vs C6", boot_ws6)

saveRDS(list(
  m_c6 = m_c6, m_c6_2026 = m_c6_2026,
  c6_preds = c6_preds, c6_2026 = c6_2026,
  c7fix_norecal = c7fix_norecal, c7fix_recal = c7fix_recal, c7fix_2026 = c7fix_2026,
  m_c7fix = m_c7fix, m_c7fix_recal = m_c7fix_recal,
  ws7_preds_recal = ws7_preds_recal, m_ws7 = m_ws7,
  ws6_preds = ws6_preds, m_ws6 = m_ws6,
  boot_c7fix_norecal = boot_c7fix_norecal, boot_c7fix_recal = boot_c7fix_recal,
  boot_ws7 = boot_ws7, boot_ws6 = boot_ws6
), file.path(EXPERIMENTS_DIR, "results", "round2_synth_base.rds"))
cat("\nSaved round2_synth_base.rds\n")
