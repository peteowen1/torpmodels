RESULTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/results"
.rds <- function(name) file.path(RESULTS_DIR, name)
torp_paths <- c("../torp", "../../torp", "../../../torp", "C:/dev/torpverse/torp")
for (p in torp_paths) if (file.exists(file.path(p, "DESCRIPTION"))) { devtools::load_all(p, quiet = TRUE); break }
source("C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/rolling_lib.R")

c6_cache <- readRDS(.rds("wide_c6_pool_confirm.rds"))
c7_cache <- readRDS(.rds("wide_c7fix_pool_confirm.rds"))

c6_norecal <- c6_cache$preds_norecal   # Input Blend, no recal
c7_norecal <- c7_cache$preds           # C7-fixed, no recal

m_c6 <- .compute_metrics(c6_norecal)
m_c7 <- .compute_metrics(c7_norecal)
cat(sprintf("C6 (no recal): MAE=%.3f Brier=%.4f Slope=%.3f\n", m_c6$mae, m_c6$brier, m_c6$slope))
cat(sprintf("C7-fixed (no recal): MAE=%.3f Brier=%.4f Slope=%.3f\n", m_c7$mae, m_c7$brier, m_c7$slope))

c6 <- c6_norecal[, c("match_id", "pred_margin", "pred_win", "margin", "home_win")]
c7 <- c7_norecal[, c("match_id", "pred_margin", "pred_win", "margin", "home_win")]
names(c6)[2:3] <- c("pred_margin_c6", "pred_win_c6")
names(c7)[2:5] <- c("pred_margin_c7", "pred_win_c7", "margin_c7", "home_win_c7")
merged <- merge(c6, c7, by = "match_id")
cat("Merged n =", nrow(merged), "\n")
stopifnot(all.equal(merged$margin, merged$margin_c7))

weights <- c(1.0, 0.85, 0.7, 0.5, 0.3, 0.15, 0.0)
cat("\n=== NO-RECAL ensemble sweep (win-prob = pure C6-no-recal) ===\n")
results <- list()
for (w in weights) {
  pm <- w * merged$pred_margin_c6 + (1 - w) * merged$pred_margin_c7
  df <- data.frame(pred_margin = pm, pred_win = merged$pred_win_c6, margin = merged$margin, home_win = merged$home_win, match_id = merged$match_id)
  m <- .compute_metrics(df)
  results[[as.character(w)]] <- list(df = df, m = m)
  cat(sprintf("w=%.2f: MAE=%.3f RMSE=%.3f Brier=%.4f Slope=%.3f\n", w, m$mae, m$rmse, m$brier, m$slope))
}
maes <- sapply(results, function(r) r$m$mae)
best_w <- names(which.min(maes))
cat("\nBest (lowest MAE) weight:", best_w, "\n")
if (best_w != "1") {
  best_df <- results[[best_w]]$df
  boot_best <- boot_mae_diff(best_df, c6_norecal)
  cat(sprintf("Boot vs C6-no-recal: deltaMAE=%+.3f CI=[%.3f, %.3f] deltaBrier=%+.4f CI=[%.4f, %.4f]\n",
              boot_best$mae_diff, boot_best$mae_ci[1], boot_best$mae_ci[2],
              boot_best$brier_diff, boot_best$brier_ci[1], boot_best$brier_ci[2]))
} else {
  cat("Best weight is w=1.0 -- ensemble adds nothing even without recal.\n")
}
