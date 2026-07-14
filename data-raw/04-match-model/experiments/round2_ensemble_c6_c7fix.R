RESULTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/results"
EXPERIMENTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
.rds <- function(name) file.path(RESULTS_DIR, name)

suppressPackageStartupMessages({ library(dplyr); library(cli); library(MLmetrics) })
torp_paths <- c("../torp", "../../torp", "../../../torp", "C:/dev/torpverse/torp")
for (p in torp_paths) if (file.exists(file.path(p, "DESCRIPTION"))) { devtools::load_all(p, quiet = TRUE); break }
source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))

base <- readRDS(.rds("round2_synth_base.rds"))

c6 <- base$c6_preds[, c("match_id","pred_margin","pred_win","margin","home_win")]
c7 <- base$c7fix_norecal[, c("match_id","pred_margin","pred_win","margin","home_win")]
names(c6)[2:3] <- c("pred_margin_c6", "pred_win_c6")
names(c7)[2:5] <- c("pred_margin_c7", "pred_win_c7", "margin_c7", "home_win_c7")
merged <- merge(c6, c7, by = "match_id")
cat("Merged pooled n =", nrow(merged), "\n")
stopifnot(all.equal(merged$margin, merged$margin_c7))

weights <- c(1.0, 0.85, 0.7, 0.5, 0.3, 0.15, 0.0)
cat("\n=== Pooled 2025:2026 (n=", nrow(merged), "): margin-blend weight w (weight on C6), win-prob = pure C6 ===\n", sep="")
results <- list()
for (w in weights) {
  pm <- w * merged$pred_margin_c6 + (1 - w) * merged$pred_margin_c7
  df <- data.frame(pred_margin = pm, pred_win = merged$pred_win_c6,
                    margin = merged$margin, home_win = merged$home_win,
                    match_id = merged$match_id)
  m <- .compute_metrics(df)
  results[[as.character(w)]] <- list(df = df, m = m)
  cat(sprintf("w=%.2f: MAE=%.3f RMSE=%.3f Brier=%.4f Bits=%.4f Slope=%.3f\n", w, m$mae, m$rmse, m$brier, m$bits, m$slope))
}

# Bootstrap CI for the best interior candidate found (if any) vs C6 champion
maes <- sapply(results, function(r) r$m$mae)
cat("\nMAE by weight:", paste(names(results), round(maes,3), collapse=" | "), "\n")
best_w <- names(which.min(maes))
cat("Best (lowest MAE) weight on this pooled screen: w =", best_w, "\n")

if (best_w != "1") {
  best_df <- results[[best_w]]$df
  boot_best <- boot_mae_diff(best_df, base$c6_preds)
  cat(sprintf("Boot vs C6 champion: deltaMAE=%+.3f CI=[%.3f, %.3f] deltaBrier=%+.4f CI=[%.4f, %.4f]\n",
              boot_best$mae_diff, boot_best$mae_ci[1], boot_best$mae_ci[2],
              boot_best$brier_diff, boot_best$brier_ci[1], boot_best$brier_ci[2]))
}
saveRDS(results, file.path(EXPERIMENTS_DIR, "results", "round2_blend_pool.rds"))
cat("\nSaved round2_blend_pool.rds\n")
