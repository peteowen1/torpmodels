RESULTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/results"
EXPERIMENTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
.rds <- function(name) file.path(RESULTS_DIR, name)

suppressPackageStartupMessages({ library(dplyr) })
base <- readRDS(.rds("round2_synth_base.rds"))

c6 <- base$c6_preds[, c("match_id","season","pred_margin","pred_win","margin","home_win")]
c7 <- base$c7fix_norecal[, c("match_id","pred_margin","pred_win","margin","home_win")]
names(c6)[3:4] <- c("pred_margin_c6", "pred_win_c6")
names(c7)[2:5] <- c("pred_margin_c7", "pred_win_c7", "margin_c7", "home_win_c7")
merged <- merge(c6, c7, by = "match_id")

weights <- c(1.0, 0.85, 0.7, 0.5, 0.3, 0.15, 0.0)
for (s in sort(unique(merged$season))) {
  sub <- merged[merged$season == s, ]
  cat(sprintf("\n=== Season %s (n=%d) ===\n", s, nrow(sub)))
  for (w in weights) {
    pm <- w * sub$pred_margin_c6 + (1 - w) * sub$pred_margin_c7
    mae <- mean(abs(pm - sub$margin))
    cat(sprintf("  w=%.2f: MAE=%.3f\n", w, mae))
  }
}
