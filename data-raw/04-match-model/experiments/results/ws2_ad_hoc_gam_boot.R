EXPERIMENTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))
library(withr); library(cli)

champ <- readRDS(file.path(EXPERIMENTS_DIR, "results/ws2_champion_roll.rds"))
feat  <- readRDS(file.path(EXPERIMENTS_DIR, "results/ws2_feature_roll.rds"))

b_gam <- boot_mae_diff(feat$gam_preds, champ$gam_preds, B = 2000)
cat(sprintf("GAM-only Elo-feature vs Champion GAM: N=%d diff=%.3f 95%%CI[%.3f, %.3f]\n",
            b_gam$n_matches, b_gam$mae_diff, b_gam$mae_ci[1], b_gam$mae_ci[2]))
cat(sprintf("Brier diff=%.4f 95%%CI[%.4f, %.4f]\n", b_gam$brier_diff, b_gam$brier_ci[1], b_gam$brier_ci[2]))

b_xgb <- boot_mae_diff(feat$xgb_preds, champ$xgb_preds, B = 2000)
cat(sprintf("XGB-only Elo-feature vs Champion XGB: N=%d diff=%.3f 95%%CI[%.3f, %.3f]\n",
            b_xgb$n_matches, b_xgb$mae_diff, b_xgb$mae_ci[1], b_xgb$mae_ci[2]))

saveRDS(list(gam_boot = b_gam, xgb_boot = b_xgb), file.path(EXPERIMENTS_DIR, "results/ws2_feature_gam_xgb_boot.rds"))
