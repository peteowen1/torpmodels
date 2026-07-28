# Do the v2 EPR ratings actually improve MATCH prediction?
# ========================================================
# The defender-value program shipped v2 (EPV position standardisation, corrected
# lineup taxonomy, PSR standardisation) on positional-calibration evidence. Its
# match-model gate reportedly passed with a small Brier cost -- but that was
# measured before the stale-vintage corruption was found, on a rating file whose
# 2022 and 2025 seasons were deflated.
#
# Both vintages are now published side by side, so the question is directly
# answerable on corrected data:
#   torp_ratings.parquet     = v2 (canonical, corrected 2026-07-28)
#   torp_ratings_v1.parquet  = v1 (preserved)
#
# Same pipeline, same games, same trainers -- ONLY the rating vintage differs.
#
# CAVEAT worth stating up front: the preserved v1 file is the pre-2026-07-27
# artifact, which carried its own documented cross-season drift (daily runs
# regenerated only the current season, so its vintages were mixed). It is the
# best available v1, not a clean one. A v2 win here is therefore partly "v2 is
# internally consistent" rather than purely "the standardisation helps" -- and
# that is worth knowing either way, since internal consistency is a real
# property of the shipped thing.
#
# Run: powershell.exe -Command 'Rscript "<this file>"'

suppressMessages({
  library(dplyr); library(data.table)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})
options(torp.local_data_dir = NA)   # releases, never the mirror

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RES <- file.path(EXP, "results")
source(file.path(EXP, "rolling_lib.R"))
source(file.path(EXP, "signal_gate.R"))
TEST_SEASONS <- 2025:2026

# build_team_mdl_df() loads ratings internally, so mirror it with the vintage
# injected. Everything else is byte-identical to the production wrapper.
build_with_ratings <- function(torp_df) {
  all_grounds <- file_reader("stadium_data", "reference-data")
  xg_df <- load_xg(TRUE); fixtures <- load_fixtures(TRUE)
  results <- load_results(TRUE); teams <- load_teams(TRUE)
  psr_df <- tryCatch({
    .compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE))
  }, error = function(e) { cli::cli_warn("PSR: {conditionMessage(e)}"); NULL })

  fix_df <- .build_fixtures_df(fixtures)
  team_rt_df <- .build_team_ratings_df(teams, torp_df, psr_df)
  team_rt_fix_df <- .build_match_features(fix_df, team_rt_df, all_grounds)
  weather_df <- .load_match_weather(fixtures, all_grounds, NULL, get_afl_season())
  anchor <- max(as.Date(fix_df$utc_start_time), na.rm = TRUE)
  .build_team_mdl_df(team_rt_fix_df, results, xg_df, weather_df, anchor)
}

.bits <- function(pw, hw) mean(ifelse(hw == 1, 1 + log2(pw),
                              ifelse(hw == 0, 1 + log2(1 - pw),
                                     1 + 0.5 * log2(pw * (1 - pw)))))

run_arm <- function(label, torp_df) {
  cli::cli_h1("Arm: {label}")
  tm <- build_with_ratings(torp_df)
  cli::cli_inform("team_mdl_df: {nrow(tm)} rows")
  print(as.data.table(tm)[!is.na(epr.x), .(n = .N, epr_sd = round(sd(epr.x), 2)),
                          by = season.x][order(season.x)], row.names = FALSE)
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams,
                           xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff",
                           cv_extra_feature_cols = "xelo_diff",
                           verbose = FALSE)
  list(tm = tm, roll = roll)
}

v2 <- as.data.frame(load_torp_ratings())
v1 <- as.data.frame(load_torp_ratings(version = "v1"))
cli::cli_inform("v2 rows: {nrow(v2)} | v1 rows: {nrow(v1)}")

a2 <- run_arm("v2 (canonical)", v2)
a1 <- run_arm("v1 (preserved)", v1)

report <- function(a, lab) {
  ib <- a$roll$input_blend_preds
  # RAW predictions only, deliberately.
  #
  # The first version of this applied fit_match_margin_calibration(a$tm) here.
  # That is wrong and it manufactured a fake result: the sidecar is fitted
  # against the PRODUCTION model trained on the whole frame, whereas these are
  # ROLLING-EVAL predictions from week-by-week refits. Different model, different
  # natural scale. It returned b = 1.391 for the v1 arm whose raw slope was
  # already 0.991, over-corrected it by ~40%, and turned a null (dMAE +0.081)
  # into an apparent 1.08-MAE v2 win. The bootstrap below disagreed with the
  # table above it, which is the only reason it was caught.
  #
  # Both arms are uncalibrated, so the comparison stays like-for-like.
  pm <- ib$pred_margin
  for (s in list(list("2026", ib$season == 2026), list("pooled", rep(TRUE, nrow(ib))))) {
    i <- s[[2]]
    cat(sprintf("%-16s %-7s MAE %.3f | bits %.4f | Brier %.4f | cor %.3f | slope %.3f\n",
                lab, s[[1]], mean(abs(pm[i] - ib$margin[i])),
                .bits(ib$pred_win[i], ib$home_win[i]),
                mean((ib$pred_win[i] - ib$home_win[i])^2),
                cor(pm[i], ib$margin[i]),
                unname(coef(lm(ib$margin[i] ~ pm[i]))[2])))
  }
}
cli::cli_h1("Does the v2 rating vintage help match prediction?")
report(a2, "v2"); report(a1, "v1")

cat("\n=== ship-gate bootstrap: v2 vs v1 (pooled) ===\n")
bt <- boot_mae_diff(a2$roll$input_blend_preds, a1$roll$input_blend_preds)
cat(sprintf("dMAE %.3f  95%% CI [%.3f, %.3f]  | dBrier %+.5f\n",
            bt$mae_diff, bt$mae_ci[1], bt$mae_ci[2], bt$brier_diff))
cat("(negative dMAE = v2 better)\n")

saveRDS(list(v2 = a2$roll, v1 = a1$roll, boot = bt),
        file.path(RES, "v1_vs_v2_match.rds"))
cli::cli_alert_success("Saved v1_vs_v2_match.rds")
