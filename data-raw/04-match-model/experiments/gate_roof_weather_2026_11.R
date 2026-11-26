# Roofed-games rolling gate (review W12, 2026-11-26).
# RESULT: measured null (log loss and margin MAE both worse with the flag on), so
# the flag was REMOVED from torp (6342210c). Kept as the record; it no longer runs
# against current torp. Results: torpdata/data/raw/match_gates/roof_weather_2026-11/.
# Its "Total points" block below scores against UNSCALED total_xpoints (wrong
# scale); the right numbers are in gate_roof_weather_2026_11_totals.R.
# Question: does zeroing the rain and wind model features under a closed roof
# (torp MATCH_ROOF_ZERO_WEATHER) improve the match model?
#
# Same harness and production feature set as gate_xrapm_2026_11_prodcols.R
# (both arms carry xelo_diff and the listed-position splits). The two arms are
# two model frames that differ ONLY in log_wind / log_precip on roofed rows:
#   off = MATCH_ROOF_ZERO_WEATHER FALSE (production today)
#   on  = MATCH_ROOF_ZERO_WEATHER TRUE
# Ship rule as for every match-model gate: margin MAE AND log loss improve on
# the pooled 2025-2026 window. Reported beside it, because the flag acts on the
# totals model: total-points MAE (pred_xtotal vs total_xpoints). The in-sample
# screen already put the ceiling at R2 +0.0017 on the totals GAM.
# Sequential path only. Each arm checkpoints per round.
#
# Run from torpverse/torp:  Rscript ../torpmodels/data-raw/04-match-model/experiments/gate_roof_weather_2026_11.R
# Output: torpdata/data/raw/match_gates/roof_weather_2026-11/ (gitignored).

t0 <- Sys.time()
say <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
suppressMessages({
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
  devtools::load_all("C:/dev/torpverse/torpmodels", quiet = TRUE)
  library(dplyr)
})
exp_dir <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
source(file.path(exp_dir, "rolling_lib.R"))
source(file.path(exp_dir, "arm_lib.R"))
out_dir <- "C:/dev/torpverse/torpdata/data/raw/match_gates/roof_weather_2026-11"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
TEST_SEASONS <- 2025:2026

options(torp.local_data_dir = NA)
stopifnot(is.null(get_local_data_dir()))
say("loading inputs (releases only)")
src <- load_match_inputs()
torp_df <- load_torp_ratings()

set_roof <- function(v) {
  assignInNamespace("MATCH_ROOF_ZERO_WEATHER", v, ns = "torp")
  stopifnot(identical(torp:::MATCH_ROOF_ZERO_WEATHER, v))
}
stopifnot(identical(torp:::MATCH_ROOF_ZERO_WEATHER, FALSE))
set_roof(FALSE); tm_off <- build_team_mdl_with(src, torp_df)
set_roof(TRUE);  tm_on  <- build_team_mdl_with(src, torp_df)
set_roof(FALSE)
say("model frame", nrow(tm_off), "rows,", ncol(tm_off), "cols")

# ---- Anchors, written before the first run ----
# 1. Same rows, same columns; only log_wind / log_precip differ, and only on roofed rows.
stopifnot(nrow(tm_on) == nrow(tm_off), identical(names(tm_on), names(tm_off)))
other <- setdiff(names(tm_on), c("log_wind", "log_precip"))
stopifnot(identical(as.data.frame(tm_on[, other]), as.data.frame(tm_off[, other])))
roof <- tm_on$is_roof %in% TRUE
stopifnot(identical(tm_on$log_wind[!roof], tm_off$log_wind[!roof]),
          identical(tm_on$log_precip[!roof], tm_off$log_precip[!roof]),
          all(tm_on$log_wind[roof] == 0), all(tm_on$log_precip[roof] == 0))
# 2. The flag reached real rows: roofed rows with outside rain in the off frame.
n_rain <- sum(roof & tm_off$log_precip > 0)
stopifnot(n_rain > 0)
say("anchors hold:", sum(roof), "roofed rows,", n_rain, "with outside rain zeroed; everything else identical")

pos_cols <- MATCH_LISTED_POS_DIFF_COLS
stopifnot(length(pos_cols) > 0, all(pos_cols %in% names(tm_off)))
feat <- c("xelo_diff", pos_cols)
frames <- list(off = tm_off, on = tm_on)
res <- list()
for (nm in names(frames)) {
  say("arm", nm, "start")
  ta <- Sys.time()
  res[[nm]] <- run_rolling_eval(frames[[nm]], test_seasons = TEST_SEASONS,
                                extra_feature_cols = feat, cv_extra_feature_cols = feat,
                                verbose = FALSE, checkpoint_dir = file.path(out_dir, "ckpt", nm))
  say("arm", nm, "done in", round(difftime(Sys.time(), ta, units = "mins"), 1), "min,",
      nrow(res[[nm]]$input_blend_preds), "matches")
}
saveRDS(res, file.path(out_dir, "results.rds"))

# ---- Metrics: pooled and by season, all four prediction sets ----
sets <- c("gam_preds", "xgb_preds", "blend_preds", "input_blend_preds")
met <- bind_rows(lapply(names(res), function(nm) bind_rows(lapply(sets, function(st) {
  p <- res[[nm]][[st]]
  bind_rows(
    cbind(arm = nm, set = st, window = "pooled", as.data.frame(.compute_metrics(p))),
    bind_rows(lapply(TEST_SEASONS, function(s)
      cbind(arm = nm, set = st, window = as.character(s), as.data.frame(.compute_metrics(p[p$season == s, ])))))
  )
}))))
write.csv(met, file.path(out_dir, "metrics.csv"), row.names = FALSE)
cat("\nMetrics (mae = margin MAE in points, logloss = win-probability log loss; lower is better for both):\n")
print(met[met$set == "input_blend_preds", intersect(c("arm", "window", "mae", "logloss", "brier", "accuracy", "bits"), names(met))])

# ---- Paired difference, served set (Input Blend), pooled ----
a <- res$on$input_blend_preds; b <- res$off$input_blend_preds
stopifnot(identical(a$match_id, b$match_id))
bd <- boot_mae_diff(a, b)
cat("\nboot_mae_diff (on minus off, Input Blend, pooled; negative = on is better):\n"); print(bd)
ll <- function(p) -mean(p$home_win * log(pmax(p$pred_win, 1e-9)) + (1 - p$home_win) * log(pmax(1 - p$pred_win, 1e-9)))
cat("logloss on", round(ll(a), 5), "off", round(ll(b), 5), "diff", round(ll(a) - ll(b), 5), "\n")

# ---- Total points, where the flag acts (lower is better) ----
tot <- dplyr::distinct(as.data.frame(tm_off)[tm_off$team_type_fac.x == "home", c("match_id", "total_xpoints")])
for (nm in names(res)) {
  p <- merge(res[[nm]]$input_blend_preds[, c("match_id", "pred_xtotal")], tot, by = "match_id")
  roofed <- p$match_id %in% tm_off$match_id[roof]
  cat(sprintf("total-points MAE, arm %-3s: all %.3f (n %d) | roofed %.3f (n %d)\n", nm,
              mean(abs(p$pred_xtotal - p$total_xpoints)), nrow(p),
              mean(abs(p$pred_xtotal[roofed] - p$total_xpoints[roofed])), sum(roofed)))
}
say("GATE DONE in", round(difftime(Sys.time(), t0, units = "mins"), 1), "min")
