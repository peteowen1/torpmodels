# xRAPM rolling gate, PRODUCTION FEATURE SET variant (2026-11-17).
# Same as gate_xrapm_2026_11.R, except both arms also carry the listed-position
# splits (MATCH_LISTED_POS_DIFF_COLS), which production's .train_match_xgb() uses
# and the harness trainer does not (pre-PR review finding). So the off arm is
# production's XGBoost feature set (xrapm_diff constant 0 = absent).
#
# Original question: does a real xrapm_diff in XGBoost improve the match model?
#
# Production builds xrapm_diff as a constant 0 (.build_match_features() drops
# xrapm). Both arms are scored on ONE model frame built with xRAPM carried
# through, so the only difference is the XGBoost feature set:
#   off = extra_feature_cols "xelo_diff"                (production today)
#   on  = extra_feature_cols c("xelo_diff", "xrapm_diff")
# GAMs are untouched in both (xrapm_diff is XGBoost-only by design).
#
# Ships only if BOTH margin MAE and log loss improve on the pooled 2025-2026
# window (NEXT-STEPS, decided 2026-11-12). Sequential path only (the parallel
# one is not ship-gate safe). Each arm checkpoints per round.
#
# Run from torpverse/torp:  Rscript ../torpmodels/data-raw/04-match-model/experiments/gate_xrapm_2026_11.R
# Output: torpdata/data/raw/match_gates/xrapm_2026-11/ (gitignored): checkpoints,
#         results.rds, metrics.csv, gate.log lines on stdout.

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
out_dir <- "C:/dev/torpverse/torpdata/data/raw/match_gates/xrapm_2026-11-prodcols"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
TEST_SEASONS <- 2025:2026

# Releases only: a local torpdata/data copy, or a cwd-relative
# data-raw/03-ratings xRAPM snapshot, silently shadows the release (the first
# launch, 2026-11-17 9:32pm, read a 2026-08-25 local snapshot with no season
# keys and stopped).
options(torp.local_data_dir = NA)
stopifnot(is.null(get_local_data_dir()))
say("loading inputs (releases only)")
src <- load_match_inputs()
torp_df <- load_torp_ratings()
xr_path <- file.path(tempdir(), "career_team_rapm_asof_AFLM.parquet")
piggyback::pb_download("career_team_rapm_asof_AFLM.parquet", dest = tempdir(),
                       repo = "peteowen1/torpdata", tag = TEAM_RAPM_ASOF_RELEASE_TAG, overwrite = TRUE)
stopifnot(file.exists(xr_path))
xr <- load_team_rapm_asof(comp = "AFLM", path = xr_path)
stopifnot(!is.null(xr), nrow(xr) > 0)
say("torp ratings", nrow(torp_df), "rows | xRAPM snapshot", nrow(xr), "rows")

tm_on  <- build_team_mdl_with(src, torp_df, xrapm_df = xr, include_xrapm = TRUE)
tm_off <- build_team_mdl_with(src, torp_df, xrapm_df = xr, include_xrapm = FALSE)
say("model frame", nrow(tm_on), "rows,", ncol(tm_on), "cols")

# ---- Anchors, written before the first run ----
# 1. The two frames differ only in xRAPM columns.
xcols <- grep("xrapm", names(tm_on), value = TRUE)
common <- setdiff(intersect(names(tm_on), names(tm_off)), xcols)
stopifnot(nrow(tm_on) == nrow(tm_off),
          identical(as.data.frame(tm_on[, common]), as.data.frame(tm_off[, common])))
# 2. Off arm is production: xrapm_diff a flat 0.
stopifnot(all(tm_off$xrapm_diff == 0))
# 3. On arm has a real feature in the test seasons: populated, varied.
cov <- tm_on |> group_by(season.x) |>
  summarise(n = n(), nonzero = mean(xrapm_diff != 0), sd = sd(xrapm_diff), .groups = "drop")
print(as.data.frame(cov))
stopifnot(all(cov$nonzero[cov$season.x %in% TEST_SEASONS] > 0.9), all(cov$sd[cov$season.x %in% TEST_SEASONS] > 0))
say("anchors hold: frames differ only in", paste(xcols, collapse = ","), "| off xrapm_diff all 0")

pos_cols <- MATCH_LISTED_POS_DIFF_COLS
stopifnot(length(pos_cols) > 0, all(pos_cols %in% names(tm_on)))
say("position split columns:", paste(pos_cols, collapse = ","))
arms <- list(off = c("xelo_diff", pos_cols), on = c("xelo_diff", pos_cols, "xrapm_diff"))
res <- list()
for (nm in names(arms)) {
  say("arm", nm, "start")
  ta <- Sys.time()
  res[[nm]] <- run_rolling_eval(tm_on, test_seasons = TEST_SEASONS,
                                extra_feature_cols = arms[[nm]], cv_extra_feature_cols = arms[[nm]],
                                verbose = FALSE, checkpoint_dir = file.path(out_dir, "ckpt", nm))
  say("arm", nm, "done in", round(difftime(Sys.time(), ta, units = "mins"), 1), "min,",
      nrow(res[[nm]]$input_blend_preds), "matches | nrounds", paste(res[[nm]]$xgb_nrounds, collapse = ","))
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
print(met[met$window == "pooled", intersect(c("arm", "set", "mae", "logloss"), names(met))])

# ---- Paired difference, served set (Input Blend), pooled ----
a <- res$on$input_blend_preds; b <- res$off$input_blend_preds
stopifnot(identical(a$match_id, b$match_id))
bd <- boot_mae_diff(a, b)
cat("\nboot_mae_diff (on minus off, Input Blend, pooled; negative = on is better):\n"); print(bd)
ll <- function(p) -mean(p$home_win * log(pmax(p$pred_win, 1e-9)) + (1 - p$home_win) * log(pmax(1 - p$pred_win, 1e-9)))
cat("logloss on", round(ll(a), 5), "off", round(ll(b), 5), "diff", round(ll(a) - ll(b), 5), "\n")
say("GATE DONE in", round(difftime(Sys.time(), t0, units = "mins"), 1), "min")
