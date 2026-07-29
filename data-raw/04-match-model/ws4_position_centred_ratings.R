# Does position-centring EPR change match prediction?
# ==================================================
# Pete's proposal (2026-07-28): set each position's TOG-weighted mean EPR to 0,
# so a rating reads "points above the average player in your position".
#
# It is a NORMALISATION, not an estimate -- position LEVELS are unidentifiable
# from match margins (FABLE-DEFENDER-VALUE-PLAN §8.3, F(5,1113)=0.47, p=0.80).
# So the question is not "is it more accurate" but "what does it COST". A
# cross-sectional shift by position should largely cancel in the home-away
# differential because teams field similar mixes; on the published ratings that
# left cor(epr_diff old, new) = 0.9966. This checks it survives the pipeline.
#
# SHIP CRITERION IS dMAE ~ 0 WITH A TIGHT CI, NOT dMAE < 0. A normalisation that
# improved prediction would mean something is wrong -- most likely that the
# shift is encoding roster shape (teams field 2-9 listed midfielders), which is
# the failure the position-split work already hit once.
#
# LEAK SAFETY: centring is done within (season, round) on that round's
# cross-section only. Ratings at (season, round) mean "entering round r" -- the
# file carries a row for the upcoming unplayed round -- so nothing here was
# unavailable pre-game.
#
# PARALLEL PATH: both arms are scored through run_rolling_eval_parallel() with
# identical worker/thread settings. That is the memory's sanctioned use --
# xgboost's tree_method="hist" is nondeterministic across thread counts, so the
# divergence shifts both arms together and largely cancels in the delta. The
# ABSOLUTE MAEs here are therefore NOT comparable with any sequentially-scored
# champion; only the arm-vs-arm delta is. Re-confirm sequentially before
# shipping anything on this.
#
# Run: powershell.exe -Command 'Rscript "<this file>"'

suppressMessages({
  library(dplyr); library(data.table)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})
options(torp.local_data_dir = NA)

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
RES <- file.path(EXP, "results")
source(file.path(EXP, "rolling_lib.R"))
source(file.path(EXP, "signal_gate.R"))
source(file.path(EXP, "arm_lib.R"))

TEST_SEASONS <- 2025:2026
N_WORKERS    <- 5L          # 5 x (2 gam + 2 xgb) = 20 of 24 cores
CH    <- c("epr_recv", "epr_disp", "epr_spoil", "epr_hitout")
CACHE <- file.path(RES, "ws4_arms_team_mdl.rds")

.bits <- function(pw, hw) mean(ifelse(hw == 1, 1 + log2(pw),
                              ifelse(hw == 0, 1 + log2(1 - pw),
                                     1 + 0.5 * log2(pw * (1 - pw)))))

#' TOG-weighted centring of each EPR channel within (season, round, position)
position_centre <- function(dt) {
  x <- copy(as.data.table(dt))
  x[, .w := pmax(dplyr::coalesce(pred_tog, 0.5), 0.01)]
  for (cc in intersect(CH, names(x))) {
    x[!is.na(position_group),
      (cc) := get(cc) - stats::weighted.mean(get(cc), .w, na.rm = TRUE),
      by = .(season, round, position_group)]
  }
  x[, epr := rowSums(as.matrix(.SD), na.rm = TRUE), .SDcols = intersect(CH, names(x))]
  x[, .w := NULL][]
}

if (file.exists(CACHE)) {
  cli::cli_alert_info("Reusing cached {basename(CACHE)} (delete it to rebuild)")
  arms <- readRDS(CACHE)
} else {
  raw <- as.data.table(load_torp_ratings())
  cen <- position_centre(raw)
  cli::cli_inform("ratings rows: {nrow(raw)}")

  cat("\n=== centring check: TOG-weighted mean EPR by position, latest round ===\n")
  lat <- cen[season == max(season)][round == max(round)]
  print(lat[!is.na(position_group), .(n = .N,
        wmean = round(stats::weighted.mean(epr, pmax(pred_tog, 0.01), na.rm = TRUE), 4)),
        by = position_group], row.names = FALSE)

  # Load the rating-independent inputs ONCE and share them across both arms.
  # Guarantees the arms differ only in the rating vintage, and skips a repeat
  # of the ~193 MB fetch.
  src <- load_match_inputs()
  arms <- list(current  = build_team_mdl_with(src, as.data.frame(raw)),
               centred  = build_team_mdl_with(src, as.data.frame(cen)))
  saveRDS(arms, CACHE)
}
for (nm in names(arms)) cli::cli_inform("{nm}: {nrow(arms[[nm]])} rows")

t0 <- Sys.time()
res <- score_arms(arms, test_seasons = TEST_SEASONS, parallel = TRUE, n_workers = N_WORKERS)
cli::cli_alert_success("Both arms scored in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")

report <- function(a, lab) {
  ib <- a$input_blend_preds
  for (s in list(list("2026", ib$season == 2026), list("pooled", rep(TRUE, nrow(ib))))) {
    i <- s[[2]]
    cat(sprintf("%-10s %-7s n=%3d MAE %.3f | bits %.4f | Brier %.4f | cor %.3f\n",
                lab, s[[1]], sum(i), mean(abs(ib$pred_margin[i] - ib$margin[i])),
                .bits(ib$pred_win[i], ib$home_win[i]),
                mean((ib$pred_win[i] - ib$home_win[i])^2),
                cor(ib$pred_margin[i], ib$margin[i])))
  }
}
cli::cli_h1("What does position-centring cost?")
for (nm in names(res)) report(res[[nm]], nm)

cat("\n=== bootstrap: centred vs current (pooled, RAW predictions) ===\n")
bt <- boot_mae_diff(res$centred$input_blend_preds, res$current$input_blend_preds)
cat(sprintf("dMAE %+.3f  95%% CI [%.3f, %.3f] | dBrier %+.5f\n",
            bt$mae_diff, bt$mae_ci[1], bt$mae_ci[2], bt$brier_diff))
cat("(want ~0 with a tight CI: the normalisation should be FREE, not better)\n")

cat("\n=== how much did the team feature move? ===\n")
j <- merge(as.data.table(arms$current)[team_type == "home", .(match_id, old = epr_diff)],
           as.data.table(arms$centred)[team_type == "home", .(match_id, new = epr_diff)],
           by = "match_id")
cat(sprintf("cor(epr_diff old, new) = %.5f | mean |change| %.2f | sd(old) %.2f\n",
            cor(j$old, j$new), mean(abs(j$new - j$old)), sd(j$old)))

saveRDS(list(res = res, boot = bt, parallel = TRUE),
        file.path(RES, "ws4_position_centred.rds"))
cli::cli_alert_success("Saved ws4_position_centred.rds")
