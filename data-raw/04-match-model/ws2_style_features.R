# WS2 -- do team style features add signal the rating family cannot?
# ==================================================================
# FABLE-MATCH-FEATURES-PLAN.md WS2, and the first genuinely NEW information
# tested in this program.
#
# WS1 established that the team-rating family is saturated: five structurally
# different estimators all converge to cor 0.55-0.56 standalone, and the full
# model subsumes every one of them. Whatever is left has to come from outside
# "team results + xScore history".
#
# The hypothesis: player-rating aggregates measure WHO IS ON THE PARK. Style
# metrics measure HOW THE TEAM PLAYS -- contest dominance, territory, pressure,
# ball use. Plausibly what separates evenly-matched sides, which is exactly
# where the model was shown to be weakest (in the games it calls within 12
# points, its correlation with the result was 0.045 against a field of 0.18-0.22).
#
# Scored on the same terms as everything else: rolling week-by-week OOS, the
# G7 incremental-signal gate against an Elo baseline, and G8's close-call
# bucket. A feature that lowers MAE without adding signal over a plain rating
# has just re-derived the rating, which WS1 showed is a real failure mode here.
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
source(file.path(EXP, "ws2_style_lib.R"))

TEST_SEASONS <- 2025:2026
SEASONS      <- 2021:2026
DF <- file.path(RES, "ws2_team_mdl_df.rds")

# ---- build ------------------------------------------------------------------
if (file.exists(DF)) {
  cli::cli_alert_info("Reusing cached {basename(DF)}")
  team_mdl_df <- readRDS(DF)
} else {
  cli::cli_h1("Building team_mdl_df (corrected ratings)")
  team_mdl_df <- build_team_mdl_df()
  saveRDS(team_mdl_df, DF)
}

cli::cli_h1("Building style features")
ps <- as.data.table(load_player_stats(SEASONS))
cli::cli_inform("player_stats: {nrow(ps)} rows")
tg <- ws2_team_game_stats(ps)
cli::cli_inform("team-game rows: {nrow(tg)}")
print(head(tg, 3))

# Targets = every (match, team) in the model frame, INCLUDING matches with no
# box score yet. History = every match that has one. See ws2_style_profiles().
targets <- unique(as.data.table(team_mdl_df)[
  , .(match_id, team = as.character(team_name.x),
      date = as.Date(substr(as.character(utc_start_time), 1, 10)))])
stopifnot(!anyNA(targets$date), !anyNA(targets$team))
cli::cli_inform("targets: {nrow(targets)} team-matches | history: {nrow(tg)} team-matches")
cli::cli_inform("targets with no box score of their own: {sum(!targets$match_id %in% tg$match_id)}")

prof <- ws2_style_profiles(tg, targets)
cli::cli_inform("profiles: {nrow(prof)} rows, {ncol(prof) - 2} metrics")

cat("\n=== vectorised prefix-sum path vs reference implementation ===\n")
set.seed(1)
samp <- targets[sample(.N, min(150L, .N))]
p_fast <- ws2_style_profiles(tg, samp)
p_ref  <- ws2_style_profiles_ref(tg, samp)
pc <- setdiff(names(p_ref), c("match_id", "team"))
stopifnot(identical(dim(p_fast), dim(p_ref)),
          identical(p_fast$match_id, p_ref$match_id), identical(p_fast$team, p_ref$team))
maxdiff <- max(vapply(pc, function(cc) max(abs(p_fast[[cc]] - p_ref[[cc]])), numeric(1)))
cat("max abs difference across", length(pc), "metrics x", nrow(samp), "targets:",
    format(maxdiff, scientific = TRUE), "\n")
if (maxdiff > 1e-8) cli::cli_abort("WS2: vectorised profiles disagree with the reference")
cat("agree -> the strictly-before boundary is preserved\n")

cat("\n=== leak test: does a target's profile depend on its OWN match? ===\n")
# The two implementations agreeing proves they match EACH OTHER, not that either
# excludes the present. So corrupt a match's box score and require its own
# profile to be bit-identical.
#
# Poison ONE match at a time. The first version of this poisoned all sampled
# matches at once and recomputed all sampled targets -- which flagged a leak that
# was not one: sampled matches span the whole date range, so a late target
# legitimately reads an early poisoned match. Poisoning per match makes
# cross-contamination impossible by construction.
tg_dt <- as.data.table(tg)
mcols <- setdiff(names(tg_dt), c("match_id", "team", "season", "round", "date"))
leak_ids <- unique(samp$match_id)
leak_ids <- leak_ids[seq_len(min(25L, length(leak_ids)))]
leak <- 0
for (mid in leak_ids) {
  tgt <- samp[match_id == mid]
  tgp <- copy(tg_dt)
  hit <- tgp$match_id == mid
  if (!any(hit)) next                          # target with no box score: nothing to poison
  for (cc in mcols) tgp[hit, (cc) := get(cc) * 1000 + 12345]
  pp <- ws2_style_profiles(tgp, tgt)
  base <- p_fast[match_id == mid]
  setkey(pp, team); setkey(base, team)
  leak <- max(leak, max(vapply(pc, function(cc) max(abs(pp[[cc]] - base[[cc]])), numeric(1))))
}
cat("poisoned", length(leak_ids), "matches one at a time; max abs change to their own profiles:",
    format(leak, scientific = TRUE), "\n")
if (leak > 0) cli::cli_abort("WS2: LEAK -- a target's profile moved when its own match was altered")
cat("no leak -> profiles are built strictly from prior matches\n")

team_mdl_df <- ws2_join_style_diffs(team_mdl_df, prof)
sdiff_cols <- grep("_sdiff$", names(team_mdl_df), value = TRUE)
cli::cli_inform("style differentials: {length(sdiff_cols)}")

cat("\n=== leak check: a differential must be antisymmetric within a match ===\n")
tm <- as.data.table(team_mdl_df)
chk <- tm[, .(n = .N), by = match_id][n == 2]
s1 <- tm[match_id %in% chk$match_id[1:200]]
worst <- max(vapply(sdiff_cols, function(cc)
  max(abs(s1[, sum(get(cc)), by = match_id]$V1)), numeric(1)))
cat("max |sum of the two rows| across metrics:", round(worst, 8), "(0 = antisymmetric)\n")

cat("\n=== do the style diffs correlate with the outcome at all? ===\n")
done <- tm[!is.na(win) & team_type == "home"]
cr <- sort(vapply(sdiff_cols, function(cc)
  suppressWarnings(cor(done[[cc]], done$score_diff, use = "complete.obs")), numeric(1)))
print(round(cr, 3))

# ---- score ------------------------------------------------------------------
cli::cli_h1("Rolling OOS: champion vs champion + style features")
base_feats <- "xelo_diff"
arms <- list(
  champion = base_feats,
  style    = c(base_feats, sdiff_cols)
)
out <- list()
for (nm in names(arms)) {
  cli::cli_h2("arm: {nm} ({length(arms[[nm]])} extra feature{?s})")
  t0 <- Sys.time()
  out[[nm]] <- run_rolling_eval(
    team_mdl_df, test_seasons = TEST_SEASONS,
    gam_trainer = .train_match_gams,          # GAM unchanged; XGB gets the features
    xgb_trainer = .train_xgb_fixed,
    extra_feature_cols = arms[[nm]],
    cv_extra_feature_cols = arms[[nm]],
    verbose = FALSE)
  cli::cli_inform("{nm} took {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
}
saveRDS(list(arms = out, sdiff_cols = sdiff_cols), file.path(RES, "ws2_style_arms.rds"))

.bits <- function(pw, hw) mean(ifelse(hw == 1, 1 + log2(pw),
                              ifelse(hw == 0, 1 + log2(1 - pw),
                                     1 + 0.5 * log2(pw * (1 - pw)))))
rep1 <- function(a, lab) {
  ib <- a$input_blend_preds
  # RAW, uncalibrated -- see the same note in ws3_v1_vs_v2_ratings.R. The margin
  # calibration sidecar is fitted against the PRODUCTION model on the full frame;
  # these are rolling-eval predictions from week-by-week refits. Applying it here
  # made the printed table disagree with the bootstrap below, which reads raw.
  # Both arms share a team_mdl_df so it hit them equally, but a transform that
  # is wrong for both is still wrong.
  pm <- ib$pred_margin
  for (s in list(list("2026", ib$season == 2026), list("pooled", rep(TRUE, nrow(ib))))) {
    i <- s[[2]]
    cat(sprintf("%-10s %-7s MAE %.3f | bits %.4f | Brier %.4f | cor %.3f\n",
                lab, s[[1]], mean(abs(pm[i] - ib$margin[i])),
                .bits(ib$pred_win[i], ib$home_win[i]),
                mean((ib$pred_win[i] - ib$home_win[i])^2),
                cor(pm[i], ib$margin[i])))
  }
}
cli::cli_h1("Results")
for (nm in names(out)) rep1(out[[nm]], nm)

cat("\n=== ship gate: style vs champion (pooled) ===\n")
bt <- boot_mae_diff(out$style$input_blend_preds, out$champion$input_blend_preds)
cat(sprintf("dMAE %.3f  95%% CI [%.3f, %.3f] | dBrier %+.5f\n",
            bt$mae_diff, bt$mae_ci[1], bt$mae_ci[2], bt$brier_diff))

cli::cli_h1("G7 -- does it add signal over the Elo baseline?")
base <- elo_baseline_preds(team_mdl_df, test_seasons = TEST_SEASONS)
for (nm in names(out)) signal_gate_report(out[[nm]]$input_blend_preds, base, label = nm)
cli::cli_alert_success("Saved ws2_style_arms.rds")
