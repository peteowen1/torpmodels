# Production match gate for the three centring fixes.
#
# WHAT IS BEING GATED, all v2, all currently behind flags:
#   ROLE_REMAP_BENCH            a bench start is not a role; 21% of player-games
#   EPV_HITOUT_CENTRE_ON_RUCK   the hitout channel cells on ruck involvement
#   EPV_RUCK_BLEND_WIDTH = 10   and blends across it rather than switching
#
# Fixes a live bug: Sean Darcy 5th in the competition because, having started on
# the bench, he was measured against benchwarmers. Max Gawn's ruck channel reads
# NEGATIVE in production and +0.65 after.
#
# EXPECT A NULL, AND THAT IS THE PASS. This changes which players get credit
# WITHIN a team, and the team total is very nearly untouched -- the same 22
# players' values are being reallocated among themselves. The match model sees
# team aggregates. So there is little for it to see, and a null means "did not
# break prediction", which is all this is being asked.
#
# The case for the change is the leaderboard (centring_leaderboard.txt): position
# mix identical, Spearman 0.9636, the biggest fallers are exactly the
# ruck-forwards who were credited against forwards, and nobody appears from
# nowhere. None of that is visible to a match gate.
#
# Both arms use IDENTICAL rating constants and pre-built rating frames, so the
# only difference is the player-game frame. ~45 min. Run detached.

suppressMessages({
  library(dplyr); library(data.table); library(arrow)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
TEST_SEASONS <- 2025:2026

con <- file(file.path(OUT_DIR, "centring_gate.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con); flush(con) }
say_dt <- function(x, n = 40) for (l in capture.output(print(utils::head(x, n)))) say(l)
set_const <- function(l) for (nm in names(l)) assignInNamespace(nm, l[[nm]], ns = "torp")

say("=== Centring fixes: production match gate ===")
say("run at ", format(Sys.time()), " | test seasons ", paste(TEST_SEASONS, collapse = "-"))

teams <- load_teams(TRUE); shared_fixtures <- load_fixtures(TRUE)
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                   error = function(e) NULL)
res <- as.data.table(load_results(TRUE))

V2 <- list(EPV_ENGINE = "v2", EPV3_CHANNELS = 3L,
           EPV3_SUB_SCALE = c(cont_aerial = 1, cont_stop = 1),
           EPV3_STOP_ZERO_SUM = FALSE,
           EPV_STANDARDISE_CHANNELS = c("recv", "disp", "spoil"),
           EPV3_POINTS_SCALE = c(recv = 1, disp = 1, cont_aerial = 1, cont_stop = 1),
           EPV_PER_CHANNEL_POINTS_SCALE = FALSE, EPV_POINTS_SCALE = 0.919,
           EPR_PRIOR_RATE_RECV = -0.7 * 0.919, EPR_PRIOR_RATE_DISP = -0.7 * 0.919,
           EPR_PRIOR_RATE_SPOIL = -0.3 * 0.919, EPR_PRIOR_RATE_HITOUT = -0.3 * 0.919,
           EPR_PRIOR_GAMES_RECV = 3, EPR_PRIOR_GAMES_DISP = 3,
           EPR_PRIOR_GAMES_SPOIL = 3, EPR_PRIOR_GAMES_HITOUT = 3)
set_const(V2)

rts <- list(
  "v2 production" = as.data.table(read_parquet(file.path(OUT_DIR, "centring_rt_before.parquet"))),
  "v2 + centring" = as.data.table(read_parquet(file.path(OUT_DIR, "centring_rt_after.parquet"))))

say(""); say("--- ARMS GUARD ---")
k <- c("player_id", "season", "round")
m <- merge(rts[[1]][, c(k, "epr"), with = FALSE], rts[[2]][, c(k, "epr"), with = FALSE],
           by = k, suffixes = c("_a", "_b"))
dd <- mean(abs(m$epr_a - m$epr_b), na.rm = TRUE)
say(sprintf("  mean|diff| in epr: %.5f over %s rows", dd, format(nrow(m), big.mark = ",")))
if (dd < 1e-9) { say("  !! IDENTICAL -- aborting"); close(con); quit(status = 1) }

build_with_ratings <- function(torp_df) {
  ag <- file_reader("stadium_data", "reference-data")
  fx <- .build_fixtures_df(shared_fixtures)
  trt <- .build_team_ratings_df(teams, torp_df, psr_df)
  trf <- .build_match_features(fx, trt, ag)
  wx <- .load_match_weather(shared_fixtures, ag, NULL, get_afl_season())
  .build_team_mdl_df(trf, load_results(TRUE), load_xg(TRUE), wx,
                     max(as.Date(fx$utc_start_time), na.rm = TRUE))
}
.bits <- function(pw, hw) mean(ifelse(hw == 1, 1 + log2(pw),
                              ifelse(hw == 0, 1 + log2(1 - pw), 1 + 0.5 * log2(pw * (1 - pw)))))

preds <- list()
for (nm in names(rts)) {
  say(""); say("=== evaluating: ", nm, " ===")
  tm <- build_with_ratings(as.data.frame(rts[[nm]]))
  ft <- grep("^(epr|psr|torp|elo|xelo).*_diff$|^(epr|psr|torp)\\.[xy]$", names(tm), value = TRUE)
  keep <- stats::complete.cases(tm[, ft, drop = FALSE]); if (any(!keep)) tm <- tm[keep, , drop = FALSE]
  degen <- ft[vapply(ft, function(v) length(unique(tm[[v]][is.finite(tm[[v]])])) < 50, logical(1))]
  if (length(degen)) say("  degenerate (reported, not removed): ", paste(degen, collapse = ", "))
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff", cv_extra_feature_cols = "xelo_diff")
  p <- unique(as.data.table(roll$input_blend_preds), by = "match_id")[, arm := nm]
  write_parquet(p, file.path(OUT_DIR,
    paste0("centring_gate_", gsub("[^a-z0-9]+", "_", tolower(nm)), ".parquet")))
  preds[[nm]] <- p
}

say(""); say("=== RESULTS ===")
allp <- rbindlist(preds, use.names = TRUE, fill = TRUE)[is.finite(pred_margin) & is.finite(margin)]
say_dt(allp[, .(n = .N, MAE = round(mean(abs(pred_margin - margin)), 4),
                RMSE = round(sqrt(mean((pred_margin - margin)^2)), 4),
                bits = round(.bits(pmin(pmax(pred_win, 1e-6), 1 - 1e-6), home_win), 4),
                tips = sum((pred_margin > 0) == (margin > 0))), by = arm], 5)

common <- Reduce(intersect, lapply(preds, function(p) p$match_id))
ba <- preds[["v2 production"]][match_id %chin% common][order(match_id)]
q  <- preds[["v2 + centring"]][match_id %chin% common][order(match_id)]
d <- abs(q$pred_margin - q$margin) - abs(ba$pred_margin - ba$margin); d <- d[is.finite(d)]
tt <- t.test(d)
say(""); say(sprintf("paired on %d matches: dMAE %+.4f  95%% CI [%+.4f, %+.4f]",
                     length(common), mean(d), tt$conf.int[1], tt$conf.int[2]))
say("(negative = the fix is better; this is a guardrail, so ~0 is the pass)")
say("")
say("A null is expected and wanted. The change reallocates credit WITHIN a team")
say("and the match model sees team aggregates, so there is little for it to see.")
say("The case for the change is the leaderboard, which no match gate can score.")

saveRDS(list(dMAE = mean(d), ci = tt$conf.int), file.path(OUT_DIR, "centring_gate.rds"))
say(""); say("done ", format(Sys.time())); close(con); cat("\nDone\n")
