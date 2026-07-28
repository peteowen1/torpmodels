# Live Squiggle leaderboard watch -- is C6 actually delivering?
# =============================================================
# FABLE-MATCH-FEATURES-PLAN.md §6.0. Network-only and fast (~5s): no model
# training, no data rebuild. Distinct from eval_squiggle_rank.R, which re-runs
# the harness to ask "where WOULD each variant rank"; this asks "where IS torp
# actually ranking, on the rounds where the current model was serving".
#
# WHY THIS EXISTS
# ---------------
# torp's 2026 record (MAE 26.49, 29th of 31) is a measurement of the PRE-C6
# model: C6 merged to main 2026-07-14 and its calibration sidecar first
# published 2026-07-20, so only rounds ~19+ reflect what is running now. A
# fresh full-season rolling backtest of the current model scored MAE 24.895 /
# bits 0.2413 against Aggregate's 24.93 / 0.2421 -- i.e. it should be at or
# just past the bar. This script watches for that showing up in live results.
#
# It deliberately reports the post-cutover sample size and a "too early to
# judge" verdict rather than a bare number, because with ~9 games a round the
# early post-C6 windows are far too small to distinguish a 0.5 MAE effect from
# noise -- and a good-looking 2-round window is exactly the kind of thing that
# invites a premature conclusion.
#
# Usage:  Rscript torpmodels/data-raw/04-match-model/watch_live_leaderboard.R [YEAR]

suppressMessages({ library(jsonlite); library(data.table) })

YEAR <- { a <- commandArgs(trailingOnly = TRUE)[1]; if (is.na(a)) 2026L else as.integer(a) }

# Round from which the current model was serving. C6's calibration sidecar was
# first published 2026-07-20; round 19 is the first full round after that.
C6_FIRST_ROUND <- 19L
OUR_SOURCE     <- "In The Game"
# Enough completed games post-cutover to say anything at all. The season-long
# gap being chased is ~1.5 MAE with a per-game sd of ~40, so distinguishing it
# needs order-40 games; below that the honest answer is "not yet".
MIN_N_TO_JUDGE <- 40L

.sq <- function(q) {
  con <- url(sprintf("https://api.squiggle.com.au/?q=%s", q),
             headers = c("User-Agent" = "torpverse-leaderboard-watch (fptpost@gmail.com)"))
  on.exit(try(close(con), silent = TRUE))
  fromJSON(paste(readLines(con, warn = FALSE), collapse = ""))
}

games <- as.data.table(.sq(sprintf("games;year=%d", YEAR))$games)
tips  <- as.data.table(.sq(sprintf("tips;year=%d", YEAR))$tips)

g <- games[complete == 100, .(gameid = as.integer(id), round,
                              hmargin = hscore - ascore)]
t <- tips[, .(gameid = as.integer(gameid), source, tip, hteam,
              margin = as.numeric(margin), hconf = as.numeric(hconfidence))]
d <- merge(t, g, by = "gameid")
d[, pred := ifelse(tip == hteam, margin, -margin)]
d[, p := pmin(pmax(hconf / 100, 1e-6), 1 - 1e-6)]
d[, hw := fifelse(hmargin > 0, 1, fifelse(hmargin == 0, 0.5, 0))]
# Squiggle's own bits convention
d[, bits := fifelse(hw == 1, 1 + log2(p),
             fifelse(hw == 0, 1 + log2(1 - p), 1 + 0.5 * log2(p * (1 - p))))]
d[, ae := abs(pred - hmargin)]

n_games <- uniqueN(d$gameid)
max_rd  <- max(g$round)

.board <- function(dat, label) {
  n_here <- uniqueN(dat$gameid)
  res <- dat[, .(n = .N, MAE = mean(ae), bits = mean(bits),
                 correct = mean(fifelse(hw == 0.5, 0.5,
                                        as.numeric((p > 0.5) == (hw == 1))))),
             by = source][n >= 0.8 * n_here][order(MAE)]
  res[, rank_mae := .I]
  res[, rank_bits := frank(-bits, ties.method = "first")]
  cli_line <- sprintf("%s  (%d games, %d qualifying sources)", label, n_here, nrow(res))
  cat("\n", strrep("=", nchar(cli_line)), "\n", cli_line, "\n",
      strrep("=", nchar(cli_line)), "\n", sep = "")
  show <- res[source %in% c(OUR_SOURCE, "Aggregate", "Wheelo Ratings") |
                rank_mae <= 3]
  print(show[order(rank_mae),
             .(source, n, MAE = round(MAE, 2), bits = round(bits, 4),
               rank_mae, rank_bits)], row.names = FALSE)
  res
}

cat(sprintf("Squiggle %d: %d completed games, latest round %d\n", YEAR, n_games, max_rd))
full <- .board(d, sprintf("FULL SEASON (rounds 0-%d)", max_rd))
post <- if (max_rd >= C6_FIRST_ROUND) {
  .board(d[round >= C6_FIRST_ROUND],
         sprintf("POST-C6 ONLY (rounds %d-%d)", C6_FIRST_ROUND, max_rd))
} else NULL

# --- verdict ------------------------------------------------------------------
cat("\n--- VERDICT ---\n")
if (is.null(post)) {
  cat("No completed rounds since C6 went live. Nothing to judge yet.\n")
} else {
  n_post <- uniqueN(d[round >= C6_FIRST_ROUND]$gameid)
  us  <- post[source == OUR_SOURCE]
  agg <- post[source == "Aggregate"]
  if (nrow(us) == 0 || nrow(agg) == 0) {
    cat("Could not find both", OUR_SOURCE, "and Aggregate in the post-C6 window.\n")
  } else {
    cat(sprintf("Post-C6: %s MAE %.2f (rank %d) / bits %.4f (rank %d)\n",
                OUR_SOURCE, us$MAE, us$rank_mae, us$bits, us$rank_bits))
    cat(sprintf("         Aggregate MAE %.2f / bits %.4f\n", agg$MAE, agg$bits))
    cat(sprintf("         delta: MAE %+.2f, bits %+.4f  (negative MAE / positive bits = we win)\n",
                us$MAE - agg$MAE, us$bits - agg$bits))
    if (n_post < MIN_N_TO_JUDGE) {
      cat(sprintf("\nTOO EARLY: %d post-C6 games, want >= %d before reading anything into this.\n",
                  n_post, MIN_N_TO_JUDGE))
      cat("A 1-2 round window cannot separate a real 0.5 MAE effect from noise.\n")
    } else {
      beat_mae  <- us$MAE  < agg$MAE
      beat_bits <- us$bits > agg$bits
      cat(sprintf("\n%d post-C6 games -- enough for a first read.\n", n_post))
      cat(sprintf("Beats Aggregate on MAE: %s | on bits: %s\n",
                  ifelse(beat_mae, "YES", "no"), ifelse(beat_bits, "YES", "no")))
      cat("Backtest expectation was MAE ~24.9 / bits ~0.241 (at or just past the bar).\n")
      cat("A large shortfall vs that points at the serve path, not the model\n")
      cat("(see FABLE-MATCH-FEATURES-PLAN.md 6.5 -- stored vs submitted divergence).\n")
    }
  }
}
