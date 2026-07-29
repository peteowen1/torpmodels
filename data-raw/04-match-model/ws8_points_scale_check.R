# Did the points-scale calibration land 1 rating point = 1 scoreboard point?
# =========================================================================
# EPV_POINTS_SCALE / PSV_POINTS_SCALE were set from slopes measured on the
# PRE-scale ratings (EPR 0.919, PSR 1.579). This rebuilds with them applied and
# re-measures. Expected: both slopes ~1.000.
#
# It is not self-evidently exact. EPR passes through .bayesian_shrink(), whose
# prior term is scaled by a matching constant rather than by construction, and
# PSR passes through centring + standardisation. Both are argued to be
# scale-equivariant; this is where that argument gets checked against numbers.
suppressMessages({
  library(data.table); library(dplyr)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
  devtools::load_all("C:/dev/torpverse/torpmodels", quiet = TRUE)
})
options(torp.local_data_dir = NA)
EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
source(file.path(EXP, "arm_lib.R"))

pgd <- adjust_epv_for_opponents(as.data.table(load_player_game_data(TRUE)))
psr_df <- .compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE))
r <- build_ratings_history(2021:2026, pgd = pgd, psr_df = psr_df, opponent_adjust = FALSE)
src <- load_match_inputs(); src$psr_df <- psr_df
tm <- as.data.table(build_team_mdl_with(src, as.data.frame(r)))

tm <- tm[is.finite(score_diff)]
m <- if ("team_type" %in% names(tm)) tm[team_type == "home"] else unique(tm, by = "match_id")
cat(sprintf("\nmatch rows: %d\n\n", nrow(m)))
cat("=== slopes AFTER scaling (target 1.000) ===\n")
for (v in c("epr_diff", "psr_diff", "torp_diff")) {
  x <- m[[v]]; y <- m$score_diff; ok <- is.finite(x) & is.finite(y)
  f <- lm(y[ok] ~ x[ok]); ci <- confint(f)[2, ]
  hit <- if (ci[1] <= 1 && ci[2] >= 1) "  <- 1:1 achieved" else "  <- still off"
  cat(sprintf("  %-10s slope %6.3f [%6.3f, %6.3f]%s\n", v, coef(f)[2], ci[1], ci[2], hit))
}
saveRDS(m, file.path(EXP, "results", "ws8_scaled_team_mdl.rds"))
cli::cli_alert_success("ws8 done")
