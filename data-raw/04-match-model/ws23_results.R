# Results for ws23, recovered from the saved arm predictions.
#
# ws23 crashed in its own summary block on `pred_home_win_prob`, which does not
# exist -- the rolling eval returns `pred_win`, and `margin` / `home_win` are
# already joined on. Every arm had already been written to parquet by then
# (run_arm() checkpoints before returning, precisely so a crash downstream does
# not cost the 20 minutes of evaluation), so nothing needs re-running.
#
# WHAT IS VALID HERE AND WHAT IS NOT. ws23's "v3 ship" arm read
# `epv3_fin_pgd_ship.parquet`, written 08-04 23:22, three contest-changing
# commits before the difficulty arms were built. So:
#
#   ship vs either difficulty arm   CONFOUNDED -- it measures the difficulty
#                                   split PLUS the contest-population changes,
#                                   and cannot separate them. ws24 redoes it
#                                   with all three arms on current code.
#   difficulty vs measured          VALID -- both built fresh, minutes apart,
#                                   differing only in the surprise share.
#
# The second is the question ws23 can actually answer: does the measured share
# (handball ~0.5, kick ~0.7-0.8) beat the assumed flat 0.5?

suppressMessages({ library(data.table); library(arrow) })
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
con <- file(file.path(OUT_DIR, "epv3_difficulty_gate_results.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con); flush(con) }
say_dt <- function(x, n = 40) for (l in capture.output(print(utils::head(x, n)))) say(l)

ARMS <- c("v3 ship" = "epv3_diffgate_v3_ship.parquet",
          "v3 + difficulty" = "epv3_diffgate_v3_difficulty.parquet",
          "v3 + measured" = "epv3_diffgate_v3_measured.parquet")

say("=== ws23 difficulty gate: results ==="); say("run at ", format(Sys.time()))
p <- lapply(ARMS, function(f) as.data.table(read_parquet(file.path(OUT_DIR, f))))
names(p) <- names(ARMS)

.bits <- function(pw, hw) mean(ifelse(hw == 1, 1 + log2(pw),
                              ifelse(hw == 0, 1 + log2(1 - pw), 1 + 0.5 * log2(pw * (1 - pw)))))

summ <- rbindlist(lapply(names(p), function(nm) {
  d <- p[[nm]][is.finite(pred_margin) & is.finite(margin)]
  pw <- pmin(pmax(d$pred_win, 1e-6), 1 - 1e-6)
  data.table(arm = nm, n = nrow(d),
             MAE = round(mean(abs(d$pred_margin - d$margin)), 4),
             RMSE = round(sqrt(mean((d$pred_margin - d$margin)^2)), 4),
             bits = round(.bits(pw, d$home_win), 4),
             tips = sum((d$pred_margin > 0) == (d$margin > 0), na.rm = TRUE))
}))
say(""); say_dt(summ, 5)

common <- Reduce(intersect, lapply(p, function(x) x$match_id))
say(""); say("common matches: ", length(common))
al <- lapply(p, function(x) x[match_id %chin% common][order(match_id)])
ae <- lapply(al, function(x) abs(x$pred_margin - x$margin))

pair <- function(a, b, label, note) {
  d <- ae[[b]] - ae[[a]]
  d <- d[is.finite(d)]
  tt <- t.test(d)
  say(sprintf("  %-34s dMAE %+.4f  95%% CI [%+.4f, %+.4f]  %s",
              label, mean(d), tt$conf.int[1], tt$conf.int[2], note))
}
say("")
say("negative = the second arm is BETTER than the first")
pair("v3 ship", "v3 + difficulty", "difficulty vs ship", "<- CONFOUNDED, see header")
pair("v3 ship", "v3 + measured",   "measured vs ship",   "<- CONFOUNDED, see header")
say("")
pair("v3 + difficulty", "v3 + measured", "measured share vs flat 0.5", "<- VALID")

say("")
say("The valid comparison isolates one thing: EPV_DIFFICULTY_SURPRISE_BY_TYPE.")
say("Both arms are the same code, the same build minute, the same everything")
say("else. A CI spanning zero is still an interval with a point estimate in it --")
say("quote both, and do not read 'spans zero' as 'no evidence'.")

say(""); say("per-season, since the pooled window has hidden a 2025/2026 split before:")
for (nm in names(p)) {
  d <- p[[nm]][is.finite(pred_margin) & is.finite(margin)]
  s <- d[, .(n = .N, MAE = round(mean(abs(pred_margin - margin)), 3),
             tips = sum((pred_margin > 0) == (margin > 0))), by = season][order(season)]
  s[, arm := nm]
  say_dt(s[, .(arm, season, n, MAE, tips)], 5)
}
say("")
say("Decide on the POOLED window. 2026-only over-promised three times in one")
say("earlier session; the per-season split is here to be seen, not to be picked from.")

saveRDS(summ, file.path(OUT_DIR, "epv3_difficulty_gate_results.rds"))
say(""); say("done ", format(Sys.time())); close(con); cat("\nDone\n")
