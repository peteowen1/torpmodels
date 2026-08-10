# Does each rating CHANNEL convert at 1 point, or only the totals?
# ================================================================
# Pete's question, 2026-07-29. The points-scale work calibrated the TOTALS: one
# EPV_POINTS_SCALE = 0.919 on all four EPV channels, one PSV_POINTS_SCALE = 1.579
# on the shared PSR/PSV betas. Nobody checked the parts.
#
# It matters because the channels are SUMMED: epr = epv_recv + epv_disp +
# epv_spoil + epv_hitout. If recv really converts at 0.80 and disp at 1.10, a
# single global 0.919 leaves them mis-weighted AGAINST EACH OTHER inside that sum
# while the total still calibrates at 1.000. A correct total is not evidence of
# correct parts.
#
# Two fits, and they answer different questions:
#   UNIVARIATE   -- what one channel's diff predicts on its own. Contaminated by
#                   the other channels through their correlation, so a channel
#                   that is merely correlated with a good one looks good.
#   MULTIVARIATE -- what a channel adds holding the others fixed. This is the one
#                   a per-channel CONSTANT would have to represent, because the
#                   constants act inside a sum.
#
# Watch for: hitout is ruck-exclusive, so its team-level diff is near-degenerate
# for most matches and its slope is poorly identified. Expect a wide CI; do not
# chase it.

suppressMessages({
  library(data.table); library(dplyr)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
  devtools::load_all("C:/dev/torpverse/torpmodels", quiet = TRUE)
})
options(torp.local_data_dir = NA)

EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
source(file.path(EXP, "arm_lib.R"))
CACHE <- file.path(EXP, "results", "ws8_scaled_team_mdl.rds")

# ws8 built exactly this frame -- post-scale, one row per match -- so reuse it
# rather than spending 20 minutes rebuilding an identical thing. #136 changed no
# rating (every flag it added defaults FALSE), so the vintage still matches.
if (file.exists(CACHE)) {
  cli::cli_alert_info("Reusing ws8's post-scale team model ({format(file.mtime(CACHE), '%Y-%m-%d %H:%M')})")
  m <- as.data.table(readRDS(CACHE))
} else {
  pgd <- adjust_epv_for_opponents(as.data.table(load_player_game_data(TRUE)))
  psr_df <- .compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE))
  r <- build_ratings_history(2021:2026, pgd = pgd, psr_df = psr_df, opponent_adjust = FALSE)
  src <- load_match_inputs(); src$psr_df <- psr_df
  tm <- as.data.table(build_team_mdl_with(src, as.data.frame(r)))
  tm <- tm[is.finite(score_diff)]
  m <- if ("team_type" %in% names(tm)) tm[team_type == "home"] else unique(tm, by = "match_id")
  saveRDS(m, CACHE)
}
# team_mdl_df carries the season under the join suffixes (season.x / season.y),
# not a bare `season` -- reading m$season gives NULL, which silently turns the
# per-season table into zero rows rather than erroring.
if (!"season" %in% names(m)) {
  src_col <- intersect(c("season.x", "game_year"), names(m))
  if (length(src_col) == 0) cli::cli_abort("No season column in the cached frame.")
  m[, season := as.integer(get(src_col[1]))]
  cli::cli_alert_info("Derived {.field season} from {.field {src_col[1]}}")
}
stopifnot(!anyNA(m$season))
cli::cli_alert_info("{nrow(m)} match rows, seasons {min(m$season)}-{max(m$season)}")

EPV_CH <- c("epr_recv_diff", "epr_disp_diff", "epr_spoil_diff", "epr_hitout_diff")
PSV_CH <- c("osr_diff", "dsr_diff")

cli::cli_h1("0. do the parts actually sum to the total?")
# If they do not, a per-channel constant cannot be reasoned about as a
# decomposition of the total at all -- so this is checked before anything else
# rather than assumed.
ok <- Reduce(`&`, lapply(c(EPV_CH, "epr_diff"), function(v) is.finite(m[[v]])))
gap <- with(m[ok], epr_diff - (epr_recv_diff + epr_disp_diff + epr_spoil_diff + epr_hitout_diff))
cli::cli_alert_info("epr_diff - sum(channels): max |gap| = {signif(max(abs(gap)), 3)} over {sum(ok)} rows")
if (max(abs(gap)) > 1e-6) {
  cli::cli_alert_danger("The channels do NOT sum to epr_diff -- treat the decomposition below as indicative only.")
}
if (all(PSV_CH %in% names(m))) {
  ok2 <- Reduce(`&`, lapply(c(PSV_CH, "psr_diff"), function(v) is.finite(m[[v]])))
  if (any(ok2)) {
    g2 <- with(m[ok2], psr_diff - (osr_diff + dsr_diff))
    cli::cli_alert_info("psr_diff - (osr+dsr): max |gap| = {signif(max(abs(g2)), 3)} over {sum(ok2)} rows")
  }
}

cli::cli_h1("1. UNIVARIATE slopes (each channel on its own)")
uni <- rbindlist(lapply(c(EPV_CH, PSV_CH, "epr_diff", "psr_diff", "torp_diff"), function(v) {
  if (!v %in% names(m)) return(NULL)
  x <- m[[v]]; y <- m$score_diff; k <- is.finite(x) & is.finite(y)
  if (sum(k) < 50 || stats::sd(x[k]) < 1e-8) {
    return(data.table(channel = v, n = sum(k), slope = NA_real_, lo = NA_real_,
                      hi = NA_real_, sd_x = round(stats::sd(x[k]), 3), r2 = NA_real_))
  }
  f <- lm(y[k] ~ x[k]); ci <- confint(f)[2, ]
  data.table(channel = v, n = sum(k), slope = round(coef(f)[2], 3),
             lo = round(ci[1], 3), hi = round(ci[2], 3),
             sd_x = round(stats::sd(x[k]), 3), r2 = round(summary(f)$r.squared, 4))
}))
uni[, hits_one := !is.na(lo) & lo <= 1 & hi >= 1]
print(uni, row.names = FALSE)

cli::cli_h1("2. why univariate is not the answer: the channels are correlated")
cm <- m[, ..EPV_CH]
cm <- cm[Reduce(`&`, lapply(EPV_CH, function(v) is.finite(cm[[v]])))]
print(round(cor(cm), 3))

cli::cli_h1("3. MULTIVARIATE decomposition (what a per-channel constant means)")
# This is the fit that matters: the constants act inside a sum, so each one has
# to represent the channel's contribution holding the others fixed.
fml <- as.formula(paste("score_diff ~", paste(EPV_CH, collapse = " + ")))
fit <- lm(fml, data = m[Reduce(`&`, lapply(c(EPV_CH, "score_diff"), function(v) is.finite(m[[v]])))])
ci <- confint(fit)
multi <- data.table(channel = rownames(ci), slope = round(coef(fit), 3),
                    lo = round(ci[, 1], 3), hi = round(ci[, 2], 3))
multi[, hits_one := lo <= 1 & hi >= 1]
print(multi[channel != "(Intercept)"], row.names = FALSE)
cli::cli_alert_info("model r2 {round(summary(fit)$r.squared, 4)}; joint F p = {signif(anova(fit)[['Pr(>F)']][1], 3)}")

# The implied correction, if this were to ship. A channel currently scaled by
# EPV_POINTS_SCALE that converts at slope s needs an extra factor of s to reach
# 1:1 -- so its own constant would be EPV_POINTS_SCALE * s.
cli::cli_h1("4. implied per-channel constants (multivariate)")
imp <- multi[channel != "(Intercept)"]
imp[, `:=`(current = EPV_POINTS_SCALE,
           implied = round(EPV_POINTS_SCALE * slope, 4),
           ratio_to_global = round(slope, 3))]
print(imp[, .(channel, current, implied, ratio_to_global, lo, hi, hits_one)], row.names = FALSE)
cli::cli_alert_info("ratio_to_global is how far each channel is from the single global constant.")
cli::cli_alert_info("A channel whose CI contains 1 is not distinguishable from the global one.")

cli::cli_h1("4b. how much each channel actually MOVES a prediction")
# A slope of 7 on a channel whose team-level diff has sd 0.6 contributes less
# than a slope of 0.6 on one with sd 14. Without this column a big ratio reads
# as a big problem, which is how a rescale gets prioritised by the wrong number.
mag <- merge(imp[, .(channel, slope)], uni[, .(channel, sd_x)], by = "channel")
mag[, `:=`(pts_per_sd_now = round(slope * sd_x, 2))]
tot_sd <- uni[channel == "epr_diff"]$sd_x
mag[, share_of_epr_sd := round(sd_x / tot_sd, 3)]
print(mag[order(-pts_per_sd_now)], row.names = FALSE)

cli::cli_h1("4c. would per-channel constants keep the TOTAL at 1:1?")
# The constants only make sense if applying them leaves the total calibrated --
# otherwise fixing the parts breaks the thing that currently works. Rescaling
# channel i by its multivariate slope should send every partial slope to 1, and
# the total follows by linearity. Check it rather than assert it.
sc_map <- setNames(imp$slope, imp$channel)
mm <- copy(m)
for (v in names(sc_map)) mm[, (v) := get(v) * sc_map[[v]]]
mm[, epr_diff_rescaled := Reduce(`+`, lapply(EPV_CH, function(v) mm[[v]]))]
k <- is.finite(mm$epr_diff_rescaled) & is.finite(mm$score_diff)
f3 <- lm(score_diff ~ epr_diff_rescaled, data = mm[k])
c3 <- confint(f3)[2, ]
cli::cli_alert_info("total slope after per-channel rescale: {round(coef(f3)[2], 3)} [{round(c3[1],3)}, {round(c3[2],3)}] (was 1.000)")
f4 <- lm(as.formula(paste("score_diff ~", paste(EPV_CH, collapse = " + "))), data = mm[k])
cli::cli_alert_info("per-channel slopes after rescale (target 1.000 each): {paste(round(coef(f4)[-1], 3), collapse = ', ')}")

cli::cli_h1("5. per-season wander (the totals already wandered: EPR 0.87-1.02)")
# Per channel this will be worse. A global constant per channel is still the
# right simplification -- per-season would be fitting noise -- but the spread is
# what says how much to trust any single number above.
#
# These are UNIVARIATE slopes, so do NOT read them against section 3's
# multivariate ones -- a univariate slope is inflated by every other channel it
# correlates with (recv/disp correlate 0.79). What they are good for is the
# question of stability: whether a channel's mis-scaling is consistent across
# seasons or an artefact of one.
seas <- rbindlist(lapply(sort(unique(m$season)), function(s) {
  d <- m[season == s]
  out <- data.table(season = s, n = nrow(d))
  for (v in EPV_CH) {
    x <- d[[v]]; y <- d$score_diff; k <- is.finite(x) & is.finite(y)
    out[[v]] <- if (sum(k) > 30 && stats::sd(x[k]) > 1e-8) round(coef(lm(y[k] ~ x[k]))[2], 3) else NA_real_
  }
  out
}))
print(seas, row.names = FALSE)

cli::cli_h1("6. osr / dsr, which are reconciled to sum to psr")
if (all(PSV_CH %in% names(m)) && any(is.finite(m$osr_diff))) {
  f2 <- lm(score_diff ~ osr_diff + dsr_diff,
           data = m[is.finite(osr_diff) & is.finite(dsr_diff) & is.finite(score_diff)])
  c2 <- confint(f2)
  p2 <- data.table(channel = rownames(c2), slope = round(coef(f2), 3),
                   lo = round(c2[, 1], 3), hi = round(c2[, 2], 3))
  p2[, hits_one := lo <= 1 & hi >= 1]
  print(p2[channel != "(Intercept)"], row.names = FALSE)
  cli::cli_alert_info("If these differ, the residual shift that forces osv + dsv = psv is absorbing a scale error.")
} else {
  cli::cli_alert_warning("osr_diff/dsr_diff absent or all NA in this frame -- the osv/dsv half is NOT answered.")
}

saveRDS(list(uni = uni, multi = multi, implied = imp, per_season = seas),
        file.path(EXP, "results", "ws13_per_channel_points_scale.rds"))
cli::cli_alert_success("ws13 done")
