# The production match gate for the v4 engine (Net Points) against v3 production.
#
# The guardrail, not the decision (docs/HOW-WE-WORK.md section 2): the
# six-season fast EPR gate already chose v4 (run_epr_gate_v3v4.R in torp), and
# Pete flipped EPV_ENGINE on 2026-09-06. This bounds the PRICE in the full
# pipeline -- five GAMs and an XGBoost consuming every channel diff plus
# epr_diff -- where the rating is one feature among many and every earlier
# engine change has gated near neutral. Read it once; do not iterate on it.
#
# Two arms, each under its own constants, frames cached from the fast gate:
#   1  v3 production  -- EPV_ENGINE v3 exactly as shipped 2026-08-18
#   2  v4             -- EPV_ENGINE v4 with EPR_UNITS_SCALE_V4 (2.40)
#
# ~20 min per arm. Run detached:
#   Start-Process Rscript -ArgumentList '"<this file>"'
suppressMessages({
  library(dplyr); library(data.table); library(arrow)
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
})
options(torp.local_data_dir = NA)
EXP <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
OUT_DIR <- "C:/dev/torpverse/torp/data-raw/outputs"
source(file.path(EXP, "rolling_lib.R"))
TEST_SEASONS <- 2025:2026
con <- file(file.path(OUT_DIR, "epv4_prod_gate.txt"), open = "wt")
say <- function(...) { m <- paste0(...); cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = con); flush(con) }
say_dt <- function(x, n = 40) for (l in capture.output(print(utils::head(x, n)))) say(l)
set_const <- function(l) for (nm in names(l)) assignInNamespace(nm, l[[nm]], ns = "torp")
rd <- function(f) as.data.table(read_parquet(file.path(OUT_DIR, f)))

say("=== Production match gate: v4 Net Points against v3 production ===")
say("run at ", format(Sys.time()), " | test seasons ", paste(TEST_SEASONS, collapse = "-"))
say("loaded engine ", EPV_ENGINE, " | units ", EPR_UNITS_SCALE_V4)

teams <- load_teams(TRUE)
shared_stat_ratings <- get_player_stat_ratings(current = FALSE)
shared_fixtures     <- load_fixtures(TRUE)
psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)), error = function(e) NULL)

# the constants each arm needs, held with the arm (see ws22 for why)
V3_CONST <- list(EPV_ENGINE = "v3",
                 EPV3_POINTS_SCALE = c(recv = 1.650438, disp = 1.421199, cont_aerial = 0.337485, cont_stop = 1.659480),
                 EPV_STANDARDISE_CHANNELS = c("recv", "disp"),
                 EPR_PRIOR_RATE_RECV = -0.7 * 1.650438, EPR_PRIOR_RATE_DISP = -0.7 * 1.421199,
                 EPR_PRIOR_RATE_SPOIL = -0.3 * 0.337485)
V4_CONST <- list(EPV_ENGINE = "v4",
                 EPV3_POINTS_SCALE = c(recv = 1, disp = 1, cont_aerial = 1, cont_stop = 1),
                 EPV_STANDARDISE_CHANNELS = c("recv", "disp"),
                 EPR_PRIOR_RATE_RECV = -0.7, EPR_PRIOR_RATE_DISP = -0.7, EPR_PRIOR_RATE_SPOIL = -0.3)

build_ratings <- function(pgd, tag, engine) {
  f <- file.path(OUT_DIR, paste0("epv4_gate_rt_", tag, ".parquet"))
  if (file.exists(f)) { cli::cli_alert_info("Reusing ratings {tag}"); return(rd(basename(f))) }
  d <- adjust_epv_for_opponents(as.data.table(copy(pgd)))
  setattr(d, "epv_engine", engine)
  if (isTRUE(EPV_LEVEL_CENTRE)) d <- centre_epv_by_position(d)
  out <- rbindlist(lapply(sort(unique(d$season)), function(s) {
    sr <- if (s >= 2024) 0 else 1
    mr <- if (s == get_afl_season()) get_afl_week(type = "next") else 28
    torp:::.build_epr_season(s, sr:mr, d, shared_stat_ratings, shared_fixtures)
  }), use.names = TRUE, fill = TRUE)
  if (isTRUE(EPR_POSITION_CENTRE)) out <- centre_epr_by_position(out)
  if (!is.null(psr_df) && nrow(psr_df) > 0 && "psr" %in% names(psr_df)) out <- calculate_torp(out, psr_df)
  out <- as.data.table(out); write_parquet(out, f); out
}

ARMS <- list(
  list(label = "v3 production", const = V3_CONST, pgd = "v3v4_pgd_v3.parquet", eng = "v3", rt = "v3"),
  list(label = "v4 net points", const = V4_CONST, pgd = "v3v4_pgd_v4.parquet", eng = "v4", rt = "v4units")
)
rts <- list()
for (a in ARMS) {
  set_const(a$const)
  pgd <- rd(a$pgd); setattr(pgd, "epv_engine", a$eng)
  rts[[a$label]] <- build_ratings(pgd, a$rt, a$eng)
  s <- rts[[a$label]][!is.na(epr)]; s <- s[season == max(season)][round == max(round)]
  say(sprintf("  %-15s ratings %d rows | latest round: mean epr %.2f sd %.2f | top: %s", a$label, nrow(rts[[a$label]]),
              mean(s$epr), sd(s$epr), paste(head(s[order(-epr)]$player_name, 5), collapse = ", ")))
}
say("--- ARMS GUARD ---")
d <- merge(rts[[1]][, .(player_id, season, round, a = epr)], rts[[2]][, .(player_id, season, round, b = epr)],
           by = c("player_id", "season", "round"))
say(sprintf("  rating tables share %d rows; mean |diff| %.3f; cor %.3f (must not be identical)", nrow(d), mean(abs(d$a - d$b), na.rm = TRUE), cor(d$a, d$b, use = "complete.obs")))

.bits <- function(pw, hw) mean(ifelse(hw == 1, 1 + log2(pw),
                              ifelse(hw == 0, 1 + log2(1 - pw),
                                     1 + 0.5 * log2(pw * (1 - pw)))))
build_with_ratings <- function(torp_df) {
  ag <- file_reader("stadium_data", "reference-data")
  fx <- .build_fixtures_df(shared_fixtures)
  trt <- .build_team_ratings_df(teams, torp_df, psr_df)
  trf <- .build_match_features(fx, trt, ag)
  wx <- .load_match_weather(shared_fixtures, ag, NULL, get_afl_season())
  .build_team_mdl_df(trf, load_results(TRUE), load_xg(TRUE), wx,
                     max(as.Date(fx$utc_start_time), na.rm = TRUE))
}
run_arm <- function(label, torp_df) {
  tm <- build_with_ratings(torp_df)
  ft <- grep("^(epr|psr|torp|elo|xelo).*_diff$|^(epr|psr|torp)\\.[xy]$", names(tm), value = TRUE)
  keep <- stats::complete.cases(tm[, ft, drop = FALSE])
  if (any(!keep)) tm <- tm[keep, , drop = FALSE]
  roll <- run_rolling_eval(tm, test_seasons = TEST_SEASONS,
                           gam_trainer = .train_match_gams, xgb_trainer = .train_xgb_fixed,
                           extra_feature_cols = "xelo_diff", cv_extra_feature_cols = "xelo_diff")
  p <- unique(as.data.table(roll$input_blend_preds), by = "match_id")
  p[, arm := label]
  write_parquet(p, file.path(OUT_DIR, paste0("epv4_prod_gate_", gsub("[^a-z0-9]+", "_", tolower(label)), ".parquet")))
  p
}
preds <- rbindlist(lapply(ARMS, function(a) { set_const(a$const); say("--- running arm: ", a$label, " at ", format(Sys.time())); run_arm(a$label, rts[[a$label]]) }),
                   use.names = TRUE, fill = TRUE)
common <- Reduce(intersect, split(preds$match_id, preds$arm))
preds <- preds[match_id %in% common]
say(""); say("--- same-games: ", length(common), " matches in every arm ---")
metrics <- function(p, seasons = NULL) {
  d <- if (is.null(seasons)) p else p[season %in% seasons]
  d <- d[is.finite(margin) & is.finite(pred_margin) & is.finite(pred_win)]
  hw <- ifelse(d$margin > 0, 1, ifelse(d$margin == 0, 0.5, 0))
  data.table(n = nrow(d), MAE = round(mean(abs(d$pred_margin - d$margin)), 4),
             RMSE = round(sqrt(mean((d$pred_margin - d$margin)^2)), 4),
             bits = round(.bits(pmin(pmax(d$pred_win, 1e-6), 1 - 1e-6), hw), 4),
             Brier = round(mean((d$pred_win - hw)^2), 4),
             tips = sum((d$pred_margin > 0) == (d$margin > 0), na.rm = TRUE))
}
say(""); say("=== GATE: pooled 2025-26 (decide on this window) ===")
say_dt(preds[, metrics(.SD), by = arm], 12)
say(""); say("--- by season, reported not decided on ---")
say_dt(preds[, metrics(.SD, 2025), by = arm], 12); say_dt(preds[, metrics(.SD, 2026), by = arm], 12)
say(""); say("--- paired against v3 production (negative = v4 BETTER) ---")
base <- preds[arm == "v3 production", .(match_id, e0 = abs(pred_margin - margin))]
x <- preds[arm == "v4 net points", .(match_id, e1 = abs(pred_margin - margin))]
m <- merge(base, x, by = "match_id"); dd <- m$e1 - m$e0; ci <- t.test(dd)$conf.int
say(sprintf("  v4 net points  dMAE %+.4f  95%% CI [%+.4f, %+.4f]  P(better) %.3f", mean(dd), ci[1], ci[2], mean(dd < 0)))
write_parquet(preds, file.path(OUT_DIR, "epv4_prod_gate_preds.parquet"))
say(""); say("done ", format(Sys.time())); close(con); cat("\nDone\n")
