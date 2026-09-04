# Re-run C6 vs C7-fixed ensemble on current, wider data (more 2026 rounds
# played since the original round-2 measurement, n=369 -> current).
# Uses CURRENT torp:::.train_match_gams()/.train_match_xgb() (with the
# season-grouped OOF-stacking fix applied earlier this session) and
# CURRENT rolling_lib.R's .train_xgb_fixed() (same fix ported).

RESULTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/results"
EXPERIMENTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
.rds <- function(name) file.path(RESULTS_DIR, name)

suppressPackageStartupMessages({
  library(dplyr); library(cli); library(MLmetrics); library(tidyverse); library(xgboost); library(mgcv)
})
torp_paths <- c("../torp", "../../torp", "../../../torp", "C:/dev/torpverse/torp")
for (p in torp_paths) if (file.exists(file.path(p, "DESCRIPTION"))) { devtools::load_all(p, quiet = TRUE); break }

source(file.path(EXPERIMENTS_DIR, "rolling_lib.R"))
source(file.path(EXPERIMENTS_DIR, "elo_lib.R"))

CONFIRM_SEASONS <- 2025:2026

cat("=== Step 1: fresh team_mdl_df ===\n")
team_mdl_df <- build_team_mdl_df()

# Known pre-existing bug (found earlier this session, unrelated to this
# measurement): unplayed finals placeholder fixtures ("Loser of QF1" etc.)
# have no real opponent, so their rating-diff features (epr_diff etc.) are
# NA. predict_all()'s model.matrix() silently DROPS NA rows, which later
# crashes a row-count-mismatch assignment back into team_mdl_df. These rows
# have win=NA so they can never be used for training OR testing anyway --
# safe to drop entirely before the eval.
bad_rows <- is.na(team_mdl_df$win) & is.na(team_mdl_df$epr_diff)
cat("Filtering", sum(bad_rows), "unplayed placeholder-fixture rows (win NA + no opponent rating) before eval\n")
team_mdl_df <- team_mdl_df[!bad_rows, ]

completed <- team_mdl_df[!is.na(team_mdl_df$win) & team_mdl_df$team_type == "home", ]
cat("Fresh team_mdl_df:", nrow(team_mdl_df), "rows;", nrow(completed), "completed matches\n")
by_season <- completed %>% count(season.x)
print(by_season)

cat("\n=== Step 2: fresh Elo table (WS2 tuned params) ===\n")
elo_stuff <- readRDS(.rds("ws2_elo_table.rds"))
best_elo <- elo_stuff$best
cat("Reusing WS2-tuned Elo params: k=", best_elo$k, " hga=", best_elo$hga, " carryover=", best_elo$carryover, "\n")
matches <- .matches_from_team_mdl_df(team_mdl_df)
fresh_elo_table <- build_team_elo(matches, k = best_elo$k, hga = best_elo$hga, carryover = best_elo$carryover, mov_mult = TRUE)
team_mdl_df_elo <- join_elo_diff_to_team_mdl_df(team_mdl_df, fresh_elo_table)
n_na_elo <- sum(is.na(team_mdl_df_elo$elo_diff))
n_incomplete <- sum(is.na(team_mdl_df_elo$win))
cat("NA elo_diff rows:", n_na_elo, " (incomplete/future rows:", n_incomplete, ")\n")
if (n_na_elo > n_incomplete) stop("NA elo_diff on completed rows beyond expected future-fixture NAs")
team_mdl_df_elo$elo_diff[is.na(team_mdl_df_elo$elo_diff)] <- 0
saveRDS(team_mdl_df_elo, .rds("wide_team_mdl_df_elo.rds"))

cat("\n=== Step 3: C6 'Everything' (current production GAM/XGB, Input Blend) ===\n")
t0 <- Sys.time()
roll <- run_rolling_eval(team_mdl_df_elo, CONFIRM_SEASONS, verbose = TRUE)
cat("run_rolling_eval completed in", round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")
saveRDS(roll, .rds("wide_run_rolling_eval.rds"))

ib_preds <- roll$input_blend_preds
cat("Input Blend n =", nrow(ib_preds), "\n")
m_ib <- .compute_metrics(ib_preds)
cat(sprintf("Input Blend (no recal): MAE=%.3f RMSE=%.3f Brier=%.4f Slope=%.3f\n", m_ib$mae, m_ib$rmse, m_ib$brier, m_ib$slope))

# ---- V1a recal (copied verbatim from round2_c7_winfix.R) ----
recal_expanding <- function(preds_all, score_idx, history_pool_idx,
                             mode = c("slope_only", "slope_intercept", "nonlinear"),
                             min_n = 30) {
  mode <- match.arg(mode)
  key <- preds_all$season * 1000L + preds_all$round
  score_idx <- score_idx[order(key[score_idx])]
  hist_key <- key[history_pool_idx]
  out <- numeric(length(score_idx))
  b_trace <- vector("list", length(score_idx))
  for (k in seq_along(score_idx)) {
    i <- score_idx[k]
    cur_key <- key[i]
    hist_idx <- history_pool_idx[hist_key < cur_key]
    n_hist <- length(hist_idx)
    if (n_hist < min_n) {
      out[k] <- preds_all$pred_margin[i]
      next
    }
    hist_df <- preds_all[hist_idx, c("pred_margin", "margin")]
    if (mode == "slope_only") {
      b <- unname(stats::coef(stats::lm(margin ~ pred_margin + 0, data = hist_df))[1])
      out[k] <- b * preds_all$pred_margin[i]
    }
  }
  list(idx = score_idx, pred_margin_recal = out)
}
.apply_recal <- function(preds_all, res) {
  out <- preds_all[res$idx, ]
  out$pred_margin <- res$pred_margin_recal
  out
}
v1a_recal_own <- function(preds) {
  idx <- seq_len(nrow(preds))
  res <- recal_expanding(preds, idx, idx, mode = "slope_only", min_n = 30)
  .apply_recal(preds, res)
}

c6_preds <- v1a_recal_own(ib_preds)
m_c6 <- .compute_metrics(c6_preds)
cat(sprintf("C6 'Everything' (Input Blend + V1a recal): MAE=%.3f RMSE=%.3f Brier=%.4f Slope=%.3f\n", m_c6$mae, m_c6$rmse, m_c6$brier, m_c6$slope))
saveRDS(list(preds = c6_preds, metrics = m_c6, preds_norecal = ib_preds, metrics_norecal = m_ib), .rds("wide_c6_pool_confirm.rds"))

cat("\n=== Step 4: C7-fixed (flat XGB margin + GAM win-head) ===\n")
c7_pool_cache <- readRDS(.rds("ws5_c7_pool_confirm.rds"))
best_nrounds_margin_pool <- c7_pool_cache$nrounds_margin
cat("Reusing round-1 tuned nrounds_margin_pool =", best_nrounds_margin_pool, "(not re-tuned for the wider window)\n")

osr_dsr_ok <- all(c("osr_diff", "dsr_diff") %in% names(team_mdl_df_elo)) && !all(is.na(team_mdl_df_elo$osr_diff))
c7_base_cols <- c(
  "team_type_fac", "game_year_decimal.x", "game_prop_through_year.x",
  "game_prop_through_month.x", "game_prop_through_day.x",
  "epr_diff", "epr_recv_diff", "epr_disp_diff", "epr_spoil_diff", "epr_hitout_diff",
  "torp_diff", "psr_diff",
  if (osr_dsr_ok) c("osr_diff", "dsr_diff"),
  "log_dist_diff", "familiarity_diff", "days_rest_diff_fac",
  "elo_diff", "temp_avg", "precipitation_total", "wind_avg", "humidity_avg"
)
reg_params <- list(objective = "reg:squarederror", eval_metric = "rmse", tree_method = "hist",
                    eta = 0.05, subsample = 0.7, colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15)

.fit_c7_winhead <- function(train_df, test_df, pred_margin_train, pred_margin_test) {
  train_df$pred_margin <- pred_margin_train
  win_gam <- mgcv::bam(
    win ~ s(team_name.x, bs = "re") + s(team_name.y, bs = "re")
      + s(team_name_season.x, bs = "re") + s(team_name_season.y, bs = "re")
      + s(pred_margin, bs = "ts", k = 5)
      + s(log_dist_diff, bs = "ts", k = 5) + s(familiarity_diff, bs = "ts", k = 5)
      + s(days_rest_diff_fac, bs = "re"),
    data = train_df, weights = train_df$weightz,
    family = "binomial", nthreads = 4L, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE, gamma = 1.4
  )
  test_df$pred_margin <- pred_margin_test
  as.numeric(predict(win_gam, newdata = test_df, type = "response"))
}
.simple_match_format <- function(df) {
  home <- df[df$team_type == "home", ]
  data.frame(
    season = home$season.x, round = home$round_number.x, match_id = home$match_id,
    home_team = as.character(home$team_name.x),
    pred_margin = home$pred_margin, pred_win = home$pred_win,
    margin = home$score_diff,
    home_win = ifelse(home$score_diff > 0, 1, ifelse(home$score_diff == 0, 0.5, 0)),
    stringsAsFactors = FALSE
  )
}
run_custom_rolling_eval <- function(df, test_seasons, fit_predict_fn, verbose = TRUE) {
  test_rounds <- df |> dplyr::filter(!is.na(win), season.x %in% test_seasons) |>
    dplyr::distinct(season.x, round_number.x) |> dplyr::arrange(season.x, round_number.x) |>
    dplyr::rename(season = season.x, round = round_number.x)
  all_preds <- list()
  for (i in seq_len(nrow(test_rounds))) {
    s <- test_rounds$season[i]; r <- test_rounds$round[i]
    train_filter <- (df$season.x < s) | (df$season.x == s & df$round_number.x < r)
    test_mask <- !is.na(df$win) & df$season.x == s & df$round_number.x == r
    n_test <- sum(test_mask) / 2
    if (n_test == 0) next
    n_train <- sum(train_filter & !is.na(df$win)) / 2
    if (verbose) cli::cli_progress_step("{s} R{r}: train={n_train}, test={n_test}")
    train_df <- df[train_filter & !is.na(df$win), ]
    test_df  <- df[test_mask, ]
    all_preds[[i]] <- fit_predict_fn(train_df, test_df)
  }
  if (verbose) cli::cli_alert_success("Custom rolling evaluation complete")
  dplyr::bind_rows(all_preds)
}
fit_predict_c7fix_pool <- function(train_df, test_df) {
  fmat_tr <- stats::model.matrix(~ . - 1, data = train_df[, c7_base_cols, drop = FALSE])
  fmat_te <- stats::model.matrix(~ . - 1, data = test_df[, c7_base_cols, drop = FALSE])
  set.seed(1234)
  m_reg <- xgboost::xgb.train(params = reg_params,
                               data = xgboost::xgb.DMatrix(fmat_tr, label = train_df$score_diff, weight = train_df$weightz),
                               nrounds = best_nrounds_margin_pool, verbose = 0)
  pred_margin_tr <- predict(m_reg, xgboost::xgb.DMatrix(fmat_tr))
  pred_margin_te <- predict(m_reg, xgboost::xgb.DMatrix(fmat_te))
  test_df$pred_margin <- pred_margin_te
  test_df$pred_win <- .fit_c7_winhead(train_df, test_df, pred_margin_tr, pred_margin_te)
  .simple_match_format(test_df)
}

t0 <- Sys.time()
c7fix_pool_preds <- run_custom_rolling_eval(team_mdl_df_elo, CONFIRM_SEASONS, fit_predict_c7fix_pool)
cat("C7-fixed pooled confirm completed in", round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")
m_c7fix_pool <- .compute_metrics(c7fix_pool_preds)
cat(sprintf("C7-fixed (no recal): MAE=%.3f RMSE=%.3f Brier=%.4f Slope=%.3f\n", m_c7fix_pool$mae, m_c7fix_pool$rmse, m_c7fix_pool$brier, m_c7fix_pool$slope))
saveRDS(list(preds = c7fix_pool_preds, metrics = m_c7fix_pool, nrounds_margin = best_nrounds_margin_pool), .rds("wide_c7fix_pool_confirm.rds"))

cat("\n=== Step 5: ensemble sweep ===\n")
c6 <- c6_preds[, c("match_id", "pred_margin", "pred_win", "margin", "home_win")]
c7 <- c7fix_pool_preds[, c("match_id", "pred_margin", "pred_win", "margin", "home_win")]
names(c6)[2:3] <- c("pred_margin_c6", "pred_win_c6")
names(c7)[2:5] <- c("pred_margin_c7", "pred_win_c7", "margin_c7", "home_win_c7")
merged <- merge(c6, c7, by = "match_id")
cat("Merged pooled n =", nrow(merged), "\n")
stopifnot(all.equal(merged$margin, merged$margin_c7))

weights <- c(1.0, 0.85, 0.7, 0.5, 0.3, 0.15, 0.0)
cat(sprintf("\n=== Pooled %s (n=%d): margin-blend weight w (weight on C6), win-prob = pure C6 ===\n", paste(range(CONFIRM_SEASONS), collapse=":"), nrow(merged)))
results <- list()
for (w in weights) {
  pm <- w * merged$pred_margin_c6 + (1 - w) * merged$pred_margin_c7
  df <- data.frame(pred_margin = pm, pred_win = merged$pred_win_c6, margin = merged$margin, home_win = merged$home_win, match_id = merged$match_id)
  m <- .compute_metrics(df)
  results[[as.character(w)]] <- list(df = df, m = m)
  cat(sprintf("w=%.2f: MAE=%.3f RMSE=%.3f Brier=%.4f Slope=%.3f\n", w, m$mae, m$rmse, m$brier, m$slope))
}
maes <- sapply(results, function(r) r$m$mae)
cat("\nMAE by weight:", paste(names(results), round(maes, 3), collapse = " | "), "\n")
best_w <- names(which.min(maes))
cat("Best (lowest MAE) weight on this pooled screen: w =", best_w, "\n")

if (best_w != "1") {
  best_df <- results[[best_w]]$df
  boot_best <- boot_mae_diff(best_df, c6_preds)
  cat(sprintf("Boot vs C6 champion: deltaMAE=%+.3f CI=[%.3f, %.3f] deltaBrier=%+.4f CI=[%.4f, %.4f]\n",
              boot_best$mae_diff, boot_best$mae_ci[1], boot_best$mae_ci[2],
              boot_best$brier_diff, boot_best$brier_ci[1], boot_best$brier_ci[2]))
  saveRDS(list(results = results, best_w = best_w, boot_best = boot_best, m_c6 = m_c6, m_c7fix_pool = m_c7fix_pool, n = nrow(merged)),
          .rds("wide_ensemble_final.rds"))
} else {
  cat("Best weight is w=1.0 (pure C6) -- ensemble adds nothing on this window.\n")
  saveRDS(list(results = results, best_w = best_w, m_c6 = m_c6, m_c7fix_pool = m_c7fix_pool, n = nrow(merged)),
          .rds("wide_ensemble_final.rds"))
}
cat("\n=== DONE ===\n")
