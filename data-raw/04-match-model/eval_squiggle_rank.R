# Squiggle Leaderboard Evaluation
# ================================
# Insert each rolling-eval variant (GAM, XGBoost, OutBlend, InBlend) into the
# Squiggle 2026 leaderboard as a candidate tipster and report rank by bits,
# MAE, and Brier. Lets us answer "where would each variant sit if we'd
# submitted it to Squiggle every week?".
#
# Splits by bucket (full season, R1-5, R6-10) so we can see ranking drift.
#
# Self-contained: sources train_match_models.R to populate gam_preds,
# xgb_preds, blend_preds, input_blend_preds, then fetches the Squiggle
# field and stitches everything together.
#
# Usage:
#   Rscript torpmodels/data-raw/04-match-model/eval_squiggle_rank.R
#
# To change the bucket split:
#   - Edit SPLIT_AT (default 5; first bucket is R1-SPLIT_AT)

# --- Parameters ----
TEST_YEAR <- 2026   # year to evaluate against Squiggle
SPLIT_AT  <- 5      # bucket boundary

# --- Source the training script (~90s) ----
.t0 <- Sys.time()
.olddir <- getwd()
.script_dir <- "C:/Users/peteo/OneDrive/Documents/torpverse/torpmodels/data-raw/04-match-model"
# Prefer relative if called from script_dir
if (file.exists("train_match_models.R")) .script_dir <- getwd()
setwd(.script_dir)

cli::cli_h1("Sourcing train_match_models.R")
.sink_path <- tempfile("eval_sq_", fileext = ".log")
.sink_con <- file(.sink_path, "w")
sink(.sink_con); sink(.sink_con, type = "message")
.err <- tryCatch(source("train_match_models.R", local = FALSE),
                 error = function(e) e)
sink(type = "message"); sink(); close(.sink_con)
setwd(.olddir)

if (inherits(.err, "error")) {
  cat("FAILED:", .err$message, "\nLast 50 lines:\n",
      paste(tail(readLines(.sink_path), 50), collapse = "\n"), "\n")
  stop("rolling eval failed")
}
cli::cli_alert_success("Training script complete in {round(as.numeric(Sys.time() - .t0, units = 'secs'), 1)}s")

# --- Squiggle field fetch ----
suppressPackageStartupMessages(library(dplyr))

cli::cli_h2("Fetching Squiggle field")
.tips <- fitzRoy::fetch_squiggle_data("tips",  year = TEST_YEAR)
.games <- fitzRoy::fetch_squiggle_data("games", year = TEST_YEAR)

.done_games <- .games |>
  filter(complete == 100) |>
  transmute(
    gameid = as.integer(id),
    round = as.integer(round),
    hteam_norm = torp_replace_teams(hteam),
    actual_margin = as.numeric(hscore) - as.numeric(ascore)
  )

.sq_tips <- .tips |>
  mutate(
    gameid = as.integer(gameid),
    hmargin = as.numeric(hmargin),
    hconfidence = as.numeric(hconfidence),
    round = as.integer(round),
    hteam_norm = torp_replace_teams(hteam)
  ) |>
  inner_join(.done_games |> select(gameid, actual_margin), by = "gameid") |>
  filter(!is.na(hconfidence), !is.na(hmargin)) |>
  transmute(source, round, hteam_norm,
            pred_margin = hmargin,
            pred_win = hconfidence / 100,
            actual_margin)

# --- Reshape rolling preds into Squiggle-tip shape ----
.rolling_to_sq <- function(df, label) {
  df |>
    mutate(hteam_norm = torp_replace_teams(as.character(home_team))) |>
    filter(!is.na(margin), !is.na(pred_win), !is.na(pred_margin)) |>
    transmute(source = label, round, hteam_norm,
              pred_margin, pred_win,
              actual_margin = margin)
}

.rolling_rows <- bind_rows(
  .rolling_to_sq(gam_preds,         "torp Rolling-GAM"),
  .rolling_to_sq(xgb_preds,         "torp Rolling-XGB"),
  .rolling_to_sq(blend_preds,       "torp Rolling-OutBlend"),
  .rolling_to_sq(input_blend_preds, "torp Rolling-InBlend")
)

.all_rows <- bind_rows(.sq_tips, .rolling_rows)

# --- Metrics + leaderboard ----
.compute_metrics <- function(df) {
  df |>
    mutate(
      p_home = pmin(pmax(pred_win, 0.001), 0.999),
      home_win = ifelse(actual_margin > 0, 1,
                        ifelse(actual_margin == 0, 0.5, 0)),
      p_winner = ifelse(home_win == 1, p_home,
                        ifelse(home_win == 0, 1 - p_home, 0.5)),
      bits = log2(2 * p_winner)
    ) |>
    summarise(
      n = n(),
      acc = mean(round(p_home) == home_win) * 100,
      mae = mean(abs(pred_margin - actual_margin)),
      brier = mean((p_home - home_win)^2),
      bits_total = sum(bits),
      .groups = "drop"
    )
}

# Build a board for one round range; returns a data frame
build_board <- function(rng, label) {
  d <- .all_rows |> filter(round >= rng[1] & round <= rng[2])
  m <- d |>
    group_by(source) |>
    .compute_metrics() |>
    mutate(
      rk_bits  = rank(-bits_total, ties.method = "min"),
      rk_mae   = rank(mae,         ties.method = "min"),
      rk_brier = rank(brier,       ties.method = "min")
    ) |>
    arrange(desc(bits_total))

  cli::cli_h2("{label} — {n_distinct(d$source)} sources")
  show <- m |>
    transmute(rk_bits, source, n,
              acc = round(acc, 1), mae = round(mae, 2),
              brier = round(brier, 4), bits = round(bits_total, 2),
              rk_mae, rk_brier)
  print(as.data.frame(show), row.names = FALSE)

  # Highlight torp variants + "In The Game" together
  cli::cli_inform("torp variants + live submission")
  focus <- show |>
    filter(source == "In The Game" | grepl("^torp ", source)) |>
    arrange(rk_bits)
  print(as.data.frame(focus), row.names = FALSE)

  invisible(m)
}

cli::cli_h1("Leaderboard")
boards <- list(
  all    = build_board(c(1, 10),         "ALL 2026 (R1-10)"),
  early  = build_board(c(1, SPLIT_AT),   sprintf("R1-%d", SPLIT_AT)),
  late   = build_board(c(SPLIT_AT + 1, 10),
                       sprintf("R%d-10", SPLIT_AT + 1))
)

# Save for downstream analyses. Uses tools::R_user_dir so the cache survives
# across R sessions — debug/drill-down scripts can load this without paying
# the 90s training cost again.
out_dir <- tools::R_user_dir("torpmodels", "cache")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, sprintf("eval_squiggle_rank_%d.rds", TEST_YEAR))
saveRDS(list(
  boards = boards,
  all_rows = .all_rows,
  rolling_rows = .rolling_rows,
  sq_tips = .sq_tips,
  done_games = .done_games,
  test_year = TEST_YEAR,
  split_at = SPLIT_AT,
  saved_at = Sys.time()
), out_path)
cli::cli_alert_success("Saved board + raw rows to {.file {out_path}}")
