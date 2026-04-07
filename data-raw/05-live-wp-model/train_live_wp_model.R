# Train Live Win Probability Model
# =================================
# Simplified WP model for browser use during live games.
# Uses only features available from Squiggle API at runtime:
#   - period (1-4)
#   - period_seconds (clock time, counts up from 0)
#   - points_diff (home score - away score)
#
# Draws: labeled as 0.5 (quasibinomial family), so tied-near-fulltime → ~50%
# Output: lookup table exported as JSON for win-prob.js

library(devtools)
library(data.table)

# Load torp
torp_path <- file.path(dirname(dirname(getwd())), "torp")
if (!file.exists(file.path(torp_path, "DESCRIPTION"))) {
  torp_path <- "C:/Users/peteo/OneDrive/Documents/torpverse/torp"
}
devtools::load_all(torp_path)

# ── Step 1: Load and clean PBP data ──────────────────────────────────────────

cli::cli_h1("Training Live Win Probability Model")
cli::cli_alert_info("Loading chains data...")
chains <- torp::load_chains(TRUE)

cli::cli_alert_info("Cleaning play-by-play data...")
pbp <- torp::clean_pbp(chains)

cli::cli_alert_info("PBP rows: {nrow(pbp)}")

# ── Step 2: Build training data ──────────────────────────────────────────────

dt <- data.table::as.data.table(pbp)

required_cols <- c("period", "period_seconds", "points_diff", "label_wp",
                   "torp_match_id", "season", "home")
missing <- setdiff(required_cols, names(dt))
if (length(missing) > 0) {
  cli::cli_abort("Missing columns: {paste(missing, collapse = ', ')}")
}

dt <- dt[!is.na(label_wp) & !is.na(period_seconds) & !is.na(points_diff),
         .(period, period_seconds, points_diff, label_wp, torp_match_id, season, home)]

n_matches <- uniqueN(dt$torp_match_id)
cli::cli_alert_info("Training rows: {nrow(dt)} across {n_matches} matches, seasons {min(dt$season)}-{max(dt$season)}")

# Handle draws: label as 0.5 instead of excluding
# This makes tied-near-fulltime predict ~50% (correct: you don't know the outcome)
n_draws <- uniqueN(dt[label_wp == 0.5]$torp_match_id)
# Check if draws are already 0.5 or if they're missing
draw_check <- dt[, .(final_diff = last(points_diff)), by = torp_match_id]
n_tied_final <- sum(draw_check$final_diff == 0)
cli::cli_alert_info("Matches ending tied (potential draws): {n_tied_final} of {n_matches}")

# If label_wp is binary (0/1), recode draws as 0.5
# A draw = game where final points_diff == 0
if (n_tied_final > 0) {
  # Find match IDs that ended in a draw
  draw_matches <- draw_check[final_diff == 0, torp_match_id]
  n_before <- sum(dt$label_wp == 0.5)
  dt[torp_match_id %in% draw_matches, label_wp := 0.5]
  n_after <- sum(dt$label_wp == 0.5)
  cli::cli_alert_info("Relabeled {n_after - n_before} rows from {length(draw_matches)} drawn matches as label_wp = 0.5")
}

# ── Step 3: Feature engineering ──────────────────────────────────────────────

dt[, total_seconds := (period - 1L) * 2000 + period_seconds]

cli::cli_h2("Feature summary")
print(summary(dt[, .(period, period_seconds, points_diff, total_seconds, label_wp)]))

# ── Step 4: Fit model ────────────────────────────────────────────────────────
# GAM with smooth interaction between margin and time
# quasibinomial handles non-integer labels (draws = 0.5)

cli::cli_alert_info("Fitting GAM (quasibinomial for draw handling)...")
library(mgcv)

fit <- bam(
  label_wp ~ te(total_seconds, points_diff, k = c(10, 10)) + home,
  family = quasibinomial(link = "logit"),
  data = dt,
  method = "fREML",
  discrete = TRUE
)

cli::cli_alert_success("GAM fitted: R-sq(adj) = {round(summary(fit)$r.sq, 3)}, deviance explained = {round(summary(fit)$dev.expl * 100, 1)}%")

# ── Step 5: Generate lookup table ────────────────────────────────────────────

cli::cli_alert_info("Generating lookup table...")

# Period seconds buckets: every 120 seconds (2 min)
ps_buckets <- seq(0, 2000, by = 120)

# Margin buckets: fine near 0 (every 1pt from -12 to +12), coarser outside
margin_fine <- seq(-12, 12, by = 1)
margin_coarse_neg <- seq(-78, -18, by = 6)
margin_coarse_pos <- seq(18, 78, by = 6)
margin_buckets <- sort(unique(c(margin_coarse_neg, margin_fine, margin_coarse_pos)))

cli::cli_alert_info("Margin buckets: {length(margin_buckets)} ({min(margin_buckets)} to {max(margin_buckets)}, fine 1pt resolution from -12 to +12)")

lookup <- data.table::CJ(
  period = 1:4,
  period_seconds = ps_buckets,
  points_diff = margin_buckets
)

lookup[, `:=`(
  total_seconds = (period - 1L) * 2000 + period_seconds,
  home = 1L
)]

lookup[, wp := predict(fit, newdata = lookup, type = "response")]

# Clamp to [0.001, 0.999] and round
lookup[, wp := pmin(pmax(round(wp, 3), 0.001), 0.999)]

# Home advantage
lookup_away <- copy(lookup)
lookup_away[, home := 0L]
lookup_away[, wp_away := predict(fit, newdata = lookup_away, type = "response")]
lookup[, home_advantage := wp - lookup_away$wp_away]

cli::cli_alert_info("Home advantage: {round(mean(lookup$home_advantage) * 100, 1)}% average")

# Export as JSON
output <- list(
  description = "AFL Live Win Probability Lookup Table",
  trained_on = paste0(min(dt$season), "-", max(dt$season)),
  n_matches = n_matches,
  n_plays = nrow(dt),
  n_draws = n_tied_final,
  home_advantage = round(mean(lookup$home_advantage), 4),
  periods = ps_buckets,
  margins = margin_buckets,
  table = lapply(1:4, function(p) {
    lapply(seq_along(ps_buckets), function(ps_idx) {
      lookup[period == p & period_seconds == ps_buckets[ps_idx], wp]
    })
  })
)

output_dir <- file.path(dirname(dirname(getwd())), "inst", "models", "core")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

json_path <- file.path(output_dir, "live_wp_lookup.json")
jsonlite::write_json(output, json_path, auto_unbox = TRUE, pretty = FALSE)
cli::cli_alert_success("Saved lookup table to {json_path} ({file.size(json_path)} bytes)")

rds_path <- file.path(output_dir, "live_wp_model.rds")
saveRDS(fit, rds_path)
cli::cli_alert_success("Saved GAM model to {rds_path}")

# ── Step 6: Calibration ─────────────────────────────────────────────────────

cli::cli_h2("Calibration check (home team, {n_matches} matches)")

test_cases <- data.table::data.table(
  label = c("Kickoff", "Q1 10min +6", "Q1 end +18",
            "HT tied", "HT +12", "HT +24", "HT -24",
            "3QT +30", "3QT -30", "3QT tied",
            "Q4 5min +6", "Q4 5min tied",
            "Q4 15min +6", "Q4 15min +1", "Q4 15min tied",
            "Q4 25min +6", "Q4 25min +1", "Q4 25min tied", "Q4 25min -1"),
  period = c(1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4),
  period_seconds = c(0, 600, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800,
                     600, 600, 1200, 1200, 1200, 1800, 1800, 1800, 1800),
  points_diff = c(0, 6, 18, 0, 12, 24, -24, 30, -30, 0, 6, 0, 6, 1, 0, 6, 1, 0, -1),
  home = 1L
)
test_cases[, total_seconds := (period - 1L) * 2000 + period_seconds]
test_cases[, predicted := round(predict(fit, newdata = test_cases, type = "response") * 100, 1)]
print(test_cases[, .(label, period, period_seconds, points_diff, predicted)])

# ── Step 7: Empirical calibration by bucket ──────────────────────────────────

cli::cli_h2("Empirical calibration: predicted vs actual win rate")

# Bucket predictions into deciles and compare to actual outcomes
dt[, predicted := predict(fit, newdata = dt, type = "response")]
dt[, pred_bucket := cut(predicted, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)]

calib <- dt[, .(
  actual_win_rate = round(mean(label_wp) * 100, 1),
  mean_predicted = round(mean(predicted) * 100, 1),
  n_plays = .N,
  n_matches = uniqueN(torp_match_id)
), by = pred_bucket][order(pred_bucket)]

print(calib)

# ── Step 8: Copy to inthegame-blog ──────────────────────────────────────────

blog_path <- "C:/Users/peteo/OneDrive/Documents/inthegame-blog/afl/live-wp-lookup.json"
file.copy(json_path, blog_path, overwrite = TRUE)
cli::cli_alert_success("Copied lookup table to {blog_path}")

cli::cli_h1("Done! ({n_matches} matches, {nrow(dt)} plays, {n_tied_final} draws)")
