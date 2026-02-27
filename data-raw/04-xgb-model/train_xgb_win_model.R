# Train Match Prediction Models (GAM pipeline)
# =============================================
# Self-contained script that:
#   1. Builds team_mdl_df using torp's data loading functions
#   2. Trains 5 sequential GAM models (production match prediction pipeline)
#   3. Saves & uploads match_gams.rds to GitHub releases
#
# Set HOLDOUT_SEASON to a finite year (e.g. 2025) to also train a parallel
# XGBoost pipeline for comparison. In production mode (Inf), only GAMs are trained.
#
# Data pipeline adapted from torp's build_match_predictions.R.

# Setup ----
library(tidyverse)
library(xgboost)
library(mgcv)
library(fitzRoy)
library(MLmetrics)
library(geosphere)
library(cli)

# Load torp for data loading functions
if (!require(torp)) {
  torp_paths <- c("../torp", "../../torp", "../../../torp")
  loaded <- FALSE
  for (p in torp_paths) {
    if (file.exists(file.path(p, "DESCRIPTION"))) {
      devtools::load_all(p)
      loaded <- TRUE
      break
    }
  }
  if (!loaded) stop("Cannot find torp package. Install it or run from torpverse workspace.")
}

# Constants ----
WEIGHT_DECAY_DAYS <- 1000
LOG_DIST_OFFSET   <- 10000
LOG_DIST_DEFAULT  <- 16
MIN_DATA_SEASON   <- 2021
MIN_DATA_ROUND    <- 14

# Position Maps ----
PHASE_MAP <- list(
  def = c("BPL", "BPR", "FB", "CHB", "HBFL", "HBFR"),
  mid = c("C", "WL", "WR", "R", "RR", "RK"),
  fwd = c("FPL", "FPR", "FF", "CHF", "HFFL", "HFFR"),
  int = c("INT", "SUB")
)
POS_GROUP_MAP <- list(
  backs = c("BPL", "BPR", "FB"), half_backs = c("HBFL", "HBFR", "CHB"),
  midfielders = c("WL", "WR", "C"), followers = c("R", "RR", "RK"),
  half_forwards = c("HFFL", "HFFR", "CHF"), forwards = c("FPL", "FPR", "FF")
)
INDIVIDUAL_POS <- c(
  "BPL", "BPR", "FB", "HBFL", "HBFR", "CHB",
  "WL", "WR", "C", "R", "RR", "RK",
  "HFFL", "HFFR", "CHF", "FPL", "FPR", "FF"
)
COMBO_POS_MAP <- list(
  CB = c("CHB", "FB"), BP = c("BPL", "BPR"), HBF = c("HBFL", "HBFR"),
  W = c("WL", "WR"), MIDS = c("C", "R", "RR"), HFF = c("HFFL", "HFFR"),
  FP = c("FPL", "FPR"), CF = c("FF", "CHF")
)
LISTED_POS_MAP <- list(
  key_def = "KEY_DEFENDER", med_def = "MEDIUM_DEFENDER", midfield = "MIDFIELDER",
  mid_fwd = "MIDFIELDER_FORWARD", med_fwd = "MEDIUM_FORWARD",
  key_fwd = "KEY_FORWARD", rucks = "RUCK"
)
POS_COLS <- c(
  names(PHASE_MAP), names(POS_GROUP_MAP), INDIVIDUAL_POS,
  names(COMBO_POS_MAP), names(LISTED_POS_MAP), "other_pos"
)

# Helpers ----
get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# ========================================================================
# BUILD team_mdl_df
# ========================================================================

cli::cli_h1("Building Match Prediction Training Data")
tictoc::tic("total")

# 1. Load Data ----
cli::cli_h2("Loading data")

all_grounds   <- torp:::file_reader("stadium_data", "reference-data")
xg_df         <- torp::load_xg(TRUE)
fixtures      <- torp::load_fixtures(TRUE)
results       <- torp::load_results(TRUE)
teams         <- torp::load_teams(TRUE)
torp_df_total <- torp::load_torp_ratings()

cli::cli_inform("Loaded: fixtures={nrow(fixtures)}, results={nrow(results)}, teams={nrow(teams)}, ratings={nrow(torp_df_total)}")

stopifnot(
  "fixtures has 0 rows" = nrow(fixtures) > 0,
  "results has 0 rows" = nrow(results) > 0,
  "teams has 0 rows" = nrow(teams) > 0,
  "torp_df_total has 0 rows" = nrow(torp_df_total) > 0,
  "xg_df has 0 rows" = nrow(xg_df) > 0
)

# 2. Build Fixtures (fix_df) ----
cli::cli_h2("Building fixture features")

team_map <- fixtures |>
  group_by(teamId = home.team.providerId) |>
  summarise(team_name = get_mode(home.team.name)) |>
  mutate(team_name = fitzRoy::replace_teams(team_name))

fix_df <- fixtures |>
  mutate(result = home.score.totalScore - away.score.totalScore) |>
  select(providerId, compSeason.year, round.roundNumber, home.team.providerId, away.team.providerId,
         utcStartTime, venue.name, venue.timezone, result) |>
  pivot_longer(cols = ends_with("team.providerId"), names_to = "team_type", values_to = "team.providerId") |>
  mutate(
    venue = fitzRoy::replace_venues(venue.name),
    team_type = substr(team_type, 1, 4),
    result = ifelse(team_type == "away", -result, result)
  ) |>
  select(providerId, season = compSeason.year, round.roundNumber, team_type, teamId = team.providerId,
         utcStartTime, venue, venue.timezone, result) |>
  left_join(team_map, by = "teamId") |>
  mutate(team_name_season = as.factor(paste(team_name, season)))

## Date Variables ----
fix_df <- fix_df |>
  mutate(utc_dt = lubridate::ymd_hms(utcStartTime, tz = "UTC")) |>
  group_by(venue.timezone) |>
  mutate(
    local_dt = lubridate::with_tz(utc_dt, tzone = first(venue.timezone)),
    game_year = lubridate::year(local_dt),
    game_month = lubridate::month(local_dt),
    game_yday = lubridate::yday(local_dt),
    game_mday = lubridate::day(local_dt),
    game_wday = lubridate::wday(local_dt, week_start = 1),
    game_wday_fac = as.factor(game_wday),
    game_hour = lubridate::hour(local_dt) + lubridate::minute(local_dt) / 60 + lubridate::second(local_dt) / 3600,
    game_prop_through_year = game_yday / ifelse(lubridate::leap_year(game_year), 366, 365),
    game_prop_through_month = game_mday / lubridate::days_in_month(game_month),
    game_prop_through_day = game_hour / 24,
    game_year_decimal = as.numeric(game_year + game_prop_through_year)
  ) |>
  ungroup() |>
  select(-local_dt)

# Weight anchor: most recent game in data
weight_anchor_date <- max(as.Date(fix_df$utcStartTime), na.rm = TRUE)
cli::cli_inform("Weight anchor date: {weight_anchor_date}")

# 3. Process Lineups ----
cli::cli_h2("Processing lineups")

torp_sum_cols <- c("torp", "torp_recv", "torp_disp", "torp_spoil", "torp_hitout")

team_lineup_df <- teams |>
  left_join(torp_df_total, by = c("player.playerId" = "player_id", "season" = "season", "round.roundNumber" = "round")) |>
  filter((position.x != "EMERG" & position.x != "SUB") | is.na(position.x)) |>
  mutate(across(all_of(torp_sum_cols), ~tidyr::replace_na(.x, 0)))

# Generate position columns from lookup tables
for (col in names(PHASE_MAP))
  team_lineup_df[[col]] <- ifelse(team_lineup_df$position.x %in% PHASE_MAP[[col]], team_lineup_df$torp, NA)
for (col in names(POS_GROUP_MAP))
  team_lineup_df[[col]] <- ifelse(team_lineup_df$position.x %in% POS_GROUP_MAP[[col]], team_lineup_df$torp, NA)
for (pos in INDIVIDUAL_POS)
  team_lineup_df[[pos]] <- ifelse(team_lineup_df$position.x == pos, team_lineup_df$torp, NA)
for (col in names(COMBO_POS_MAP))
  team_lineup_df[[col]] <- ifelse(team_lineup_df$position.x %in% COMBO_POS_MAP[[col]], team_lineup_df$torp, NA)
for (col in names(LISTED_POS_MAP))
  team_lineup_df[[col]] <- ifelse(team_lineup_df$position.y == LISTED_POS_MAP[[col]], team_lineup_df$torp, NA)
team_lineup_df$other_pos <- ifelse(is.na(team_lineup_df$position.y), team_lineup_df$torp, NA)

## Aggregate Lineups ----
team_rt_df <- team_lineup_df |>
  filter(!is.na(player.playerId)) |>
  mutate(team_name_adj = fitzRoy::replace_teams(teamName)) |>
  group_by(providerId, teamId, season, round.roundNumber, teamType) |>
  summarise(
    venue = fitzRoy::replace_venues(max(venue.name)),
    team_name_adj = max(team_name_adj),
    across(all_of(c(torp_sum_cols, POS_COLS)), ~sum(.x, na.rm = TRUE)),
    count = n(),
    .groups = "drop"
  )

# 4. Feature Engineering ----
cli::cli_h2("Feature engineering")

## Home Ground ----
home_ground <- team_rt_df |>
  group_by(teamId, team_name_adj) |>
  summarise(home_ground = get_mode(venue), .groups = "drop") |>
  mutate(venue_adj = fitzRoy::replace_venues(as.character(home_ground))) |>
  left_join(
    all_grounds |> mutate(venue_adj = fitzRoy::replace_venues(as.character(Ground))),
    by = "venue_adj"
  )

## Familiarity ----
ground_prop <- team_rt_df |>
  arrange(teamId, season, round.roundNumber) |>
  group_by(teamId) |>
  mutate(cum_total_games = row_number() - 1) |>
  group_by(teamId, venue) |>
  mutate(cum_venue_games = row_number() - 1) |>
  ungroup() |>
  mutate(familiarity = ifelse(cum_total_games > 0, cum_venue_games / cum_total_games, 0)) |>
  select(teamId, season, round.roundNumber, venue, familiarity)

## Distance Traveled ----
team_dist_df <- fix_df |>
  mutate(venue = ifelse(venue == "Adelaide Arena at Jiangwan Stadium", "Jiangwan Stadium", venue)) |>
  left_join(all_grounds |> select(venue, venue_lat = Latitude, venue_lon = Longitude), by = "venue") |>
  left_join(home_ground |> select(teamId, team_lat = Latitude, team_lon = Longitude), by = "teamId") |>
  mutate(
    distance = purrr::pmap_dbl(
      list(venue_lon, venue_lat, team_lon, team_lat),
      ~ geosphere::distHaversine(c(..1, ..2), c(..3, ..4))
    ),
    log_dist = log(distance + LOG_DIST_OFFSET),
    log_dist = tidyr::replace_na(log_dist, LOG_DIST_DEFAULT)
  ) |>
  left_join(ground_prop, by = c("teamId", "season", "round.roundNumber", "venue")) |>
  mutate(familiarity = tidyr::replace_na(familiarity, 0))

## Days Rest ----
days_rest <- fix_df |>
  arrange(teamId, utcStartTime) |>
  group_by(teamId, season) |>
  mutate(days_rest = as.numeric(difftime(utcStartTime, lag(utcStartTime), units = "days"))) |>
  ungroup() |>
  mutate(days_rest = tidyr::replace_na(days_rest, 21))

# 5. Combine Features (team_rt_fix_df) ----
cli::cli_h2("Combining features")

team_rt_fix_df <- fix_df |>
  mutate(team_name = fitzRoy::replace_teams(team_name)) |>
  left_join(team_dist_df |> select(providerId, teamId, log_dist, familiarity), by = c("providerId", "teamId")) |>
  left_join(days_rest |> select(providerId, teamId, days_rest), by = c("providerId", "teamId")) |>
  left_join(
    team_rt_df |> select(providerId, teamId, season, round.roundNumber,
                          all_of(c(torp_sum_cols, POS_COLS, "count"))),
    by = c("providerId", "teamId", "season", "round.roundNumber")
  ) |>
  group_by(teamId) |>
  tidyr::fill(torp, torp_recv, torp_disp, torp_spoil, torp_hitout) |>
  mutate(
    def = ifelse(def == 0, lag(def), def),
    mid = ifelse(mid == 0, lag(mid), mid),
    fwd = ifelse(fwd == 0, lag(fwd), fwd),
    int = ifelse(int == 0, lag(int), int),
    team_type_fac = as.factor(team_type)
  ) |>
  tidyr::fill(def, mid, fwd, int) |>
  ungroup()

# 6. Build Model Dataset (team_mdl_df) ----
cli::cli_h2("Building model dataset")

opp_cols <- c(
  "providerId", "team_type",
  "torp", "torp_recv", "torp_disp", "torp_spoil", "torp_hitout",
  "def", "mid", "fwd", "int", INDIVIDUAL_POS,
  "team_name", "team_name_season",
  "log_dist", "familiarity", "days_rest",
  "team_type_fac", "season", "round.roundNumber", "venue", "count",
  "game_year_decimal", "game_prop_through_year", "game_prop_through_month",
  "game_wday_fac", "game_prop_through_day"
)

team_mdl_df_tot <- team_rt_fix_df |>
  left_join(
    team_rt_fix_df |>
      select(all_of(opp_cols)) |>
      mutate(type_anti = if_else(team_type == "home", "away", "home")),
    by = c("providerId" = "providerId", "team_type" = "type_anti")
  ) |>
  mutate(
    torp_diff = torp.x - torp.y,
    torp_ratio = log(pmax(torp.x, 0.01) / pmax(torp.y, 0.01)),
    torp_recv_diff = torp_recv.x - torp_recv.y,
    torp_disp_diff = torp_disp.x - torp_disp.y,
    torp_spoil_diff = torp_spoil.x - torp_spoil.y,
    torp_hitout_diff = torp_hitout.x - torp_hitout.y
  ) |>
  left_join(
    results |> select(
      match.matchId,
      homeTeamScore.matchScore.totalScore, homeTeamScore.matchScore.goals, homeTeamScore.matchScore.behinds,
      awayTeamScore.matchScore.totalScore, awayTeamScore.matchScore.goals, awayTeamScore.matchScore.behinds,
      match.utcStartTime
    ),
    by = c("providerId" = "match.matchId")
  ) |>
  left_join(xg_df, by = c("providerId" = "match_id")) |>
  mutate(
    home_shots = homeTeamScore.matchScore.goals + homeTeamScore.matchScore.behinds,
    away_shots = awayTeamScore.matchScore.goals + awayTeamScore.matchScore.behinds,
    score_diff = ifelse(team_type == "home",
      homeTeamScore.matchScore.totalScore - awayTeamScore.matchScore.totalScore,
      awayTeamScore.matchScore.totalScore - homeTeamScore.matchScore.totalScore),
    shot_diff = ifelse(team_type == "home", home_shots - away_shots, away_shots - home_shots),
    team_shots = ifelse(team_type == "home", home_shots, away_shots),
    harmean_shots = torp::harmonic_mean(home_shots, away_shots),
    shot_conv = ifelse(team_type == "home",
      homeTeamScore.matchScore.goals / pmax(home_shots, 1),
      awayTeamScore.matchScore.goals / pmax(away_shots, 1)),
    shot_conv_diff = ifelse(team_type == "home",
      (homeTeamScore.matchScore.goals / pmax(home_shots, 1)) - (awayTeamScore.matchScore.goals / pmax(away_shots, 1)),
      (awayTeamScore.matchScore.goals / pmax(away_shots, 1)) - (homeTeamScore.matchScore.goals / pmax(home_shots, 1))),
    xscore_diff = ifelse(team_type == "home", xscore_diff, -xscore_diff),
    team_xscore = ifelse(team_type == "home", home_xscore, away_xscore),
    win = ifelse(score_diff > 0, 1, ifelse(score_diff == 0, 0.5, 0)),
    hoff_adef = pmax(pmin((fwd.x - def.y), 20), -5),
    hmid_amid = pmax(pmin((mid.x - mid.y), 12), -12),
    hdef_afwd = pmax(pmin((def.x - fwd.y), 5), -20),
    hint_aint = pmax(pmin((int.x - int.y), 10), -10),
    BPL_diff = BPL.x - BPL.y, BPR_diff = BPR.x - BPR.y, FB_diff = FB.x - FB.y,
    HBFL_diff = HBFL.x - HBFL.y, HBFR_diff = HBFR.x - HBFR.y, CHB_diff = CHB.x - CHB.y,
    WL_diff = WL.x - WL.y, WR_diff = WR.x - WR.y, C_diff = C.x - C.y,
    R_diff = R.x - R.y, RR_diff = RR.x - RR.y, RK_diff = RK.x - RK.y,
    HFFL_diff = HFFL.x - HFFL.y, HFFR_diff = HFFR.x - HFFR.y, CHF_diff = CHF.x - CHF.y,
    FPL_diff = FPL.x - FPL.y, FPR_diff = FPR.x - FPR.y, FF_diff = FF.x - FF.y,
    int_diff = int.x - int.y,
    team_type_fac = team_type_fac.x,
    total_score = homeTeamScore.matchScore.totalScore + awayTeamScore.matchScore.totalScore,
    total_shots = home_shots + away_shots,
    team_name.x = as.factor(team_name.x),
    team_name.y = as.factor(team_name.y),
    log_dist_diff = log_dist.x - log_dist.y,
    familiarity_diff = familiarity.x - familiarity.y,
    days_rest_diff = days_rest.x - days_rest.y,
    days_rest_diff_fac = as.factor(round(ifelse(days_rest_diff > 3, 4, ifelse(days_rest_diff < -3, -4, days_rest_diff)))),
    weightz = exp(as.numeric(-(weight_anchor_date - as.Date(match.utcStartTime))) / WEIGHT_DECAY_DAYS),
    weightz = weightz / mean(weightz, na.rm = TRUE),
    shot_weightz = (harmean_shots / mean(harmean_shots, na.rm = TRUE)) * weightz
  )

## Filter early matches ----
team_mdl_df <- team_mdl_df_tot |>
  filter(season.x > MIN_DATA_SEASON | (season.x == MIN_DATA_SEASON & round.roundNumber.x >= MIN_DATA_ROUND))

## Adjust total_xpoints ----
xpoints_scale <- mean(team_mdl_df$total_points, na.rm = TRUE) / mean(team_mdl_df$total_xpoints, na.rm = TRUE)
if (!is.finite(xpoints_scale)) {
  stop("Cannot compute xpoints scaling factor (NaN). ",
       "total_points non-NA: ", sum(!is.na(team_mdl_df$total_points)),
       ", total_xpoints non-NA: ", sum(!is.na(team_mdl_df$total_xpoints)),
       ". Check xg_df join.")
}
team_mdl_df <- team_mdl_df |>
  mutate(
    total_xpoints_adj = total_xpoints * xpoints_scale,
    venue_fac = as.factor(venue.x)
  )

cli::cli_inform("team_mdl_df: {nrow(team_mdl_df)} rows, {ncol(team_mdl_df)} cols")
cli::cli_inform("Seasons: {paste(sort(unique(team_mdl_df$season.x)), collapse = ', ')}")

# ========================================================================
# TRAIN MODELS (temporal holdout: train < HOLDOUT_SEASON, test >= HOLDOUT_SEASON)
# ========================================================================

# Set to Inf for production (train on all data). Set to e.g. 2025 for evaluation.
HOLDOUT_SEASON <- Inf

gam_df  <- team_mdl_df |> filter(!is.na(win), season.x < HOLDOUT_SEASON)
test_df <- team_mdl_df |> filter(!is.na(win), season.x >= HOLDOUT_SEASON)
cli::cli_inform("Train: {nrow(gam_df)} rows ({paste(sort(unique(gam_df$season.x)), collapse=', ')})")
if (is.finite(HOLDOUT_SEASON)) {
  cli::cli_inform("Test:  {nrow(test_df)} rows (season >= {HOLDOUT_SEASON})")
} else {
  cli::cli_inform("Production mode: training on all data (no holdout)")
}

# 7. Train GAM Models ----
cli::cli_h2("Training GAM pipeline (5 sequential models)")
train_mask <- !is.na(team_mdl_df$win) & team_mdl_df$season.x < HOLDOUT_SEASON

## Total xPoints Model ----
cli::cli_progress_step("Training total xPoints model")
afl_total_xpoints_mdl <- mgcv::bam(
  total_xpoints_adj ~
    s(team_type_fac.x, bs = "re")
    + s(game_year_decimal.x, bs = "ts")
    + s(game_prop_through_year.x, bs = "cc")
    + s(game_prop_through_month.x, bs = "cc")
    + s(game_wday_fac.x, bs = "re")
    + s(game_prop_through_day.x, bs = "cc")
    + s(team_name.x, bs = "re") + s(team_name.y, bs = "re")
    + s(team_name_season.x, bs = "re") + s(team_name_season.y, bs = "re")
    + s(abs(torp_diff), bs = "ts", k = 5)
    + s(abs(torp_recv_diff), bs = "ts", k = 5)
    + s(abs(torp_disp_diff), bs = "ts", k = 5)
    + s(abs(torp_spoil_diff), bs = "ts", k = 5)
    + s(abs(torp_hitout_diff), bs = "ts", k = 5)
    + s(torp.x, bs = "ts", k = 5) + s(torp.y, bs = "ts", k = 5)
    + s(venue_fac, bs = "re")
    + s(log_dist.x, bs = "ts", k = 5) + s(log_dist.y, bs = "ts", k = 5)
    + s(familiarity.x, bs = "ts", k = 5) + s(familiarity.y, bs = "ts", k = 5)
    + s(log_dist_diff, bs = "ts", k = 5)
    + s(familiarity_diff, bs = "ts", k = 5)
    + s(days_rest_diff_fac, bs = "re"),
  data = gam_df, weights = gam_df$weightz,
  family = gaussian(), nthreads = 4, select = TRUE, discrete = TRUE,
  drop.unused.levels = FALSE
)
team_mdl_df$pred_tot_xscore <- predict(afl_total_xpoints_mdl, newdata = team_mdl_df, type = "response")

## xScore Diff Model ----
cli::cli_progress_step("Training xScore diff model")
gam_df$pred_tot_xscore <- team_mdl_df$pred_tot_xscore[train_mask]
afl_xscore_diff_mdl <- mgcv::bam(
  xscore_diff ~
    s(team_type_fac, bs = "re")
    + s(team_name.x, bs = "re") + s(team_name.y, bs = "re")
    + s(team_name_season.x, bs = "re") + s(team_name_season.y, bs = "re")
    + ti(torp_diff, pred_tot_xscore, bs = c("ts", "ts"), k = 4)
    + s(pred_tot_xscore, bs = "ts", k = 5)
    + s(torp_diff, bs = "ts", k = 5)
    + s(torp_recv_diff, bs = "ts", k = 5)
    + s(torp_disp_diff, bs = "ts", k = 5)
    + s(torp_spoil_diff, bs = "ts", k = 5)
    + s(torp_hitout_diff, bs = "ts", k = 5)
    + s(log_dist_diff, bs = "ts", k = 5) + s(familiarity_diff, bs = "ts", k = 5) + s(days_rest_diff_fac, bs = "re"),
  data = gam_df, weights = gam_df$weightz,
  family = gaussian(), nthreads = 4, select = TRUE, discrete = TRUE,
  drop.unused.levels = FALSE
)
team_mdl_df$pred_xscore_diff <- predict(afl_xscore_diff_mdl, newdata = team_mdl_df, type = "response")

## Conversion Model ----
cli::cli_progress_step("Training conversion model")
gam_df$pred_xscore_diff <- team_mdl_df$pred_xscore_diff[train_mask]
afl_conv_mdl <- mgcv::bam(
  shot_conv_diff ~
    s(team_type_fac.x, bs = "re")
    + s(game_year_decimal.x, bs = "ts")
    + s(game_prop_through_year.x, bs = "cc")
    + s(game_prop_through_month.x, bs = "cc")
    + s(game_wday_fac.x, bs = "re")
    + s(game_prop_through_day.x, bs = "cc")
    + s(team_name.x, bs = "re") + s(team_name.y, bs = "re")
    + s(team_name_season.x, bs = "re") + s(team_name_season.y, bs = "re")
    + ti(torp_diff, pred_tot_xscore, bs = c("ts", "ts"), k = 4)
    + s(torp_diff, bs = "ts", k = 5)
    + s(torp_recv_diff, bs = "ts", k = 5)
    + s(torp_disp_diff, bs = "ts", k = 5)
    + s(torp_spoil_diff, bs = "ts", k = 5)
    + s(torp_hitout_diff, bs = "ts", k = 5)
    + s(pred_tot_xscore, bs = "ts", k = 5)
    + s(pred_xscore_diff, bs = "ts", k = 5)
    + s(venue_fac, bs = "re")
    + s(log_dist_diff, bs = "ts", k = 5) + s(familiarity_diff, bs = "ts", k = 5) + s(days_rest_diff_fac, bs = "re"),
  data = gam_df, weights = gam_df$shot_weightz,
  family = gaussian(), nthreads = 4, select = TRUE, discrete = TRUE,
  drop.unused.levels = FALSE
)
team_mdl_df$pred_conv_diff <- predict(afl_conv_mdl, newdata = team_mdl_df, type = "response")

## Score Diff Model ----
cli::cli_progress_step("Training score diff model")
gam_df$pred_conv_diff <- team_mdl_df$pred_conv_diff[train_mask]
afl_score_mdl <- mgcv::bam(
  score_diff ~
    s(team_type_fac, bs = "re")
    + s(team_name.x, bs = "re") + s(team_name.y, bs = "re")
    + s(team_name_season.x, bs = "re") + s(team_name_season.y, bs = "re")
    + ti(pred_xscore_diff, pred_conv_diff, bs = "ts", k = 5)
    + ti(pred_tot_xscore, pred_conv_diff, bs = "ts", k = 5)
    + s(pred_xscore_diff)
    + s(log_dist_diff, bs = "ts", k = 5) + s(familiarity_diff, bs = "ts", k = 5) + s(days_rest_diff_fac, bs = "re"),
  data = gam_df, weights = gam_df$weightz,
  family = "gaussian", nthreads = 4, select = TRUE, discrete = TRUE,
  drop.unused.levels = FALSE
)
team_mdl_df$pred_score_diff <- predict(afl_score_mdl, newdata = team_mdl_df, type = "response")

## Win Probability Model (GAM) ----
cli::cli_progress_step("Training win probability model (GAM)")
gam_df$pred_score_diff <- team_mdl_df$pred_score_diff[train_mask]
afl_win_mdl <- mgcv::bam(
  win ~
    +s(team_name.x, bs = "re") + s(team_name.y, bs = "re")
    + s(team_name_season.x, bs = "re") + s(team_name_season.y, bs = "re")
    + ti(pred_tot_xscore, pred_score_diff, bs = c("ts", "ts"), k = 4)
    + s(pred_score_diff, bs = "ts", k = 5)
    + s(log_dist_diff, bs = "ts", k = 5) + s(familiarity_diff, bs = "ts", k = 5) + s(days_rest_diff_fac, bs = "re"),
  data = gam_df, weights = gam_df$weightz,
  family = "binomial", nthreads = 4, select = TRUE, discrete = TRUE,
  drop.unused.levels = FALSE
)
team_mdl_df$pred_win <- predict(afl_win_mdl, newdata = team_mdl_df, type = "response")

cli::cli_alert_success("GAM pipeline complete (5 models trained on {nrow(gam_df)} rows)")

# 8. Train XGBoost Pipeline (5 sequential models, mirroring GAM pipeline) ----
# Only runs with a finite HOLDOUT_SEASON for GAM vs XGBoost comparison
if (is.finite(HOLDOUT_SEASON)) {
cli::cli_h2("Training XGBoost pipeline (5 sequential models)")

xgb_complete <- team_mdl_df |>
  filter(!is.na(win), !is.na(total_xpoints_adj), !is.na(xscore_diff),
         !is.na(shot_conv_diff), !is.na(score_diff))
xgb_df   <- xgb_complete |> filter(season.x < HOLDOUT_SEASON)
xgb_test <- xgb_complete |> filter(season.x >= HOLDOUT_SEASON)
cli::cli_inform("XGBoost train: {nrow(xgb_df)} rows, test: {nrow(xgb_test)} rows")

# Base feature columns — no team names, no venue factor, no GAM predictions
xgb_base_cols <- c(
  "team_type_fac",
  "game_year_decimal.x", "game_prop_through_year.x",
  "game_prop_through_month.x", "game_prop_through_day.x",
  "torp_diff", "torp_recv_diff", "torp_disp_diff",
  "torp_spoil_diff", "torp_hitout_diff", "torp.x", "torp.y",
  "log_dist.x", "log_dist.y", "log_dist_diff",
  "familiarity.x", "familiarity.y", "familiarity_diff",
  "days_rest_diff_fac"
)

## Shared hyperparameters ----
xgb_reg_params <- list(
  objective = "reg:squarederror", eval_metric = "rmse",
  tree_method = "hist", eta = 0.05, subsample = 0.7,
  colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15
)
xgb_cls_params <- list(
  objective = "binary:logistic", eval_metric = "logloss",
  tree_method = "hist", eta = 0.05, subsample = 0.7,
  colsample_bytree = 0.8, max_depth = 3, min_child_weight = 15
)

## Season-grouped CV folds (each season = one fold) ----
train_seasons <- sort(unique(xgb_df$season.x))
cli::cli_inform("Season folds: {paste(train_seasons, collapse=', ')} ({length(train_seasons)} folds)")
xgb_season_folds <- xgb_df$season.x
xgb_folds <- lapply(train_seasons, function(s) which(xgb_season_folds == s))

## Helper: train one XGBoost step with CV for nrounds ----
train_xgb_step <- function(df, label, weights, feature_cols, params, folds, step_name) {
  fmat <- model.matrix(~ . - 1, data = df[, feature_cols, drop = FALSE])
  dtrain <- xgb.DMatrix(data = fmat, label = label, weight = weights)

  cli::cli_progress_step("CV for {step_name}")
  set.seed(1234)
  cv <- xgb.cv(params = params, data = dtrain, nrounds = 1000, folds = folds,
                early_stopping_rounds = 30, print_every_n = 50, verbose = 1)

  metric_col <- paste0("test_", params$eval_metric, "_mean")
  best_n <- which.min(cv$evaluation_log[[metric_col]])
  best_score <- min(cv$evaluation_log[[metric_col]])
  cli::cli_inform("{step_name}: nrounds={best_n}, CV {params$eval_metric}={round(best_score, 4)}")

  set.seed(1234)
  model <- xgb.train(params = params, data = dtrain, nrounds = best_n, print_every_n = 50)
  preds <- predict(model, dtrain)

  list(model = model, preds = preds, best_n = best_n, cv_score = best_score)
}

## Helper: predict on new data using a trained step's model ----
predict_xgb_new <- function(model, df, feature_cols) {
  mat <- model.matrix(~ . - 1, data = df[, feature_cols, drop = FALSE])
  predict(model, xgb.DMatrix(data = mat))
}

## Step 1: Predict total xPoints ----
step1 <- train_xgb_step(xgb_df, xgb_df$total_xpoints_adj, xgb_df$weightz,
                          xgb_base_cols, xgb_reg_params, xgb_folds, "Step 1: total_xpoints")
xgb_df$xgb_pred_tot_xscore   <- step1$preds
xgb_test$xgb_pred_tot_xscore <- predict_xgb_new(step1$model, xgb_test, xgb_base_cols)

## Step 2: Predict xScore diff ----
step2_cols <- c(xgb_base_cols, "xgb_pred_tot_xscore")
step2 <- train_xgb_step(xgb_df, xgb_df$xscore_diff, xgb_df$weightz,
                          step2_cols, xgb_reg_params, xgb_folds, "Step 2: xscore_diff")
xgb_df$xgb_pred_xscore_diff   <- step2$preds
xgb_test$xgb_pred_xscore_diff <- predict_xgb_new(step2$model, xgb_test, step2_cols)

## Step 3: Predict conversion diff ----
step3_cols <- c(xgb_base_cols, "xgb_pred_tot_xscore", "xgb_pred_xscore_diff")
step3 <- train_xgb_step(xgb_df, xgb_df$shot_conv_diff, xgb_df$shot_weightz,
                          step3_cols, xgb_reg_params, xgb_folds, "Step 3: conv_diff")
xgb_df$xgb_pred_conv_diff   <- step3$preds
xgb_test$xgb_pred_conv_diff <- predict_xgb_new(step3$model, xgb_test, step3_cols)

## Step 4: Predict score diff ----
step4_cols <- c(xgb_base_cols, "xgb_pred_xscore_diff", "xgb_pred_conv_diff", "xgb_pred_tot_xscore")
step4 <- train_xgb_step(xgb_df, xgb_df$score_diff, xgb_df$weightz,
                          step4_cols, xgb_reg_params, xgb_folds, "Step 4: score_diff")
xgb_df$xgb_pred_score_diff   <- step4$preds
xgb_test$xgb_pred_score_diff <- predict_xgb_new(step4$model, xgb_test, step4_cols)

## Step 5: Predict win ----
step5_cols <- c(xgb_base_cols, "xgb_pred_tot_xscore", "xgb_pred_score_diff")
step5 <- train_xgb_step(xgb_df, as.numeric(xgb_df$win), xgb_df$weightz,
                          step5_cols, xgb_cls_params, xgb_folds, "Step 5: win")
xgb_df$xgb_pred_win   <- step5$preds
xgb_test$xgb_pred_win <- predict_xgb_new(step5$model, xgb_test, step5_cols)

cli::cli_alert_success("XGBoost pipeline complete (5 sequential models)")

# 9. Temporal Holdout Evaluation ----
# (still inside is.finite(HOLDOUT_SEASON) block)
cli::cli_h1("Temporal Holdout: Train < {HOLDOUT_SEASON}, Test >= {HOLDOUT_SEASON}")

# GAM test predictions come from team_mdl_df (GAMs predict on all rows)
# Align to xgb_test rows (which also filter for complete responses)
xgb_test_mask <- !is.na(team_mdl_df$win) & !is.na(team_mdl_df$total_xpoints_adj) &
                 !is.na(team_mdl_df$xscore_diff) & !is.na(team_mdl_df$shot_conv_diff) &
                 !is.na(team_mdl_df$score_diff) & team_mdl_df$season.x >= HOLDOUT_SEASON

gam_test_win   <- team_mdl_df$pred_win[xgb_test_mask]
gam_test_score <- team_mdl_df$pred_score_diff[xgb_test_mask]
test_labels     <- as.numeric(xgb_test$win)
test_score_diff <- xgb_test$score_diff

# GAM holdout metrics
gam_logloss  <- MLmetrics::LogLoss(gam_test_win, test_labels)
gam_accuracy <- mean(round(gam_test_win) == test_labels)
gam_brier    <- mean((gam_test_win - test_labels)^2)
gam_mae      <- mean(abs(gam_test_score - test_score_diff))
gam_rmse     <- sqrt(mean((gam_test_score - test_score_diff)^2))

# XGBoost holdout metrics
xgb_logloss  <- MLmetrics::LogLoss(xgb_test$xgb_pred_win, test_labels)
xgb_accuracy <- mean(round(xgb_test$xgb_pred_win) == test_labels)
xgb_brier    <- mean((xgb_test$xgb_pred_win - test_labels)^2)
xgb_mae      <- mean(abs(xgb_test$xgb_pred_score_diff - test_score_diff))
xgb_rmse     <- sqrt(mean((xgb_test$xgb_pred_score_diff - test_score_diff)^2))

comparison <- data.frame(
  Metric = c("Win LogLoss", "Win Accuracy (%)", "Win Brier", "Score Diff MAE", "Score Diff RMSE"),
  GAM = c(round(gam_logloss, 4), round(gam_accuracy * 100, 1), round(gam_brier, 4), round(gam_mae, 1), round(gam_rmse, 1)),
  XGBoost = c(round(xgb_logloss, 4), round(xgb_accuracy * 100, 1), round(xgb_brier, 4), round(xgb_mae, 1), round(xgb_rmse, 1)),
  stringsAsFactors = FALSE
)

cat("\n=== Holdout Test Set Comparison ===\n")
cat("Train:", nrow(xgb_df), "rows | Test:", nrow(xgb_test), "rows\n\n")
print(comparison, row.names = FALSE)

cat("\n=== XGBoost CV Scores (on train set, per step) ===\n")
cat(sprintf("  Step 1 (total_xpoints) RMSE:    %.4f  nrounds: %d\n", step1$cv_score, step1$best_n))
cat(sprintf("  Step 2 (xscore_diff)   RMSE:    %.4f  nrounds: %d\n", step2$cv_score, step2$best_n))
cat(sprintf("  Step 3 (conv_diff)     RMSE:    %.4f  nrounds: %d\n", step3$cv_score, step3$best_n))
cat(sprintf("  Step 4 (score_diff)    RMSE:    %.4f  nrounds: %d\n", step4$cv_score, step4$best_n))
cat(sprintf("  Step 5 (win)           LogLoss: %.4f  nrounds: %d\n", step5$cv_score, step5$best_n))

# Feature importance for final win step
importance <- xgb.importance(model = step5$model)
cat("\nTop 10 XGBoost Features (win step):\n")
print(head(importance, 10))

} else {
  cli::cli_inform("Skipping XGBoost pipeline & evaluation (production mode, HOLDOUT_SEASON = Inf)")
}

# 10. Save Models ----
cli::cli_h2("Saving models")

output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

## Save GAM pipeline ----
match_gams <- list(
  total_xpoints = afl_total_xpoints_mdl,
  xscore_diff   = afl_xscore_diff_mdl,
  conv_diff      = afl_conv_mdl,
  score_diff     = afl_score_mdl,
  win            = afl_win_mdl
)
gam_path <- file.path(output_dir, "match_gams.rds")
saveRDS(match_gams, gam_path)
cli::cli_inform("Saved GAM pipeline to {gam_path}")

## Save XGBoost pipeline (evaluation mode only) ----
if (is.finite(HOLDOUT_SEASON)) {
  match_xgb <- list(
    total_xpoints = step1$model,
    xscore_diff   = step2$model,
    conv_diff      = step3$model,
    score_diff     = step4$model,
    win            = step5$model
  )
  xgb_path <- file.path(output_dir, "match_xgb_pipeline.rds")
  saveRDS(match_xgb, xgb_path)
  cli::cli_inform("Saved XGBoost pipeline to {xgb_path}")
}

## Upload to GitHub releases ----
if (requireNamespace("piggyback", quietly = TRUE)) {
  cli::cli_inform("Uploading GAM models to GitHub release...")
  tryCatch({
    piggyback::pb_upload(gam_path, repo = "peteowen1/torpmodels", tag = "core-models")
    cli::cli_alert_success("Upload complete!")
  }, error = function(e) {
    cli::cli_warn(c(
      "Upload to GitHub failed: {e$message}",
      "i" = "Models saved locally at {gam_path}",
      "i" = "Upload manually with: piggyback::pb_upload('{gam_path}', repo='peteowen1/torpmodels', tag='core-models')"
    ))
  })
} else {
  cli::cli_warn(c(
    "piggyback package not installed -- skipping upload to GitHub releases.",
    "i" = "Models saved locally at {gam_path}"
  ))
}

tictoc::toc()

# Summary ----
cat("\n=== Final Summary ===\n")
cat("GAM trained on:", nrow(gam_df), "rows\n")
if (is.finite(HOLDOUT_SEASON)) {
  cat("Holdout test:", nrow(xgb_test), "rows (season >=", HOLDOUT_SEASON, ")\n")
  cat("GAM holdout LogLoss:", round(gam_logloss, 4), "\n")
  cat("XGBoost holdout LogLoss:", round(xgb_logloss, 4), "\n")
  cat("Saved: match_gams.rds + match_xgb_pipeline.rds\n")
} else {
  cat("Production mode: all data used for training\n")
  cat("Saved & uploaded: match_gams.rds\n")
}
