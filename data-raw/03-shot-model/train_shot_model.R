# Train Shot Outcome Model
# ========================
# This script trains the shot outcome ordered categorical model using GAM.
# Predicts: clanger (miss), behind, goal

library(dplyr)
library(forcats)
library(mgcv)

# Load torp for data loading functions
if (!require(torp)) {
  devtools::load_all("../../torp")  # Load from sibling directory
}

# Load training data
cli::cli_inform("Loading play-by-play data...")
shots_prep <- torp::load_pbp(seasons = TRUE, rounds = TRUE)

# Filter to shots only
shots_all <- shots_prep %>%
  dplyr::filter(!is.na(points_shot) | !is.na(shot_at_goal))

shots <- shots_all %>%
  dplyr::filter(
    !is.na(shot_at_goal),
    x > 0,
    goal_x < 65,
    abs_y < 45
  )

# Create shot category variable
shots <- shots %>%
  dplyr::mutate(
    scored_shot = ifelse(!is.na(points_shot), 1, 0),
    shot_cat = dplyr::case_when(
      is.na(points_shot) ~ 1,  # Miss/clanger
      points_shot == 1 ~ 2,    # Behind
      points_shot == 6 ~ 3     # Goal
    )
  )

# Create player factor with lumping for rare players
shots$player_id_shot <- forcats::fct_lump_min(shots$player_id, 10, other_level = "Other")

# Create player name mapping
player_name_mapping <- shots %>%
  dplyr::group_by(player_id_shot = player_id) %>%
  dplyr::summarise(player_name_shot = dplyr::last(player_name))

shot_player_df <- tibble::tibble(
  player_id_shot = levels(shots$player_id_shot)
) %>%
  dplyr::left_join(player_name_mapping)

# Train the ordered categorical GAM
cli::cli_inform("Training shot outcome model (this may take a while)...")
shot_ocat_mdl <- mgcv::bam(
  shot_cat ~
    ti(goal_x, abs_y, by = phase_of_play, bs = "ts")
    + ti(goal_x, abs_y, bs = "ts")
    + s(goal_x, bs = "ts")
    + s(abs_y, bs = "ts")
    + ti(lag_goal_x, lag_y)
    + s(lag_goal_x, bs = "ts")
    + s(lag_y, bs = "ts")
    + s(play_type, bs = "re")
    + s(phase_of_play, bs = "re")
    + s(player_position_fac, bs = "re")
    + s(player_id_shot, bs = "re"),
  data = shots,
  family = ocat(R = 3),
  nthreads = 4,
  select = TRUE,
  discrete = TRUE,
  drop.unused.levels = FALSE
)

# Save the model
output_dir <- file.path(getwd(), "inst", "models", "core")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

output_path <- file.path(output_dir, "shot_ocat_mdl.rds")
saveRDS(shot_ocat_mdl, output_path)
cli::cli_inform("Saved shot model to {output_path}")

# Also save the player mapping (needed for predictions)
player_path <- file.path(output_dir, "shot_player_df.rds")
saveRDS(shot_player_df, player_path)
cli::cli_inform("Saved player mapping to {player_path}")

# Upload to GitHub release (if piggyback is available)
if (requireNamespace("piggyback", quietly = TRUE)) {
  cli::cli_inform("Uploading to GitHub release...")
  piggyback::pb_upload(
    output_path,
    repo = "peteowen1/torpmodels",
    tag = "core-models"
  )
  cli::cli_inform("Upload complete!")
}
