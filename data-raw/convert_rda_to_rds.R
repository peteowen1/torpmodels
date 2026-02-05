# Convert .rda models from torp to .rds format for torpmodels
# ============================================================
# This script converts the existing models from torp's data/*.rda format
# to the .rds format used by torpmodels for GitHub releases.

# Set paths
torp_data_dir <- "../torp/data"
output_core_dir <- "inst/models/core"
output_stat_dir <- "inst/models/stat-models"

# Create output directories
dir.create(output_core_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_stat_dir, recursive = TRUE, showWarnings = FALSE)

# Models to convert
core_models <- c("ep_model", "wp_model", "shot_ocat_mdl", "xgb_win_model")

cli::cli_h1("Converting core models from torp/data/")

for (model_name in core_models) {
  rda_path <- file.path(torp_data_dir, paste0(model_name, ".rda"))
  rds_path <- file.path(output_core_dir, paste0(model_name, ".rds"))

  if (file.exists(rda_path)) {
    # Load the .rda file
    env <- new.env()
    load(rda_path, envir = env)

    # Get the model object (should be named same as file)
    if (exists(model_name, envir = env)) {
      model <- get(model_name, envir = env)

      # Save as .rds
      saveRDS(model, rds_path)

      file_size <- round(file.size(rds_path) / 1024^2, 2)
      cli::cli_inform("Converted {model_name}: {file_size} MB")
    } else {
      cli::cli_warn("Model object '{model_name}' not found in {rda_path}")
    }
  } else {
    cli::cli_warn("File not found: {rda_path}")
  }
}

# Convert stat-models from torp/data-raw/stat-models/
stat_models_dir <- "../torp/data-raw/stat-models"

if (dir.exists(stat_models_dir)) {
  cli::cli_h1("Converting stat-models from torp/data-raw/stat-models/")

  stat_files <- list.files(stat_models_dir, pattern = "\\.rds$", full.names = TRUE)

  for (stat_file in stat_files) {
    file_name <- basename(stat_file)
    output_path <- file.path(output_stat_dir, file_name)

    # Copy the file (already in .rds format)
    file.copy(stat_file, output_path, overwrite = TRUE)

    file_size <- round(file.size(output_path) / 1024^2, 2)
    cli::cli_inform("Copied {file_name}: {file_size} MB")
  }

  cli::cli_inform("Converted {length(stat_files)} stat models")
}

cli::cli_h1("Conversion complete!")
cli::cli_inform("Core models saved to: {output_core_dir}")
cli::cli_inform("Stat models saved to: {output_stat_dir}")

# Summary
core_files <- list.files(output_core_dir, pattern = "\\.rds$")
stat_files <- list.files(output_stat_dir, pattern = "\\.rds$")

cli::cli_inform("Total core models: {length(core_files)}")
cli::cli_inform("Total stat models: {length(stat_files)}")
