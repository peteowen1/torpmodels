# Model Loading Functions
# =======================
# Functions for loading pre-trained models from local cache or GitHub releases

# Package-level constants for available models
# ---------------------------------------------

#' @noRd
.CORE_MODELS <- c(
  "ep" = "Expected Points (EP) model - XGBoost multiclass for predicting expected points from field position",
  "wp" = "Win Probability (WP) model - predicts probability of winning from game state",
  "shot" = "Shot outcome model - ordered categorical model for shot results",
  "xgb_win" = "XGBoost match prediction model"
)

#' @noRd
.STAT_MODELS <- c(
  "goals", "behinds", "disposals", "kicks", "handballs", "marks",
  "contested_marks", "tackles", "hitouts", "frees_for", "frees_against",
  "inside50s", "rebound50s", "clearances_total_clearances",
  "clearances_centre_clearances", "clearances_stoppage_clearances",
  "contested_possessions", "uncontested_possessions", "clangers",
  "bounces", "one_percenters", "goal_assists", "marks_inside50",
  "tackles_inside50", "shots_at_goal", "goal_accuracy", "turnovers",
  "intercepts", "score_involvements", "disposal_efficiency",
  "time_on_ground_percentage", "total_possessions",
  # Extended stats models
  "extended_stats_centre_bounce_attendances",
  "extended_stats_contest_def_loss_percentage",
  "extended_stats_contest_def_losses",
  "extended_stats_contest_def_one_on_ones",
  "extended_stats_contest_off_one_on_ones",
  "extended_stats_contest_off_wins",
  "extended_stats_contest_off_wins_percentage",
  "extended_stats_contested_possession_rate",
  "extended_stats_def_half_pressure_acts",
  "extended_stats_effective_disposals",
  "extended_stats_effective_kicks",
  "extended_stats_f50ground_ball_gets",
  "extended_stats_ground_ball_gets",
  "extended_stats_hitout_to_advantage_rate",
  "extended_stats_hitout_win_percentage",
  "extended_stats_hitouts_to_advantage",
  "extended_stats_intercept_marks",
  "extended_stats_kick_efficiency",
  "extended_stats_kick_to_handball_ratio",
  "extended_stats_kickins",
  "extended_stats_kickins_playon",
  "extended_stats_marks_on_lead",
  "extended_stats_pressure_acts",
  "extended_stats_ruck_contests",
  "extended_stats_score_launches",
  "extended_stats_spoils"
)

#' Get the torpmodels repository
#'
#' @return Character string of the repository in format "owner/repo"
#' @keywords internal
get_torpmodels_repo <- function() {
  getOption("torpmodels.repo", "peteowen1/torpmodels")
}

#' Get the local models directory
#'
#' Returns the path to the local models cache directory. By default uses
#' the inst/models directory within the package, but can be overridden
#' via the torpmodels.cache_dir option.
#'
#' @return Character string path to local models directory
#' @keywords internal
get_models_dir <- function() {

  # Check for user-specified cache directory
  cache_dir <- getOption("torpmodels.cache_dir", NULL)

  if (!is.null(cache_dir)) {
    if (!dir.exists(cache_dir)) {
      dir.create(cache_dir, recursive = TRUE)
    }
    return(cache_dir)
  }

  # Default to user cache directory
  cache_base <- tools::R_user_dir("torpmodels", "cache")
  models_dir <- file.path(cache_base, "models")

  # Create directory if it doesn't exist

  if (!dir.exists(models_dir)) {
    dir.create(models_dir, recursive = TRUE)
  }

  return(models_dir)
}

#' Load a TORP Model
#'
#' Loads a pre-trained model from local cache or downloads from GitHub releases.
#' Models are cached locally after first download for faster subsequent loads.
#'
#' @param model_name Character. Name of the model to load. One of:
#'   - "ep" or "ep_model" - Expected Points model
#'   - "wp" or "wp_model" - Win Probability model
#'   - "shot" or "shot_ocat_mdl" - Shot outcome classification model
#'   - "xgb_win" or "xgb_win_model" - XGBoost match prediction model
#' @param force_download Logical. If TRUE, downloads fresh copy even if cached locally.
#' @param verbose Logical. If TRUE, prints status messages.
#'
#' @return The loaded model object
#' @export
#'
#' @examples
#' \dontrun{
#' ep_model <- load_torp_model("ep")
#' wp_model <- load_torp_model("wp")
#' }
load_torp_model <- function(model_name, force_download = FALSE, verbose = TRUE) {
  # Normalize model name
  model_info <- normalize_model_name(model_name)

  if (is.null(model_info)) {
    cli::cli_abort("Unknown model: {model_name}. Available models: ep, wp, shot, xgb_win")
  }

  model_file <- model_info$file
  release_tag <- model_info$tag

  # Check local cache first
  local_path <- file.path(get_models_dir(), "core", model_file)

  if (file.exists(local_path) && !force_download) {
    if (verbose) {
      cli::cli_inform("Loading {model_name} from local cache")
    }
    return(safe_read_rds(local_path, model_name))
  }

  # Download from GitHub release
  if (verbose) {
    cli::cli_inform("Downloading {model_name} from GitHub releases...")
  }

  download_model_from_release(model_file, release_tag, local_path, verbose)

  if (!file.exists(local_path)) {
    cli::cli_abort("Failed to download model: {model_name}")
  }

  return(safe_read_rds(local_path, model_name))
}

#' Load a Stat Model
#'
#' Loads a per-statistic GAM model from local cache or downloads from GitHub releases.
#' These models predict individual player statistics.
#'
#' @param stat_name Character. Name of the statistic model to load (e.g., "goals", "disposals").
#' @param force_download Logical. If TRUE, downloads fresh copy even if cached locally.
#' @param verbose Logical. If TRUE, prints status messages.
#'
#' @return The loaded GAM model object
#' @export
#'
#' @examples
#' \dontrun{
#' goals_model <- load_stat_model("goals")
#' disposals_model <- load_stat_model("disposals")
#' }
load_stat_model <- function(stat_name, force_download = FALSE, verbose = TRUE) {
  if (!grepl("^[a-z_]+$", stat_name)) {
    cli::cli_abort("Invalid stat name: {stat_name}. Must contain only lowercase letters and underscores.")
  }

  known_stats <- list_available_models()$stat_models
  if (!stat_name %in% known_stats) {
    cli::cli_abort("Unknown stat: {stat_name}. See {.fn list_available_models} for available stats.")
  }

  model_file <- paste0(stat_name, ".rds")
  release_tag <- "stat-models"

  # Check local cache first
  local_path <- file.path(get_models_dir(), "stat-models", model_file)

  if (file.exists(local_path) && !force_download) {
    if (verbose) {
      cli::cli_inform("Loading stat model '{stat_name}' from local cache")
    }
    return(safe_read_rds(local_path, stat_name))
  }

  # Download from GitHub release
  if (verbose) {
    cli::cli_inform("Downloading stat model '{stat_name}' from GitHub releases...")
  }

  download_model_from_release(model_file, release_tag, local_path, verbose)

  if (!file.exists(local_path)) {
    cli::cli_abort("Failed to download stat model: {stat_name}")
  }

  return(safe_read_rds(local_path, stat_name))
}

#' List Available Models
#'
#' Returns a list of available models that can be loaded.
#'
#' @return A list with two elements: core_models and stat_models
#' @export
#'
#' @examples
#' list_available_models()
list_available_models <- function() {
  list(
    core_models = .CORE_MODELS,
    stat_models = .STAT_MODELS
  )
}

#' Check Model Cache Status
#'
#' Shows which models are cached locally and their file sizes.
#'
#' @return A data frame with model names, cached status, and sizes
#' @export
#'
#' @examples
#' check_model_cache()
check_model_cache <- function() {
  models_dir <- get_models_dir()
  core_dir <- file.path(models_dir, "core")
  stat_dir <- file.path(models_dir, "stat-models")

  core_files <- c("ep_model.rds", "wp_model.rds", "shot_ocat_mdl.rds", "xgb_win_model.rds")

  results <- data.frame(
    model = character(),
    type = character(),
    cached = logical(),
    size_mb = numeric(),
    stringsAsFactors = FALSE
  )

  # Check core models
  for (f in core_files) {
    path <- file.path(core_dir, f)
    cached <- file.exists(path)
    size <- if (cached) file.size(path) / 1024^2 else NA
    results <- rbind(results, data.frame(
      model = gsub("\\.rds$", "", f),
      type = "core",
      cached = cached,
      size_mb = round(size, 2),
      stringsAsFactors = FALSE
    ))
  }

  # Check stat models directory
  if (dir.exists(stat_dir)) {
    stat_files <- list.files(stat_dir, pattern = "\\.rds$")
    for (f in stat_files) {
      path <- file.path(stat_dir, f)
      size <- file.size(path) / 1024^2
      results <- rbind(results, data.frame(
        model = gsub("\\.rds$", "", f),
        type = "stat",
        cached = TRUE,
        size_mb = round(size, 2),
        stringsAsFactors = FALSE
      ))
    }
  }

  return(results)
}

#' Clear Model Cache
#'
#' Removes cached models from local storage.
#'
#' @param type Character. One of "all", "core", or "stat" to specify which models to clear.
#' @param verbose Logical. If TRUE, prints status messages.
#'
#' @return Invisible NULL
#' @export
#'
#' @examples
#' \dontrun{
#' clear_model_cache("all")
#' clear_model_cache("stat")
#' }
clear_model_cache <- function(type = "all", verbose = TRUE) {
  type <- match.arg(type, c("all", "core", "stat"))
  models_dir <- get_models_dir()

  if (type %in% c("all", "core")) {
    core_dir <- file.path(models_dir, "core")
    if (dir.exists(core_dir)) {
      files <- list.files(core_dir, full.names = TRUE)
      if (length(files) > 0) {
        unlink(files)
        if (verbose) cli::cli_inform("Cleared {length(files)} core model(s)")
      }
    }
  }

  if (type %in% c("all", "stat")) {
    stat_dir <- file.path(models_dir, "stat-models")
    if (dir.exists(stat_dir)) {
      files <- list.files(stat_dir, full.names = TRUE)
      if (length(files) > 0) {
        unlink(files)
        if (verbose) cli::cli_inform("Cleared {length(files)} stat model(s)")
      }
    }
  }

  invisible(NULL)
}

# Internal helper functions

#' Normalize model name to file and tag
#' @keywords internal
normalize_model_name <- function(model_name) {
  model_name <- tolower(model_name)

  model_map <- list(
    ep = list(file = "ep_model.rds", tag = "core-models"),
    ep_model = list(file = "ep_model.rds", tag = "core-models"),
    wp = list(file = "wp_model.rds", tag = "core-models"),
    wp_model = list(file = "wp_model.rds", tag = "core-models"),
    shot = list(file = "shot_ocat_mdl.rds", tag = "core-models"),
    shot_ocat_mdl = list(file = "shot_ocat_mdl.rds", tag = "core-models"),
    xgb_win = list(file = "xgb_win_model.rds", tag = "core-models"),
    xgb_win_model = list(file = "xgb_win_model.rds", tag = "core-models")
  )

  return(model_map[[model_name]])
}

#' Safely read an RDS file with error handling
#'
#' Wraps readRDS() in tryCatch to handle corrupted downloads gracefully.
#' Deletes the corrupted file so the next attempt starts fresh.
#'
#' @param path Path to the RDS file
#' @param label Label for error messages (e.g., model name)
#' @return The deserialized R object
#' @keywords internal
safe_read_rds <- function(path, label = basename(path)) {
  tryCatch(
    readRDS(path),
    error = function(e) {
      unlink(path)
      cli::cli_abort(
        "Model file for {label} is corrupted or unreadable: {e$message}. Cache cleared, try again."
      )
    }
  )
}

#' Download model from GitHub release
#' @keywords internal
#' @importFrom cli cli_inform cli_warn cli_abort
#' @importFrom utils download.file
download_model_from_release <- function(file_name, release_tag, local_path, verbose = TRUE) {
  repo <- get_torpmodels_repo()

  # Ensure parent directory exists
  parent_dir <- dirname(local_path)
  if (!dir.exists(parent_dir)) {
    dir.create(parent_dir, recursive = TRUE)
  }

  pb_error <- NULL
  url_error <- NULL

  # Try piggyback first (preferred method)
  tryCatch({
    # Download to temp location first
    temp_dir <- tempdir()

    piggyback::pb_download(
      file = file_name,
      repo = repo,
      tag = release_tag,
      dest = temp_dir
    )

    temp_path <- file.path(temp_dir, file_name)
    if (file.exists(temp_path)) {
      file.copy(temp_path, local_path, overwrite = TRUE)
      unlink(temp_path)
      if (verbose) cli::cli_inform("Successfully downloaded {file_name}")
      return(invisible(TRUE))
    }
  }, error = function(e) {
    pb_error <<- e$message
    if (verbose) cli::cli_warn("piggyback download failed: {e$message}")
  })

  # Fallback to direct URL download
  tryCatch({
    url <- paste0(
      "https://github.com/", repo, "/releases/download/",
      release_tag, "/", file_name
    )

    if (verbose) cli::cli_inform("Trying direct download from {url}")

    download.file(url, local_path, mode = "wb", quiet = !verbose)

    if (file.exists(local_path) && file.size(local_path) > 0) {
      if (verbose) cli::cli_inform("Successfully downloaded {file_name}")
      return(invisible(TRUE))
    }
  }, error = function(e) {
    url_error <<- e$message
    cli::cli_warn("Direct download failed: {e$message}")
  })

  # Both methods failed - report both errors
  details <- character()
  if (!is.null(pb_error)) details <- c(details, paste0("piggyback: ", pb_error))
  if (!is.null(url_error)) details <- c(details, paste0("direct URL: ", url_error))
  detail_msg <- paste(details, collapse = "; ")

  cli::cli_abort("Failed to download {file_name} from release {release_tag}. {detail_msg}")
}
