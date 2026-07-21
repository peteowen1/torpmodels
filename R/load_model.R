# Model Loading Functions
# =======================
# Functions for loading pre-trained models from local cache or GitHub releases

# Package-level constants for available models
# ---------------------------------------------

#' @noRd
.CORE_MODELS <- c(
  "ep" = "Expected Points (EP) model - XGBoost multiclass for predicting expected points from field position",
  "wp" = "Win Probability (WP) model - predicts probability of winning from game state",
  "wp_calibration" = "WP recalibration sidecar - two-parameter Platt-on-logit correction for temporal calibration drift (torpverse/docs/plans/FABLE-RECAL-PLAN.md)",
  "shot" = "Shot outcome model - ordered categorical model for shot results",
  "match_gams" = "Sequential GAM pipeline for match predictions (5 models: total_xpoints, xscore_diff, conv_diff, score_diff, win)",
  "match_xgb_pipeline" = "XGBoost pipeline (5 models) for match predictions - evaluation/comparison only",
  "match_margin_calibration" = "Match margin recalibration sidecar - single-slope temporal-holdout correction for the blended match-model margin (torpverse/docs/plans/FABLE-MATCH-MAE-PLAN.md)",
  "xgb_win" = "Legacy XGBoost match prediction model (superseded by match_gams)",
  "shot_player_df" = "Shot player lookup table - maps player IDs to lumped factor levels for shot model"
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
#' \code{tools::R_user_dir("torpmodels", "cache")}, but can be overridden
#' via the \code{torpmodels.cache_dir} option.
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

# Manifest-verified cache freshness (ECOSYSTEM-FIX-PLAN.md M3/M4)
# -----------------------------------------------------------------
# torpmodels' commit record is models_manifest.json (R/publish.R), NOT
# versebus's generic bus_manifest.json -- same sha256-per-artifact idea, a
# different asset name/schema (`artifacts` keyed by file name, not `assets`
# keyed by a `name` field). `.get_models_manifest()` fetches it via
# publish.R's `.fetch_manifest()` (404 -> NULL means legacy mode), rate
# limited to one fetch per (repo, tag) per session per 15-minute window.
# `.fetch_manifest()` aborts on transient errors when called from its
# original producer path (never risk clobbering the ledger) -- here, as a
# read-only cache-freshness check, that would wrongly hard-fail a load that
# could otherwise succeed from a perfectly good local cache, so the abort is
# caught and degraded to "couldn't verify this session" instead. A tag with
# no manifest at all (stat-models, as of this writing) keeps exactly the
# pre-M3 existence-only cache behaviour, plus one cli_warn for the whole
# session -- never a hard failure.

#' @noRd
.tm_manifest_state <- new.env(parent = emptyenv())

#' @noRd
.tm_manifest_ttl_secs <- 900L

#' Session-cached, rate-limited fetch of a tag's models_manifest.json
#' @keywords internal
.get_models_manifest <- function(repo, tag, verbose = TRUE) {
  key <- paste0(repo, "@", tag)
  cached <- .tm_manifest_state[[key]]
  if (!is.null(cached) &&
      as.numeric(Sys.time() - cached$fetched_at, units = "secs") < .tm_manifest_ttl_secs) {
    return(cached$manifest)
  }

  manifest <- tryCatch(
    .fetch_manifest(repo, tag),
    error = function(e) {
      if (verbose) {
        cli::cli_warn("Could not fetch models_manifest.json for {.val {tag}} ({conditionMessage(e)}) -- skipping cache verification this session")
      }
      NULL
    }
  )
  .tm_manifest_state[[key]] <- list(manifest = manifest, fetched_at = Sys.time())

  if (is.null(manifest) && verbose && !isTRUE(.tm_manifest_state$.legacy_warned)) {
    cli::cli_warn("{.val {tag}} on {.val {repo}} has no models_manifest.json -- running unverified (legacy mode)")
    .tm_manifest_state$.legacy_warned <- TRUE
  }

  manifest
}

#' Is a locally cached model file still valid against models_manifest.json?
#'
#' TRUE (serve from cache) when there's no manifest to check against (legacy
#' mode) or no entry for this specific file (an untracked artifact -- can't
#' validate it, so don't punish it); otherwise delegates to versebus's
#' sidecar-based `vb_cache_validate()`.
#' @keywords internal
.tm_cache_is_fresh <- function(repo, tag, file_name, local_path, verbose = TRUE) {
  manifest <- .get_models_manifest(repo, tag, verbose)
  if (is.null(manifest) || is.null(manifest$artifacts)) return(TRUE)
  entry <- manifest$artifacts[[file_name]]
  if (is.null(entry) || is.null(entry$sha256)) return(TRUE)
  vb_cache_validate(local_path, entry)
}

#' Load a TORP Model
#'
#' Loads a pre-trained model from local cache or downloads from GitHub releases.
#' Models are cached locally after first download for faster subsequent loads.
#'
#' @param model_name Character. Name of the model to load. One of:
#'   - "ep" or "ep_model" - Expected Points model
#'   - "wp" or "wp_model" - Win Probability model
#'   - "wp_calibration" - WP recalibration sidecar (published atomically with "wp")
#'   - "shot" or "shot_ocat_mdl" - Shot outcome classification model
#'   - "match_gams" - Sequential GAM pipeline for match predictions (production)
#'   - "match_margin_calibration" - Match margin recalibration sidecar (published atomically with "match_gams")
#'   - "xgb_win" or "xgb_win_model" - Legacy XGBoost match prediction model
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
    cli::cli_abort("Unknown model: {model_name}. Available models: ep, wp, wp_calibration, shot, match_gams, match_margin_calibration, xgb_win (legacy)")
  }

  model_file <- model_info$file
  release_tag <- model_info$tag
  repo <- get_torpmodels_repo()

  # Check local cache first
  local_path <- file.path(get_models_dir(), "core", model_file)

  if (file.exists(local_path) && !force_download) {
    if (.tm_cache_is_fresh(repo, release_tag, model_file, local_path, verbose)) {
      if (verbose) {
        cli::cli_inform("Loading {model_name} from local cache")
      }
      return(.load_with_provenance(local_path, model_name, verbose))
    }
    if (verbose) {
      cli::cli_inform("Cached {model_name} does not match models_manifest.json -- re-downloading")
    }
  }

  # Download from GitHub release
  if (verbose) {
    cli::cli_inform("Downloading {model_name} from GitHub releases...")
  }

  had_cache <- file.exists(local_path)
  dl_result <- tryCatch(
    download_model_from_release(model_file, release_tag, local_path, verbose),
    error = function(e) e
  )
  if (inherits(dl_result, "error")) {
    # A stale/mismatched manifest that made the cache look non-fresh should
    # never turn a genuine network failure into "no model at all" when a
    # perfectly loadable file already sits in the cache -- on main this
    # always served the cached RDS; restore that as the fallback here too.
    if (had_cache && !isTRUE(force_download)) {
      cli::cli_warn("Download of {model_name} failed ({conditionMessage(dl_result)}) -- serving existing local cache instead")
      return(.load_with_provenance(local_path, model_name, verbose))
    }
    stop(dl_result)
  }

  if (!file.exists(local_path)) {
    cli::cli_abort("Failed to download model: {model_name}")
  }

  return(.load_with_provenance(local_path, model_name, verbose))
}

#' Read a model RDS and report its provenance stamp
#'
#' Never hard-fails on a meta-less artifact -- old models predate provenance
#' stamping and must keep loading, just with a warning.
#' @keywords internal
.load_with_provenance <- function(local_path, model_name, verbose) {
  object <- safe_read_rds(local_path, model_name)
  meta <- model_meta(object)
  if (verbose) {
    if (!is.null(meta)) {
      describe_model_meta(meta)
    } else {
      cli::cli_warn("{model_name} has no torp_meta -- trained before provenance stamping or via a non-canonical path")
    }
  }
  object
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
  repo <- get_torpmodels_repo()

  # Check local cache first
  local_path <- file.path(get_models_dir(), "stat-models", model_file)

  if (file.exists(local_path) && !force_download) {
    if (.tm_cache_is_fresh(repo, release_tag, model_file, local_path, verbose)) {
      if (verbose) {
        cli::cli_inform("Loading stat model '{stat_name}' from local cache")
      }
      return(safe_read_rds(local_path, stat_name))
    }
    if (verbose) {
      cli::cli_inform("Cached stat model '{stat_name}' does not match models_manifest.json -- re-downloading")
    }
  }

  # Download from GitHub release
  if (verbose) {
    cli::cli_inform("Downloading stat model '{stat_name}' from GitHub releases...")
  }

  had_cache <- file.exists(local_path)
  dl_result <- tryCatch(
    download_model_from_release(model_file, release_tag, local_path, verbose),
    error = function(e) e
  )
  if (inherits(dl_result, "error")) {
    if (had_cache && !isTRUE(force_download)) {
      cli::cli_warn("Download of stat model '{stat_name}' failed ({conditionMessage(dl_result)}) -- serving existing local cache instead")
      return(safe_read_rds(local_path, stat_name))
    }
    stop(dl_result)
  }

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

  core_files <- unique(vapply(
    names(.CORE_MODELS),
    function(m) normalize_model_name(m)$file,
    character(1)
  ))

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
    wp_calibration = list(file = "wp_calibration.rds", tag = "core-models"),
    shot = list(file = "shot_ocat_mdl.rds", tag = "core-models"),
    shot_ocat_mdl = list(file = "shot_ocat_mdl.rds", tag = "core-models"),
    xgb_win = list(file = "xgb_win_model.rds", tag = "core-models"),
    xgb_win_model = list(file = "xgb_win_model.rds", tag = "core-models"),
    match_gams = list(file = "match_gams.rds", tag = "core-models"),
    match_xgb_pipeline = list(file = "match_xgb_pipeline.rds", tag = "core-models"),
    match_margin_calibration = list(file = "match_margin_calibration.rds", tag = "core-models"),
    shot_player_df = list(file = "shot_player_df.rds", tag = "core-models")
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
      msg <- conditionMessage(e)
      # Only delete cached file for actual corruption/deserialization errors
      # Don't delete for environment issues (missing packages, OOM, etc.)
      is_corruption <- grepl(
        "unknown input format|not an RDS file|decompression|bad restore file|unexpected end|ascii85|error reading from connection",
        msg, ignore.case = TRUE
      )
      if (is_corruption) {
        unlink(path)
        unlink(paste0(path, ".sha256"))
        cli::cli_abort(
          "Model file for {label} is corrupted: {msg}. Cache cleared, try again."
        )
      }
      cli::cli_abort(
        "Failed to load model {label}: {msg}"
      )
    }
  )
}

#' Download model from GitHub release
#'
#' Verifies sha256 against `models_manifest.json` when the tag's manifest
#' tracks this file -- replacing the old `file.size > 1000` heuristic, which
#' is now only a last-resort fallback for tags with no manifest yet (e.g.
#' stat-models, as of this writing). Each download attempt lands in a
#' tempdir beside the destination and is moved into place atomically via
#' `vb_atomic_write()`, with a `<local_path>.sha256` sidecar written
#' alongside on success. Three retry layers, innermost first: (1) the raw
#' fetch call itself (`piggyback::pb_download()` / `download.file()`) is
#' wrapped in `.vb_retry()`, which retries up to 3 attempts total with 2s
#' then 5s backoff on transient errors -- this is the fix for
#' torpdata#66/#68, where GitHub's release-asset CDN throws sporadic 5xx
#' errors that clear on their own within a few seconds; a confirmed-absent
#' (404) error is excluded via `vb_classify_error()` and re-raised
#' immediately, never retried. (2) A failed integrity check (corrupt/short
#' download that DID complete a fetch) deletes the temp and retries the
#' whole attempt (including its own inner `.vb_retry()`) once more for the
#' SAME method. (3) If piggyback still fails after both layers, falls
#' through to a direct release URL. A pre-existing `local_path` is never
#' touched by a failed download.
#' @keywords internal
#' @importFrom cli cli_inform cli_warn cli_abort
download_model_from_release <- function(file_name, release_tag, local_path, verbose = TRUE) {
  repo <- get_torpmodels_repo()

  # Ensure parent directory exists
  parent_dir <- dirname(local_path)
  if (!dir.exists(parent_dir)) {
    dir.create(parent_dir, recursive = TRUE)
  }

  manifest <- .get_models_manifest(repo, release_tag, verbose)
  entry <- if (!is.null(manifest) && !is.null(manifest$artifacts)) manifest$artifacts[[file_name]] else NULL

  # One fetch+verify+place attempt. `fetch_fn(tmpdir)` must leave `file_name`
  # inside `tmpdir`; raises a vb_error_integrity (via .vb_abort(), vendored
  # in versebus.R) on a corrupt/undersized/mismatched download, otherwise
  # propagates whatever error the download call itself raised (network,
  # 404, ...).
  attempt <- function(fetch_fn) {
    tmpdir <- tempfile(".tm_dl_", tmpdir = parent_dir)
    dir.create(tmpdir)
    on.exit(unlink(tmpdir, recursive = TRUE), add = TRUE)

    # Transient-network retry layer (torpdata#66/#68): GitHub's release-asset
    # CDN throws sporadic 5xx errors from piggyback::pb_download() /
    # download.file() that usually clear within a few seconds. This wraps
    # only the raw fetch call -- confirmed-absent (404) errors are excluded
    # via should_retry and propagate immediately, never retried here.
    .vb_retry(
      function() fetch_fn(tmpdir),
      should_retry = function(e) vb_classify_error(e) != "absent"
    )

    tmp <- file.path(tmpdir, file_name)
    if (!file.exists(tmp) || file.size(tmp) == 0L) {
      .vb_abort("{file_name}: download produced no/empty file", "vb_error_integrity")
    }
    # RDS validity check independent of file size: saveRDS() defaults to
    # gzip compression, so any real .rds -- even a ~200-byte calibration
    # sidecar list of a few scalars -- starts with the gzip magic bytes
    # (0x1f 0x8b). An HTML error page or truncated download won't. This
    # replaces a flat file.size <= 1000L floor that rejected legitimate
    # small sidecars (wp_calibration.rds, match_margin_calibration.rds).
    is_valid_rds <- function(path) {
      con <- file(path, "rb")
      on.exit(close(con), add = TRUE)
      magic <- readBin(con, "raw", 2L)
      length(magic) == 2L && identical(as.integer(magic), c(0x1fL, 0x8bL))
    }
    if (!is_valid_rds(tmp)) {
      .vb_abort("{file_name}: downloaded file is not a valid .rds (likely an error page or truncated download)", "vb_error_integrity")
    }
    if (!is.null(entry) && !is.null(entry$sha256)) {
      got <- vb_sha256(tmp)
      if (!identical(got, entry$sha256)) {
        # A stale models_manifest.json entry (e.g. an upload that skipped
        # update_models_manifest()) is far more likely than corruption --
        # the RDS-validity check above already caught the corruption case.
        # Warn rather than permanently brick the asset until someone
        # manually republishes the manifest.
        cli::cli_warn(
          "{file_name}: sha256 mismatch vs models_manifest.json (got {substr(got, 1, 12)}..., want {substr(entry$sha256, 1, 12)}...) -- manifest may be stale, trusting the verified-valid download"
        )
      }
    }

    vb_atomic_write(function(p) file.copy(tmp, p, overwrite = TRUE), local_path)
    writeLines(vb_sha256(local_path), paste0(local_path, ".sha256"))
    invisible(TRUE)
  }

  with_retry <- function(fetch_fn, label) {
    result <- tryCatch(attempt(fetch_fn), error = function(e) e)
    if (inherits(result, "vb_error_integrity")) {
      if (verbose) {
        cli::cli_warn("{label} download of {file_name} failed integrity check ({conditionMessage(result)}); retrying once")
      }
      result <- tryCatch(attempt(fetch_fn), error = function(e) e)
    }
    result
  }

  # Try piggyback first (preferred method)
  pb_result <- with_retry(function(tmpdir) {
    piggyback::pb_download(
      file = file_name, repo = repo, tag = release_tag,
      dest = tmpdir, overwrite = TRUE
    )
  }, "piggyback")

  if (!inherits(pb_result, "error")) {
    if (verbose) cli::cli_inform("Successfully downloaded {file_name}")
    return(invisible(TRUE))
  }
  if (verbose) cli::cli_warn("piggyback download failed: {conditionMessage(pb_result)}")

  # Fallback to direct URL download
  url_result <- with_retry(function(tmpdir) {
    url <- paste0(
      "https://github.com/", repo, "/releases/download/",
      release_tag, "/", file_name
    )
    if (verbose) cli::cli_inform("Trying direct download from {url}")
    # Qualified on purpose (not an unqualified `@importFrom` binding): a
    # bare `download.file()` resolves to the copy captured in this
    # package's namespace at load time, which testthat's
    # `local_mocked_bindings(.package = "utils")` cannot reach -- only a
    # live `utils::` lookup sees the mocked binding.
    utils::download.file(url, file.path(tmpdir, file_name), mode = "wb", quiet = !verbose)
  }, "direct URL")

  if (!inherits(url_result, "error")) {
    if (verbose) cli::cli_inform("Successfully downloaded {file_name}")
    return(invisible(TRUE))
  }

  # Both methods failed -- report both; type as vb_error_integrity if either
  # failure was a corruption signal (never silently downgrade that to a
  # generic network error class).
  details <- paste0(
    "piggyback: ", conditionMessage(pb_result), "; ",
    "direct URL: ", conditionMessage(url_result)
  )
  is_integrity <- inherits(pb_result, "vb_error_integrity") || inherits(url_result, "vb_error_integrity")
  cli::cli_abort(
    "Failed to download {file_name} from release {release_tag}. {details}",
    class = if (is_integrity) c("vb_error_integrity", "vb_error") else "vb_error"
  )
}
