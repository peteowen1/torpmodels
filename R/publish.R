# Atomic Model Publish + Manifest
# ================================
# Uploads are atomic groups: an artifact and its sidecar(s) either both
# publish or the publish aborts (the F3 shot-model / shot_player_df gap).
# Every publish also appends to a release-level models_manifest.json so a
# silent overwrite (like June's wp_model.rds) becomes detectable by
# inspection via check_manifest_sync().

#' @noRd
.MODEL_GROUPS <- list(
  ep    = "ep_model.rds",
  wp    = c("wp_model.rds", "wp_calibration.rds"),
  shot  = c("shot_ocat_mdl.rds", "shot_player_df.rds"),
  match = c("match_gams.rds", "match_xgb_pipeline.rds", "match_margin_calibration.rds")
)

#' Publish a model group to a GitHub release as an atomic unit
#'
#' Uploads every file in the named group, or none: if any file is missing
#' from `dir`, aborts before attempting any upload. If an upload fails
#' partway through, aborts loudly listing what did/didn't upload (piggyback
#' re-upload is idempotent per file, so a re-run recovers).
#'
#' @param group Character. One of `names(.MODEL_GROUPS)`.
#' @param dir Character. Directory containing the model files.
#' @param repo Character. `"owner/repo"`. Defaults to [get_torpmodels_repo()].
#' @param tag Character. Release tag.
#' @param update_manifest Logical. If `TRUE` (default), call
#'   [update_models_manifest()] after a successful upload.
#'
#' @return Character vector of uploaded file names, invisibly.
#' @export
publish_model_group <- function(group, dir, repo = get_torpmodels_repo(),
                                tag = "core-models", update_manifest = TRUE) {
  files <- .MODEL_GROUPS[[group]]
  if (is.null(files)) {
    cli::cli_abort("Unknown model group: {group}. Known groups: {paste(names(.MODEL_GROUPS), collapse = ', ')}")
  }

  present <- file.exists(file.path(dir, files))
  if (!all(present)) {
    cli::cli_abort(c(
      "publish_model_group({.val {group}}) aborted: not all group members present in {dir}",
      "x" = "Missing: {paste(files[!present], collapse = ', ')}",
      "i" = "A partial group is never uploaded (this is the F3 guard)."
    ))
  }

  Sys.setenv(piggyback_cache_duration = 1)

  uploaded <- character(0)
  upload_error <- NULL

  for (f in files) {
    path <- file.path(dir, f)
    ok <- tryCatch({
      piggyback::pb_upload(path, repo = repo, tag = tag)
      TRUE
    }, error = function(e) {
      upload_error <<- conditionMessage(e)
      FALSE
    })
    if (isTRUE(ok)) {
      uploaded <- c(uploaded, f)
    } else {
      break
    }
  }

  if (!is.null(upload_error)) {
    remaining <- setdiff(files, uploaded)
    cli::cli_abort(c(
      "publish_model_group({.val {group}}) failed uploading {remaining[1]}: {upload_error}",
      "i" = "Uploaded before failure: {if (length(uploaded)) paste(uploaded, collapse = ', ') else 'none'}",
      "i" = "Remaining: {paste(remaining, collapse = ', ')}",
      "i" = "Re-run publish_model_group() to retry -- piggyback uploads are idempotent per file."
    ))
  }

  if (update_manifest) {
    manifest_result <- tryCatch(
      update_models_manifest(files, dir, repo, tag),
      error = function(e) e
    )
    if (inherits(manifest_result, "error")) {
      # Retry once (mirrors versebus vb_publish's own manifest-last retry),
      # then degrade to a loud warning rather than aborting: the group's
      # assets are ALREADY live on the release at this point, so treating
      # this as a publish failure would be misleading -- the caller would
      # re-run publish_model_group() and re-upload assets that don't need
      # re-uploading. What actually needs a retry is update_models_manifest()
      # alone.
      Sys.sleep(5)
      manifest_result <- tryCatch(
        update_models_manifest(files, dir, repo, tag),
        error = function(e) e
      )
      if (inherits(manifest_result, "error")) {
        cli::cli_warn(c(
          "publish_model_group({.val {group}}): assets uploaded successfully, but models_manifest.json update failed twice: {conditionMessage(manifest_result)}",
          "i" = "{paste(uploaded, collapse = ', ')} are live but untracked -- re-run {.fn update_models_manifest} for this group once the transient issue clears."
        ))
      }
    }
  }

  invisible(uploaded)
}

#' Publish stat models to a GitHub release
#'
#' Unlike [publish_model_group()]'s fixed-group atomicity (built for
#' sidecar-pair artifacts like wp_model+wp_calibration, where a partial
#' upload is worse than none), the 58 per-stat GAMs are independent files --
#' one bad upload shouldn't block the other ~57. Continues past individual
#' upload failures, collects them into a single warning, and records every
#' successfully uploaded file in `models_manifest.json` via
#' [update_models_manifest()].
#'
#' Quality-gated by default: each file's freshly-stamped `cv_metric` (a
#' temporal-holdout RMSE -- lower is better) is compared against the
#' PREVIOUSLY published version's `cv_metric` for that same file, read from
#' the existing manifest. Unlike the WP/match-margin gates (an absolute
#' threshold against a fixed calibration target of 1.0), RMSE has no
#' meaningful absolute target -- it's scale-dependent per stat (a goals
#' model and a disposals model have unrelated "good" RMSE values) -- so this
#' gates on RELATIVE regression vs each stat's own history instead. A file
#' with no prior manifest entry (first-ever publish) always passes -- there
#' is nothing to regress against. A gated file is skipped (old version stays
#' live) with a loud warning, same "continue past one bad file" philosophy
#' as upload failures.
#'
#' @param files Character vector of file names (relative to `dir`) to publish.
#' @param dir Character. Directory containing the model files.
#' @param repo Character. `"owner/repo"`. Defaults to [get_torpmodels_repo()].
#' @param tag Character. Release tag, defaults to `"stat-models"`.
#' @param update_manifest Logical. If `TRUE` (default), call
#'   [update_models_manifest()] after upload.
#' @param gate Logical. If `TRUE` (default), skip publishing any file whose
#'   `cv_metric` regressed by more than `gate_max_regression` vs the
#'   previously published version. Set `FALSE` to disable (e.g. a
#'   deliberate re-baseline after a feature/formula change).
#' @param gate_max_regression Numeric, default `0.20`. Maximum tolerated
#'   relative RMSE increase (new/old - 1) before a file is gated.
#'
#' @return Character vector of successfully uploaded file names, invisibly.
#' @export
publish_stat_models <- function(files, dir, repo = get_torpmodels_repo(),
                                tag = "stat-models", update_manifest = TRUE,
                                gate = TRUE, gate_max_regression = 0.20) {
  present <- file.exists(file.path(dir, files))
  if (!any(present)) {
    cli::cli_abort("publish_stat_models: none of the {length(files)} requested files exist in {dir}")
  }
  if (!all(present)) {
    cli::cli_warn("publish_stat_models: {sum(!present)} file(s) missing from {dir}, skipping: {paste(files[!present], collapse = ', ')}")
  }
  files <- files[present]

  # Tracked outside the `if (isTRUE(gate))` block (not just declared inside
  # it) so it's always defined for the attr() on the final return, even when
  # gate = FALSE.
  gated <- character(0)
  if (isTRUE(gate)) {
    prev_manifest <- tryCatch(.fetch_manifest(repo, tag), error = function(e) NULL)
    for (f in files) {
      prev_metric <- prev_manifest$artifacts[[f]]$cv_metric
      if (is.null(prev_metric) || is.na(prev_metric)) next  # nothing to regress against
      new_meta <- tryCatch(model_meta(readRDS(file.path(dir, f))), error = function(e) NULL)
      new_metric <- new_meta$cv_metric
      if (is.null(new_metric) || is.na(new_metric)) next  # can't gate without a new metric either
      if (new_metric > prev_metric * (1 + gate_max_regression)) {
        cli::cli_warn(c(
          "publish_stat_models: gating {.val {f}} -- cv_metric regressed {round((new_metric / prev_metric - 1) * 100, 1)}% ({round(prev_metric, 4)} -> {round(new_metric, 4)}), exceeds {gate_max_regression * 100}% tolerance",
          "i" = "Previously published version stays live. Re-run with gate = FALSE to force-publish anyway."
        ))
        gated <- c(gated, f)
      }
    }
    files <- setdiff(files, gated)
    if (length(files) == 0) {
      cli::cli_warn("publish_stat_models: all requested files were gated -- nothing to publish")
      empty <- character(0)
      attr(empty, "gated") <- gated
      return(invisible(empty))
    }
  }

  Sys.setenv(piggyback_cache_duration = 1)

  uploaded <- character(0)
  failed <- character(0)

  for (f in files) {
    path <- file.path(dir, f)
    ok <- tryCatch({
      piggyback::pb_upload(path, repo = repo, tag = tag)
      TRUE
    }, error = function(e) {
      cli::cli_warn("publish_stat_models: upload failed for {.val {f}}: {conditionMessage(e)}")
      FALSE
    })
    if (isTRUE(ok)) uploaded <- c(uploaded, f) else failed <- c(failed, f)
  }

  if (length(failed) > 0) {
    cli::cli_warn(c(
      "publish_stat_models: {length(failed)}/{length(files)} uploads failed: {paste(failed, collapse = ', ')}",
      "i" = "Uploaded {length(uploaded)} successfully -- re-run publish_stat_models() with just the failed file names to retry (piggyback uploads are idempotent per file)."
    ))
  }

  if (update_manifest && length(uploaded) > 0) {
    manifest_result <- tryCatch(update_models_manifest(uploaded, dir, repo, tag), error = function(e) e)
    if (inherits(manifest_result, "error")) {
      Sys.sleep(5)
      manifest_result <- tryCatch(update_models_manifest(uploaded, dir, repo, tag), error = function(e) e)
      if (inherits(manifest_result, "error")) {
        cli::cli_warn(c(
          "publish_stat_models: assets uploaded successfully, but models_manifest.json update failed twice: {conditionMessage(manifest_result)}",
          "i" = "{paste(uploaded, collapse = ', ')} are live but untracked -- re-run {.fn update_models_manifest} for these once the transient issue clears."
        ))
      }
    }
  }

  # Gated files are attached as an attribute (not folded into `uploaded`)
  # so the return value's plain length stays "how many were actually
  # published" -- a caller comparing that against its own trained-file
  # count needs to also subtract attr(result, "gated") to avoid treating a
  # deliberate quality-gate skip as an upload failure.
  attr(uploaded, "gated") <- gated
  invisible(uploaded)
}

#' @noRd
.manifest_asset_name <- "models_manifest.json"

#' Fetch the existing manifest, or NULL for a fresh one
#' @keywords internal
.fetch_manifest <- function(repo, tag, manifest_name = .manifest_asset_name) {
  tryCatch({
    tmp_dir <- tempfile("torpmodels_manifest_")
    dir.create(tmp_dir)
    on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)
    piggyback::pb_download(file = manifest_name, repo = repo, tag = tag, dest = tmp_dir)
    path <- file.path(tmp_dir, manifest_name)
    if (!file.exists(path)) return(NULL)
    jsonlite::fromJSON(path, simplifyVector = FALSE)
  }, error = function(e) {
    msg <- conditionMessage(e)
    if (grepl("404|not found", msg, ignore.case = TRUE)) {
      return(NULL)
    }
    cli::cli_abort("Could not fetch existing {.val {manifest_name}} ({msg}); refusing to overwrite the ledger on a transient error.")
  })
}

#' Read-modify-write the models_manifest.json ledger
#'
#' Downloads the current manifest (a 404 means "start fresh"; any other
#' download error aborts rather than risking a clobber), records a new
#' entry per file (sha256, size, timestamp, and a meta subset when the file
#' carries a `torp_meta` stamp), moves the previous entry for that artifact
#' onto its `history` (capped at 20), and re-uploads the manifest last so
#' artifacts always land before the ledger claims them.
#'
#' @param files Character vector of file names (relative to `dir`).
#' @param dir Character. Directory containing the files.
#' @param repo Character. `"owner/repo"`.
#' @param tag Character. Release tag.
#'
#' @return The updated manifest list, invisibly.
#' @export
update_models_manifest <- function(files, dir, repo, tag) {
  manifest <- .fetch_manifest(repo, tag)
  if (is.null(manifest)) {
    manifest <- list(schema_version = 1L, updated_at = NA_character_, artifacts = list())
  }
  if (is.null(manifest$artifacts)) manifest$artifacts <- list()

  now <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")

  for (f in files) {
    path <- file.path(dir, f)
    if (!file.exists(path)) next

    obj_meta <- tryCatch(model_meta(readRDS(path)), error = function(e) NULL)

    entry <- list(
      sha256 = digest::digest(path, algo = "sha256", file = TRUE),
      size = file.info(path)$size,
      uploaded_at = now
    )
    if (!is.null(obj_meta)) {
      entry$model <- obj_meta$model
      entry$script <- obj_meta$script
      entry$seasons <- obj_meta$seasons
      entry$torp_sha <- obj_meta$torp_sha
      entry$torpmodels_sha <- obj_meta$torpmodels_sha
      entry$params_hash <- digest::digest(obj_meta$params)
      entry$cv_metric <- obj_meta$cv_metric
    }

    prev <- manifest$artifacts[[f]]
    if (!is.null(prev)) {
      prev_history <- prev$history
      prev$history <- NULL
      history <- c(list(prev), prev_history)
      if (length(history) > 20) history <- history[seq_len(20)]
      entry$history <- history
    }

    manifest$artifacts[[f]] <- entry
  }

  manifest$updated_at <- now

  tmp <- tempfile(fileext = ".json")
  jsonlite::write_json(manifest, tmp, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null")
  piggyback::pb_upload(tmp, repo = repo, tag = tag, name = .manifest_asset_name)

  invisible(manifest)
}

#' Detect artifacts uploaded outside the canonical publish path
#'
#' Compares each release asset's `updated_at`/`size` against its
#' `models_manifest.json` record. This is the check that would have caught
#' June's silent wp_model.rds overwrite the day it happened.
#'
#' @param repo Character. `"owner/repo"`. Defaults to [get_torpmodels_repo()].
#' @param tag Character. Release tag.
#'
#' @return Invisibly, a list with `stale_or_outside_path` and
#'   `manifest_without_asset` character vectors (empty when clean).
#' @export
check_manifest_sync <- function(repo = get_torpmodels_repo(), tag = "core-models") {
  manifest <- .fetch_manifest(repo, tag)
  if (is.null(manifest)) {
    # (.manifest_asset_name) parens: cli's glue interpolation reserves a
    # bare {.foo...} for its own inline-markup style classes, so a
    # dot-prefixed variable name needs parens to be evaluated as an
    # expression rather than misparsed as an unknown style and hard-error
    # ("Invalid cli literal") -- discovered 2026-07-26 via the identical bug
    # in wt_av_modelling.R, which crashed a real training run.
    cli::cli_abort("No {.val {(.manifest_asset_name)}} found for {repo}@{tag} -- nothing to check.")
  }

  # stdout goes to a file: system2(stdout = TRUE) splits very long lines on
  # Windows, and re-joining with "\n" injects newlines inside JSON strings.
  gh_json <- tempfile(fileext = ".json")
  gh_err <- tempfile(fileext = ".txt")
  on.exit(unlink(c(gh_json, gh_err)), add = TRUE)
  status <- tryCatch(
    system2("gh", c("api", sprintf("repos/%s/releases/tags/%s", repo, tag)),
            stdout = gh_json, stderr = gh_err),
    error = function(e) -1L
  )
  if (!identical(status, 0L) || !file.exists(gh_json) || file.size(gh_json) == 0) {
    err_txt <- if (file.exists(gh_err)) paste(readLines(gh_err, warn = FALSE), collapse = " ") else ""
    cli::cli_abort("check_manifest_sync: `gh api` failed to list release assets for {repo}@{tag}. {err_txt}")
  }
  release <- jsonlite::fromJSON(gh_json, simplifyVector = FALSE)
  assets <- release$assets %||% list()
  asset_names <- vapply(assets, function(a) a$name, character(1))
  names(assets) <- asset_names

  stale_or_outside_path <- character(0)
  manifest_without_asset <- character(0)

  for (nm in names(manifest$artifacts)) {
    entry <- manifest$artifacts[[nm]]
    asset <- assets[[nm]]
    if (is.null(asset)) {
      manifest_without_asset <- c(manifest_without_asset, nm)
      next
    }
    asset_updated <- tryCatch(as.POSIXct(asset$updated_at, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                              error = function(e) NA)
    manifest_updated <- tryCatch(as.POSIXct(entry$uploaded_at, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                                 error = function(e) NA)
    size_mismatch <- !is.null(asset$size) && !is.null(entry$size) && asset$size != entry$size
    time_mismatch <- !is.na(asset_updated) && !is.na(manifest_updated) &&
      as.numeric(asset_updated - manifest_updated, units = "secs") > 60
    if (isTRUE(time_mismatch) || isTRUE(size_mismatch)) {
      stale_or_outside_path <- c(stale_or_outside_path, nm)
    }
  }

  # Assets on the release with ZERO manifest entry (e.g. published via a
  # bare piggyback::pb_upload() that bypassed update_models_manifest()
  # entirely, such as match_margin_calibration.rds's current upload path in
  # torp) are a distinct, more severe gap than a stale/drifted entry -- the
  # loop above can only ever find drift on files the manifest already
  # tracks. Surface files with no entry at all, excluding the manifest
  # asset itself and any non-model housekeeping files.
  untracked_assets <- setdiff(asset_names, c(names(manifest$artifacts), .manifest_asset_name, "bus_manifest.json"))

  if (length(stale_or_outside_path) == 0 && length(manifest_without_asset) == 0 &&
      length(untracked_assets) == 0) {
    cli::cli_alert_success("check_manifest_sync: {repo}@{tag} is clean -- every asset matches its manifest record.")
  } else {
    if (length(stale_or_outside_path) > 0) {
      cli::cli_warn("Uploaded outside the canonical path (asset newer/different-size than manifest record): {paste(stale_or_outside_path, collapse = ', ')}")
    }
    if (length(manifest_without_asset) > 0) {
      cli::cli_warn("Manifest entries with no matching release asset: {paste(manifest_without_asset, collapse = ', ')}")
    }
    if (length(untracked_assets) > 0) {
      cli::cli_warn("Release assets with NO manifest entry at all -- zero provenance/history tracking: {paste(untracked_assets, collapse = ', ')}")
    }
  }

  invisible(list(stale_or_outside_path = stale_or_outside_path,
                 manifest_without_asset = manifest_without_asset,
                 untracked_assets = untracked_assets))
}
