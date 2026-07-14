# Model Provenance Metadata
# =========================
# Every trained artifact carries a `torp_meta` attribute (script, git SHAs,
# window, params, CV metric, timestamp) so a silent overwrite (like the
# March->June wp_model.rds swap) is detectable by inspection, not forensics.

#' @noRd
`%||%` <- function(x, y) if (is.null(x)) y else x

#' Locate a sibling repo root by matching its DESCRIPTION Package field
#' @keywords internal
.locate_repo_root <- function(pkg_name) {
  candidates <- c(".", pkg_name, file.path("..", pkg_name), file.path("../..", pkg_name))
  for (p in candidates) {
    desc_path <- file.path(p, "DESCRIPTION")
    if (file.exists(desc_path)) {
      pkg <- tryCatch(read.dcf(desc_path, fields = "Package")[1], error = function(e) NA_character_)
      if (!is.na(pkg) && identical(pkg, pkg_name)) return(p)
    }
  }
  NA_character_
}

#' Short git SHA for a repo root, NA if unavailable
#' @keywords internal
.git_sha_at <- function(path) {
  if (is.na(path) || !dir.exists(path)) return(NA_character_)
  out <- tryCatch(
    system2("git", c("-C", path, "rev-parse", "--short", "HEAD"),
            stdout = TRUE, stderr = FALSE),
    error = function(e) NA_character_,
    warning = function(w) NA_character_
  )
  if (length(out) != 1 || !is.character(out) || !nzchar(out)) return(NA_character_)
  out
}

#' Build model provenance metadata
#'
#' Returns a list of provenance fields to be stamped onto a trained model
#' object via [stamp_model_meta()]. This is the single source of truth for
#' what "we trained this model, here's how" means across every trainer.
#'
#' @param model_name Character. Short model id, e.g. "wp", "ep", "shot".
#' @param seasons Integer vector of training seasons, or a pre-formatted
#'   "YYYY-YYYY" character range.
#' @param params List of hyperparameters used to train the model (includes
#'   the monotone constraints string for WP).
#' @param feature_names Character vector of feature names in model order.
#' @param cv_metric Numeric. Cross-validated metric (logloss/mlogloss).
#' @param n_rows Integer. Number of training rows.
#' @param n_matches Integer. Number of training matches.
#' @param extra Named list merged into the returned meta. Supports a
#'   `script` element naming the training script (recorded as the top-level
#'   `script` field); any other elements are merged in as-is, e.g.
#'   `ep_source`, `ep_artifact_sha256`.
#'
#' @return A named list with the required provenance fields.
#' @keywords internal
build_model_meta <- function(model_name, seasons, params, feature_names,
                             cv_metric = NA_real_, n_rows = NA_integer_,
                             n_matches = NA_integer_, extra = list()) {
  if (missing(model_name) || is.null(model_name) || length(model_name) != 1 || !nzchar(model_name)) {
    cli::cli_abort("build_model_meta: {.arg model_name} is required")
  }
  if (missing(seasons) || is.null(seasons)) {
    cli::cli_abort("build_model_meta: {.arg seasons} is required")
  }
  if (missing(params)) {
    cli::cli_abort("build_model_meta: {.arg params} is required")
  }
  if (missing(feature_names)) {
    cli::cli_abort("build_model_meta: {.arg feature_names} is required")
  }

  seasons_range <- if (is.character(seasons) && length(seasons) == 1) {
    seasons
  } else if (length(seasons) > 0 && !all(is.na(seasons))) {
    paste(range(seasons, na.rm = TRUE), collapse = "-")
  } else {
    NA_character_
  }

  script <- extra[["script"]] %||% NA_character_
  extra[["script"]] <- NULL

  torp_root <- .locate_repo_root("torp")
  torpmodels_root <- .locate_repo_root("torpmodels")

  meta <- list(
    model = model_name,
    schema_version = 1L,
    trained_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    script = script,
    seasons = seasons_range,
    n_rows = n_rows,
    n_matches = n_matches,
    params = params,
    feature_names = feature_names,
    cv_metric = cv_metric,
    torp_sha = .git_sha_at(torp_root),
    torpmodels_sha = .git_sha_at(torpmodels_root),
    torp_version = as.character(tryCatch(utils::packageVersion("torp"), error = function(e) NA_character_)),
    r_version = as.character(getRversion()),
    xgboost_version = as.character(tryCatch(utils::packageVersion("xgboost"), error = function(e) NA_character_))
  )

  c(meta, extra)
}

#' Stamp provenance metadata onto a model object
#'
#' @param object Any R object (xgb.Booster, bam, tibble, ...). RDS
#'   round-trips arbitrary attributes, so this survives saveRDS/readRDS.
#' @param meta A metadata list, typically from [build_model_meta()].
#' @return `object` with a `torp_meta` attribute set.
#' @keywords internal
stamp_model_meta <- function(object, meta) {
  attr(object, "torp_meta") <- meta
  object
}

#' Retrieve provenance metadata from a model object
#'
#' @param object Any R object previously stamped via [stamp_model_meta()].
#' @return The `torp_meta` list, or `NULL` if absent.
#' @export
model_meta <- function(object) {
  attr(object, "torp_meta", exact = TRUE)
}

#' One-line human-readable summary of a model's provenance
#'
#' @param meta A metadata list, typically from [model_meta()].
#' @return `meta`, invisibly.
#' @keywords internal
describe_model_meta <- function(meta) {
  if (is.null(meta)) {
    cli::cli_inform("no torp_meta available")
    return(invisible(meta))
  }
  cli::cli_inform(paste0(
    "{meta$model %||% '?'}: trained {meta$trained_at %||% '?'} via ",
    "{meta$script %||% 'unknown script'} | seasons {meta$seasons %||% '?'} | ",
    "cv_metric {format(meta$cv_metric %||% NA)} | torp_sha {meta$torp_sha %||% NA}"
  ))
  invisible(meta)
}
