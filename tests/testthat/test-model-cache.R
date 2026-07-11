# Regression tests for the manifest-verified model cache (ECOSYSTEM-FIX-PLAN.md
# M3: cache-hit freshness vs models_manifest.json; M4: verified/atomic
# downloads). All piggyback/utils network calls are mocked -- no network.

make_fake_models_manifest <- function(artifacts = list()) {
  list(schema_version = 1L, updated_at = "2026-07-11T00:00:00Z", artifacts = artifacts)
}

reset_manifest_state <- function() {
  rm(list = ls(envir = torpmodels:::.tm_manifest_state, all.names = TRUE),
     envir = torpmodels:::.tm_manifest_state)
}

# Reset both before AND after each test (via withr::defer) -- this state is
# a package-level env shared across every test file in the session, so a
# leftover cached/legacy-warned entry from one test must never leak forward
# into test-model_meta.R or a later test-model-cache.R case.
local_reset_manifest_state <- function(env = parent.frame()) {
  reset_manifest_state()
  withr::defer(reset_manifest_state(), envir = env)
}

local_model_cache_dir <- function(env = parent.frame()) {
  dir <- withr::local_tempdir(.local_envir = env)
  withr::local_options(list(torpmodels.cache_dir = dir), .local_envir = env)
  dir
}

test_that("a stale cache (sha256 sidecar mismatch vs models_manifest.json) triggers re-download", {
  local_reset_manifest_state()
  cache_dir <- local_model_cache_dir()
  core_dir <- file.path(cache_dir, "core")
  dir.create(core_dir, recursive = TRUE)
  local_path <- file.path(core_dir, "wp_model.rds")
  saveRDS(list(stale = TRUE), local_path)
  writeLines(strrep("0", 64), paste0(local_path, ".sha256"))  # deliberately wrong sha

  src_dir <- withr::local_tempdir()
  fresh_obj <- list(fresh = TRUE)
  fresh_path <- file.path(src_dir, "wp_model.rds")
  saveRDS(fresh_obj, fresh_path)
  fresh_sha <- vb_sha256(fresh_path)

  manifest <- make_fake_models_manifest(artifacts = list(
    wp_model.rds = list(sha256 = fresh_sha, size = file.size(fresh_path),
                        uploaded_at = "2026-07-11T00:00:00Z")
  ))

  manifest_calls <- 0L
  download_calls <- 0L
  testthat::local_mocked_bindings(
    pb_download = function(file, repo, tag, dest, ...) {
      if (identical(file, "models_manifest.json")) {
        manifest_calls <<- manifest_calls + 1L
        jsonlite::write_json(manifest, file.path(dest, file), auto_unbox = TRUE, null = "null")
      } else {
        download_calls <<- download_calls + 1L
        file.copy(fresh_path, file.path(dest, file))
      }
      invisible(NULL)
    },
    .package = "piggyback"
  )

  result <- load_torp_model("wp", verbose = FALSE)
  expect_identical(result, fresh_obj)
  expect_identical(download_calls, 1L)
  expect_identical(manifest_calls, 1L)
  expect_identical(readLines(paste0(local_path, ".sha256")), fresh_sha)

  # A second load within the 15-minute TTL window must not re-fetch the
  # manifest (session-cached), and the now-fresh cache must not re-download.
  result2 <- load_torp_model("wp", verbose = FALSE)
  expect_identical(result2, fresh_obj)
  expect_identical(manifest_calls, 1L)
  expect_identical(download_calls, 1L)
})

test_that("a corrupt download that would pass the old size-only heuristic raises vb_error_integrity, is deleted, and is retried once", {
  local_reset_manifest_state()
  cache_dir <- local_model_cache_dir()
  local_path <- file.path(cache_dir, "core", "wp_model.rds")

  manifest <- make_fake_models_manifest(artifacts = list(
    wp_model.rds = list(sha256 = strrep("a", 64), size = 2000, uploaded_at = "2026-07-11T00:00:00Z")
  ))

  pb_calls <- 0L
  testthat::local_mocked_bindings(
    pb_download = function(file, repo, tag, dest, ...) {
      if (identical(file, "models_manifest.json")) {
        jsonlite::write_json(manifest, file.path(dest, file), auto_unbox = TRUE, null = "null")
      } else {
        pb_calls <<- pb_calls + 1L
        # 2000 bytes comfortably clears the OLD `file.size > 1000` heuristic --
        # only the new sha256-vs-manifest check catches this as corrupt.
        writeLines(strrep("x", 2000), file.path(dest, file))
      }
      invisible(NULL)
    },
    .package = "piggyback"
  )
  url_calls <- 0L
  testthat::local_mocked_bindings(
    download.file = function(url, destfile, ...) {
      url_calls <<- url_calls + 1L
      writeLines(strrep("y", 2000), destfile)
      0L
    },
    .package = "utils"
  )

  expect_error(
    load_torp_model("wp", verbose = FALSE),
    class = "vb_error_integrity"
  )
  expect_false(file.exists(local_path))
  expect_false(file.exists(paste0(local_path, ".sha256")))
  # retry invoked: the original attempt plus one retry, per download method
  expect_identical(pb_calls, 2L)
  expect_identical(url_calls, 2L)
})

test_that("a mid-download failure never leaves a partial file at the cache path", {
  local_reset_manifest_state()
  cache_dir <- local_model_cache_dir()
  local_path <- file.path(cache_dir, "core", "wp_model.rds")

  testthat::local_mocked_bindings(
    pb_download = function(file, repo, tag, dest, ...) {
      if (identical(file, "models_manifest.json")) {
        stop("404 Not Found")  # legacy mode: nothing to verify sha256 against
      }
      # Write a partial file, then blow up mid-transfer -- what a dropped
      # connection looks like from the caller's side.
      writeLines("PARTIAL-GARBAGE", file.path(dest, file))
      stop("connection reset by peer")
    },
    .package = "piggyback"
  )
  testthat::local_mocked_bindings(
    download.file = function(url, destfile, ...) {
      writeLines("PARTIAL-GARBAGE-2", destfile)
      stop("transfer closed with outstanding read data remaining")
    },
    .package = "utils"
  )

  expect_error(load_torp_model("wp", verbose = FALSE))
  expect_false(file.exists(local_path))
  expect_false(file.exists(paste0(local_path, ".sha256")))
  # the per-attempt tempdirs (created beside the destination) must not leak
  core_dir <- dirname(local_path)
  leftovers <- if (dir.exists(core_dir)) list.files(core_dir, pattern = "^\\.tm_dl_", all.files = TRUE) else character(0)
  expect_length(leftovers, 0)
})

test_that("legacy mode (tag has no models_manifest.json) loads from cache with exactly one session-wide warning", {
  local_reset_manifest_state()
  cache_dir <- local_model_cache_dir()
  core_dir <- file.path(cache_dir, "core")
  dir.create(core_dir, recursive = TRUE)
  cached_obj <- stamp_model_meta(list(legacy = TRUE), list(model = "wp"))
  local_path <- file.path(core_dir, "wp_model.rds")
  saveRDS(cached_obj, local_path)
  # no .sha256 sidecar -- nothing to compare against in legacy mode anyway

  pb_calls <- 0L
  testthat::local_mocked_bindings(
    pb_download = function(file, repo, tag, dest, ...) {
      pb_calls <<- pb_calls + 1L
      stop("404 Not Found")  # stat-models/core-models today: no manifest yet
    },
    .package = "piggyback"
  )

  warnings_seen <- testthat::capture_warnings({
    result1 <- load_torp_model("wp", verbose = TRUE)
    result2 <- load_torp_model("wp", verbose = TRUE)
  })

  expect_identical(unclass(result1), unclass(cached_obj))
  expect_identical(unclass(result2), unclass(cached_obj))
  expect_length(warnings_seen, 1)
  expect_true(any(grepl("models_manifest.json", warnings_seen, fixed = TRUE)))
  # one manifest-fetch attempt (404, cached for the rest of the session);
  # never a model-file re-download -- both calls were legacy-mode cache hits
  expect_identical(pb_calls, 1L)
})

test_that("load_stat_model() also validates the cache against models_manifest.json", {
  local_reset_manifest_state()
  cache_dir <- local_model_cache_dir()
  stat_dir <- file.path(cache_dir, "stat-models")
  dir.create(stat_dir, recursive = TRUE)
  local_path <- file.path(stat_dir, "goals.rds")
  saveRDS(list(stale = TRUE), local_path)
  writeLines(strrep("0", 64), paste0(local_path, ".sha256"))

  src_dir <- withr::local_tempdir()
  fresh_obj <- list(fresh = TRUE)
  fresh_path <- file.path(src_dir, "goals.rds")
  saveRDS(fresh_obj, fresh_path)
  fresh_sha <- vb_sha256(fresh_path)

  manifest <- make_fake_models_manifest(artifacts = list(
    goals.rds = list(sha256 = fresh_sha, size = file.size(fresh_path),
                     uploaded_at = "2026-07-11T00:00:00Z")
  ))

  testthat::local_mocked_bindings(
    pb_download = function(file, repo, tag, dest, ...) {
      if (identical(file, "models_manifest.json")) {
        jsonlite::write_json(manifest, file.path(dest, file), auto_unbox = TRUE, null = "null")
      } else {
        file.copy(fresh_path, file.path(dest, file))
      }
      invisible(NULL)
    },
    .package = "piggyback"
  )

  result <- load_stat_model("goals", verbose = FALSE)
  expect_identical(result, fresh_obj)
  expect_identical(readLines(paste0(local_path, ".sha256")), fresh_sha)
})

test_that("safe_read_rds() deletes the .sha256 sidecar alongside a corrupted cache file", {
  withr::with_tempdir({
    path <- file.path(getwd(), "corrupt.rds")
    writeLines("not a valid rds file", path)
    writeLines(strrep("a", 64), paste0(path, ".sha256"))
    expect_error(torpmodels:::safe_read_rds(path, "broken_model"), "corrupted")
    expect_false(file.exists(path))
    expect_false(file.exists(paste0(path, ".sha256")))
  })
})
