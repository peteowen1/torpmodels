test_that("publish_model_group aborts when any group member is missing", {  # F3 regression
  dir <- withr::local_tempdir(); saveRDS(1, file.path(dir, "shot_ocat_mdl.rds"))

  upload_calls <- 0L
  testthat::local_mocked_bindings(
    pb_upload = function(...) { upload_calls <<- upload_calls + 1L },
    .package = "piggyback"
  )

  expect_error(publish_model_group("shot", dir), "shot_player_df")
  # no upload attempted
  expect_identical(upload_calls, 0L)
})

test_that("publish_model_group('wp') aborts without its wp_calibration.rds sidecar", {  # torpverse/docs/plans/FABLE-RECAL-PLAN.md D3
  dir <- withr::local_tempdir(); saveRDS(1, file.path(dir, "wp_model.rds"))

  upload_calls <- 0L
  testthat::local_mocked_bindings(
    pb_upload = function(...) { upload_calls <<- upload_calls + 1L },
    .package = "piggyback"
  )

  expect_error(publish_model_group("wp", dir), "wp_calibration")
  expect_identical(upload_calls, 0L)   # no partial upload -- neither file goes up

  # once both members are present, the group publishes atomically
  saveRDS(list(a = 0, b = 1.2), file.path(dir, "wp_calibration.rds"))
  uploaded <- publish_model_group("wp", dir, update_manifest = FALSE)
  expect_setequal(uploaded, c("wp_model.rds", "wp_calibration.rds"))
  expect_identical(upload_calls, 2L)
})

test_that("update_models_manifest treats 404 as fresh and aborts on transient errors", {
  dir <- withr::local_tempdir()
  saveRDS(1, file.path(dir, "wp_model.rds"))

  testthat::local_mocked_bindings(
    pb_download = function(...) stop("404 Not Found"),
    pb_upload = function(...) invisible(NULL),
    .package = "piggyback"
  )
  # (a) 404 -> fresh manifest is built and published without error
  expect_no_error(update_models_manifest("wp_model.rds", dir, "peteowen1/torpmodels", "core-models"))

  testthat::local_mocked_bindings(
    pb_download = function(...) stop("Timeout was reached"),
    .package = "piggyback"
  )
  # (b) non-404 -> abort, ledger never clobbered
  expect_error(
    update_models_manifest("wp_model.rds", dir, "peteowen1/torpmodels", "core-models"),
    "transient"
  )
})

test_that("publish_stat_models uploads all present files and updates the manifest", {
  dir <- withr::local_tempdir()
  saveRDS(1, file.path(dir, "goals.rds"))
  saveRDS(2, file.path(dir, "disposals.rds"))

  upload_calls <- character(0)
  testthat::local_mocked_bindings(
    pb_upload = function(file, repo, tag, name = basename(file), ...) {
      upload_calls <<- c(upload_calls, basename(file))
    },
    .package = "piggyback"
  )
  manifest_calls <- 0L
  testthat::local_mocked_bindings(
    update_models_manifest = function(files, dir, repo, tag) {
      manifest_calls <<- manifest_calls + 1L
      expect_setequal(files, c("goals.rds", "disposals.rds"))
      invisible(NULL)
    }
  )

  uploaded <- publish_stat_models(c("goals.rds", "disposals.rds"), dir, gate = FALSE)
  expect_setequal(uploaded, c("goals.rds", "disposals.rds"))
  expect_setequal(upload_calls, c("goals.rds", "disposals.rds"))
  expect_identical(manifest_calls, 1L)
})

test_that("publish_stat_models continues past an individual upload failure -- not all-or-nothing like publish_model_group", {
  dir <- withr::local_tempdir()
  saveRDS(1, file.path(dir, "goals.rds"))
  saveRDS(2, file.path(dir, "disposals.rds"))

  testthat::local_mocked_bindings(
    pb_upload = function(file, repo, tag, name = basename(file), ...) {
      if (basename(file) == "goals.rds") stop("simulated upload failure")
      invisible(NULL)
    },
    .package = "piggyback"
  )
  manifest_files <- NULL
  testthat::local_mocked_bindings(
    update_models_manifest = function(files, dir, repo, tag) {
      manifest_files <<- files
      invisible(NULL)
    }
  )

  expect_warning(
    uploaded <- publish_stat_models(c("goals.rds", "disposals.rds"), dir, gate = FALSE),
    "goals.rds"
  )
  expect_identical(as.character(uploaded), "disposals.rds")
  expect_identical(manifest_files, "disposals.rds")  # only the successful upload gets tracked
})

test_that("publish_stat_models warns and skips missing files rather than aborting the whole batch", {
  dir <- withr::local_tempdir()
  saveRDS(1, file.path(dir, "goals.rds"))

  testthat::local_mocked_bindings(
    pb_upload = function(...) invisible(NULL),
    .package = "piggyback"
  )
  testthat::local_mocked_bindings(
    update_models_manifest = function(files, dir, repo, tag) invisible(NULL)
  )

  expect_warning(
    uploaded <- publish_stat_models(c("goals.rds", "missing_stat.rds"), dir, gate = FALSE),
    "missing_stat.rds"
  )
  expect_identical(as.character(uploaded), "goals.rds")
})

test_that("publish_stat_models aborts only when NONE of the requested files exist", {
  dir <- withr::local_tempdir()
  expect_error(publish_stat_models("nope.rds", dir, gate = FALSE), "none of")
})

test_that("publish_stat_models gates a file whose cv_metric regressed beyond the tolerance", {
  dir <- withr::local_tempdir()
  # Previous published RMSE was 1.0; new is 1.5 -- a 50% regression, over
  # the default 20% tolerance.
  saveRDS(stamp_model_meta(1, build_model_meta("goals", 2021:2025, list(), c("a"), cv_metric = 1.5)),
          file.path(dir, "goals.rds"))

  existing_manifest <- list(schema_version = 1L, updated_at = "2026-01-01T00:00:00Z",
                            artifacts = list(goals.rds = list(cv_metric = 1.0)))
  testthat::local_mocked_bindings(
    pb_download = function(file, repo, tag, dest, ...) {
      jsonlite::write_json(existing_manifest, file.path(dest, file), auto_unbox = TRUE, null = "null")
      invisible(NULL)
    },
    pb_upload = function(...) stop("should not upload a gated file"),
    .package = "piggyback"
  )

  expect_warning(
    uploaded <- publish_stat_models("goals.rds", dir),
    "regressed"
  )
  expect_length(uploaded, 0)
  expect_identical(attr(uploaded, "gated"), "goals.rds")
})

test_that("publish_stat_models publishes a file that improved or stayed within tolerance", {
  dir <- withr::local_tempdir()
  saveRDS(stamp_model_meta(1, build_model_meta("goals", 2021:2025, list(), c("a"), cv_metric = 1.05)),
          file.path(dir, "goals.rds"))

  existing_manifest <- list(schema_version = 1L, updated_at = "2026-01-01T00:00:00Z",
                            artifacts = list(goals.rds = list(cv_metric = 1.0)))
  upload_calls <- character(0)
  testthat::local_mocked_bindings(
    pb_download = function(file, repo, tag, dest, ...) {
      jsonlite::write_json(existing_manifest, file.path(dest, file), auto_unbox = TRUE, null = "null")
      invisible(NULL)
    },
    pb_upload = function(file, repo, tag, name = basename(file), ...) {
      upload_calls <<- c(upload_calls, basename(file))
    },
    .package = "piggyback"
  )
  testthat::local_mocked_bindings(update_models_manifest = function(...) invisible(NULL))

  uploaded <- publish_stat_models("goals.rds", dir)
  expect_identical(as.character(uploaded), "goals.rds")
  expect_length(attr(uploaded, "gated"), 0)
  expect_identical(upload_calls, "goals.rds")
})

test_that("publish_stat_models never gates a file with no prior manifest entry (first-ever publish)", {
  dir <- withr::local_tempdir()
  saveRDS(stamp_model_meta(1, build_model_meta("brand_new_stat", 2021:2025, list(), c("a"), cv_metric = 99)),
          file.path(dir, "brand_new_stat.rds"))

  testthat::local_mocked_bindings(
    pb_download = function(...) stop("404 Not Found"),  # no manifest yet
    pb_upload = function(file, repo, tag, name = basename(file), ...) invisible(NULL),
    .package = "piggyback"
  )
  testthat::local_mocked_bindings(update_models_manifest = function(...) invisible(NULL))

  uploaded <- publish_stat_models("brand_new_stat.rds", dir)
  expect_identical(as.character(uploaded), "brand_new_stat.rds")
})

test_that("publish_stat_models with gate = FALSE force-publishes a regressed file", {
  dir <- withr::local_tempdir()
  saveRDS(stamp_model_meta(1, build_model_meta("goals", 2021:2025, list(), c("a"), cv_metric = 1.5)),
          file.path(dir, "goals.rds"))

  upload_calls <- character(0)
  testthat::local_mocked_bindings(
    pb_upload = function(file, repo, tag, name = basename(file), ...) {
      upload_calls <<- c(upload_calls, basename(file))
    },
    .package = "piggyback"
  )
  testthat::local_mocked_bindings(update_models_manifest = function(...) invisible(NULL))

  # gate = FALSE means .fetch_manifest() (and its pb_download call) is never
  # reached, so no pb_download mock is needed here.
  uploaded <- publish_stat_models("goals.rds", dir, gate = FALSE)
  expect_identical(as.character(uploaded), "goals.rds")
  expect_identical(upload_calls, "goals.rds")
})

test_that("manifest write moves prior entry to history and records sha256", {
  dir <- withr::local_tempdir()
  saveRDS(
    stamp_model_meta(list(dummy = TRUE),
                     build_model_meta("wp", 2021:2025, list(eta = 0.025), c("a", "b"), cv_metric = 0.4)),
    file.path(dir, "wp_model.rds")
  )

  existing_manifest <- list(
    schema_version = 1L,
    updated_at = "2026-01-01T00:00:00Z",
    artifacts = list(
      wp_model.rds = list(sha256 = "deadbeef", size = 123, uploaded_at = "2026-01-01T00:00:00Z",
                          model = "wp", script = "old_script.R")
    )
  )

  uploaded_paths <- character(0)
  testthat::local_mocked_bindings(
    pb_download = function(file, repo, tag, dest, ...) {
      jsonlite::write_json(existing_manifest, file.path(dest, file), auto_unbox = TRUE, null = "null")
      invisible(NULL)
    },
    pb_upload = function(file, repo, tag, name = basename(file), ...) {
      uploaded_paths <<- c(uploaded_paths, file)
      invisible(NULL)
    },
    .package = "piggyback"
  )

  manifest <- update_models_manifest("wp_model.rds", dir, "peteowen1/torpmodels", "core-models")

  expect_length(uploaded_paths, 1)
  entry <- manifest$artifacts[["wp_model.rds"]]
  expect_true(nchar(entry$sha256) > 0)
  expect_identical(entry$model, "wp")
  expect_length(entry$history, 1)
  expect_identical(entry$history[[1]]$sha256, "deadbeef")
})
