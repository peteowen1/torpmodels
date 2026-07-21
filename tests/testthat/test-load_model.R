test_that("normalize_model_name() returns correct mappings", {
  ep <- torpmodels:::normalize_model_name("ep")
  expect_equal(ep$file, "ep_model.rds")
  expect_equal(ep$tag, "core-models")

  wp <- torpmodels:::normalize_model_name("wp_model")
  expect_equal(wp$file, "wp_model.rds")
  expect_equal(wp$tag, "core-models")

  shot <- torpmodels:::normalize_model_name("shot")
  expect_equal(shot$file, "shot_ocat_mdl.rds")

  xgb <- torpmodels:::normalize_model_name("xgb_win")
  expect_equal(xgb$file, "xgb_win_model.rds")

  gams <- torpmodels:::normalize_model_name("match_gams")
  expect_equal(gams$file, "match_gams.rds")
  expect_equal(gams$tag, "core-models")
})

test_that("normalize_model_name() returns NULL for unknown models", {
  expect_null(torpmodels:::normalize_model_name("nonexistent"))
  expect_null(torpmodels:::normalize_model_name(""))
  expect_null(torpmodels:::normalize_model_name("random_name"))
})

test_that("normalize_model_name() is case-insensitive", {
  expect_equal(torpmodels:::normalize_model_name("EP")$file, "ep_model.rds")
  expect_equal(torpmodels:::normalize_model_name("WP")$file, "wp_model.rds")
})

test_that("list_available_models() returns expected structure", {
  models <- list_available_models()

  expect_type(models, "list")
  expect_named(models, c("core_models", "stat_models"))

  expect_true("ep" %in% names(models$core_models))
  expect_true("wp" %in% names(models$core_models))
  expect_true("shot" %in% names(models$core_models))
  expect_true("xgb_win" %in% names(models$core_models))
  expect_true("match_gams" %in% names(models$core_models))

  expect_true("goals" %in% models$stat_models)
  expect_true("disposals" %in% models$stat_models)
  expect_length(models$stat_models, 58)
})

test_that("list_available_models() includes extended stats", {
  models <- list_available_models()
  expect_true("extended_stats_pressure_acts" %in% models$stat_models)
  expect_true("extended_stats_effective_disposals" %in% models$stat_models)
})

test_that("clear_model_cache() rejects invalid type via match.arg()", {
  expect_error(clear_model_cache(type = "invalid"), "should be one of")
})

test_that("clear_model_cache() deletes correct files by type", {
  withr::with_tempdir({
    cache_dir <- file.path(getwd(), "test_cache")
    withr::with_options(list(torpmodels.cache_dir = cache_dir), {
      core_dir <- file.path(cache_dir, "core")
      stat_dir <- file.path(cache_dir, "stat-models")
      dir.create(core_dir, recursive = TRUE)
      dir.create(stat_dir, recursive = TRUE)
      writeLines("fake", file.path(core_dir, "ep_model.rds"))
      writeLines("fake", file.path(stat_dir, "goals.rds"))

      clear_model_cache("core", verbose = FALSE)
      expect_equal(length(list.files(core_dir)), 0)
      expect_equal(length(list.files(stat_dir)), 1)

      clear_model_cache("stat", verbose = FALSE)
      expect_equal(length(list.files(stat_dir)), 0)
    })
  })
})

test_that("get_models_dir() creates directory and respects option override", {
  withr::with_options(list(torpmodels.cache_dir = NULL), {
    dir <- get_models_dir()
    expect_true(dir.exists(dir))
    expect_true(grepl("torpmodels", dir))
  })

  withr::with_tempdir({
    custom_dir <- file.path(getwd(), "custom_models")
    withr::with_options(list(torpmodels.cache_dir = custom_dir), {
      dir <- get_models_dir()
      expect_equal(dir, custom_dir)
      expect_true(dir.exists(dir))
    })
  })
})

test_that("load_torp_model() rejects unknown model names", {
  expect_error(load_torp_model("nonexistent"), "Unknown model")
})

test_that("normalize_model_name('wp_calibration') maps to the sidecar file", {  # torpverse/docs/plans/FABLE-RECAL-PLAN.md D3
  info <- normalize_model_name("wp_calibration")
  expect_equal(info$file, "wp_calibration.rds")
  expect_equal(info$tag, "core-models")
})

test_that("load_torp_model('wp_calibration') round-trips the object and prints meta", {
  cache_dir <- withr::local_tempdir()
  core_dir <- file.path(cache_dir, "core")
  dir.create(core_dir, recursive = TRUE)
  withr::local_options(torpmodels.cache_dir = cache_dir)

  # As in test-model_meta.R's equivalent test: these cache fixtures are
  # hand-built, not real published artifacts, so prime legacy (no-manifest)
  # mode up front rather than let the real network-hitting freshness check
  # run.
  reset_tm_manifest_state <- function() {
    rm(list = ls(envir = torpmodels:::.tm_manifest_state, all.names = TRUE),
       envir = torpmodels:::.tm_manifest_state)
  }
  reset_tm_manifest_state()
  withr::defer(reset_tm_manifest_state())
  testthat::local_mocked_bindings(
    pb_download = function(...) stop("404 Not Found"),
    .package = "piggyback"
  )
  suppressWarnings(torpmodels:::.get_models_manifest(get_torpmodels_repo(), "core-models"))

  calib <- list(a = 0.05, b = 1.22, formula = "plogis(a + b*qlogis(p))",
                fitted_on = "temporal-oos", gate_season = 2025L, n_fit = 50000L,
                slope_before = 1.14, slope_after = 1.01,
                slope_q4close_before = 1.26, slope_q4close_after = 0.95)
  meta <- build_model_meta("wp_calibration", 2021:2025, list(formula = "plogis(a + b*qlogis(p))"),
                           c("a", "b"), extra = list(gate_season = 2025L))
  saveRDS(stamp_model_meta(calib, meta), file.path(core_dir, "wp_calibration.rds"))

  expect_message(
    result <- load_torp_model("wp_calibration", verbose = TRUE),
    "trained"
  )
  expect_equal(result$a, 0.05)
  expect_equal(result$b, 1.22)
  expect_equal(model_meta(result)$model, "wp_calibration")
})

test_that("download_model_from_release() retries transient piggyback failures before giving up", {  # peteowen1/torpdata#66, #68
  withr::with_tempdir({
    local_path <- file.path(getwd(), "ep_model.rds")

    reset_tm_manifest_state <- function() {
      rm(list = ls(envir = torpmodels:::.tm_manifest_state, all.names = TRUE),
         envir = torpmodels:::.tm_manifest_state)
    }
    reset_tm_manifest_state()
    withr::defer(reset_tm_manifest_state())

    calls <- 0L
    testthat::local_mocked_bindings(
      pb_download = function(file, repo, tag, dest, ...) {
        if (identical(file, "models_manifest.json")) {
          stop("404 Not Found")  # legacy mode: no models_manifest.json for this tag
        }
        calls <<- calls + 1L
        if (calls < 3L) stop("500 Internal Server Error")
        saveRDS(list(ok = TRUE), file.path(dest, file))
        invisible(NULL)
      },
      .package = "piggyback"
    )
    testthat::local_mocked_bindings(
      Sys.sleep = function(...) invisible(NULL),
      .package = "base"
    )

    result <- suppressWarnings(torpmodels:::download_model_from_release(
      "ep_model.rds", "core-models", local_path, verbose = FALSE
    ))

    expect_true(isTRUE(result))
    expect_identical(calls, 3L)
    expect_true(file.exists(local_path))
    expect_true(readRDS(local_path)$ok)
  })
})

test_that("download_model_from_release() does not retry a confirmed-absent (404) error", {
  withr::with_tempdir({
    local_path <- file.path(getwd(), "ep_model.rds")

    reset_tm_manifest_state <- function() {
      rm(list = ls(envir = torpmodels:::.tm_manifest_state, all.names = TRUE),
         envir = torpmodels:::.tm_manifest_state)
    }
    reset_tm_manifest_state()
    withr::defer(reset_tm_manifest_state())

    # A structured http_error_404 condition is how vb_classify_error()
    # positively confirms "absent" (see R/versebus.R) -- should_retry must
    # exclude it and re-raise on the first attempt, unlike a plain transient
    # error string (which defaults to "transient" and IS retried).
    absent_404 <- function(...) {
      stop(structure(
        class = c("http_error_404", "error", "condition"),
        list(message = "404 Not Found", call = NULL)
      ))
    }

    pb_calls <- 0L
    testthat::local_mocked_bindings(
      pb_download = function(file, repo, tag, dest, ...) {
        if (identical(file, "models_manifest.json")) stop("404 Not Found")
        pb_calls <<- pb_calls + 1L
        absent_404()
      },
      .package = "piggyback"
    )
    dl_calls <- 0L
    testthat::local_mocked_bindings(
      download.file = function(...) {
        dl_calls <<- dl_calls + 1L
        absent_404()
      },
      .package = "utils"
    )

    expect_error(
      suppressWarnings(torpmodels:::download_model_from_release(
        "ep_model.rds", "core-models", local_path, verbose = FALSE
      )),
      "Failed to download"
    )
    # No retry on a confirmed-absent (404) error at either layer.
    expect_identical(pb_calls, 1L)
    expect_identical(dl_calls, 1L)
    expect_false(file.exists(local_path))
  })
})

test_that("load_stat_model() validates stat_name format", {
  expect_error(load_stat_model("GOALS"), "Invalid stat name")
  expect_error(load_stat_model("goals-per-game"), "Invalid stat name")
  expect_error(load_stat_model("goals 123"), "Invalid stat name")
})

test_that("load_stat_model() rejects empty string", {
  expect_error(load_stat_model(""), "Invalid stat name")
})

test_that("load_stat_model() rejects unknown stat names", {
  expect_error(load_stat_model("nonexistent_stat"), "Unknown stat")
})

test_that("safe_read_rds() returns object for valid RDS file", {
  withr::with_tempdir({
    path <- file.path(getwd(), "valid.rds")
    saveRDS(list(a = 1, b = "test"), path)
    result <- torpmodels:::safe_read_rds(path, "test_model")
    expect_equal(result$a, 1)
    expect_true(file.exists(path))
  })
})

test_that("safe_read_rds() deletes corrupted file and raises error", {
  withr::with_tempdir({
    path <- file.path(getwd(), "corrupt.rds")
    writeLines("not a valid rds file", path)
    expect_true(file.exists(path))
    expect_error(
      torpmodels:::safe_read_rds(path, "broken_model"),
      "corrupted"
    )
    expect_false(file.exists(path))
  })
})

test_that("safe_read_rds() does not delete file on non-corruption errors", {
  withr::with_tempdir({
    path <- file.path(getwd(), "valid.rds")
    # Save an object that references a non-existent class
    obj <- structure(list(x = 1), class = "NonExistentS3Class")
    saveRDS(obj, path)
    # This should load fine since S3 classes don't need registration
    result <- torpmodels:::safe_read_rds(path, "test_model")
    expect_true(file.exists(path))
  })
})

test_that("EP model description says XGBoost, not GAM", {
  models <- list_available_models()
  expect_true(grepl("XGBoost", models$core_models[["ep"]]))
  expect_false(grepl("GAM", models$core_models[["ep"]]))
})

test_that("check_model_cache() returns expected data.frame structure", {
  withr::with_tempdir({
    withr::with_options(list(torpmodels.cache_dir = file.path(getwd(), "empty_cache")), {
      result <- check_model_cache()
      expect_s3_class(result, "data.frame")
      expect_named(result, c("model", "type", "cached", "size_mb"))
      expect_true(all(!result$cached[result$type == "core"]))
    })
  })
})
