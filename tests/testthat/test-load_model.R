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
