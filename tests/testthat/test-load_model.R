test_that("normalize_model_name() returns correct mappings", {
  ep <- normalize_model_name("ep")
  expect_equal(ep$file, "ep_model.rds")
  expect_equal(ep$tag, "core-models")

  wp <- normalize_model_name("wp_model")
  expect_equal(wp$file, "wp_model.rds")
  expect_equal(wp$tag, "core-models")

  shot <- normalize_model_name("shot")
  expect_equal(shot$file, "shot_ocat_mdl.rds")

  xgb <- normalize_model_name("xgb_win")
  expect_equal(xgb$file, "xgb_win_model.rds")
})

test_that("normalize_model_name() returns NULL for unknown models", {
  expect_null(normalize_model_name("nonexistent"))
  expect_null(normalize_model_name(""))
  expect_null(normalize_model_name("random_name"))
})

test_that("normalize_model_name() is case-insensitive", {
  expect_equal(normalize_model_name("EP")$file, "ep_model.rds")
  expect_equal(normalize_model_name("WP")$file, "wp_model.rds")
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
  expect_true(length(models$stat_models) > 20)
})

test_that("clear_model_cache() rejects invalid type via match.arg()", {
  expect_error(clear_model_cache(type = "invalid"), "should be one of")
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
    })
  })
})

test_that("load_stat_model() validates stat_name format", {
  expect_error(load_stat_model("GOALS"), "Invalid stat name")
  expect_error(load_stat_model("goals-per-game"), "Invalid stat name")
  expect_error(load_stat_model("goals 123"), "Invalid stat name")
})

test_that("load_stat_model() rejects unknown stat names", {
  expect_error(load_stat_model("nonexistent_stat"), "Unknown stat")
})

test_that("EP model description says XGBoost, not GAM", {
  models <- list_available_models()
  expect_true(grepl("XGBoost", models$core_models[["ep"]]))
  expect_false(grepl("GAM", models$core_models[["ep"]]))
})
