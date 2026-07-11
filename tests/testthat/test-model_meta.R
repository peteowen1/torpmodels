test_that("meta stamp survives an RDS round-trip on an xgboost booster", {
  skip_if_not_installed("xgboost")
  X <- matrix(rnorm(200), ncol = 2, dimnames = list(NULL, c("a", "b")))
  m <- xgboost::xgb.train(params = list(objective = "binary:logistic"),
                          data = xgboost::xgb.DMatrix(X, label = rbinom(100, 1, .5)),
                          nrounds = 2, verbose = 0)
  meta <- build_model_meta("wp", 2021:2025, list(eta = 0.025), c("a", "b"), cv_metric = 0.45)
  path <- withr::local_tempfile(fileext = ".rds")
  saveRDS(stamp_model_meta(m, meta), path)
  got <- model_meta(readRDS(path))
  expect_identical(got$model, "wp"); expect_identical(got$seasons, "2021-2025")
  for (f in c("trained_at", "script", "params", "feature_names",
              "torp_sha", "r_version", "schema_version")) expect_true(f %in% names(got))
})

test_that("build_model_meta rejects missing required inputs", {
  expect_error(build_model_meta(seasons = 2021:2025), "model")   # etc.
})

test_that("load_torp_model warns on meta-less artifact and prints provenance on stamped one", {
  cache_dir <- withr::local_tempdir()
  core_dir <- file.path(cache_dir, "core")
  dir.create(core_dir, recursive = TRUE)
  withr::local_options(torpmodels.cache_dir = cache_dir)

  # Unstamped artifact (pre-provenance model) -- must warn but still load
  saveRDS(list(dummy = TRUE), file.path(core_dir, "ep_model.rds"))
  expect_warning(
    result <- load_torp_model("ep", verbose = TRUE),
    "no torp_meta"
  )
  expect_identical(result, list(dummy = TRUE))

  # Stamped artifact -- must print provenance, no warning
  meta <- build_model_meta("wp", 2021:2025, list(eta = 0.025), c("a", "b"), cv_metric = 0.4)
  saveRDS(stamp_model_meta(list(dummy = TRUE), meta), file.path(core_dir, "wp_model.rds"))
  expect_message(
    load_torp_model("wp", verbose = TRUE),
    "trained"
  )
})
