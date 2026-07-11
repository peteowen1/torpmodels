# train_lib.R is side-effect-free at source time, so tests can source() it.
# Load dev torp first (sibling repo) so torp::: calls inside train_lib.R
# resolve against current source rather than a possibly-stale installed copy.
.torp_test_paths <- c("../../../torp", "../../torp", "../torp")
.torp_test_path <- NULL
for (.p in .torp_test_paths) {
  if (file.exists(file.path(.p, "DESCRIPTION"))) {
    .torp_test_path <- .p
    break
  }
}
if (!is.null(.torp_test_path)) {
  suppressMessages(devtools::load_all(.torp_test_path, quiet = TRUE))
}

lib <- testthat::test_path("..", "..", "data-raw", "lib", "train_lib.R")
env <- new.env()
source(lib, local = env)

test_that("wp_params derives constraints from torp and validate_wp_spec enforces width", {
  skip_if_not_installed("torp")   # skip_if(!requireNamespace) on CI without torp
  p <- env$wp_params()
  expect_identical(p$eta, 0.025)
  X_bad <- matrix(0, 1, 15)
  X_ok <- matrix(0, 1, 18, dimnames = list(NULL, torp:::WP_MODEL_FEATURES))
  expect_error(env$validate_wp_spec(X_bad, p))               # the exact F1 shape
  expect_silent(env$validate_wp_spec(X_ok, p))
})

test_that("default_training_seasons excludes the in-progress season", {   # F4
  skip_if_not_installed("torp")
  # get_afl_season() returns double (lubridate::year()); coerce for expect_identical
  expect_identical(max(env$default_training_seasons()), as.integer(torp::get_afl_season() - 1L))
  expect_identical(min(env$default_training_seasons()), 2021L)
})

test_that("insample EP source forces upload = FALSE", {
  skip_if_not_installed("torp")

  fake_epv <- data.frame(torp_match_id = rep(1:4, each = 5))

  # Mock the heavy pieces (network/data/xgboost) so this stays a fast unit
  # test of the upload guard itself, not an end-to-end training run.
  env$load_training_pbp <- function(seasons) fake_epv
  env$fit_wp <- function(model_data_wp, ...) {
    list(model = "MOCK_WP", optimal_nrounds = 1L, cv_logloss = 0.2,
        X = matrix(0, 1, 1), y = 0, folds = list(1))
  }

  testthat::local_mocked_bindings(
    add_epv_vars = function(df) df,
    .package = "torp"
  )
  testthat::local_mocked_bindings(
    clean_model_data_wp = function(df) df,
    .package = "torp"
  )

  publish_calls <- 0L
  testthat::local_mocked_bindings(
    build_model_meta = function(...) list(model = "wp"),
    stamp_model_meta = function(object, meta) object,
    publish_model_group = function(...) { publish_calls <<- publish_calls + 1L },
    .package = "torpmodels"
  )

  expect_message(
    result <- env$train_core_models(models = "wp", seasons = 2024L, upload = TRUE,
                                    wp_ep_source = "insample",
                                    output_dir = withr::local_tempdir()),
    "forces upload = FALSE"
  )
  expect_identical(publish_calls, 0L)   # publish never invoked
  expect_identical(result$wp$model, "wp")
})
