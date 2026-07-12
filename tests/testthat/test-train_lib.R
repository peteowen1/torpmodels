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

# -----------------------------------------------------------------------
# FABLE-RECAL-PLAN.md Step 1 -- WP recalibration + temporal slope gate
# -----------------------------------------------------------------------
# Pure-function tests below run FIRST and must stay ahead of the wiring
# tests further down: those mutate env$wp_gate_slope / env$fit_wp_calibration
# / env$validate_wp_temporal_slope / env$cv_wp_oos_preds directly (plain
# `env$x <- ` reassignment, not local_mocked_bindings, so nothing restores
# it at test_that() boundaries) -- run after them and these tests would be
# exercising a mock instead of the real implementation.

test_that("wp_gate_slope: all-rows cell is a plain non-dedup slope excluding draws", {
  set.seed(1)
  n <- 400
  x <- stats::qlogis(stats::runif(n, 0.05, 0.95))
  p_true <- stats::plogis(0.1 + 1.2 * x)
  y <- stats::rbinom(n, 1, p_true)
  preds <- stats::plogis(x)
  meta <- data.frame(period = 4L, points_diff = 0, est_match_elapsed = 0,
                     match_id = paste0("m", seq_len(n)))

  # inject draw rows -- must be excluded, not error
  y_with_draws <- c(y, 0.5, 0.5)
  preds_with_draws <- c(preds, 0.5, 0.5)
  meta_with_draws <- rbind(meta, meta[1:2, ])

  got <- env$wp_gate_slope(preds_with_draws, y_with_draws, meta_with_draws, cell = "all")
  expected <- unname(stats::coef(stats::glm(y ~ stats::qlogis(preds), family = stats::binomial()))[2])
  expect_equal(got, expected, tolerance = 1e-6)
})

test_that("wp_gate_slope: q4close cell filters to period 4 / |margin|<=12 and dedups by (match_id, bucket)", {
  meta <- data.frame(
    match_id = c("m1", "m1", "m2", "m3"),
    period = c(4, 4, 4, 1),          # m3 excluded: not Q4
    points_diff = c(5, 5, 20, 5),    # m2 excluded: |margin| > 12
    est_match_elapsed = c(3300, 3300, 3300, 3300)  # m1's two rows share a bucket
  )
  preds  <- c(0.9, 0.2, 0.5, 0.5)
  labels <- c(1,   0,   1,   0)

  # Only m1 survives the cell filter; its two rows collapse to the LAST one
  # via the dedup -- one row is too few for a GLM, so expect NA, not a
  # coefficient computed from both (undeduped) rows.
  got <- env$wp_gate_slope(preds, labels, meta, cell = "q4close")
  expect_true(is.na(got))
})

test_that("wp_gate_slope q4close: keeps the LAST row per (match_id, bucket), not the first", {
  meta <- data.frame(
    match_id = c("m1", "m1", "m2", "m2"),
    period = 4, points_diff = 5,
    est_match_elapsed = c(3300, 3300, 3300, 3300)
  )
  preds  <- c(0.9, 0.15, 0.1, 0.85)   # kept (last): m1 -> 0.15, m2 -> 0.85
  labels <- c(1,   0,    0,   1)      # kept (last): m1 -> 0,    m2 -> 1

  got <- env$wp_gate_slope(preds, labels, meta, cell = "q4close")
  expected <- unname(stats::coef(stats::glm(c(0, 1) ~ stats::qlogis(c(0.15, 0.85)),
                                            family = stats::binomial()))[2])
  expect_equal(got, expected, tolerance = 1e-6)
})

test_that("fit_wp_calibration recovers a known (a, b) and drops draw rows", {
  set.seed(42)
  n <- 3000
  x <- stats::qlogis(stats::runif(n, 0.02, 0.98))
  true_a <- -0.3; true_b <- 1.35
  y <- stats::rbinom(n, 1, stats::plogis(true_a + true_b * x))
  preds <- stats::plogis(x)   # so qlogis(preds) == x

  # draw rows that would corrupt a naive (non-filtering) fit
  y_all <- c(y, rep(0.5, 50))
  preds_all <- c(preds, rep(0.5, 50))

  fit <- env$fit_wp_calibration(preds_all, y_all)
  expect_true(is.finite(fit$a) && is.finite(fit$b) && fit$b > 0)
  expect_equal(fit$a, true_a, tolerance = 0.15)
  expect_equal(fit$b, true_b, tolerance = 0.15)
})

test_that("fit_wp_calibration's stopifnot(b > 0) rejects an inverted relationship", {
  set.seed(7)
  n <- 200
  x <- stats::qlogis(stats::runif(n, 0.05, 0.95))
  y <- stats::rbinom(n, 1, stats::plogis(-2 * x))   # inverted -> b < 0
  preds <- stats::plogis(x)
  expect_error(env$fit_wp_calibration(preds, y))
})

test_that("validate_wp_temporal_slope passes on well-calibrated data and aborts on a breach", {
  set.seed(11)
  n <- 3000
  meta <- data.frame(period = 4L, points_diff = sample(-10:10, n, replace = TRUE),
                     est_match_elapsed = sample(3000:3600, n, replace = TRUE),
                     match_id = paste0("m", seq_len(n)))   # every match_id unique -> dedup is a no-op
  x <- stats::qlogis(stats::runif(n, 0.05, 0.95))
  y <- stats::rbinom(n, 1, stats::plogis(x))   # slope 1 by construction
  preds <- stats::plogis(x)

  result <- env$validate_wp_temporal_slope(preds, y, meta, threshold = 0.10)
  expect_true(abs(result$slope_all - 1) <= 0.10)
  expect_true(abs(result$slope_q4close - 1) <= 0.10)

  # A badly flat calibration (b << 1, the exact defect this gate exists for)
  flat_preds <- stats::plogis(0.3 * x)
  expect_error(env$validate_wp_temporal_slope(flat_preds, y, meta, threshold = 0.10), "FAILED")
})

test_that("cv_wp_oos_preds returns a full-length OOS vector via per-fold refit (tiny xgboost fixture)", {
  skip_if_not_installed("xgboost")
  set.seed(3)
  n <- 60
  X <- matrix(rnorm(n * 2), ncol = 2, dimnames = list(NULL, c("a", "b")))
  y <- rbinom(n, 1, 0.5)
  folds <- list(1:20, 21:40, 41:60)
  params <- list(objective = "binary:logistic", eval_metric = "logloss")

  got <- env$cv_wp_oos_preds(X, y, folds, params, nrounds = 2)
  expect_length(got, n)
  expect_true(all(got >= 0 & got <= 1))
  expect_false(anyNA(got))
})

# -----------------------------------------------------------------------
# train_core_models() WP-branch wiring (mocked -- network-free, no real
# xgboost training). Order matters within this block only insofar as each
# test fully re-mocks every env$ binding it touches, so leftover state from
# an earlier wiring test can't leak in.
# -----------------------------------------------------------------------

.fake_epv_for_wiring_tests <- function() {
  data.frame(
    torp_match_id = rep(1:4, each = 5), match_id = rep(1:4, each = 5),
    period = 4L, points_diff = 0, est_match_elapsed = 0,
    label_wp = rep(c(0, 1), 10)
  )
}

test_that("calibrate = FALSE forces upload = FALSE and never runs the temporal variant", {
  skip_if_not_installed("torp")
  fake_epv <- .fake_epv_for_wiring_tests()

  env$load_training_pbp <- function(seasons) fake_epv
  env$fit_ep <- function(model_data_epv, ...) {
    list(model = "MOCK_EP", optimal_nrounds = 1L, cv_logloss = 0.1,
        X = matrix(0, nrow(model_data_epv), 1), y = rep(0, nrow(model_data_epv)),
        folds = list(seq_len(nrow(model_data_epv))))
  }
  env$cv_ep_oos_preds <- function(...) matrix(0.2, nrow(fake_epv), 5)
  env$build_wp_data <- function(model_data_epv, oos_ep_preds) model_data_epv
  env$fit_wp <- function(model_data_wp, ...) {
    list(model = "MOCK_WP", optimal_nrounds = 1L, cv_logloss = 0.2,
        X = matrix(0, nrow(model_data_wp), 1), y = rep(0, nrow(model_data_wp)),
        folds = list(seq_len(nrow(model_data_wp))))
  }
  temporal_called <- FALSE
  env$fit_wp_temporal_variant <- function(...) { temporal_called <<- TRUE; NULL }

  publish_calls <- 0L
  testthat::local_mocked_bindings(
    build_model_meta = function(model_name, ...) list(model = model_name),
    stamp_model_meta = function(object, meta) object,
    publish_model_group = function(...) { publish_calls <<- publish_calls + 1L },
    .package = "torpmodels"
  )

  expect_message(
    result <- env$train_core_models(models = "wp", seasons = 2021:2024, upload = TRUE,
                                    wp_ep_source = "cv", calibrate = FALSE,
                                    output_dir = withr::local_tempdir()),
    "forces upload = FALSE"
  )
  expect_false(temporal_called)
  expect_identical(publish_calls, 0L)
  expect_null(result$wp_calibration)
  expect_identical(result$wp$model, "wp")
})

test_that("calibrate = TRUE (default): fits + gates + saves wp_calibration.rds and publishes the pair", {
  skip_if_not_installed("torp")
  fake_epv <- .fake_epv_for_wiring_tests()

  env$load_training_pbp <- function(seasons) fake_epv
  env$fit_ep <- function(model_data_epv, ...) {
    list(model = "MOCK_EP", optimal_nrounds = 1L, cv_logloss = 0.1,
        X = matrix(0, nrow(model_data_epv), 1), y = rep(0, nrow(model_data_epv)),
        folds = list(seq_len(nrow(model_data_epv))))
  }
  env$cv_ep_oos_preds <- function(...) matrix(0.2, nrow(fake_epv), 5)
  env$build_wp_data <- function(model_data_epv, oos_ep_preds) model_data_epv
  env$fit_wp <- function(model_data_wp, ...) {
    list(model = "MOCK_WP", optimal_nrounds = 1L, cv_logloss = 0.2,
        X = matrix(0, nrow(model_data_wp), 1), y = rep(0, nrow(model_data_wp)),
        folds = list(seq_len(nrow(model_data_wp))))
  }

  fake_meta <- data.frame(period = 4L, points_diff = 0, est_match_elapsed = 0, match_id = 1:4)
  temporal_called_with <- NULL
  env$fit_wp_temporal_variant <- function(model_data_epv, gate_season, ...) {
    temporal_called_with <<- gate_season
    list(preds = rep(0.6, 4), labels = c(0, 1, 0, 1), meta_cols = fake_meta)
  }
  env$fit_wp_calibration <- function(preds, labels) list(a = 0, b = 1.2)
  env$wp_gate_slope <- function(preds, labels, meta_cols, cell) 1.0
  env$validate_wp_temporal_slope <- function(calibrated_preds, labels, meta_cols, threshold = 0.10) {
    list(slope_all = 1.0, slope_q4close = 1.0)
  }
  env$cv_wp_oos_preds <- function(...) rep(0.5, 4)

  testthat::local_mocked_bindings(load_chains = function(...) data.frame(), .package = "torp")
  publish_groups <- character(0)
  testthat::local_mocked_bindings(
    build_model_meta = function(model_name, ...) list(model = model_name),
    stamp_model_meta = function(object, meta) { attr(object, "torp_meta") <- meta; object },
    publish_model_group = function(group, ...) { publish_groups <<- c(publish_groups, group) },
    .package = "torpmodels"
  )

  out_dir <- withr::local_tempdir()
  result <- env$train_core_models(models = "wp", seasons = 2021:2024, upload = TRUE,
                                  wp_ep_source = "cv", output_dir = out_dir)

  expect_identical(temporal_called_with, 2024L)
  expect_true(file.exists(file.path(out_dir, "wp_calibration.rds")))
  expect_true(file.exists(file.path(out_dir, "wp_model.rds")))
  expect_identical(publish_groups, "wp")
  expect_identical(result$wp_calibration$model, "wp_calibration")

  saved_calib <- readRDS(file.path(out_dir, "wp_calibration.rds"))
  expect_equal(saved_calib$a, 0)
  expect_equal(saved_calib$b, 1.2)
  expect_equal(saved_calib$slope_after, 1.0)
})

test_that("slope_gate = FALSE warns loudly but never calls validate_wp_temporal_slope", {
  skip_if_not_installed("torp")
  fake_epv <- .fake_epv_for_wiring_tests()

  env$load_training_pbp <- function(seasons) fake_epv
  env$fit_ep <- function(model_data_epv, ...) {
    list(model = "MOCK_EP", optimal_nrounds = 1L, cv_logloss = 0.1,
        X = matrix(0, nrow(model_data_epv), 1), y = rep(0, nrow(model_data_epv)),
        folds = list(seq_len(nrow(model_data_epv))))
  }
  env$cv_ep_oos_preds <- function(...) matrix(0.2, nrow(fake_epv), 5)
  env$build_wp_data <- function(model_data_epv, oos_ep_preds) model_data_epv
  env$fit_wp <- function(model_data_wp, ...) {
    list(model = "MOCK_WP", optimal_nrounds = 1L, cv_logloss = 0.2,
        X = matrix(0, nrow(model_data_wp), 1), y = rep(0, nrow(model_data_wp)),
        folds = list(seq_len(nrow(model_data_wp))))
  }
  fake_meta <- data.frame(period = 4L, points_diff = 0, est_match_elapsed = 0, match_id = 1:4)
  env$fit_wp_temporal_variant <- function(...) {
    list(preds = rep(0.6, 4), labels = c(0, 1, 0, 1), meta_cols = fake_meta)
  }
  env$fit_wp_calibration <- function(preds, labels) list(a = 0, b = 1.2)
  # a slope that WOULD breach the 0.10 gate -- proves the gate never ran
  env$wp_gate_slope <- function(preds, labels, meta_cols, cell) 1.35
  env$validate_wp_temporal_slope <- function(...) stop("gate must not run when slope_gate = FALSE")
  env$cv_wp_oos_preds <- function(...) rep(0.5, 4)

  testthat::local_mocked_bindings(load_chains = function(...) data.frame(), .package = "torp")
  testthat::local_mocked_bindings(
    build_model_meta = function(model_name, ...) list(model = model_name),
    stamp_model_meta = function(object, meta) { attr(object, "torp_meta") <- meta; object },
    publish_model_group = function(...) invisible(NULL),
    .package = "torpmodels"
  )

  expect_message(
    result <- env$train_core_models(models = "wp", seasons = 2021:2024, upload = TRUE,
                                    wp_ep_source = "cv", slope_gate = FALSE,
                                    output_dir = withr::local_tempdir()),
    "slope_gate = FALSE"
  )
  expect_identical(result$wp_calibration$model, "wp_calibration")
})
