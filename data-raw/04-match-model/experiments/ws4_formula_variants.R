# ws4_formula_variants.R — WS4: structural formula variants (ti() interactions
# and chain shape), FABLE-MATCH-MAE-PLAN.md
# =====================================================================
# Hypothesis: the ti(epr_diff/torp_diff/psr_diff, gam_pred_tot_xscore) tensor
# smooths in models 2-4 of .train_match_gams() amplify rating diffs into
# oversized margins exactly where games turn out close (diagnosis Finding 3).
#
# Three gam_trainer overrides, each a copy of torp:::.train_match_gams()
# (plan G5 -- production torp/R/*.R is not touched in this pass) with the
# formula edited per variant:
#   V4a  drop ti(epr_diff/torp_diff/psr_diff, gam_pred_tot_xscore) from
#        models 2-4; keep the main-effect smooths (s(gam_pred_tot_xscore),
#        s(epr_diff), s(torp_diff), s(psr_diff)) and model 4's own
#        second-order stack tensors untouched.
#   V4b  = V4a, plus drop model 4's second-order stack tensors
#        ti(gam_pred_xscore_diff, gam_pred_conv_diff) and
#        ti(gam_pred_tot_xscore, gam_pred_conv_diff).
#   V4c  flat chain: models 1-3 unchanged (still trained; model 1's
#        gam_pred_tot_xscore still feeds model 5's win formula), but model 4
#        (score_diff) is refit directly on model-4's non-stack features only
#        (team REs, epr_diff family, torp_diff, psr/osr/dsr_diff optional,
#        travel/rest) -- zero gam_pred_* stack terms at all. Tests how much
#        the 4-stage chain buys over one honest model for score_diff.
#
# Do NOT resweep gamma_arg (plan: tuned 2026-05, commit 47bc397) unless a
# structural winner emerges here.
#
# Screening per plan G2: TEST_SEASONS <- 2026 only. A 2025:2026 confirmation
# run is added at the bottom, gated on a structural winner emerging.

# Setup (mirrors train_match_models.R's setup section) ----
library(tidyverse)
library(xgboost)
library(mgcv)
library(MLmetrics)
library(geosphere)
library(cli)

torp_paths <- c("../../../torp", "../../torp", "../torp", "C:/dev/torpverse/torp")
torp_loaded <- FALSE
for (p in torp_paths) {
  if (file.exists(file.path(p, "DESCRIPTION"))) {
    devtools::load_all(p)
    torp_loaded <- TRUE
    break
  }
}
if (!torp_loaded) stop("Cannot find torp package (tried: ", paste(torp_paths, collapse = ", "), ")")

source("C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/rolling_lib.R")

RESULTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/results"
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR, recursive = TRUE)

# .train_match_gams_ws4 ----

#' WS4 gam_trainer override: structural formula variants
#'
#' Full copy of torp:::.train_match_gams() (plan G5) with models 2-4's
#' formulas edited per `variant`. Models 1 and 5 are byte-identical to
#' production in all three variants.
#'
#' @keywords internal
.train_match_gams_ws4 <- function(team_mdl_df, train_filter = NULL, nthreads = 4L,
                                   gamma_arg = 1.4, variant = c("v4a", "v4b", "v4c")) {
  variant <- match.arg(variant)
  loadNamespace("mgcv")

  if (is.null(train_filter)) {
    train_mask <- !is.na(team_mdl_df$win)
  } else {
    train_mask <- train_filter & !is.na(team_mdl_df$win)
  }

  gam_df <- team_mdl_df[train_mask, ]
  cli::cli_inform("[ws4:{variant}] Training on {nrow(gam_df)} completed matches")
  if (nrow(gam_df) == 0) {
    cli::cli_abort("Cannot train GAM models: 0 completed matches after filtering")
  }

  # Same optional-term uniqueness guard as production ----
  optional_smooth_terms <- list(
    "s(psr.x, bs = \"ts\", k = 5)"           = list(var = "psr.x", k = 5),
    "s(psr.y, bs = \"ts\", k = 5)"           = list(var = "psr.y", k = 5),
    "s(log_wind, bs = \"ts\", k = 5)"        = list(var = "log_wind", k = 5),
    "s(log_precip, bs = \"ts\", k = 5)"      = list(var = "log_precip", k = 5),
    "s(temp_avg, bs = \"ts\", k = 5)"        = list(var = "temp_avg", k = 5),
    "s(humidity_avg, bs = \"ts\", k = 5)"     = list(var = "humidity_avg", k = 5),
    "s(abs(psr_diff), bs = \"ts\", k = 5)"   = list(var = "psr_diff", k = 5),
    "s(abs(osr_diff), bs = \"ts\", k = 5)"   = list(var = "osr_diff", k = 5),
    "s(abs(dsr_diff), bs = \"ts\", k = 5)"   = list(var = "dsr_diff", k = 5),
    "s(psr_diff, bs = \"ts\", k = 5)"        = list(var = "psr_diff", k = 5),
    "s(osr_diff, bs = \"ts\", k = 5)"        = list(var = "osr_diff", k = 5),
    "s(dsr_diff, bs = \"ts\", k = 5)"        = list(var = "dsr_diff", k = 5),
    "ti(psr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)" = list(var = "psr_diff", k = 4)
  )
  drop_terms <- character(0)
  for (term_str in names(optional_smooth_terms)) {
    info <- optional_smooth_terms[[term_str]]
    vals <- gam_df[[info$var]]
    n_unique <- length(unique(vals[!is.na(vals)]))
    if (n_unique < info$k) {
      drop_terms <- c(drop_terms, term_str)
    }
  }

  .add_optional <- function(base_terms, optional_terms) {
    keep <- setdiff(optional_terms, drop_terms)
    if (length(keep) > 0) {
      paste(base_terms, "+", paste(keep, collapse = " + "))
    } else {
      base_terms
    }
  }

  # Model 1: Total expected points -- unchanged in all variants ----
  cli::cli_progress_step("[ws4:{variant}] Training total xPoints model")
  m1_base <- paste(
    "total_xpoints_adj ~",
    "s(team_type_fac, bs = \"re\")",
    "+ s(game_year_decimal.x, bs = \"ts\")",
    "+ s(game_prop_through_year.x, bs = \"cc\")",
    "+ s(game_prop_through_month.x, bs = \"cc\")",
    "+ s(game_wday_fac.x, bs = \"re\")",
    "+ s(game_prop_through_day.x, bs = \"cc\")",
    "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
    "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
    "+ s(abs(epr_diff), bs = \"ts\", k = 5)",
    "+ s(abs(epr_recv_diff), bs = \"ts\", k = 5)",
    "+ s(abs(epr_disp_diff), bs = \"ts\", k = 5)",
    "+ s(abs(epr_spoil_diff), bs = \"ts\", k = 5)",
    "+ s(abs(epr_hitout_diff), bs = \"ts\", k = 5)",
    "+ s(epr.x, bs = \"ts\", k = 5) + s(epr.y, bs = \"ts\", k = 5)",
    "+ s(abs(torp_diff), bs = \"ts\", k = 5)",
    "+ s(torp.x, bs = \"ts\", k = 5) + s(torp.y, bs = \"ts\", k = 5)",
    "+ s(venue_fac, bs = \"re\")",
    "+ s(log_dist.x, bs = \"ts\", k = 5) + s(log_dist.y, bs = \"ts\", k = 5)",
    "+ s(familiarity.x, bs = \"ts\", k = 5) + s(familiarity.y, bs = \"ts\", k = 5)",
    "+ s(log_dist_diff, bs = \"ts\", k = 5)",
    "+ s(familiarity_diff, bs = \"ts\", k = 5)",
    "+ s(days_rest_diff_fac, bs = \"re\")"
  )
  m1_optional <- c(
    "s(psr.x, bs = \"ts\", k = 5)", "s(psr.y, bs = \"ts\", k = 5)",
    "s(abs(psr_diff), bs = \"ts\", k = 5)",
    "s(abs(osr_diff), bs = \"ts\", k = 5)", "s(abs(dsr_diff), bs = \"ts\", k = 5)",
    "s(log_wind, bs = \"ts\", k = 5)", "s(log_precip, bs = \"ts\", k = 5)",
    "s(temp_avg, bs = \"ts\", k = 5)", "s(humidity_avg, bs = \"ts\", k = 5)"
  )
  m1_formula <- stats::as.formula(.add_optional(m1_base, m1_optional))

  afl_total_xpoints_mdl <- mgcv::bam(
    m1_formula,
    data = gam_df, weights = gam_df$weightz,
    family = gaussian(), nthreads = nthreads, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE,
    gamma = gamma_arg
  )
  team_mdl_df$gam_pred_tot_xscore <- predict(afl_total_xpoints_mdl, newdata = team_mdl_df, type = "response")

  # Model 2: xScore differential ----
  # v4a/v4b drop ti(epr_diff/torp_diff/psr_diff, gam_pred_tot_xscore), keep
  # the main-effect smooths (s(gam_pred_tot_xscore), s(epr_diff), s(torp_diff)).
  # v4c leaves model 2 exactly as production.
  cli::cli_progress_step("[ws4:{variant}] Training xScore diff model")
  gam_df$gam_pred_tot_xscore <- team_mdl_df$gam_pred_tot_xscore[train_mask]

  drop_ti_tot <- variant %in% c("v4a", "v4b")

  if (drop_ti_tot) {
    m2_terms <- c(
      "xscore_diff ~ s(team_type_fac, bs = \"re\")",
      "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
      "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
      "+ s(gam_pred_tot_xscore, bs = \"ts\", k = 5)",
      "+ s(epr_diff, bs = \"ts\", k = 5)",
      "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
      "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
      "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
      "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
      "+ s(torp_diff, bs = \"ts\", k = 5)",
      "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
      "+ s(days_rest_diff_fac, bs = \"re\")"
    )
    m2_optional <- c("s(psr_diff, bs = \"ts\", k = 5)",
                      "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  } else {
    m2_terms <- c(
      "xscore_diff ~ s(team_type_fac, bs = \"re\")",
      "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
      "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
      "+ ti(epr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
      "+ s(gam_pred_tot_xscore, bs = \"ts\", k = 5)",
      "+ s(epr_diff, bs = \"ts\", k = 5)",
      "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
      "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
      "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
      "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
      "+ s(torp_diff, bs = \"ts\", k = 5)",
      "+ ti(torp_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
      "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
      "+ s(days_rest_diff_fac, bs = \"re\")"
    )
    m2_optional <- c("s(psr_diff, bs = \"ts\", k = 5)",
                      "ti(psr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
                      "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  }
  m2_base <- paste(m2_terms, collapse = " ")
  m2_formula <- stats::as.formula(.add_optional(m2_base, m2_optional))

  afl_xscore_diff_mdl <- mgcv::bam(
    m2_formula,
    data = gam_df, weights = gam_df$weightz,
    family = gaussian(), nthreads = nthreads, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE,
    gamma = gamma_arg
  )
  team_mdl_df$gam_pred_xscore_diff <- predict(afl_xscore_diff_mdl, newdata = team_mdl_df, type = "response")

  # Model 3: Conversion differential ----
  # Same drop rule as model 2 for v4a/v4b; v4c leaves model 3 as production.
  cli::cli_progress_step("[ws4:{variant}] Training conversion model")
  gam_df$gam_pred_xscore_diff <- team_mdl_df$gam_pred_xscore_diff[train_mask]

  if (drop_ti_tot) {
    m3_terms <- c(
      "shot_conv_diff ~ s(team_type_fac, bs = \"re\")",
      "+ s(game_year_decimal.x, bs = \"ts\")",
      "+ s(game_prop_through_year.x, bs = \"cc\")",
      "+ s(game_prop_through_month.x, bs = \"cc\")",
      "+ s(game_wday_fac.x, bs = \"re\")",
      "+ s(game_prop_through_day.x, bs = \"cc\")",
      "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
      "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
      "+ s(epr_diff, bs = \"ts\", k = 5)",
      "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
      "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
      "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
      "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
      "+ s(torp_diff, bs = \"ts\", k = 5)",
      "+ s(gam_pred_tot_xscore, bs = \"ts\", k = 5)",
      "+ s(gam_pred_xscore_diff, bs = \"ts\", k = 5)",
      "+ s(venue_fac, bs = \"re\")",
      "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
      "+ s(days_rest_diff_fac, bs = \"re\")"
    )
    m3_optional <- c("s(psr_diff, bs = \"ts\", k = 5)",
                      "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  } else {
    m3_terms <- c(
      "shot_conv_diff ~ s(team_type_fac, bs = \"re\")",
      "+ s(game_year_decimal.x, bs = \"ts\")",
      "+ s(game_prop_through_year.x, bs = \"cc\")",
      "+ s(game_prop_through_month.x, bs = \"cc\")",
      "+ s(game_wday_fac.x, bs = \"re\")",
      "+ s(game_prop_through_day.x, bs = \"cc\")",
      "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
      "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
      "+ ti(epr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
      "+ s(epr_diff, bs = \"ts\", k = 5)",
      "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
      "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
      "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
      "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
      "+ s(torp_diff, bs = \"ts\", k = 5)",
      "+ ti(torp_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
      "+ s(gam_pred_tot_xscore, bs = \"ts\", k = 5)",
      "+ s(gam_pred_xscore_diff, bs = \"ts\", k = 5)",
      "+ s(venue_fac, bs = \"re\")",
      "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
      "+ s(days_rest_diff_fac, bs = \"re\")"
    )
    m3_optional <- c("s(psr_diff, bs = \"ts\", k = 5)",
                      "ti(psr_diff, gam_pred_tot_xscore, bs = c(\"ts\", \"ts\"), k = 4)",
                      "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  }
  m3_base <- paste(m3_terms, collapse = " ")
  m3_formula <- stats::as.formula(.add_optional(m3_base, m3_optional))

  afl_conv_mdl <- mgcv::bam(
    m3_formula,
    data = gam_df, weights = gam_df$shot_weightz,
    family = gaussian(), nthreads = nthreads, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE,
    gamma = gamma_arg
  )
  team_mdl_df$gam_pred_conv_diff <- predict(afl_conv_mdl, newdata = team_mdl_df, type = "response")

  # Model 4: Score differential ----
  # v4a: drop ti(epr_diff/torp_diff/psr_diff, gam_pred_tot_xscore) only; keep
  #      model 4's own second-order stack tensors + s(gam_pred_xscore_diff).
  # v4b: v4a + also drop ti(gam_pred_xscore_diff, gam_pred_conv_diff) and
  #      ti(gam_pred_tot_xscore, gam_pred_conv_diff); keep the main-effect
  #      s(gam_pred_xscore_diff).
  # v4c: flat -- zero gam_pred_* stack terms at all (no ti(), no s() on any
  #      gam_pred_* column). score_diff regressed directly on model-4's
  #      non-stack features (team REs, rating diffs, travel/rest).
  cli::cli_progress_step("[ws4:{variant}] Training score diff model")
  gam_df$gam_pred_conv_diff <- team_mdl_df$gam_pred_conv_diff[train_mask]

  if (variant == "v4c") {
    m4_terms <- c(
      "score_diff ~ s(team_type_fac, bs = \"re\")",
      "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
      "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
      "+ s(epr_diff, bs = \"ts\", k = 5)",
      "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
      "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
      "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
      "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
      "+ s(torp_diff, bs = \"ts\", k = 5)",
      "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
      "+ s(days_rest_diff_fac, bs = \"re\")"
    )
    m4_optional <- c("s(psr_diff, bs = \"ts\", k = 5)",
                      "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  } else {
    m4_terms <- c(
      "score_diff ~ s(team_type_fac, bs = \"re\")",
      "+ s(team_name.x, bs = \"re\") + s(team_name.y, bs = \"re\")",
      "+ s(team_name_season.x, bs = \"re\") + s(team_name_season.y, bs = \"re\")",
      if (variant == "v4a") "+ ti(gam_pred_xscore_diff, gam_pred_conv_diff, bs = \"ts\", k = 5)",
      if (variant == "v4a") "+ ti(gam_pred_tot_xscore, gam_pred_conv_diff, bs = \"ts\", k = 5)",
      "+ s(gam_pred_xscore_diff)",
      "+ s(epr_diff, bs = \"ts\", k = 5)",
      "+ s(epr_recv_diff, bs = \"ts\", k = 5)",
      "+ s(epr_disp_diff, bs = \"ts\", k = 5)",
      "+ s(epr_spoil_diff, bs = \"ts\", k = 5)",
      "+ s(epr_hitout_diff, bs = \"ts\", k = 5)",
      "+ s(torp_diff, bs = \"ts\", k = 5)",
      "+ s(log_dist_diff, bs = \"ts\", k = 5) + s(familiarity_diff, bs = \"ts\", k = 5)",
      "+ s(days_rest_diff_fac, bs = \"re\")"
    )
    m4_optional <- c("s(psr_diff, bs = \"ts\", k = 5)",
                      "s(osr_diff, bs = \"ts\", k = 5)", "s(dsr_diff, bs = \"ts\", k = 5)")
  }
  m4_base <- paste(m4_terms, collapse = " ")
  m4_formula <- stats::as.formula(.add_optional(m4_base, m4_optional))

  afl_score_mdl <- mgcv::bam(
    m4_formula,
    data = gam_df, weights = gam_df$weightz,
    family = "gaussian", nthreads = nthreads, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE,
    gamma = gamma_arg
  )
  team_mdl_df$gam_pred_score_diff <- predict(afl_score_mdl, newdata = team_mdl_df, type = "response")

  # Model 5: Win probability -- unchanged in all variants ----
  cli::cli_progress_step("[ws4:{variant}] Training win probability model")
  gam_df$pred_tot_xscore  <- gam_df$gam_pred_tot_xscore
  gam_df$pred_score_diff  <- team_mdl_df$gam_pred_score_diff[train_mask]
  afl_win_mdl <- mgcv::bam(
    win ~
      +s(team_name.x, bs = "re") + s(team_name.y, bs = "re")
      + s(team_name_season.x, bs = "re") + s(team_name_season.y, bs = "re")
      + ti(pred_tot_xscore, pred_score_diff, bs = c("ts", "ts"), k = 4)
      + s(pred_score_diff, bs = "ts", k = 5)
      + s(log_dist_diff, bs = "ts", k = 5) + s(familiarity_diff, bs = "ts", k = 5)
      + s(days_rest_diff_fac, bs = "re"),
    data = gam_df, weights = gam_df$weightz,
    family = "binomial", nthreads = nthreads, select = TRUE, discrete = TRUE,
    drop.unused.levels = FALSE,
    gamma = gamma_arg
  )
  team_mdl_df$pred_tot_xscore  <- team_mdl_df$gam_pred_tot_xscore
  team_mdl_df$pred_xscore_diff <- team_mdl_df$gam_pred_xscore_diff
  team_mdl_df$pred_conv_diff   <- team_mdl_df$gam_pred_conv_diff
  team_mdl_df$pred_score_diff  <- team_mdl_df$gam_pred_score_diff

  team_mdl_df$gam_pred_win <- predict(afl_win_mdl, newdata = team_mdl_df, type = "response")
  team_mdl_df$pred_win     <- team_mdl_df$gam_pred_win

  team_mdl_df$bits <- dplyr::case_when(
    team_mdl_df$win == 1   ~ 1 + log2(team_mdl_df$pred_win),
    team_mdl_df$win == 0   ~ 1 + log2(1 - team_mdl_df$pred_win),
    TRUE                   ~ 1 + 0.5 * log2(team_mdl_df$pred_win * (1 - team_mdl_df$pred_win))
  )

  models <- list(
    total_xpoints = afl_total_xpoints_mdl,
    xscore_diff   = afl_xscore_diff_mdl,
    conv_diff     = afl_conv_mdl,
    score_diff    = afl_score_mdl,
    win           = afl_win_mdl
  )

  list(models = models, data = team_mdl_df)
}

# Trainer factory (run_rolling_eval's gam_trainer signature is
# function(team_mdl_df, train_filter, nthreads)) ----
.make_ws4_trainer <- function(variant) {
  force(variant)
  function(team_mdl_df, train_filter = NULL, nthreads = 4L) {
    .train_match_gams_ws4(team_mdl_df, train_filter = train_filter, nthreads = nthreads,
                           gamma_arg = 1.4, variant = variant)
  }
}

# Build data once ----
cli::cli_h1("WS4: Building match model data")
team_mdl_df <- build_team_mdl_df()
cli::cli_inform("Seasons: {paste(sort(unique(team_mdl_df$season.x)), collapse = ', ')}")

# Smoke test: 3 rounds of 2026 only, catches formula/typo errors cheaply ----
cli::cli_h1("WS4: Smoke test (2026 rounds 1-3) on all three variants")
smoke_df <- team_mdl_df |>
  dplyr::filter(season.x < 2026 | (season.x == 2026 & round_number.x <= 3))

for (v in c("v4a", "v4b", "v4c")) {
  cli::cli_h2("Smoke: {v}")
  smoke_roll <- run_rolling_eval(smoke_df, 2026, gam_trainer = .make_ws4_trainer(v), verbose = FALSE)
  cli::cli_alert_success("{v} smoke test OK: {nrow(smoke_roll$gam_preds)} matches")
}

# Full screen: TEST_SEASONS <- 2026 (plan G2) ----
TEST_SEASONS <- 2026

run_variant <- function(label, gam_trainer) {
  cli::cli_h1("WS4: Running {label}")
  t0 <- Sys.time()
  roll <- run_rolling_eval(team_mdl_df, TEST_SEASONS, gam_trainer = gam_trainer)
  cli::cli_inform("{label} completed in {round(difftime(Sys.time(), t0, units = 'mins'), 1)} min")
  roll
}

roll_baseline <- run_variant("Baseline (champion trainers, unmodified)", .train_match_gams)
saveRDS(roll_baseline, file.path(RESULTS_DIR, "ws4_roll_baseline.rds"))

roll_v4a <- run_variant("V4a (drop ti(*, gam_pred_tot_xscore) from models 2-4)", .make_ws4_trainer("v4a"))
saveRDS(roll_v4a, file.path(RESULTS_DIR, "ws4_roll_v4a.rds"))

roll_v4b <- run_variant("V4b (V4a + drop model 4 second-order stack tensors)", .make_ws4_trainer("v4b"))
saveRDS(roll_v4b, file.path(RESULTS_DIR, "ws4_roll_v4b.rds"))

roll_v4c <- run_variant("V4c (flat single-GAM chain for score_diff)", .make_ws4_trainer("v4c"))
saveRDS(roll_v4c, file.path(RESULTS_DIR, "ws4_roll_v4c.rds"))

# Metrics ----
variants <- list(
  Baseline = roll_baseline,
  V4a      = roll_v4a,
  V4b      = roll_v4b,
  V4c      = roll_v4c
)

metrics_table <- purrr::imap_dfr(variants, function(roll, label) {
  gam_m <- .compute_metrics(roll$gam_preds)
  ib_m  <- .compute_metrics(roll$input_blend_preds)
  dplyr::bind_rows(
    data.frame(Variant = label, Model = "GAM-only", N = nrow(roll$gam_preds),
               MAE = gam_m$mae, RMSE = gam_m$rmse, Brier = gam_m$brier,
               Slope = gam_m$slope, Cor = gam_m$cor, SDRatio = gam_m$sd_ratio,
               CloseMAE = gam_m$close_mae, CloseN = gam_m$close_n),
    data.frame(Variant = label, Model = "Input Blend", N = nrow(roll$input_blend_preds),
               MAE = ib_m$mae, RMSE = ib_m$rmse, Brier = ib_m$brier,
               Slope = ib_m$slope, Cor = ib_m$cor, SDRatio = ib_m$sd_ratio,
               CloseMAE = ib_m$close_mae, CloseN = ib_m$close_n)
  )
})
metrics_table[ , -(1:2)] <- round(metrics_table[ , -(1:2)], 4)

cat("\n=== WS4 Screening Results (2026 rolling OOS) ===\n")
print(metrics_table, row.names = FALSE)
write.csv(metrics_table, file.path(RESULTS_DIR, "ws4_metrics_2026.csv"), row.names = FALSE)

# Margin calibration by predicted-margin bucket, GAM-only (attribution view) ----
cat("\n=== Margin Calibration by Predicted-Margin Bucket (GAM-only) ===\n")
bucket_table <- purrr::imap_dfr(variants, function(roll, label) {
  margin_calibration_by_pred_bucket(roll$gam_preds) |> dplyr::mutate(Variant = label, .before = 1)
})
print(as.data.frame(bucket_table), row.names = FALSE)
write.csv(bucket_table, file.path(RESULTS_DIR, "ws4_bucket_2026.csv"), row.names = FALSE)

# Bootstrap CIs vs baseline, both on GAM-only (attribution) and Input Blend (ship gate) ----
cat("\n=== boot_mae_diff() vs Baseline (2026) ===\n")
boot_results <- list()
for (v in c("V4a", "V4b", "V4c")) {
  b_gam <- boot_mae_diff(variants[[v]]$gam_preds, roll_baseline$gam_preds, B = 2000)
  b_ib  <- boot_mae_diff(variants[[v]]$input_blend_preds, roll_baseline$input_blend_preds, B = 2000)
  boot_results[[v]] <- list(gam = b_gam, ib = b_ib)
  cat(sprintf(
    "%s  GAM-only:  dMAE=%+.3f  95%%CI[%+.3f, %+.3f]   Input Blend: dMAE=%+.3f  95%%CI[%+.3f, %+.3f]\n",
    v, b_gam$mae_diff, b_gam$mae_ci[1], b_gam$mae_ci[2],
    b_ib$mae_diff, b_ib$mae_ci[1], b_ib$mae_ci[2]
  ))
}
saveRDS(boot_results, file.path(RESULTS_DIR, "ws4_boot_results.rds"))

# Attribution summary: which variant moves GAM-only slope closest to 1? ----
base_gam_m <- .compute_metrics(roll_baseline$gam_preds)
cat("\n=== WS4 Attribution Summary ===\n")
cat(sprintf("Baseline GAM-only slope: %.3f (Input Blend slope: %.3f)\n",
            base_gam_m$slope, .compute_metrics(roll_baseline$input_blend_preds)$slope))
for (v in c("V4a", "V4b", "V4c")) {
  v_gam_m <- .compute_metrics(variants[[v]]$gam_preds)
  v_ib_m  <- .compute_metrics(variants[[v]]$input_blend_preds)
  cat(sprintf(
    "%s  GAM-only slope: %.3f (delta %+.3f from baseline)  MAE %.2f (delta %+.3f)  | Input Blend slope: %.3f  MAE %.2f (delta %+.3f)\n",
    v, v_gam_m$slope, v_gam_m$slope - base_gam_m$slope, v_gam_m$mae, boot_results[[v]]$gam$mae_diff,
    v_ib_m$slope, v_ib_m$mae, boot_results[[v]]$ib$mae_diff
  ))
}

cli::cli_alert_success("WS4 screening complete. Results in {RESULTS_DIR}")
