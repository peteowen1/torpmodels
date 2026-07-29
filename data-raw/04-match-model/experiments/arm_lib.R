# Shared plumbing for multi-arm rating experiments
# ===============================================
# Every experiment that scores N rating variants (WS3 v1-vs-v2, WS4 position
# centring, ...) rebuilds team_mdl_df once per arm via its own copy of a
# build_with_ratings() helper. Only `torp_df` differs between arms; the other
# seven inputs -- stadiums, xG, fixtures, results, teams, PSR, weather -- are
# identical, and one of them is a ~193 MB download. A two-arm run therefore
# spends ~5 minutes re-fetching bytes it already has, and an N-arm sweep spends
# N times that.
#
# It is also a correctness improvement, not only a speed one: hoisting the loads
# makes it PROVABLE that every arm saw identical inputs, rather than merely
# likely. Arms that each load their own "identical" copy are only identical if
# nothing moved between the loads.
#
# Usage:
#   src <- load_match_inputs()                     # once
#   tm1 <- build_team_mdl_with(src, ratings_a)
#   tm2 <- build_team_mdl_with(src, ratings_b)

suppressMessages({ library(dplyr) })

#' Load every match-model input that does NOT depend on the rating vintage
#' @return list of the shared inputs, plus `loaded_at` for provenance
load_match_inputs <- function() {
  t0 <- Sys.time()
  all_grounds <- file_reader("stadium_data", "reference-data")
  xg_df     <- load_xg(TRUE)
  fixtures  <- load_fixtures(TRUE)
  results   <- load_results(TRUE)
  teams     <- load_teams(TRUE)
  psr_df <- tryCatch(.compute_psr_from_stat_ratings(load_player_stat_ratings(TRUE)),
                     error = function(e) { cli::cli_warn("PSR: {conditionMessage(e)}"); NULL })
  fix_df <- .build_fixtures_df(fixtures)
  weather_df <- .load_match_weather(fixtures, all_grounds, NULL, get_afl_season())
  anchor <- max(as.Date(fix_df$utc_start_time), na.rm = TRUE)
  cli::cli_alert_success(
    "Shared match inputs loaded in {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  list(all_grounds = all_grounds, xg_df = xg_df, fixtures = fixtures,
       results = results, teams = teams, psr_df = psr_df, fix_df = fix_df,
       weather_df = weather_df, anchor = anchor, loaded_at = t0)
}

#' Build team_mdl_df for one rating vintage, reusing pre-loaded shared inputs
#'
#' Mirrors torp's build_team_mdl_df() exactly; the ONLY difference is that the
#' rating-independent inputs arrive pre-loaded instead of being re-fetched.
#' @param src Output of load_match_inputs()
#' @param torp_df The rating frame for this arm
build_team_mdl_with <- function(src, torp_df) {
  team_rt_df <- .build_team_ratings_df(src$teams, torp_df, src$psr_df)
  team_rt_fix_df <- .build_match_features(src$fix_df, team_rt_df, src$all_grounds)
  .build_team_mdl_df(team_rt_fix_df, src$results, src$xg_df, src$weather_df, src$anchor)
}

#' Score several arms on the rolling harness with one consistent path
#'
#' `parallel = TRUE` runs every arm through run_rolling_eval_parallel(). That is
#' safe ONLY because all arms in one comparison share the path and thread count,
#' so xgboost's thread-count nondeterminism shifts them together and largely
#' cancels in the arm-vs-arm delta. NEVER compare an arm scored here with
#' parallel = TRUE against a champion scored sequentially -- the divergence is
#' structural (tree_method = "hist"), ~0.1-0.25 MAE, and sized like a real
#' effect. Re-confirm any winner sequentially before shipping.
score_arms <- function(arms, test_seasons, extra_feature_cols = "xelo_diff",
                       parallel = FALSE, n_workers = NULL, verbose = FALSE) {
  out <- list()
  for (nm in names(arms)) {
    cli::cli_h2("scoring arm: {nm}")
    t0 <- Sys.time()
    out[[nm]] <- if (isTRUE(parallel)) {
      run_rolling_eval_parallel(arms[[nm]], test_seasons = test_seasons,
                                extra_feature_cols = extra_feature_cols,
                                cv_extra_feature_cols = extra_feature_cols,
                                n_workers = n_workers, verbose = verbose)
    } else {
      run_rolling_eval(arms[[nm]], test_seasons = test_seasons,
                       extra_feature_cols = extra_feature_cols,
                       cv_extra_feature_cols = extra_feature_cols,
                       verbose = verbose)
    }
    cli::cli_inform("{nm}: {round(difftime(Sys.time(), t0, units='mins'), 2)} min")
  }
  attr(out, "parallel") <- isTRUE(parallel)
  out
}
