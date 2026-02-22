#' torpmodels: Pre-trained Models for AFL Analytics
#'
#' The torpmodels package provides pre-trained machine learning models for
#' Australian Football League (AFL) analytics. Models are distributed via
#' GitHub releases and cached locally after first download.
#'
#' @section Core Models:
#' \itemize{
#'   \item **ep_model** - Expected Points XGBoost multiclass model
#'   \item **wp_model** - Win Probability model
#'   \item **shot_ocat_mdl** - Shot outcome ordered categorical model
#'   \item **xgb_win_model** - XGBoost match prediction model
#' }
#'
#' @section Stat Models:
#' 58 per-statistic GAM models for predicting individual player statistics
#' (goals, disposals, tackles, etc.)
#'
#' @section Usage:
#' \preformatted{
#' # Load a core model
#' ep_model <- load_torp_model("ep")
#'
#' # Load a stat model
#' goals_model <- load_stat_model("goals")
#'
#' # Check what's cached
#' check_model_cache()
#' }
#'
#' @keywords internal
"_PACKAGE"
