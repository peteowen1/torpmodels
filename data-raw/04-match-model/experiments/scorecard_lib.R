# Standard scorecard for match-prediction arms
# ============================================
# Every eval should report the same metric set against the same reference points,
# so results are comparable across sessions and nobody has to remember which
# columns a given script happened to print.
#
# Metrics: MAE, RMSE, bits, logloss, Brier, accuracy.
# References, always included:
#   - the current production-configuration arm
#   - Squiggle Aggregate over the SAME games (the external bar)
#   - a trivial baseline: margin 0 and p = 0.5 for every game
#
# The trivial baseline matters more than it looks. It pins the scale: bits is 0
# and Brier 0.25 BY CONSTRUCTION for a 50% tip, so any model's bits/Brier is
# read directly as "better than knowing nothing". A model that cannot beat it is
# not a weak model, it is a broken one -- and on MAE the bar is higher than
# instinct suggests, because predicting 0 every week is a genuinely respectable
# ~30 points in AFL.

suppressMessages({ library(data.table); library(jsonlite) })

SQ_TEAM_MAP <- c(
  "Adelaide" = "Adelaide Crows", "Brisbane Lions" = "Brisbane Lions",
  "Carlton" = "Carlton Blues", "Collingwood" = "Collingwood Magpies",
  "Essendon" = "Essendon Bombers", "Fremantle" = "Fremantle Dockers",
  "Geelong" = "Geelong Cats", "Gold Coast" = "Gold Coast Suns",
  "Greater Western Sydney" = "GWS Giants", "Hawthorn" = "Hawthorn Hawks",
  "Melbourne" = "Melbourne Demons", "North Melbourne" = "North Melbourne Kangaroos",
  "Port Adelaide" = "Port Adelaide Power", "Richmond" = "Richmond Tigers",
  "St Kilda" = "St Kilda Saints", "Sydney" = "Sydney Swans",
  "West Coast" = "West Coast Eagles", "Western Bulldogs" = "Western Bulldogs")

.sq_get <- function(q) {
  con <- url(sprintf("https://api.squiggle.com.au/?q=%s", q),
             headers = c("User-Agent" = "torpverse-scorecard (fptpost@gmail.com)"))
  on.exit(try(close(con), silent = TRUE))
  jsonlite::fromJSON(paste(readLines(con, warn = FALSE), collapse = ""))
}

#' Fetch one Squiggle source's tips, keyed to join against harness predictions
#'
#' @param seasons Integer vector of seasons.
#' @param src_name Squiggle source name, e.g. "Aggregate". NOT named `source`:
#'   the tips table has a `source` column, and a parameter of the same name is
#'   shadowed by it inside `[`. `..source` does not rescue this -- that prefix
#'   works for column SELECTION in `j`, not for a value lookup in `i` -- and the
#'   failure is a silent "object '..source' not found" that drops every Squiggle
#'   row from the scorecard rather than erroring loudly.
#' @return data.table(season, round, home_team, pred_margin, pred_win, margin, home_win)
fetch_squiggle_source <- function(seasons, src_name = "Aggregate") {
  want <- as.character(src_name)   # plain vector, no NSE ambiguity
  out <- lapply(seasons, function(y) {
    g <- as.data.table(.sq_get(sprintf("games;year=%d", y))$games)[
          complete == 100, .(gameid = as.integer(id), round, hteam, hmargin = hscore - ascore)]
    ti <- as.data.table(.sq_get(sprintf("tips;year=%d", y))$tips)
    ti <- ti[source %chin% want, .(gameid = as.integer(gameid), tip, hteam_t = hteam,
                                   margin = as.numeric(margin),
                                   hconf = as.numeric(hconfidence))]
    if (nrow(ti) == 0) stop("no Squiggle tips found for source '", want, "' in ", y)
    d <- merge(ti, g, by = "gameid")
    bad <- setdiff(unique(d$hteam), names(SQ_TEAM_MAP))
    if (length(bad)) stop("unmapped Squiggle team(s): ", paste(bad, collapse = ", "))
    d[, .(season = y, round, home_team = unname(SQ_TEAM_MAP[hteam]),
          pred_margin = ifelse(tip == hteam_t, margin, -margin),
          pred_win = pmin(pmax(hconf / 100, 1e-6), 1 - 1e-6),
          margin = hmargin,
          home_win = fifelse(hmargin > 0, 1, fifelse(hmargin == 0, 0.5, 0)))]
  })
  rbindlist(out)
}

#' Score one set of predictions
#' @param d data.table with pred_margin, pred_win, margin, home_win
.metrics <- function(d) {
  p  <- pmin(pmax(d$pred_win, 1e-6), 1 - 1e-6)
  hw <- d$home_win
  bits <- fifelse(hw == 1, 1 + log2(p),
           fifelse(hw == 0, 1 + log2(1 - p), 1 + 0.5 * log2(p * (1 - p))))
  ll <- -(hw * log(p) + (1 - hw) * log(1 - p))
  # Accuracy: a correct tip; a drawn game, or a prediction of exactly 0, scores 0.5.
  correct <- fifelse(d$margin == 0 | d$pred_margin == 0, 0.5,
              fifelse(sign(d$pred_margin) == sign(d$margin), 1, 0))
  data.table(n = nrow(d),
             MAE      = mean(abs(d$pred_margin - d$margin)),
             RMSE     = sqrt(mean((d$pred_margin - d$margin)^2)),
             bits     = mean(bits),
             logloss  = mean(ll),
             Brier    = mean((p - hw)^2),
             accuracy = mean(correct))
}

#' Build the full scorecard for a set of arms
#'
#' @param arms Named list of prediction frames (harness `input_blend_preds`
#'   shape: season, round, home_team, pred_margin, pred_win, margin, home_win).
#' @param squiggle_sources Character vector of Squiggle sources to include.
#' @param restrict_to_common If TRUE (default) every row set is reduced to the
#'   games ALL sources share, so the comparison is like-for-like. This matters:
#'   Squiggle only covers completed games it knows about, and an unmatched
#'   comparison silently scores different fixtures against each other.
#' @return data.table, one row per source, sorted by MAE
scorecard <- function(arms, squiggle_sources = "Aggregate",
                      restrict_to_common = TRUE, verbose = TRUE) {
  key <- c("season", "round", "home_team")
  arms <- lapply(arms, function(a) {
    d <- as.data.table(a)
    stopifnot(all(c(key, "pred_margin", "pred_win", "margin", "home_win") %in% names(d)))
    d[, c(key, "pred_margin", "pred_win", "margin", "home_win"), with = FALSE]
  })
  seasons <- sort(unique(unlist(lapply(arms, function(d) unique(d$season)))))
  for (s in squiggle_sources) {
    arms[[s]] <- tryCatch(fetch_squiggle_source(seasons, src_name = s),
                          error = function(e) { cli::cli_warn("Squiggle {s}: {conditionMessage(e)}"); NULL })
  }
  dropped <- names(arms)[vapply(arms, is.null, logical(1))]
  if (length(dropped)) {
    # Loud, because a silently-missing external reference turns the scorecard
    # into a self-comparison while still looking complete.
    cli::cli_warn("Scorecard is MISSING {length(dropped)} reference source{?s}: {.val {dropped}}")
  }
  arms <- arms[!vapply(arms, is.null, logical(1))]

  if (isTRUE(restrict_to_common)) {
    common <- Reduce(function(x, y) merge(x, y, by = key),
                     lapply(arms, function(d) unique(d[, ..key])))
    if (verbose) cli::cli_inform("common games across {length(arms)} source{?s}: {nrow(common)}")
    arms <- lapply(arms, function(d) merge(d, common, by = key))
  }

  # Trivial reference, built on the SAME games so it is directly comparable.
  base <- copy(arms[[1]])[, `:=`(pred_margin = 0, pred_win = 0.5)]
  arms[["0 margin / 50%"]] <- base

  res <- rbindlist(lapply(names(arms), function(nm) cbind(source = nm, .metrics(arms[[nm]]))))
  res[order(MAE)]
}

#' Print a scorecard with sensible rounding
print_scorecard <- function(sc, title = "SCORECARD") {
  x <- copy(sc)
  for (c in c("MAE","RMSE")) x[[c]] <- round(x[[c]], 3)
  for (c in c("bits","logloss","Brier")) x[[c]] <- round(x[[c]], 4)
  x$accuracy <- paste0(round(100 * x$accuracy, 1), "%")
  cat("\n===== ", title, " =====\n", sep = "")
  print(x, row.names = FALSE)
  cat("\n(bits: higher better, 0 = a coin flip. logloss/Brier/MAE/RMSE: lower better.)\n")
  invisible(x)
}
