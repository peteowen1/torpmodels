# WS2 -- team "style" features from box-score stats
# =================================================
# FABLE-MATCH-FEATURES-PLAN.md WS2. The match model's feature set is entirely
# player-rating aggregates, travel/rest/weather, static team random effects and
# one team rating. WS1 established that the team-rating family is SATURATED:
# five structurally different estimators (win-based, points-based, xScore,
# dual-timescale, offence/defence) all converge to cor 0.55-0.56 standalone and
# the full model subsumes all of them. So further gains have to come from
# information outside "team results + xScore history".
#
# The hypothesis: player-rating aggregates measure WHO IS ON THE PARK. These
# measure HOW THE TEAM PLAYS -- contest dominance, territory, pressure, ball
# use. Different question, plausibly what separates evenly-matched sides.
#
# LEAK SAFETY is the whole game here, and it is easy to get wrong:
#   - a team's profile for match M uses only matches STRICTLY BEFORE M;
#   - exponential decay by days, so recent form dominates without a hard window;
#   - league centring uses the same strictly-prior set, so era drift (scoring
#     rates move a lot across seasons) cannot leak backwards.
# Every feature is a DIFFERENTIAL (team minus opponent) on a per-game rate, not
# a raw total: totals encode how many disposals a game happened to have, which
# is a pace artifact rather than a style signal.

suppressMessages({ library(data.table) })

# The metric set. Deliberately small and spread across distinct football
# concepts -- a wide net invites the model to fit noise, and the G7 gate asks
# for incremental signal over an Elo baseline, not for maximum R2.
WS2_METRICS <- c(
  contested_possession_rate = "contested_possession_rate",  # contest dominance
  centre_clearances         = "centre_clearances",          # stoppage: centre
  stoppage_clearances       = "stoppage_clearances",        # stoppage: around ground
  inside50s                 = "inside50s",                  # territory
  marks_inside50            = "marks_inside50",             # territory quality
  intercepts                = "intercepts",                 # defensive interception
  pressure_acts             = "pressure_acts",              # pressure applied
  def_half_pressure_acts    = "def_half_pressure_acts",     # pressure, defensive half
  disposal_efficiency       = "disposal_efficiency",        # ball use
  metres_gained             = "metres_gained",              # ball movement
  turnovers                 = "turnovers",                  # giveaways
  score_launches            = "score_launches"              # scoring chains generated
)

# ws2_team_game_stats ----

#' Aggregate player box scores to one row per team-match
#'
#' Rate metrics (`*_rate`, `*_efficiency`) are averaged weighted by time on
#' ground; counts are summed. Summing a rate would be meaningless, and
#' averaging a count would erase the thing being measured.
#'
#' @param ps Player stats from load_player_stats()
#' @return data.table(match_id, team, season, round, date, <metrics>)
ws2_team_game_stats <- function(ps) {
  dt <- as.data.table(ps)
  have <- intersect(unname(WS2_METRICS), names(dt))
  missing <- setdiff(unname(WS2_METRICS), names(dt))
  if (length(missing) > 0) {
    cli::cli_warn("WS2: {length(missing)} metric{?s} absent from player_stats: {.val {missing}}")
  }
  if (length(have) == 0) cli::cli_abort("WS2: none of the requested metrics are present")

  # player_stats identifies a player's team by team_id (CD_T120-style codes) plus
  # team_status, NOT by name. The match model joins on canonical names
  # (team_name.x), so resolve status against home_team_name/away_team_name --
  # those are already canonical and were verified to match team_mdl_df's 18
  # levels exactly, in both directions, with zero unmatched.
  if (all(c("team_status", "home_team_name", "away_team_name") %in% names(dt))) {
    st <- tolower(as.character(dt$team_status))
    if (!all(st %in% c("home", "away"))) {
      cli::cli_abort("WS2: unexpected team_status {.val {setdiff(unique(st), c('home','away'))}}")
    }
    dt[, .team := fifelse(st == "home",
                          as.character(home_team_name), as.character(away_team_name))]
  } else {
    tm_col <- intersect(c("team_name", "team", "team_abbr"), names(dt))[1]
    if (is.na(tm_col)) cli::cli_abort("WS2: cannot resolve a team column in player_stats")
    dt[, .team := as.character(get(tm_col))]
  }

  tog <- if ("time_on_ground_percentage" %in% names(dt)) {
    pmax(as.numeric(dt$time_on_ground_percentage), 1) / 100
  } else rep(1, nrow(dt))
  dt[, .tog := tog]

  # Take dates from player_stats, not from the model frame: history should span
  # every match with a box score (1233), not only those in team_mdl_df (1157).
  if (!"utc_start_time" %in% names(dt)) cli::cli_abort("WS2: player_stats has no utc_start_time")
  dt[, .date := as.Date(substr(as.character(utc_start_time), 1, 10))]
  if (anyNA(dt$.date)) cli::cli_abort("WS2: {sum(is.na(dt$.date))} unparseable utc_start_time value{?s}")

  is_rate <- grepl("_rate$|_efficiency$", have)
  agg <- dt[, c(
    lapply(have[!is_rate], function(cc) sum(as.numeric(get(cc)), na.rm = TRUE)),
    lapply(have[is_rate],  function(cc) stats::weighted.mean(as.numeric(get(cc)), .tog, na.rm = TRUE))
  ), by = .(match_id, team = .team, season, round = round_number, date = .date)]
  setnames(agg, c("match_id", "team", "season", "round", "date", have[!is_rate], have[is_rate]))
  agg[]
}

# ws2_style_profiles ----

#' Decayed, league-centred, strictly-prior team style profiles
#'
#' For each (match, team), the decay-weighted mean of that team's metric over
#' matches BEFORE this one, expressed relative to the league mean over the same
#' strictly-prior set. A team with no prior history gets 0 (league average),
#' which is the honest "we know nothing yet" value rather than an extrapolation.
#'
#' TARGETS vs HISTORY are deliberately separate sets. A profile for match M is
#' built from matches strictly BEFORE M, so M's own box score is never needed --
#' and must not be, since in production every prediction target is a match that
#' has not been played. Keying profiles off the history set instead silently
#' handed league-average (0) features to the 36 team_mdl_df matches with no
#' player_stats rows yet, which is exactly the population we most care about.
#'
#' @param tg History: output of ws2_team_game_stats(), joined to a `date`.
#' @param targets data.table(match_id, team, date) -- rows needing a profile.
#' @param half_life_days Decay half-life. 180d ~ one season, so a team's
#'   profile is dominated by the current season while retaining some carry-in.
#' @return data.table(match_id, team, <metric>_prof ...)
ws2_style_profiles <- function(tg, targets, half_life_days = 180) {
  h <- as.data.table(tg); tg_dt <- as.data.table(targets)
  setorder(h, date, match_id)
  metrics <- setdiff(names(h), c("match_id", "team", "season", "round", "date"))
  lambda <- log(2) / half_life_days

  # exp(-lambda*(d0 - dj)) = exp(-lambda*d0) * exp(lambda*dj), and the leading
  # factor cancels in any weighted MEAN. So the decayed mean over "all history
  # strictly before d0" is a ratio of PREFIX SUMS of u_j = exp(lambda*dj) --
  # O(n log n) instead of re-filtering the whole table once per row. Dates are
  # offset to their own minimum purely to keep exp() in a safe range.
  hd <- as.numeric(h$date); d0 <- as.numeric(tg_dt$date)
  dmin <- min(c(hd, d0), na.rm = TRUE)
  u <- exp(lambda * (hd - dmin))

  # Count of history rows strictly earlier than each target date. left.open
  # makes the interval (v[i], v[i+1]], so the returned index IS that count --
  # same-day matches are excluded, which is the leak guard.
  k_all <- findInterval(d0, hd, left.open = TRUE)

  teams <- unique(c(h$team, tg_dt$team))
  h_idx <- split(seq_len(nrow(h)), factor(h$team, levels = teams))

  out <- data.table(match_id = tg_dt$match_id, team = tg_dt$team)
  for (m in metrics) {
    v <- as.numeric(h[[m]]); ok <- !is.na(v); v[!ok] <- 0
    cs_uv <- c(0, cumsum(u * v)); cs_u <- c(0, cumsum(u * ok))   # 1-based via the 0 pad
    lg <- ifelse(cs_u[k_all + 1L] > 0, cs_uv[k_all + 1L] / cs_u[k_all + 1L], NA_real_)

    own <- rep(NA_real_, nrow(tg_dt))
    for (tmn in teams) {
      ii <- h_idx[[tmn]]
      sel <- which(tg_dt$team == tmn)
      if (length(sel) == 0) next
      if (length(ii) == 0) next                 # team with no history at all
      ut <- u[ii]; vt <- v[ii]; okt <- ok[ii]
      csv <- c(0, cumsum(ut * vt)); csu <- c(0, cumsum(ut * okt))
      kt <- findInterval(d0[sel], hd[ii], left.open = TRUE)
      own[sel] <- ifelse(csu[kt + 1L] > 0, csv[kt + 1L] / csu[kt + 1L], NA_real_)
    }
    p <- own - lg
    p[is.na(p)] <- 0                            # no prior history == league average
    out[[paste0(m, "_prof")]] <- p
  }
  out[]
}

#' Reference implementation of ws2_style_profiles (slow, obviously-correct)
#'
#' Kept solely so the vectorised version above can be verified against a direct
#' transcription of the definition. This is leak-sensitive code where a prefix-sum
#' off-by-one would shift the boundary from "strictly before" to "including
#' today" and quietly leak the match being predicted -- a bug that would show up
#' as a *better* score, not an error. Not used in the scoring path.
ws2_style_profiles_ref <- function(tg, targets, half_life_days = 180) {
  h <- as.data.table(tg); tg_dt <- as.data.table(targets)
  metrics <- setdiff(names(h), c("match_id", "team", "season", "round", "date"))
  lambda <- log(2) / half_life_days
  out <- vector("list", nrow(tg_dt))
  for (i in seq_len(nrow(tg_dt))) {
    d0 <- tg_dt$date[i]; tm <- tg_dt$team[i]
    prior <- h[date < d0]                       # STRICTLY before -- the leak guard
    row <- list(match_id = tg_dt$match_id[i], team = tm)
    if (nrow(prior) == 0) {
      for (m in metrics) row[[paste0(m, "_prof")]] <- 0
    } else {
      w_all <- exp(-lambda * as.numeric(d0 - prior$date))
      own <- prior$team == tm
      for (m in metrics) {
        v <- as.numeric(prior[[m]])
        league <- stats::weighted.mean(v, w_all, na.rm = TRUE)
        row[[paste0(m, "_prof")]] <- if (!any(own)) 0 else {
          tv <- stats::weighted.mean(v[own], w_all[own], na.rm = TRUE)
          if (is.na(tv) || is.na(league)) 0 else tv - league
        }
      }
    }
    out[[i]] <- row
  }
  rbindlist(out)
}

# ws2_join_style_diffs ----

#' Join style profiles onto team_mdl_df as symmetric differentials
#'
#' Same convention as elo_diff/xelo_diff: this team's profile minus its
#' opponent's, so the feature is antisymmetric between the two rows of a match
#' and carries no home-advantage component (team_type_fac already does).
#'
#' @param team_mdl_df Complete model dataset
#' @param profiles Output of ws2_style_profiles()
#' @return team_mdl_df with `<metric>_sdiff` columns added
ws2_join_style_diffs <- function(team_mdl_df, profiles) {
  p <- as.data.table(profiles)
  prof_cols <- setdiff(names(p), c("match_id", "team"))
  key <- function(mid, tm) paste(mid, tm)
  lut <- lapply(prof_cols, function(cc) stats::setNames(p[[cc]], key(p$match_id, p$team)))
  names(lut) <- prof_cols

  tx <- as.character(team_mdl_df$team_name.x)
  ty <- as.character(team_mdl_df$team_name.y)
  kx <- key(team_mdl_df$match_id, tx); ky <- key(team_mdl_df$match_id, ty)

  n_miss <- 0L
  for (cc in prof_cols) {
    vx <- unname(lut[[cc]][kx]); vy <- unname(lut[[cc]][ky])
    n_miss <- n_miss + sum(is.na(vx) | is.na(vy))
    vx[is.na(vx)] <- 0; vy[is.na(vy)] <- 0
    team_mdl_df[[sub("_prof$", "_sdiff", cc)]] <- vx - vy
  }
  if (n_miss > 0) {
    cli::cli_inform("ws2: {n_miss} profile lookup{?s} missing across {length(prof_cols)} metric{?s} -- treated as league-average (0)")
  }
  team_mdl_df
}
