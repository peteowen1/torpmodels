# WS0 — Incremental-signal gate (G7/G8)
# =====================================
# FABLE-MATCH-FEATURES-PLAN.md WS0. Prerequisite for every round-3 workstream.
#
# WHY THIS EXISTS
# ---------------
# Rounds 1-2 scored every candidate on ΔMAE against the reigning champion, with
# a block-bootstrap CI. That gate is necessary but not sufficient: it cannot
# distinguish "this candidate knows something new" from "this candidate
# re-derives a plain team-Elo rating slightly more accurately".
#
# The 2026-07-28 diagnosis established that torp's live margin predictions add
# NOTHING on top of a results-only team-Elo system: regressing
# `actual ~ Wheelo + In The Game` over the 171 completed 2026 games gives torp
# a coefficient of -0.063, and an honest expanding-window blend of the two
# (24.80) is WORSE than Wheelo alone (24.74). A model can improve its MAE
# indefinitely while staying inside Elo's information set, and none of that
# progress is worth anything.
#
# So: G7 measures a candidate against an ELO BASELINE on incremental
# information, not against the champion on MAE.
#
# See docs/reviews/2026-MATCH-SIGNAL-REDUNDANCY-DIAGNOSIS.md §4.
#
# USAGE
#   source("experiments/rolling_lib.R")
#   source("experiments/signal_gate.R")
#   base <- elo_baseline_preds(team_mdl_df, test_seasons = 2025:2026)
#   roll <- run_rolling_eval(team_mdl_df, test_seasons = 2025:2026)
#   signal_gate_report(roll$input_blend_preds, base, label = "champion (Input Blend)")
#
# All prediction sets use the harness's own one-row-per-match schema
# (.format_match_preds() + run_rolling_eval()'s home_win mutate): columns
# season, round, match_id, pred_margin, pred_win, margin, home_win.

# elo_baseline_preds ----

#' Standalone team-Elo margin predictor, in harness prediction-set shape
#'
#' The reference every round-3 candidate is measured against. Built from
#' production's own `torp:::build_team_elo()` so the baseline is the Elo torp
#' actually ships, not a research variant.
#'
#' Leak safety has two independent parts and both are enforced:
#' 1. `elo_pre` is a function only of strictly-earlier matches, by
#'    construction (see torp/R/team_elo.R) — so the ratings table can be built
#'    once over all matches without leaking.
#' 2. The points-per-Elo scale is NOT global. It is refit for every test round
#'    on completed matches strictly before that round, mirroring the rolling
#'    harness's own discipline (plan G6). A single global scale fit would leak
#'    the test window's own margin distribution into the baseline and make the
#'    baseline look better than it is — which, since the baseline is the thing
#'    candidates must BEAT, would make G7 spuriously hard to pass.
#'
#' @param team_mdl_df Complete model dataset from .build_team_mdl_df()
#' @param test_seasons Integer vector of seasons to produce predictions for
#'   (same value passed to run_rolling_eval(), so the two prediction sets
#'   cover the same matches)
#' @param k,hga,carryover,mov_mult Elo hyperparameters; default to production's
#'   shipped constants (closed 2026-07-14 — re-tuning them is a non-goal)
#' @param min_train_matches Minimum prior completed matches before the scale is
#'   fitted; below this the round is skipped (cannot form an honest baseline)
#' @return tibble(season, round, match_id, pred_margin, pred_win, margin,
#'   home_win), one row per match, home-team perspective
#' @keywords internal
elo_baseline_preds <- function(team_mdl_df,
                               test_seasons,
                               k = torp:::ELO_K,
                               hga = torp:::ELO_HGA,
                               carryover = torp:::ELO_CARRYOVER,
                               mov_mult = TRUE,
                               min_train_matches = 100L) {
  matches <- torp:::.matches_from_team_mdl_df(team_mdl_df)
  matches <- matches[!is.na(matches$home_margin), ]

  elo <- torp:::build_team_elo(matches, k = k, hga = hga,
                               carryover = carryover, mov_mult = mov_mult)
  by_match <- elo$by_match

  # Rating lookup keyed on "match_id team_name" (avoids factor-join hazards —
  # the same approach elo_lib.R::tune_team_elo() uses).
  key <- stats::setNames(by_match$elo_pre,
                         paste(by_match$match_id, by_match$team_name))
  matches$elo_diff_home <- unname(
    key[paste(matches$match_id, matches$home_team)] -
      key[paste(matches$match_id, matches$away_team)]
  )

  # Order matches so "strictly prior" is well defined across the season
  # boundary as well as within a season.
  matches <- matches[order(matches$season, matches$round, matches$date,
                           matches$match_id), ]

  test_idx <- which(matches$season %in% test_seasons)
  if (length(test_idx) == 0) {
    cli::cli_abort("elo_baseline_preds: no completed matches in test_seasons {test_seasons}")
  }

  # Group test matches by (season, round) and refit the scale per round on
  # everything strictly before that round.
  test_rounds <- unique(matches[test_idx, c("season", "round")])
  out <- vector("list", nrow(test_rounds))

  for (i in seq_len(nrow(test_rounds))) {
    s <- test_rounds$season[i]
    r <- test_rounds$round[i]

    prior <- matches$season < s | (matches$season == s & matches$round < r)
    this  <- matches$season == s & matches$round == r
    if (sum(prior) < min_train_matches) next

    fit <- elo_lib_fit_scale(matches$elo_diff_home[prior], hga,
                             matches$home_margin[prior])
    ed  <- matches$elo_diff_home[this]

    out[[i]] <- tibble::tibble(
      season      = s,
      round       = r,
      match_id    = matches$match_id[this],
      pred_margin = unname(stats::predict(
        fit, newdata = data.frame(elo_diff_hga = ed + hga))),
      pred_win    = 1 / (1 + 10^(-(ed + hga) / 400)),
      margin      = matches$home_margin[this]
    )
  }

  res <- dplyr::bind_rows(out)
  if (nrow(res) == 0) {
    cli::cli_abort("elo_baseline_preds: every test round fell below min_train_matches ({min_train_matches})")
  }
  res$home_win <- ifelse(res$margin > 0, 1, ifelse(res$margin == 0, 0.5, 0))
  res
}

# elo_lib_fit_scale ----

#' Points-per-Elo scale (through the origin)
#'
#' Local copy of elo_lib.R::fit_elo_margin_scale() so signal_gate.R can be
#' sourced without elo_lib.R (whose build_team_elo() has a different return
#' shape from production's and would shadow it depending on source order).
#'
#' @keywords internal
elo_lib_fit_scale <- function(elo_diff_home, hga, home_margin) {
  stats::lm(home_margin ~ elo_diff_hga + 0,
            data = data.frame(elo_diff_hga = elo_diff_home + hga,
                              home_margin = home_margin))
}

# incremental_signal ----

#' Does a candidate carry information the baseline does not?
#'
#' Fits `margin ~ baseline_pred + candidate_pred` on the matches both cover and
#' reports what the candidate adds. This is a static (whole-window) fit — it is
#' the *diagnostic*; rolling_blend_check() is the honest out-of-sample version.
#' Report both: a candidate that looks good here but fails there is fitting the
#' window.
#'
#' Reading the output: `beta_candidate` near zero (or negative) with a
#' non-significant p-value means the candidate is inside the baseline's
#' information set, regardless of how good its standalone MAE looks.
#'
#' @param candidate_preds,baseline_preds Harness-shaped prediction sets
#' @return list of r2_baseline, r2_candidate, r2_joint, delta_r2,
#'   beta_baseline, beta_candidate, p_candidate, cor_preds, n_matches
#' @keywords internal
incremental_signal <- function(candidate_preds, baseline_preds) {
  j <- .join_pred_sets(candidate_preds, baseline_preds)

  m_base <- stats::lm(margin ~ pred_base, data = j)
  m_cand <- stats::lm(margin ~ pred_cand, data = j)
  m_both <- stats::lm(margin ~ pred_base + pred_cand, data = j)
  sm     <- summary(m_both)

  list(
    n_matches      = nrow(j),
    r2_baseline    = summary(m_base)$r.squared,
    r2_candidate   = summary(m_cand)$r.squared,
    r2_joint       = sm$r.squared,
    delta_r2       = sm$r.squared - summary(m_base)$r.squared,
    beta_baseline  = unname(stats::coef(m_both)[["pred_base"]]),
    beta_candidate = unname(stats::coef(m_both)[["pred_cand"]]),
    p_candidate    = unname(sm$coefficients["pred_cand", "Pr(>|t|)"]),
    cor_preds      = stats::cor(j$pred_cand, j$pred_base)
  )
}

# rolling_blend_check ----

#' Honest expanding-window test: does adding the candidate to the baseline help?
#'
#' The out-of-sample counterpart to incremental_signal(). For each test round
#' (after `min_train_rounds` have accumulated), fits the blend
#' `margin ~ baseline + candidate` on all STRICTLY PRIOR rounds and applies it
#' to the current round. This is the exact test that showed torp+Wheelo (24.80)
#' losing to Wheelo alone (24.74) on the live 2026 record.
#'
#' Note the bootstrap CI here is on ΔMAE only. The blend's `pred_win` is carried
#' over from the baseline unchanged (this is a margin-space test), so the
#' returned Brier delta is structurally zero and carries no information — do not
#' quote it.
#'
#' @param candidate_preds,baseline_preds Harness-shaped prediction sets
#' @param min_train_rounds Rounds of history required before the first
#'   prediction (default 6 — matches the diagnosis's own setting)
#' @param B Bootstrap draws for the ΔMAE CI
#' @return list(n_matches, mae_candidate, mae_baseline, mae_blend,
#'   delta_mae (blend - baseline), delta_mae_ci, by_round)
#' @keywords internal
rolling_blend_check <- function(candidate_preds, baseline_preds,
                                min_train_rounds = 6L, B = 2000) {
  j <- .join_pred_sets(candidate_preds, baseline_preds)
  j <- j[order(j$season, j$round, j$match_id), ]

  rounds <- unique(j[, c("season", "round")])
  rounds <- rounds[order(rounds$season, rounds$round), ]
  if (nrow(rounds) <= min_train_rounds) {
    cli::cli_abort("rolling_blend_check: only {nrow(rounds)} round{?s} available, need > {min_train_rounds}")
  }

  acc <- vector("list", nrow(rounds))
  for (i in seq(min_train_rounds + 1L, nrow(rounds))) {
    s <- rounds$season[i]; r <- rounds$round[i]
    prior <- j$season < s | (j$season == s & j$round < r)
    this  <- j$season == s & j$round == r
    if (!any(this) || sum(prior) < 20) next

    fit <- stats::lm(margin ~ pred_base + pred_cand, data = j[prior, ])
    te  <- j[this, ]
    te$pred_blend <- unname(stats::predict(fit, newdata = te))
    acc[[i]] <- te
  }

  res <- dplyr::bind_rows(acc)
  if (nrow(res) == 0) {
    cli::cli_abort("rolling_blend_check: no test rounds survived the history requirements")
  }

  blend_set <- data.frame(match_id = res$match_id, pred_margin = res$pred_blend,
                          pred_win = res$pred_win_base, margin = res$margin,
                          home_win = res$home_win)
  base_set  <- data.frame(match_id = res$match_id, pred_margin = res$pred_base,
                          pred_win = res$pred_win_base, margin = res$margin,
                          home_win = res$home_win)
  bt <- boot_mae_diff(blend_set, base_set, B = B)

  by_round <- res |>
    dplyr::group_by(season, round) |>
    dplyr::summarise(
      n         = dplyr::n(),
      candidate = mean(abs(pred_cand - margin)),
      baseline  = mean(abs(pred_base - margin)),
      blend     = mean(abs(pred_blend - margin)),
      .groups   = "drop"
    )

  list(
    n_matches     = nrow(res),
    mae_candidate = mean(abs(res$pred_cand - res$margin)),
    mae_baseline  = mean(abs(res$pred_base - res$margin)),
    mae_blend     = mean(abs(res$pred_blend - res$margin)),
    delta_mae     = bt$mae_diff,
    delta_mae_ci  = bt$mae_ci,
    by_round      = by_round
  )
}

# close_call_metrics ----

#' G8 — metrics restricted to the games the model itself calls close
#'
#' The diagnosis located the entire correlation deficit here: in the 60 live
#' 2026 games torp called within 12 points, torp's correlation with the result
#' was 0.045 against a field of 0.18-0.22. Aggregate tables hide this
#' completely, which is why G8 makes it mandatory in every results table.
#'
#' Caveat carried from the diagnosis: conditioning on a model's OWN prediction
#' range-restricts that model more than a comparison model, attenuating its
#' within-bucket correlation for purely statistical reasons. Use this to track
#' a single model across candidates (where the confound is constant), not to
#' declare one model better than another.
#'
#' @param preds Harness-shaped prediction set
#' @param threshold Absolute predicted margin defining "close" (default 12)
#' @keywords internal
close_call_metrics <- function(preds, threshold = 12) {
  m <- abs(preds$pred_margin) <= threshold
  if (sum(m) < 10) {
    return(list(n = sum(m), mae = NA_real_, cor = NA_real_,
                sd_pred = NA_real_, sd_actual = NA_real_))
  }
  list(
    n         = sum(m),
    mae       = mean(abs(preds$pred_margin[m] - preds$margin[m])),
    cor       = stats::cor(preds$pred_margin[m], preds$margin[m]),
    sd_pred   = stats::sd(preds$pred_margin[m]),
    sd_actual = stats::sd(preds$margin[m])
  )
}

# g7_verdict ----

#' G7 pass/fail decision rule
#'
#' TODO(Pete) — this is the judgement call that governs every ship decision in
#' round 3, so it should be yours rather than a default I picked.
#'
#' Inputs available: `sig` (incremental_signal() output — delta_r2,
#' beta_candidate, p_candidate, cor_preds) and `roll` (rolling_blend_check()
#' output — mae_blend, mae_baseline, delta_mae, delta_mae_ci).
#'
#' Return: list(pass = TRUE/FALSE, reason = "<one line>").
#'
#' The trade-off to weigh, concretely. The pooled 2025:2026 window is n=369
#' matches, and round 2 measured the XGBoost retraining noise floor at ~0.157
#' MAE — so this window genuinely cannot resolve small effects.
#'
#'   STRICT (e.g. require p_candidate < 0.05 AND delta_mae_ci upper bound < 0):
#'     will reject real-but-small features. Round 2 showed almost every
#'     individually-plausible candidate has a CI straddling zero on this
#'     window; a strict rule may reject everything for several sessions and
#'     tell you nothing about direction.
#'
#'   LENIENT (e.g. require only beta_candidate > 0 and mae_blend < mae_baseline):
#'     lets you accumulate directional evidence across features and compose
#'     survivors — but it is exactly the reasoning that let round 2 ship a
#'     composed "everything" candidate whose individual parts were each nulls.
#'
#'   TWO-TIER is the option I'd lean toward if you want a suggestion: a
#'     "keep exploring" bar (directional, cheap) distinct from a "ship it" bar
#'     (bootstrap-confirmed), so a feature can survive into composition without
#'     being ship-approved. That keeps round 2's failure mode visible instead of
#'     laundering it through a composite.
#'
#' Whatever you choose, it should be written down here rather than applied
#' case-by-case — a gate re-argued per candidate is not a gate.
#'
#' IMPLEMENTED 2026-07-28 as the two-tier rule (D-M1). Rationale for choosing
#' it over the strict alternative, recorded so it can be argued with later:
#' WS1b produced a candidate improving all six pooled metrics whose MAE CI
#' still spanned zero — and always will, because the effect (~0.11) is smaller
#' than the measured XGBoost retraining noise floor (~0.157) on the largest
#' window this repo can build. A strict CI-exclusion rule rejects every such
#' candidate permanently, which forecloses the only mechanism by which a model
#' sitting at parity accumulates its way past parity.
#'
#' The two tiers:
#'   EXPLORE — directional evidence, cheap to earn, means "carry into
#'     composition and keep testing". Requires the candidate to be genuinely
#'     additive to the baseline, not merely correlated with the target.
#'   SHIP    — bootstrap-confirmed, means "port to production". Deliberately
#'     unchanged from G3 so this rule cannot launder a null into a release,
#'     which is round 2's failure mode.
#'
#' A candidate at EXPLORE is NOT ship-approved. Composites built from EXPLORE
#' candidates must clear SHIP on their own before release — the composite is
#' the thing being shipped, so it is the thing that must be confirmed.
#'
#' @param sig Output of incremental_signal()
#' @param roll Output of rolling_blend_check()
#' @return list(pass, tier, reason) — `pass` TRUE at either tier (so the report
#'   reads as a pass), `tier` one of "ship"/"explore"/"fail" for programmatic use
#' @keywords internal
g7_verdict <- function(sig, roll) {
  additive   <- sig$beta_candidate > 0 && sig$p_candidate < 0.05
  helps_oos  <- roll$mae_blend < roll$mae_baseline
  ci_excl_0  <- roll$delta_mae_ci[2] < 0

  if (additive && helps_oos && ci_excl_0) {
    return(list(pass = TRUE, tier = "ship", reason = sprintf(
      "SHIP - adds signal over baseline (beta %.3f, p %.2g) and blend CI excludes zero [%.3f, %.3f]",
      sig$beta_candidate, sig$p_candidate, roll$delta_mae_ci[1], roll$delta_mae_ci[2])))
  }
  if (additive && helps_oos) {
    return(list(pass = TRUE, tier = "explore", reason = sprintf(
      "EXPLORE - adds signal (beta %.3f, p %.2g) and blend beats baseline (%.3f vs %.3f), but CI [%.3f, %.3f] includes zero. Carry forward; not ship-approved.",
      sig$beta_candidate, sig$p_candidate, roll$mae_blend, roll$mae_baseline,
      roll$delta_mae_ci[1], roll$delta_mae_ci[2])))
  }
  list(pass = FALSE, tier = "fail", reason = sprintf(
    "FAIL - %s",
    if (!additive) sprintf("not additive to baseline (beta %.3f, p %.2g)",
                           sig$beta_candidate, sig$p_candidate)
    else sprintf("blend does not beat baseline (%.3f vs %.3f)",
                 roll$mae_blend, roll$mae_baseline)))
}

# signal_gate_report ----

#' Full G7 + G8 report for one candidate against the Elo baseline
#'
#' Prints every number the gate rests on, then the verdict (once g7_verdict()
#' is defined). Returns the components invisibly for programmatic use.
#'
#' @param candidate_preds,baseline_preds Harness-shaped prediction sets
#' @param label Human-readable candidate name for the printed header
#' @param B Bootstrap draws
#' @keywords internal
signal_gate_report <- function(candidate_preds, baseline_preds,
                               label = "candidate", B = 2000) {
  sig  <- incremental_signal(candidate_preds, baseline_preds)
  roll <- rolling_blend_check(candidate_preds, baseline_preds, B = B)
  cm_c <- close_call_metrics(candidate_preds)
  cm_b <- close_call_metrics(baseline_preds)
  verdict <- g7_verdict(sig, roll)

  cli::cli_h1("Signal gate: {label}")

  cli::cli_h2("G7a — incremental signal over Elo baseline (static fit, n={sig$n_matches})")
  cli::cli_text("r2: baseline {round(sig$r2_baseline, 4)} | candidate alone {round(sig$r2_candidate, 4)} | joint {round(sig$r2_joint, 4)}")
  cli::cli_text("delta r2 = {round(sig$delta_r2, 4)}")
  cli::cli_text("beta(baseline) = {round(sig$beta_baseline, 3)} | beta(candidate) = {round(sig$beta_candidate, 3)} | p(candidate) = {signif(sig$p_candidate, 3)}")
  cli::cli_text("cor(candidate, baseline) = {round(sig$cor_preds, 3)}")

  cli::cli_h2("G7b — honest expanding-window blend (n={roll$n_matches})")
  cli::cli_text("MAE: candidate {round(roll$mae_candidate, 3)} | baseline {round(roll$mae_baseline, 3)} | rolling blend {round(roll$mae_blend, 3)}")
  cli::cli_text("delta MAE (blend - baseline) = {round(roll$delta_mae, 3)}, 95% CI [{round(roll$delta_mae_ci[1], 3)}, {round(roll$delta_mae_ci[2], 3)}]")

  cli::cli_h2("G8 — close-call bucket (|pred| <= 12)")
  cli::cli_text("candidate: n={cm_c$n}, MAE {round(cm_c$mae, 2)}, cor {round(cm_c$cor, 3)}")
  cli::cli_text("baseline:  n={cm_b$n}, MAE {round(cm_b$mae, 2)}, cor {round(cm_b$cor, 3)}")

  tier <- if (is.null(verdict$tier)) "unknown" else verdict$tier
  switch(tier,
    ship    = cli::cli_alert_success("G7: {verdict$reason}"),
    explore = cli::cli_alert_warning("G7: {verdict$reason}"),
    fail    = cli::cli_alert_danger("G7: {verdict$reason}"),
    cli::cli_alert_warning("G7: {verdict$reason}")
  )

  invisible(list(signal = sig, rolling = roll,
                 close_candidate = cm_c, close_baseline = cm_b,
                 verdict = verdict))
}

# .join_pred_sets ----

#' Inner-join two harness prediction sets on match_id
#'
#' Aborts rather than silently proceeding when the overlap is small — a
#' candidate scored on a different match set than the baseline produces a
#' meaningless comparison, and the failure is otherwise invisible.
#'
#' @keywords internal
.join_pred_sets <- function(candidate_preds, baseline_preds) {
  need <- c("season", "round", "match_id", "pred_margin", "pred_win", "margin", "home_win")
  for (nm in c("candidate_preds", "baseline_preds")) {
    x <- get(nm)
    missing <- setdiff(need, names(x))
    if (length(missing) > 0) {
      cli::cli_abort("{nm} is missing required column{?s}: {missing}")
    }
  }

  cand <- candidate_preds[, need]
  names(cand)[names(cand) == "pred_margin"] <- "pred_cand"
  names(cand)[names(cand) == "pred_win"]    <- "pred_win_cand"

  base <- baseline_preds[, c("match_id", "pred_margin", "pred_win")]
  names(base) <- c("match_id", "pred_base", "pred_win_base")

  j <- merge(as.data.frame(cand), as.data.frame(base), by = "match_id")
  if (anyDuplicated(j$match_id)) {
    cli::cli_abort(".join_pred_sets: duplicate match_id after join — both inputs must be one row per match")
  }
  overlap <- nrow(j) / min(nrow(candidate_preds), nrow(baseline_preds))
  if (nrow(j) == 0) {
    cli::cli_abort(".join_pred_sets: no overlapping match_id between the two prediction sets")
  }
  if (overlap < 0.9) {
    cli::cli_abort(c(
      ".join_pred_sets: only {round(100 * overlap)}% of matches overlap ({nrow(j)} of {min(nrow(candidate_preds), nrow(baseline_preds))}).",
      "!" = "Comparing a candidate and baseline scored on different match sets is not interpretable.",
      "i" = "Check that both were produced with the same test_seasons."
    ))
  }
  j
}
