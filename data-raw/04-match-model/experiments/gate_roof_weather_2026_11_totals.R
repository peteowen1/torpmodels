# Total-points scoring for the roof gate (review W12, 2026-11-26), from saved results.
# Ran 26 Nov 12:10pm before the flag was removed from torp (6342210c); it no longer
# runs against current torp. Output: roof_weather_2026-11/totals_rescored.csv.
# gate_roof_weather_2026_11.R as first run scored pred_xtotal against the UNSCALED
# total_xpoints, but the totals models train on and predict total_xpoints_adj
# (torp R/match_data_prep.R:1128; rolling_lib.R:143). This recomputes the
# total-points lines on the right scale, and against actual total points, from
# results.rds. Margin MAE and log loss were unaffected and are not recomputed.
#
# Run from torpverse/torp:  Rscript ../torpmodels/data-raw/04-match-model/experiments/gate_roof_weather_2026_11_totals.R

suppressMessages({
  devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
  devtools::load_all("C:/dev/torpverse/torpmodels", quiet = TRUE)
  library(dplyr)
})
exp_dir <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments"
source(file.path(exp_dir, "rolling_lib.R"))
source(file.path(exp_dir, "arm_lib.R"))
out_dir <- "C:/dev/torpverse/torpdata/data/raw/match_gates/roof_weather_2026-11"
res <- readRDS(file.path(out_dir, "results.rds"))
stopifnot(all(c("off", "on") %in% names(res)))

options(torp.local_data_dir = NA)
stopifnot(is.null(get_local_data_dir()))
src <- load_match_inputs()
torp_df <- load_torp_ratings()
stopifnot(identical(torp:::MATCH_ROOF_ZERO_WEATHER, FALSE))
tm_off <- build_team_mdl_with(src, torp_df)
stopifnot(all(c("total_xpoints_adj", "total_points", "is_roof") %in% names(tm_off)))
roof_ids <- unique(tm_off$match_id[tm_off$is_roof %in% TRUE])

tot <- dplyr::distinct(as.data.frame(tm_off)[tm_off$team_type_fac.x == "home",
                                             c("match_id", "total_xpoints_adj", "total_points")])
stopifnot(!anyDuplicated(tot$match_id))

# Total-points MAE in points; lower is better.
out <- bind_rows(lapply(names(res), function(nm) {
  p <- merge(res[[nm]]$input_blend_preds[, c("match_id", "pred_xtotal")], tot, by = "match_id")
  stopifnot(nrow(p) == nrow(res[[nm]]$input_blend_preds))
  r <- p$match_id %in% roof_ids
  data.frame(arm = nm, n_all = nrow(p), n_roofed = sum(r),
             bias_vs_adj = mean(p$pred_xtotal - p$total_xpoints_adj),
             mae_adj_all = mean(abs(p$pred_xtotal - p$total_xpoints_adj)),
             mae_adj_roofed = mean(abs(p$pred_xtotal[r] - p$total_xpoints_adj[r])),
             mae_actual_all = mean(abs(p$pred_xtotal - p$total_points), na.rm = TRUE),
             mae_actual_roofed = mean(abs(p$pred_xtotal[r] - p$total_points[r]), na.rm = TRUE))
}))
print(out, digits = 5)
write.csv(out, file.path(out_dir, "totals_rescored.csv"), row.names = FALSE)
