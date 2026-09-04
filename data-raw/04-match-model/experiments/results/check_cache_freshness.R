RESULTS_DIR <- "C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/results"
.rds <- function(name) file.path(RESULTS_DIR, name)
suppressPackageStartupMessages({ library(dplyr) })
torp_paths <- c("../torp", "../../torp", "../../../torp", "C:/dev/torpverse/torp")
for (p in torp_paths) if (file.exists(file.path(p, "DESCRIPTION"))) { devtools::load_all(p, quiet = TRUE); break }

cached <- readRDS(.rds("team_mdl_df_cache.rds"))
cat("CACHED team_mdl_df_cache.rds:\n")
cat("  n rows:", nrow(cached), "\n")
cat("  seasons:", paste(sort(unique(cached$season.x)), collapse=", "), "\n")
completed <- cached[!is.na(cached$win) & cached$team_type == "home", ]
cat("  completed home-rows (=matches):", nrow(completed), "\n")
by_season <- completed %>% count(season.x)
print(by_season)
cat("  max round in 2026:", max(completed$round_number.x[completed$season.x == 2026], na.rm=TRUE), "\n")
cat("  file mtime:", format(file.info(.rds("team_mdl_df_cache.rds"))$mtime), "\n")

cat("\n--- attempting fresh load via load_data() to compare ---\n")
fresh_results <- tryCatch(torp::load_results(seasons = TRUE), error = function(e) { cat("load_results failed:", conditionMessage(e), "\n"); NULL })
if (!is.null(fresh_results)) {
  cat("fresh load_results n:", nrow(fresh_results), "\n")
  cat("fresh seasons:", paste(sort(unique(fresh_results$season)), collapse=", "), "\n")
  fresh_completed <- fresh_results[!is.na(fresh_results$home_score) & !is.na(fresh_results$away_score), ]
  cat("fresh completed matches:", nrow(fresh_completed), "\n")
  fresh_2026 <- fresh_completed[fresh_completed$season == 2026, ]
  cat("fresh 2026 completed matches:", nrow(fresh_2026), " max round:", max(fresh_2026$round_number, na.rm=TRUE), "\n")
}
