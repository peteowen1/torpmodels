# Smoke-test the ws30 parallel worker init WITHOUT running the 90-minute gate.
# The first parallel attempt died after the expensive builds with
# 'could not find function "run_rolling_eval"' -- it is sourced, not packaged.
# This proves a worker can reach every function eval_arm calls, in seconds.
cl <- parallel::makeCluster(2L)
on.exit(parallel::stopCluster(cl), add = TRUE)

ok <- parallel::clusterEvalQ(cl, {
  suppressMessages({
    library(data.table); library(arrow)
    devtools::load_all("C:/dev/torpverse/torp", quiet = TRUE)
    devtools::load_all("C:/dev/torpverse/torpmodels", quiet = TRUE)
    source("C:/dev/torpverse/torpmodels/data-raw/04-match-model/experiments/rolling_lib.R")
    source("C:/dev/torpverse/torp/data-raw/04-analysis/cache_guard.R")
  })
  needed <- c("run_rolling_eval", ".train_match_gams", ".train_xgb_fixed",
              ".build_team_ratings_df", ".build_match_features",
              ".build_team_mdl_df", "file_reader", "write_parquet")
  found <- vapply(needed, function(f) exists(f, mode = "function"), logical(1))
  list(missing = names(found)[!found], n_ok = sum(found))
})

for (i in seq_along(ok)) {
  cat(sprintf("worker %d: %d/8 functions found", i, ok[[i]]$n_ok))
  if (length(ok[[i]]$missing)) cat("  MISSING: ", paste(ok[[i]]$missing, collapse = ", "))
  cat("\n")
}
stopifnot(all(vapply(ok, function(x) length(x$missing) == 0L, logical(1))))
cat("\nWORKER INIT OK - every function eval_arm needs is reachable\n")
