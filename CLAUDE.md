# CLAUDE.md

This file provides guidance to Claude Code when working in the torpmodels package.

## Package Overview

**torpmodels** provides pre-trained ML models for AFL analytics, served via GitHub releases with local caching. It has two exported model loaders and no data processing logic.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full documentation, Mermaid diagrams, and model catalog.

### R Code (2 files)
- `R/load_model.R` - All exported functions: `load_torp_model()`, `load_stat_model()`, `list_available_models()`, `check_model_cache()`, `clear_model_cache()`
- `R/torpmodels-package.R` - Package-level docs

### Model Types
- **Core models** (tag: `core-models`): `ep`, `wp`, `wp_calibration`, `shot`, `match_gams`, `match_xgb_pipeline`, `match_margin_calibration`, `shot_player_df`, `xgb_win_model` — loaded via `load_torp_model()`
- **Stat models** (tag: `stat-models`): 58 per-statistic GAMs (goals, disposals, etc.) — loaded via `load_stat_model()`

### Caching
Models cache to `tools::R_user_dir("torpmodels", "cache")/models/`. Use `force_download = TRUE` to bypass cache.

### Model Training Scripts (`data-raw/`)

**`data-raw/train_models.R` is THE canonical entry point for EP/WP/shot training** — replaces the old standalone `train_ep_model*.R` / `train_wp_model*.R` / `train_shot_model*.R` scripts (deleted; git history retains them). Run from the torpmodels root:
```
Rscript data-raw/train_models.R ep wp shot                 # canonical full retrain + upload
Rscript data-raw/train_models.R wp --no-upload              # local-only
Rscript data-raw/train_models.R ep wp --seasons 2021 2024   # explicit window
Rscript data-raw/train_models.R wp --insample-ep            # legacy comparison; never uploads
Rscript data-raw/train_models.R wp --skip-slope-gate         # disable the temporal Q4/close release gate (emergencies only)
Rscript data-raw/train_models.R wp --no-calibrate            # skip WP recalibration entirely; forces --no-upload
```
It `devtools::load_all()`s dev torp (and dev torpmodels) itself — no separate `*_run.R` wrapper needed. Every model it saves carries a `torp_meta` provenance stamp (see `R/model_meta.R`) and publishes atomically via `publish_model_group()` (see `R/publish.R`), so a shot model can never upload without its `shot_player_df` sidecar, a WP model can never upload without its `wp_calibration` sidecar, and a silent overwrite is detectable via `check_manifest_sync()`.

**WP recalibration + temporal slope gate** ([`../docs/plans/FABLE-RECAL-PLAN.md`](../docs/plans/FABLE-RECAL-PLAN.md), on by default for every WP training run): `train_core_models()`'s WP branch trains a temporal variant (EP+WP on seasons before the last completed season, scored on that held-out season), fits a two-parameter Platt-on-logit calibration on the honest OOS predictions, and gates the release on `|calibrated slope - 1| <= 0.10` in both the Q4/close cell and all-rows — a breach aborts before anything is written or published. On a pass, `(a, b)` ships as `wp_calibration.rds`. `--skip-slope-gate` disables the abort (emergencies only, slopes still print); `--no-calibrate` (or `--insample-ep`) skips the whole layer and forces `--no-upload`. torp applies the calibration at serve time in `get_wp_preds()` with an identity fallback when the sidecar is absent — see `torp/CLAUDE.md`.

- `data-raw/lib/train_lib.R` - The fitting functions (`fit_ep()`, `fit_wp()`, `fit_shot()`, `train_core_models()` orchestrator, etc), plus the WP recalibration/gate functions (`fit_wp_temporal_variant()`, `fit_wp_calibration()`, `wp_gate_slope()`, `validate_wp_temporal_slope()`, `cv_wp_oos_preds()`). No side effects at source time — both `train_models.R` and `torp/data-raw/rebuild_everything.R` Phase 4 source it. WP's `monotone_constraints` are *derived* from `torp:::WP_MODEL_FEATURES`/`WP_MONOTONE_INCREASING`, never hand-inlined (this is the fix for the 15-vs-18-entry constraint bug that shipped in production).

**Match margin recalibration + slope gate** (2026-07, [`../docs/plans/FABLE-MATCH-MAE-PLAN.md`](../docs/plans/FABLE-MATCH-MAE-PLAN.md), analogous to the WP pattern above but lives in **torp**, not torpmodels): `run_predictions_pipeline()` (`torp/R/match_model.R`) fits a single-slope temporal-holdout calibration (`fit_match_margin_calibration()`, `torp/R/match_calibration.R`) each retrain — train on all seasons but the most recent, score that season OOS, fit `b = coef(lm(margin ~ pred_margin + 0))`. Gated on the raw OOS slope staying inside `MATCH_MARGIN_SLOPE_GATE` (`torp/R/constants_match.R`); a breach skips uploading a new sidecar (keeps the previous one, or identity) rather than aborting the whole pipeline — weekly predictions still need to ship, unlike the standalone EP/WP/shot retrain. Applied at serve time via `apply_match_margin_calibration()` with an identity fallback, same contract as `get_wp_preds()`.
- `02-wp-model/validate_cv_ep_wp.R` - Compares WP trained with in-sample vs cross-validated EP predictions (sources `lib/train_lib.R` for the shared data/fold/CV plumbing); quantifies the CV-EP improvement.
- `04-match-model/train_match_models.R` - Rolling week-by-week out-of-sample evaluation (GAM/XGBoost/blends vs Squiggle). Set `TEST_SEASONS` to a finite range for the eval window.
- `01-ep-model/train_ep_model_live_v2.R` - Live EP model (13 features; drops `lag_goal_x`, adds `phase_of_play` + `chain_action_num`) → `ep_model_live_v2.json`/`.rds`. Save/export happens before the Daicos-R6 sanity-check diagnostics, which are `tryCatch`-wrapped so a brittle hardcoded-match lookup can never undo the save.
- `05-live-wp-model/train_live_wp_model.R` - Live WP model (GAM → `live_wp_lookup.json` lookup table for browser)
- `05-live-wp-model/train_live_wp_chain_v4.R` - Chain-aware live WP export (possession-POV) → `wp-model-chain.json`
- `convert_rda_to_rds.R` - Utility to convert legacy `.rda` model files to `.rds` format

One script per live JSON artifact now (the superseded `train_ep_model_live.R` v1, `train_live_wp_chain.R`/`_v2.R`/`_v3.R`, and the experimental `train_live_wp_xgb.R` are deleted; git history retains them). Each envelope carries `trained_on`, `exported_at`, `torp_sha`, and `script` metadata.

**`data-raw/debug/`** holds scratch/experimental WP-comparison and calibration scripts — not part of the pipeline; ignore when tracing the training flow.

**Training order matters:** EP must be trained before WP (WP uses EP predictions as features) — `train_core_models()` enforces this automatically.

### Live Model Export
Live models are exported as JSON for browser/Worker inference on inthegame-blog:
- EP: `ep-model-live.json` (XGBoost tree structure → Worker tree walk in `worker/src/ep-model.js`)
- WP: `live-wp-lookup.json` (GAM lookup table → browser)
- xG: `xg-lookup.json` (GAM grid → browser, generated by `torp/scripts/live-model-export.R`)

### Weather Data Dependency
Weather data is loaded from torpdata's `weather-data` release via `load_weather()`. Run `torp/data-raw/01-data/get_weather_data.R` to refresh it.

### Releasing Models
```r
piggyback::pb_upload("ep_model.rds", repo = "peteowen1/torpmodels", tag = "core-models")
```

## Testing

- `tests/testthat/test-load_model.R` — tests model loading, caching, and fallback behavior
- `tests/testthat/test-train_lib.R` — sources `data-raw/lib/train_lib.R` directly (side-effect-free at source time); covers the WP recalibration/gate pure functions and the `train_core_models()` WP-branch wiring (all mocked, network-free)

## CI/CD

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `test-package.yml` | Push to `main`/`dev`, PR to `main`, manual | R CMD check on ubuntu-latest (R release) |

Models themselves are trained manually and released ad-hoc via `piggyback::pb_upload()` — no automated training workflow.

## Known Issues
- XGBoost models saved as RDS can be incompatible across XGBoost versions. Use `force_download = TRUE` to re-download.
- Shot model requires `library(mgcv)` loaded before `predict()` — otherwise `Xbd` function not found.
