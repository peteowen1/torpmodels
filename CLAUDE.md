# CLAUDE.md

This file provides guidance to Claude Code when working with the torpmodels package.

## Package Overview

**torpmodels** stores and distributes pre-trained machine learning models for AFL analytics. Models are served via GitHub releases for on-demand download with local caching.

## Development Commands

```r
# Load package for development
devtools::load_all()

# Run tests
devtools::test()

# Package check
devtools::check()

# Generate documentation
devtools::document()
```

## Architecture

### Model Distribution
- Models are stored in GitHub releases (not tracked in git)
- Two release tags:
  - `core-models` - EP, WP, shot, XGBoost models
  - `stat-models` - 58 per-statistic GAM models
- Local cache at `tools::R_user_dir("torpmodels", "cache")/models/`

### Model Loading Flow
1. `load_torp_model("ep")` or `load_stat_model("goals")`
2. Check local cache first
3. If not cached (or force_download=TRUE), download from GitHub release
4. Cache locally for future use
5. Return model object

### Key Files
- `R/load_model.R` - Model loading functions (`load_torp_model()`, `load_stat_model()`)
- `R/torpmodels-package.R` - Package documentation
- `data-raw/` - Model training scripts (moved from torp)

### Model Types
| Model | File | Description |
|-------|------|-------------|
| EP | ep_model.rds | Expected Points GAM (~600KB) |
| WP | wp_model.rds | Win Probability XGBoost (~100KB) |
| Shot | shot_ocat_mdl.rds | Ordered categorical GAM (~16MB) |
| XGB Win | xgb_win_model.rds | Match prediction XGBoost (~27KB) |
| Stat models | *.rds | 58 GAM models (~3MB each, 177MB total) |

## data-raw/ Workflow

Training scripts are organized by model type:
- `01-ep-model/` - Expected Points model training
- `02-wp-model/` - Win Probability model training
- `03-shot-model/` - Shot outcome model training
- `04-xgb-model/` - XGBoost match prediction training
- `stat-models/` - Per-statistic GAM model training

To retrain and release models:
```r
# Train model (example)
source("data-raw/01-ep-model/train_ep_model.R")

# Upload to GitHub release
piggyback::pb_upload("ep_model.rds", repo = "peteowen1/torpmodels", tag = "core-models")
```

## Debug Scripts

When running ad-hoc R scripts for debugging or testing, write them to `data-raw/debug/` and execute via:
```bash
Rscript "data-raw/debug/script_name.R"
```

## Dependencies

- **piggyback** - GitHub releases for model distribution
- **mgcv** - GAM models (Suggests)
- **xgboost** - XGBoost models (Suggests)

## Relationship to Other Packages

- **torp** - Uses torpmodels for loading models (torpmodels in Suggests)
- **torpdata** - Separate package for data (not models)
