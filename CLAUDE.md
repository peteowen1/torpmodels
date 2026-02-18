# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

**torpmodels** stores and distributes pre-trained machine learning models for AFL analytics. Models are served via GitHub releases for on-demand download with local caching.

## Development Commands

```r
# Load package for development
devtools::load_all()

# Package check
devtools::check()

# Generate documentation
devtools::document()

# Run tests (when tests exist)
devtools::test()

# Model cache utilities (after loading package)
check_model_cache()         # See what's cached locally
clear_model_cache("all")    # Clear cache (useful when testing download logic)
list_available_models()     # List available models
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
| Model | File | Type | Description |
|-------|------|------|-------------|
| EP | ep_model.rds | XGBoost (multiclass, 5 classes) | Expected Points from field position (~600KB) |
| WP | wp_model.rds | XGBoost (binary) | Win Probability from game state (~100KB) |
| Shot | shot_ocat_mdl.rds | GAM (ocat) | Ordered categorical: miss/behind/goal (~16MB) |
| XGB Win | xgb_win_model.rds | XGBoost (binary) | Match prediction model (~27KB) |
| Stat models | *.rds | GAM | 58 per-statistic models (~3MB each, 177MB total) |

### XGBoost Model Details

**EP Model** (Expected Points):
- Objective: `multi:softprob` with 5 classes
- Predicts point probabilities for different scoring outcomes

**WP Model** (Win Probability):
- Objective: `binary:logistic`
- Uses monotone constraints to ensure logical behavior (e.g., higher EP differential → higher win probability)
- Constraint string: `(0,0,0,1,1,1,0,1,0,0,0,0,0,0,0)` - positive constraints on EP and score differential features

**XGB Win Model** (Match Prediction):
- Objective: `binary:logistic`
- Full hyperparameter tuning via grid search
- Requires pre-built `team_mdl_df` from torp's match prediction workflow

### Cross-Validation Strategy
All XGBoost training uses **match-grouped CV folds** to prevent data leakage. Rows from the same match are always in the same fold, preventing the model from learning match-specific patterns.

**Known limitation (EP→WP leakage):** The EP model is trained on all data before its predictions are used as WP features. This means WP's CV metrics may be slightly optimistic because EP predictions on the WP training set are in-sample for EP. The practical impact is small if EP generalizes well, but true out-of-sample WP performance may be marginally worse than CV suggests.

## Training Scripts (data-raw/)

Training scripts are organized by model type:
- `01-ep-model/` - Expected Points model (XGBoost multiclass)
- `02-wp-model/` - Win Probability model (XGBoost) - requires EP model via `torp::add_epv_vars()`
- `03-shot-model/` - Shot outcome model (ordered categorical GAM)
- `04-xgb-model/` - Match prediction model (XGBoost) - special prerequisites below
- `convert_rda_to_rds.R` - Utility to convert legacy .rda model files to .rds format
- `debug/` - Ad-hoc analysis scripts (gitignored). Contains `train_all_models.R` (EP+WP together) and `train_shot_xgb_models.R`.

### Training Prerequisites

**All scripts** require torp package for data loading and preprocessing. Scripts attempt `devtools::load_all("../../torp")` if torp is not installed.

**XGB Win Model** has additional requirements:
1. Run `torp/data-raw/02-models/build_match_predictions.R` first
2. This creates `team_mdl_df` in memory
3. Then source `train_xgb_win_model.R` in the same R session

**Training dependencies**: tidyverse, zoo, janitor, lubridate, xgboost (or mgcv for GAM models), caret, MLmetrics, purrr, glue, data.table (for train_all_models.R)

### Training Order
EP → WP (WP model requires EP predictions via `torp::add_epv_vars()`)

Alternative: Use `data-raw/debug/train_all_models.R` to train EP and WP together in sequence (keeps EP model in memory for WP predictions).

### Retrain and Release Models
```r
# Individual training scripts save to inst/models/core/ and auto-upload via piggyback
source("data-raw/01-ep-model/train_ep_model.R")

# Train EP and WP together (keeps EP in memory for WP predictions)
source("data-raw/debug/train_all_models.R")

# Manual upload if needed
piggyback::pb_upload("inst/models/core/ep_model.rds", repo = "peteowen1/torpmodels", tag = "core-models")
```

## Running Scripts

**WSL/Bash Workaround:** Arrow package causes R segfaults via bash. Run R scripts via PowerShell:
```bash
powershell.exe -Command 'Rscript "data-raw/01-ep-model/train_ep_model.R"'
```

## Dependencies

Runtime (Imports):
- **cli** - Status messages and errors
- **piggyback** - GitHub releases for model download

Model loading (Suggests):
- **mgcv** - Required to use GAM models (shot, stat models)
- **xgboost** - Required to use XGBoost models (ep, wp, xgb_win)

Training only (not in DESCRIPTION):
- **tidyverse**, **zoo**, **janitor**, **lubridate** - Data prep
- **caret**, **MLmetrics** - XGB win model hyperparameter tuning

## Relationship to Other Packages

- **torp** - Uses torpmodels for loading models (torpmodels in Suggests). Training scripts depend on torp for data loading (`load_chains()`, `load_pbp()`) and preprocessing (`clean_pbp()`, `clean_model_data_epv()`, `clean_model_data_wp()`, `select_epv_model_vars()`, `select_wp_model_vars()`)
- **torpdata** - Separate package for data (not models)

## Gitignored Directories

- `inst/models/` - Model files are stored in GitHub releases, not git
- `data-raw/debug/` - Ad-hoc analysis scripts for exploratory work. Contains `train_all_models.R` for training EP+WP together.

## Troubleshooting

**Model download fails**: Check GitHub authentication - piggyback uses `GITHUB_PAT` environment variable for authenticated downloads. Public releases should work without auth.

**"mgcv not found" or "xgboost not found"**: These are in Suggests, not Imports. Install the relevant package for the model type you're using.

**Testing download logic**: Clear the cache first with `clear_model_cache("all")` to force re-download.
