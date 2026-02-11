# torpmodels

<!-- badges: start -->
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

Pre-trained machine learning models for AFL analytics, used by [torp](https://github.com/peteowen1/torp).

## Installation

```r
# install.packages("devtools")
devtools::install_github("peteowen1/torpmodels")
```

## Available Models

### Core Models

| Model | Type | Description | Size |
|-------|------|-------------|------|
| `ep_model` | XGBoost (multiclass) | Expected Points from field position | ~600 KB |
| `wp_model` | XGBoost (binary) | Win Probability from game state | ~100 KB |
| `shot_ocat_mdl` | GAM (ordered categorical) | Shot outcome: miss/behind/goal | ~16 MB |
| `xgb_win_model` | XGBoost (binary) | Match prediction from team ratings | ~27 KB |

### Stat Models

58 per-statistic GAM models for projecting individual player statistics, including goals, disposals, tackles, marks, clearances, and more.

## Usage

```r
library(torpmodels)

# Load core models
ep_model <- load_torp_model("ep")
wp_model <- load_torp_model("wp")
shot_model <- load_torp_model("shot")

# Load stat models
goals_model <- load_stat_model("goals")
disposals_model <- load_stat_model("disposals")

# List all available models
list_available_models()

# Check what's cached locally
check_model_cache()

# Clear cache if needed
clear_model_cache("all")
```

## Model Storage

Models are distributed via GitHub releases and cached locally after first download. The local cache lives at:

```r
tools::R_user_dir("torpmodels", "cache")
```

This keeps the package lightweight (~50 KB installed) while providing fast access after initial load.

## Related Packages

- [torp](https://github.com/peteowen1/torp) -- Core AFL analytics package (uses these models)
- [torpdata](https://github.com/peteowen1/torpdata) -- Processed AFL data via GitHub releases
