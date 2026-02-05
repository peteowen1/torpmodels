# torpmodels

Pre-trained machine learning models for AFL analytics.

## Installation

```r
# Install from GitHub
devtools::install_github("peteowen1/torpmodels")
```

## Usage

```r
library(torpmodels)

# Load core models
ep_model <- load_torp_model("ep")
wp_model <- load_torp_model("wp")

# Load stat models
goals_model <- load_stat_model("goals")
disposals_model <- load_stat_model("disposals")

# Check what's cached locally
check_model_cache()

# Clear cache if needed
clear_model_cache("all")
```

## Available Models

### Core Models
- **ep_model** - Expected Points GAM model
- **wp_model** - Win Probability model
- **shot_ocat_mdl** - Shot outcome classification model
- **xgb_win_model** - XGBoost match prediction model

### Stat Models
58 per-statistic GAM models for predicting individual player statistics including goals, disposals, tackles, marks, and more.

## Model Storage

Models are distributed via GitHub releases and cached locally after first download. This keeps the package lightweight while providing fast access after initial load.

## Related Packages

- [torp](https://github.com/peteowen1/torp) - AFL analytics package that uses these models
- [torpdata](https://github.com/peteowen1/torpdata) - AFL data repository
