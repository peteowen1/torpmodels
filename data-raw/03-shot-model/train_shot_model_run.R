# Wrapper: train Shot model using local dev torp
# ===============================================

library(mgcv)

# Load dev torp first
devtools::load_all("C:/Users/peteo/OneDrive/Documents/torpverse/torp")

setwd("C:/Users/peteo/OneDrive/Documents/torpverse/torpmodels")
source("data-raw/03-shot-model/train_shot_model.R")
