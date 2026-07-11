# Wrapper: train Shot model using local dev torp
# ===============================================

library(mgcv)

# Load dev torp first
devtools::load_all("C:/dev/torpverse/torp")

setwd("C:/dev/torpverse/torpmodels")
source("data-raw/03-shot-model/train_shot_model.R")
