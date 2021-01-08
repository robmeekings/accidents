if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
# https://cran.r-project.org/web/packages/suncalc/suncalc.pdf
if(!require(suncalc)) install.packages("suncalc", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
if(!require(pscl)) install.packages("pscl", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

# Load libraries
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(kableExtra)
library(suncalc)
library(dplyr)
library(MASS)
library(pscl)
library(randomForest)

##########################################################
# Create training set and validation hold-out test set
##########################################################

# Copies of these datasets can be downloaded from github using this code

dl <- tempfile()        # create a tempfile to receive the downloaded file
download.file("https://github.com/robmeekings/accidents/raw/main/training.rds", dl, mode="wb")  # download the file at the given url
sampled_training <- readRDS(dl)

dl <- tempfile()        # create a tempfile to receive the downloaded file
download.file("https://github.com/robmeekings/accidents/raw/main/validation.rds", dl, mode="wb")  # download the file at the given url
sampled_validation <- readRDS(dl)

# define a function to calculate rmse between observed and predicted data
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, rm.na=TRUE))
}

# calcualte the mean number of casualties
mean_casualties <- mean(sampled_training$Number_of_Casualties)

# calculate an RMSE score for using the mean as a predictor
rmse_mean <- RMSE(sampled_validation$Number_of_Casualties, mean_casualties)

rmse_results <- tibble(method = "Just the mean", RMSE = rmse_mean)

model.linear <- lm(Number_of_Casualties ~
            Police_Force +
            Other_Vehicles + 
            Speed_limit + 
            First_Road_Class +
            Second_Road_Class +
            Weather_Conditions + 
            Light_Conditions + 
            Road_Surface_Conditions +
            Carriageway_Hazards + 
            Urban_or_Rural_Area + 
            time_of_day + 
            Day_of_Week +
            rush_hour +
            Age_Band_of_Driver + 
            Sex_of_Driver +
            Vehicle_Type + 
            Age_of_Vehicle +
            Driver_IMD_Decile +
            year +
            month,
          data=sampled_training)

pred.linear <- data.frame(casualties = 
                            predict.lm(model.linear, newdata=sampled_validation))

miss_idx <- is.na(pred.linear$casualties)

rmse_linear <- RMSE(sampled_validation[!miss_idx,]$Number_of_Casualties, pred.linear[!miss_idx,])

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Linear model",  
                                     RMSE = rmse_linear))

model.poisson <- glm(Number_of_Casualties ~  Police_Force +
                       Other_Vehicles + 
                       Speed_limit + 
                       First_Road_Class +
                       Second_Road_Class +
                       First_Road_Class:Second_Road_Class +
                       #            Junction_Detail +
                       #            Weather_Conditions:Light_Conditions + 
                       Road_Surface_Conditions +
                       #            Carriageway_Hazards + 
                       #            First_Road_Class:Urban_or_Rural_Area + 
                       time_of_day + 
                       Day_of_Week +
                       rush_hour +
                       Age_Band_of_Driver + 
                       Vehicle_Type + 
                       Age_of_Vehicle +
                       Driver_IMD_Decile +
                       year +
                       month,
               data=sampled_training,
               family=poisson(link=log))

pred.poisson <- data.frame(casualties = 
                            predict(model.poisson, newdata=sampled_validation))

miss_idx <- is.na(pred.poisson$casualties)

rmse_poisson <- RMSE(sampled_validation[!miss_idx,]$Number_of_Casualties, pred.poisson[!miss_idx,])

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Poisson model",  
                                     RMSE = rmse_poisson))

summary(model.poisson)

model.negbin <- glm.nb(Number_of_Casualties ~ Police_Force +
                    Other_Vehicles + 
                    Speed_limit + 
                    First_Road_Class +
                    Second_Road_Class +
                    First_Road_Class:Second_Road_Class +
                    #            Junction_Detail +
                    #            Weather_Conditions:Light_Conditions + 
                    Road_Surface_Conditions +
                    Carriageway_Hazards + 
                    #            First_Road_Class:Urban_or_Rural_Area + 
                    time_of_day * Day_of_Week +
                    rush_hour +
                    Age_Band_of_Driver + 
                    Vehicle_Type + 
                    Age_of_Vehicle +
                    Driver_IMD_Decile +
                    year +
                    month,
                  
                  data=sampled_training)

pred.negbin <- data.frame(casualties = 
                            predict(model.negbin, newdata=sampled_validation))

miss_idx <- is.na(pred.negbin$casualties)

rmse_negbin <- RMSE(sampled_validation[!miss_idx,]$Number_of_Casualties, pred.negbin[!miss_idx,])

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Negative binomial model",  
                                     RMSE = rmse_negbin))

summary(model.negbin)

# ANOVA poisson vs neg bin
anova(model.poisson, model.negbin)

sampled_training$casualties <- sampled_training$Number_of_Casualties - 1

model.negbin_adj <- glm.nb(casualties ~ Police_Force +
                    Other_Vehicles + 
                    Speed_limit + 
                    First_Road_Class +
                    Second_Road_Class +
                    First_Road_Class:Second_Road_Class +
                    #            Junction_Detail +
                    #            Weather_Conditions:Light_Conditions + 
                    Road_Surface_Conditions +
                    Carriageway_Hazards + 
                    #            First_Road_Class:Urban_or_Rural_Area + 
                    time_of_day * Day_of_Week +
                    rush_hour +
                    Age_Band_of_Driver + 
                    Vehicle_Type + 
                    Age_of_Vehicle +
                    Driver_IMD_Decile +
                    year +
                    month,
                  
                  data=sampled_training)

summary(model.negbin_adj)

pred.negbin_adj <- data.frame(casualties = 
                            predict(model.negbin_adj, newdata=sampled_validation) + 1)

miss_idx <- is.na(pred.negbin_adj$casualties)

rmse_negbin_adj <- RMSE(sampled_validation[!miss_idx,]$Number_of_Casualties, pred.negbin_adj[!miss_idx,])

rmse_negbin_adj <- RMSE(sampled_validation$Number_of_Casualties, pred.negbin_adj$casualties)

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Neg bin model, response adj",  
                                     RMSE = rmse_negbin_adj))

model.negbin_zin <- zeroinfl(casualties ~ Police_Force +
                               Other_Vehicles + 
                               Speed_limit + 
                               First_Road_Class +
                               Second_Road_Class +
                               First_Road_Class:Second_Road_Class +
                               #            Junction_Detail +
                               #            Weather_Conditions:Light_Conditions + 
                               Road_Surface_Conditions +
                               Carriageway_Hazards + 
                               #            First_Road_Class:Urban_or_Rural_Area + 
                               time_of_day * Day_of_Week +
                               rush_hour +
                               Age_Band_of_Driver + 
                               Vehicle_Type + 
                               Age_of_Vehicle +
                               Driver_IMD_Decile +
                               year +
                               month,
                             
                             data=sampled_training,
                             dist = "negbin", EM=FALSE)

pred.negbin_zin <- data.frame(casualties = 
                                predict(model.negbin_zin, newdata=sampled_validation) + 1)

miss_idx <- is.na(pred.negbin_zin$casualties)

rmse_negbin_zin <- RMSE(sampled_validation[!miss_idx,]$Number_of_Casualties, pred.negbin_zin[!miss_idx,])

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Neg bin model, zero-inflated",  
                                     RMSE = rmse_negbin_zin))

vuong(model.negbin_adj, model.negbin_zin)

set.seed(1, sample.kind="Rounding")
# Create sampled partition index
subsample_index <- createDataPartition(y = sampled_training$Number_of_Casualties, 
                                     times = 1, p = 0.1, list = FALSE)
# Create the training set as those obs not in our sample
subsample_training <- sampled_training[subsample_index,]
# Create the validation set as those obs in our sample
#subsample_validation <- sampled_training[subsample_index,]

set.seed(1, sample.kind="Rounding")
fit <- randomForest(Number_of_Casualties~Police_Force +
                      Other_Vehicles + 
                      Speed_limit + 
                      First_Road_Class +
                      Second_Road_Class +
                      Weather_Conditions + 
                      Light_Conditions + 
                      Road_Surface_Conditions +
                      Carriageway_Hazards + 
                      Urban_or_Rural_Area + 
                      time_of_day + 
                      Day_of_Week +
                      rush_hour +
                      Age_Band_of_Driver + 
                      Sex_of_Driver +
                      Vehicle_Type + 
                      Age_of_Vehicle +
                      Driver_IMD_Decile +
                      year +
                      month, 
                    data = subsample_training, 
                    na.action = na.omit) 

pred.rf <- data.frame(casualties = predict(fit, sampled_validation))

miss_idx <- is.na(pred.rf$casualties)

rmse_rf <- RMSE(sampled_validation[!miss_idx,]$Number_of_Casualties, pred.rf[!miss_idx,])

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Random Forest",  
                                     RMSE = rmse_rf))

varImp(fit)

rf_lin <- ifelse(is.na(pred.rf$casualties + pred.linear$casualties), mean_casualties, (pred.rf$casualties + pred.linear$casualties) / 2)

miss_idx <- is.na(rf_lin)

rmse_rf_lin <- RMSE(sampled_validation$Number_of_Casualties, rf_lin)

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Random Forest and linear",  
                                 RMSE = rmse_rf_lin))

rmse_results <- as.data.frame(rmse_results)

saveRDS(rmse_results, "rmse_results.rds")

rmse_results
