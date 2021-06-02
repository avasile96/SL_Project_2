### LOADING PACKAGES ###
library(tidyverse) # really dunno
library(eqs2lavaan) # for plotting covariance
library(GGally) # plotting correlation
library(FNN) # knn
library(readr) # string to number
library(Metrics)
library(caret) # normalization
library(Directional) # another knn (regression tuning)
library(dplyr) # for finding strings in column names
library(imager) # dealing with images

### LOADING DATA ###
setwd("D:\\Uni\\SL\\SL_Project_2\\")


ADCN_train <- as_tibble(read.csv(".\\dataset\\ADCNtrain.csv"))
ADCN_test <- as_tibble(read.csv(".\\dataset\\ADCNtest.csv"))

x_ADCN_train <- ADCN_train %>% select(2:567)
x_ADCN_train$Labels <- as.integer(ADCN_train$Labels == 'CN')
y_ADCN_train <- as.integer(ADCN_train$Labels == 'CN')

### EXPLORING DATA ###

# correlation
train_cor = cor(x_ADCN_train)
ggcorr(x_ADCN_train[2,10])

remove(train_cor) #decluttering

### FEATURE ENGINEERING ###


# Normalization
norm_params <- preProcess(x_ADCN_train) # getting normalization parameters
x_ADCN_train <- predict(norm_params, x_ADCN_train) # applying them
ADCN_test <- predict(norm_params, ADCN_test) # applying them to test data

### TES - TRAIN - VAL SPLIT ###
# Generating big dataset again with engineerd features


# PCA #
# pca <- prcomp(x_ADCN_train) # pca list of objects 
# pc <- matrix(unlist(pca[5]), ncol = 9, byrow = FALSE) # matrix of PCs

### LM REGRESSION ###
# Define training control
set.seed(42) 
train.control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
# Train the model
model <- train(x_ADCN_train, y_ADCN_train, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)

### NEW STRATEGY - KNN - Automation - Pipelining ###

samp = list()
sampling_meth = list()
i = 1
for (var in c('cv', 'boot', 'optimism_boot',
              'boot_all', 'repeatedcv', 'LOOCV', 'LGOCV')){
  print(var)
  # Define training control
  set.seed(42) 
  train.control <- trainControl(method = var, number = 5)
  # Train the model
  knn.fit <- train(x_ADCN_train, y_ADCN_train, method = "knn",
                   trControl = train.control,
                   tuneGrid = expand.grid(k = seq(from = 1, to = 10, by = 1))
  )
  # Summarize the results
  print(knn.fit)
  print(rmse(y_ADCN_train, predict(knn.fit, x_ADCN_train)))
  
  samp[i] = knn.fit$results$RMSE[unlist(knn.fit$bestTune)]
  sampling_meth[i] = var
  i <- i+1
}
plot(factor(unlist(sampling_meth)), 
     unlist(samp), xlab = 'sampling method', ylab = 'knn rmse')

############# FINAL TRAINING OF THE KNN ####################################################################
train.control <- trainControl(method = "LOOCV", number = 5)
knn.fit <- train(x_ADCN_train, y_ADCN_train, method = "knn",
                 trControl = train.control,
                 tuneGrid = expand.grid(k = seq(from = 1, to = 10, by = 1))
)
print(rmse(y_ADCN_train, predict(knn.fit, x_ADCN_train)))
print(knn.fit)
knn_pred <- predict(knn.fit, x_ADCN_train)

### SAVING ###
save(fit, lm_pred, knn_pred, file = "0063828_Vasile_challenge2_ADCNres.RData") # saving results
save(fit, lm_pred, knn_pred, file = "0063828_Vasile_challenge2_ADCNfeat.RData") # saving training features (models)

### TESTING SAVE ###
load("0063828_Vasile_challenge2_ADCNres.RData")
load("0063828_Vasile_challenge2_ADCNfeat.RData")
