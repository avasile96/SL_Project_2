### LOADING PACKAGES ###
library(tidyverse) # really dunno
library(eqs2lavaan) # for plotting covariance
library(GGally) # plotting correlation
library(FNN) # knn
library(readr) # string to number
library(Metrics)
library(caret) # normalization
library(Directional) # another knn (regression tuning)

### LOADING DATA ###
setwd("D:\\Uni\\SL\\SL_Project_2\\")


ADCN_train <- as_tibble(read.csv(".\\dataset\\ADCNtrain.csv"))
ADCN_test <- as_tibble(read.csv(".\\dataset\\ADCNtest.csv"))

ADMCI_train <- as_tibble(read.csv(".\\dataset\\ADMCItrain.csv"))
ADMCI_test <- as_tibble(read.csv(".\\dataset\\ADMCItest.csv"))

MCICN_train <- as_tibble(read.csv(".\\dataset\\MCICNtrain.csv"))
MCICN_test <- as_tibble(read.csv(".\\dataset\\MCICNtest.csv"))


x_train_final <- training_data %>% select(2:10)
y_train_final <- training_data %>% select(11)

x_test <- test_data %>% select(2:10)


remove(training_data, test_data) #decluttering

### EXPLORING DATA ###

# correlation
train_cor = cor(x_train_final)
ggcorr(x_train_final)

remove(train_cor, train_cov) #decluttering

### FEATURE ENGINEERING ###

eliminated_final <- x_train_final
eliminated_Y_final <- y_train_final

for (v in names(x_train_final)){
  # print(v)
  Q <- quantile(x_train_final[[v]], probs=c(.25, .75), na.rm = FALSE)
  iqr <- IQR(x_train_final[[v]])
  
  # tol = t
  tol = 1.5
  
  up <-  Q[2]+tol*iqr # Upper Range
  low<- Q[1]-tol*iqr # Lower Range
  
  eliminated_Y_final <- subset(eliminated_Y_final, eliminated_final[[v]] > (Q[1] - tol*iqr) & eliminated_final[[v]] < (Q[2]+tol*iqr))
  eliminated_final <- subset(eliminated_final, eliminated_final[[v]] > (Q[1] - tol*iqr) & eliminated_final[[v]] < (Q[2]+tol*iqr))
}

# Normalization
norm_params_train_final <- preProcess(eliminated_final) # getting normalization parameters
scaled.x_train_final <- predict(norm_params_train_final, eliminated_final) # applying them
scaled.x_test <- predict(norm_params_train_final, x_test) # applying them to test data

### TES - TRAIN - VAL SPLIT ###
# Generating big dataset again with engineerd features


# PCA #
pca <- prcomp(x_train) # pca list of objects 
pc <- matrix(unlist(pca[5]), ncol = 9, byrow = FALSE) # matrix of PCs

### LM REGRESSION ###
# Define training control
set.seed(42) 
train.control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
# Train the model
model <- train(scaled.x_train_final[,-c(5,7)], unlist(training_data$Y), method = "lm",
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
  knn.fit <- train(scaled.x_train_final[,-c(5,7)], unlist(training_data$Y), method = "knn",
                   trControl = train.control,
                   tuneGrid = expand.grid(k = seq(from = 1, to = 10, by = 1))
  )
  # Summarize the results
  print(knn.fit)
  print(rmse(unlist(training_data$Y), predict(knn.fit, scaled.x_train_final[,-c(5,7)])))
  
  samp[i] = knn.fit$results$RMSE[unlist(knn.fit$bestTune)]
  sampling_meth[i] = var
  i <- i+1
}
plot(factor(unlist(sampling_meth)), 
     unlist(samp), xlab = 'sampling method', ylab = 'knn rmse')

############# FINAL TRAINING OF THE KNN ####################################################################
train.control <- trainControl(method = "LOOCV", number = 5)
knn.fit <- train(scaled.x_train_final[,-c(5,7)], unlist(training_data$Y), method = "knn",
                 trControl = train.control,
                 tuneGrid = expand.grid(k = seq(from = 1, to = 10, by = 1))
)
print(rmse(unlist(training_data$Y), predict(knn.fit, scaled.x_train_final[,-c(5,7)])))
print(knn.fit)
knn_pred <- predict(knn.fit, scaled.x_test[,-c(5,7)])

### SAVING ###
save(fit, lm_pred, knn_pred, file = "Vasile_challenge1.RData")

### TESTING SAVE ###
load("Vasile_challenge1.RData")

