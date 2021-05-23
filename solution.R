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

# tolerance <- seq(from = 0, to = 4, by = 0.1)
# i <- 1
# l_lm <- list()
# l_knn <- list()
# 
# for (t in tolerance){
# print(t)

training_data <- as_tibble(read.csv("D:\\Uni\\SL\\SL_Project\\train_ch.csv"))
test_data <- as_tibble(read.csv("D:\\Uni\\SL\\SL_Project\\test_ch.csv"))


x_train_final <- training_data %>% select(2:10)
y_train_final <- training_data %>% select(11)

x_test <- test_data %>% select(2:10)


remove(training_data, test_data) #decluttering

### EXPLORING DATA ###
# # covarience
# train_cov = cov(x_train_final)
# plotCov(train_cov)
# 
# correlation
train_cor = cor(x_train_final)
ggcorr(x_train_final)

remove(train_cor, train_cov) #decluttering

"OBS: There's high correelation between
v1 and v5
v1 and v7
v5 and v7
So we can discard v5 and v7"

### FEATURE ENGINEERING ###

# Eliminating outliers in training and FINAL TRAINING data
# eliminated <- x_train
# eliminated_Y <- y_train

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
training_data <- scaled.x_train_final
training_data$Y <- eliminated_Y_final

remove(eliminated_Y_final, eliminated_final,
       iqr, low, Q, tol, up, v, x_test, norm_params_train_final) #decluttering

set.seed(42) # because 42 it's the answer to everything
spec = c(train = .9, validate = .1)
g = sample(cut(
  seq(nrow(training_data)), 
  nrow(training_data)*cumsum(c(0,spec)),
  labels = names(spec)
))
splt = split(training_data, g)
train = as.data.frame(splt[1])
val = as.data.frame(splt[2])
names(train) <- names(training_data)
names(val) <- names(training_data)

x_train <- train %>% select(1:9)
y_train <- train %>% select(10)
x_val <- val %>% select(1:9)
y_val <- val %>% select(10)

remove(splt, train, val)

# Multipurpouse test --> with tol = 5, we let outliers disturb the fit, stick with tol = 4
# --> tol_5_rmse = 0.6935136 // tol_4_rmse = 0.6728488 // tol_4.5_rmse = 0.7013478
fit <- lm(unlist(y_train) ~ ., data = x_train)
val_pred = predict(fit, x_val)
rmse_pred <- rmse(unlist(y_val),val_pred)
print(rmse_pred)

# PCA #
pca <- prcomp(x_train) # pca list of objects 
pc <- matrix(unlist(pca[5]), ncol = 9, byrow = FALSE) # matrix of PCs

### LM REGRESSION ###
### Trying out K-fold Cross Validation ###
# testing to see if eliminating v5 and v7 helps --> Yeap, it helps
# Define training control
set.seed(42) 
train.control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
# Train the model
model <- train(scaled.x_train_final[,-c(5,7)], unlist(training_data$Y), method = "lm",
               trControl = train.control)
# Summarize the results
print(model)

# #################### KNN ###############
# 
# train.control <- trainControl(method = "LOOCV", number = 5)
# knn.fit <- train(scaled.x_train_final[,-c(5,7)], unlist(training_data$Y), method = "knn",
#                  trControl = train.control,
#                  tuneGrid = expand.grid(k = seq(from = 2, to = 3, by = 1))
# )
# print(rmse(unlist(training_data$Y), predict(knn.fit, scaled.x_train_final[,-c(5,7)])))
# print(knn.fit)

#   
#   l_lm[[i]] <- model$results$RMSE
#   l_knn[[i]] <- knn.fit$results$RMSE
#   i <- i + 1
# }
# plot(tolerance, l_lm, xlab = 'tolerance', ylab = 'lm rmse')
# 
# l_knn_fine <- list()
# i = 1
# for (var in 1:41){
#   l_knn_fine[[i]] <- l_knn[[var]][2]
#   i <- i+1
# }
# plot(tolerance, l_knn_fine, xlab = 'tolerance', ylab = 'knn rmse')


set.seed(42) 
train.control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
# Train the model
model <- train(scaled.x_train_final, unlist(training_data$Y), method = "lm",
               trControl = train.control)
# Summarize the results
print(model)

samp = list()
sampling_meth = list()
i = 1
# Testing different sampling algorithms
for (var in c('cv', 'boot', 'optimism_boot',
              'boot_all', 'repeatedcv', 'LOOCV', 'LGOCV')){
  print(var)
  # Define training control
  set.seed(42) 
  train.control <- trainControl(method = var, number = 10)
  # Train the model
  fit <- train(scaled.x_train_final[,-c(5,7)], unlist(training_data$Y), method = "lm",
               trControl = train.control)
  # Summarize the results
  print(fit)
  samp[i] = fit$results$RMSE
  sampling_meth[i] = var
  i <- i+1
  # print(rmse(unlist(training_data$Y), predict(fit, scaled.x_train_final[,-c(5,7)])))
}

# sampling_meth = c('cv', 'boot', 'optimism_boot','boot_all', 'repeatedcv', 'LOOCV', 'LGOCV')
plot(factor(unlist(sampling_meth)), 
     unlist(samp), xlab = 'sampling method', ylab = 'lm rmse')
# legend(1, 95, legend=c('cv', 'boot', 'optimism_boot','boot_all', 'repeatedcv', 'LOOCV', 'LGOCV'),
#        col=c("red", "blue"), lty=1:2, cex=0.8)

############### Final Fit & Predicting with lm #############################################################
set.seed(42) 
train.control <- trainControl(method = "cv", number = 10)
fit <- train(scaled.x_train_final[,-c(5,7)], unlist(training_data$Y), method = "lm",
             trControl = train.control)
print(fit)
print(rmse(unlist(training_data$Y), predict(fit, scaled.x_train_final[,-c(5,7)])))
lm_pred <- predict(fit, scaled.x_test)

### KNN ###
# testing the correct number of neightbours --> 3: 0.412 rmse
# for (var in 3:20){
#   print(var)
#   knn.fit <- knn.reg(scaled.x_train_final[,-c(5,7)], test = NULL, unlist(training_data$Y), k = var, algorithm=c("kd_tree", "cover_tree", "brute"))
#   plot(unlist(training_data$Y), knn.fit$pred, xlab="y", ylab=expression(hat(unlist(training_data$Y))))
#   print(rmse(unlist(training_data$Y), knn.fit$pred))
# }

# testing if eliminating v5 and v7 helps --> it does
# for (var in 3:20){
#   knn.fit <- knn.reg(scaled.x_train_final[,-c(5,7)], test = NULL, scaled.y_train, k = var, algorithm=c("kd_tree", "cover_tree", "brute"))
#   plot(scaled.y_train, knn.fit$pred, xlab="y", ylab=expression(hat(scaled.y_train)))
#   print(rmse(scaled.y_train, knn.fit$pred))
# }

# testing best algorithm 
# --> tied: kd_tree == brute
# for (var in c("kd_tree", "cover_tree", "brute")){
#   knn.fit <- knn.reg(scaled.x_train_final[,-c(5,7)], test = NULL, scaled.y_train, k = 3, algorithm=var)
#   plot(scaled.y_train, knn.fit$pred, xlab="y", ylab=expression(hat(scaled.y_train)))
#   print(rmse(scaled.y_train, knn.fit$pred))
# }

# testing PCA performance --> curious, the best PC to approximate data was PC4
# --> performance increases drastically with the addition of PC4 and PC5
# --> best result is achieved with all PC
# for (var in 1:7){
#   knn.fit <- knn.reg(pc[,var], test = NULL, scaled.y_train, k = 3, algorithm='kd_tree')
#   plot(scaled.y_train, knn.fit$pred, xlab="y", ylab=expression(hat(scaled.y_train)))
#   print(rmse(scaled.y_train, knn.fit$pred))
# }


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

