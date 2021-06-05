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
library(ggplot2) # plotting
library(cowplot) # for nicer ggplots
library(e1071) # SVM
library (ROCR)
library(MLmetrics)
library(AUC)
library(mltools)
library(rpart)
library(pROC)
library(imbalance)
library(performanceEstimation)
library(DMwR)

### LOADING DATA / DATABASE SELECTION ###
setwd("D:\\Uni\\SL\\SL_Project_2\\")


db_train <- as.data.frame(read.csv(".\\dataset\\ADCNtrain.csv"))
db_test <- as.data.frame(read.csv(".\\dataset\\ADCNtest.csv"))

# db_train <- as.data.frame(read.csv(".\\dataset\\ADMCItrain.csv"))
# db_test <- as.data.frame(read.csv(".\\dataset\\ADMCItest.csv"))

# db_train <- as.data.frame(read.csv(".\\dataset\\MCICNtrain.csv"))
# db_test <- as.data.frame(read.csv(".\\dataset\\MCICNtest.csv"))

db_train$Labels <- as.factor(db_train$Labels)
### EXPLORING DATA ###

# correlation
train_cor = cor(db_train[-c(1,ncol(db_train))])
heatmap(train_cor) # correlation matrix

remove(train_cor) # decluttering

# data imbalance exploration
numPositive <-length(which(db_train$Labels==levels(db_train$Labels)[1])) # ex: levels(db_train$Labels)[1] = "AD" for ADCN
numNegative <-length(which(db_train$Labels==levels(db_train$Labels)[2])) # coded for dynamic interchange of datasets
imb_idx <- imbalanceRatio(db_train, classAttr = "Labels")

### FEATURE ENGINEERING & SELECTION ###

# solving data imbalance
# balanced.train.data <- SMOTE(Labels ~., smote_data, perc.over = 100, perc.under = 200)
# “RACOG”, “RWO”, “ADASYN”, “ANSMOTE”, “SMOTE”, “MWMOTE”, “BLSMOTE”, “DBSMOTE”, “SLMOTE”, “RSLSMOTE”
newSamples <- oversample(db_train[2:ncol(db_train)] ,method = "SMOTE",classAttr = 'Labels', ratio = imb_idx*2.5) # ADCN
# newSamples <- oversample(db_train[2:ncol(db_train)] ,method = "SMOTE",classAttr = 'Labels', ratio = imb_idx*4) # ADMCI
# newSamples <- db_train[2:ncol(db_train)] # MCICN, with an imbalance index of ~0.58 is better left like this (rename just for semi-automation purpouses)

# new imbalance ratio
newSamples_imb_dx <- imbalanceRatio(newSamples, classAttr = "Labels") # check

# Normalization
norm_params <- preProcess(newSamples[-c(1,ncol(db_train))]) # getting normalization parameters
newSamples[-c(1,ncol(db_train))] <- predict(norm_params, newSamples[-c(1,ncol(db_train))]) # applying them
db_test <- predict(norm_params, db_test) # applying them to test data

# Renaming
x_train <- newSamples
y_train <- as.factor(x_train$Labels)

# Feature Selection based on medical insight
a <- c(names(select(x_train,matches("HLA"))),
       names(select(x_train,matches("SLC6"))),
       names(select(x_train,matches("SRP"))),
       names(select(x_train,matches("hippoca"))),
       names(select(x_train,matches("frontal"))),
       names(select(x_train,matches("tempo"))),
       names(select(x_train,matches("parie"))),
       names(select(x_train,matches("amyg"))),
       names(select(x_train,matches("call"))))

x_train <- select(x_train, a)
############################### KNN ############################################
set.seed(42) # because 42 it's the answer to everything
train.control <- trainControl(method = "cv", number = 5)
knn.fit <- train(x_train[1:ncol(x_train)-1], y_train, method = "knn",
                 trControl = train.control,
                 tuneGrid = expand.grid(k = seq(from = 1, to = 20, by = 2)),
                 metric = "Accuracy")
print(knn.fit)

y_pred_knn <- predict(knn.fit, x_train) # predicting on training data

roc_knn <- roc(as.numeric(y_train), as.numeric(y_pred_knn)) # builds a ROC curve and returns a “roc” object
plot(roc_knn, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(roc_knn$auc[[1]],2))) # plot AUC
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

mcc_knn <- mcc(y_pred_knn, y_train)

# pred_knn <- prediction(as.numeric(y_pred_knn), as.numeric(y_train)) # prediction object for ROCR lib
# f1_knn <- performance(pred_knn,"f")
# acc_knn <- performance(pred_knn,"acc", "rec")
# spec_knn <- specificity(y_train, y_pred_knn)
# sens_knn <- performance(pred_knn,"sens")
# prec_knn <- performance(pred_knn,"prec")

# a <- table(c(mcc_knn, roc_knn$auc[1]))

############################ adaboost #####################################
set.seed(42) # because 42 it's the answer to everything
train.control <- trainControl(method = "cv", number = 5)
ada.fit <- train(x_train[1:ncol(x_train)-1], y_train, method = "adaboost",
                 trControl = train.control,
                 tuneGrid = expand.grid(nIter = seq(from = 100, to = 150, by = 50),
                                        method = c("Adaboost.M1","Real adaboost")),
                 metric = "Accuracy")
print(ada.fit)

y_pred_ada <- predict(ada.fit, x_train) # predicting on training data

roc_ada <- roc(as.numeric(y_train), as.numeric(y_pred_ada)) # builds a ROC curve and returns a “roc” object
plot(roc_ada, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(roc_ada$auc[[1]],2))) # plot AUC
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

mcc_ada <- mcc(y_pred_ada, y_train)



# pred_ada <- prediction(as.numeric(y_pred_ada), as.numeric(y_train)) # prediction object for ROCR lib
# f1_ada <- performance(pred_ada,"f")
# acc_ada <- performance(pred_ada,"acc")
# spec_ada<- performance(pred_ada,"spec")
# sens_ada <- performance(pred_ada, "sens")
# prec_ada <- performance(pred_ada,"prec")

########################## nodeHarvest #######################################
set.seed(42) # because 42 it's the answer to everything
train.control <- trainControl(method = "cv", number = 5)
nh.fit <- train(x_train[1:ncol(x_train)-1], y_train, method = "nodeHarvest",
                 trControl = train.control,
                 metric = "Accuracy")
print(nh.fit)

y_pred_nh <- predict(nh.fit, x_train) # predicting on training data

roc_nh <- roc(as.numeric(y_train), as.numeric(y_pred_nh)) # builds a ROC curve and returns a “roc” object
plot(roc_nh, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(roc_nh$auc[[1]],2))) # plot AUC
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

mcc_nh <- mcc(y_pred_nh, y_train)

# pred_nh <- prediction(as.numeric(y_pred_nh), as.numeric(y_train)) # prediction object for ROCR lib
# f1_nh <- performance(pred_nh,"f")
# acc_nh <- performance(pred_nh,"acc")
# spec_nh<- performance(pred_nh,"spec")
# sens_nh <- performance(pred_nh, "sens")
# prec_nh <- performance(pred,"prec")

############################ xgbDART #####################################
set.seed(42) # because 42 it's the answer to everything
train.control <- trainControl(method = "cv", number = 5)
xgb.fit <- train(x_train[1:ncol(x_train)-1], y_train, method = "xgbDART",
                trControl = train.control,
                metric = "Accuracy")
print(xgb.fit)

y_pred_xgb <- predict(xgb.fit, x_train) # predicting on training data

roc_xgb <- roc(as.numeric(y_train), as.numeric(y_pred_xgb)) # builds a ROC curve and returns a “roc” object
plot(roc_xgb, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(roc_xgb$auc[[1]],2))) # plot AUC
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

mcc_xgb <- mcc(y_pred_xgb, y_train)

# pred_xgb <- prediction(as.numeric(y_pred_xgb), as.numeric(y_train)) # prediction object for ROCR lib
# f1_xgb <- performance(pred_xgb,"f")
# acc_xgb <- performance(pred_xgb,"acc")
# spec_xgb<- performance(pred_xgb,"spec")
# sens_xgb <- performance(pred_xgb, "sens")
# prec_xbg <- performance(pred_xgb,"prec")
# 
# save(y_pred_xgb, roc_xgb, mcc_xgb, pred_xgb, f1_xgb, acc_xgb, spec_xgb, sens_xgb, prec_xbg,
#      file = "xgb_stuff.RData") # saving training features (models)



############################## SVMS #########################################
x_train$Labels <- y_train # necesary only for SVM training format
set.seed(42) # because 42 it's the answer to everything
train.control <- trainControl(method = "cv", repeats = 5, number = 5,
                              classProbs =  TRUE)

svmRadial.fit <- train(Labels ~ .,
                data = x_train,
                method = "svmRadial",
                # preProc = c("center", "scale"),
                tuneGrid = expand.grid(sigma = seq(from = 0.0016, to = 0.003, by = 0.0002),
                                       C = seq(from = 1.5, to = 3, by = 0.10)),
                trControl = train.control)

print(svmRadial.fit)

y_pred_svmRadial <- predict(svmRadial.fit, x_train) # predicting on training data

roc_svmRadial <- roc(as.numeric(y_train), as.numeric(y_pred_svmRadial)) # builds a ROC curve and returns a “roc” object
plot(roc_svmRadial, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(roc_svmRadial$auc[[1]],2))) # plot AUC
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

mcc_svmRadial <- mcc(y_pred_svmRadial, y_train)

## SAVE ##
final_pred <- predict(svmRadial.fit, db_test)
final_db <- as.data.frame(db_test[1])
final_db$Labels <- final_pred
save(final_db, file = "0063828_Vasile_challenge2_ADCNres.RData") # saving training features (models)
best_features = list()
i = 1
for (var in a){
  best_features[i] <- which(colnames(db_train) == var)
  i = i+1
}
Best_features <- unlist(best_features)
save(Best_features, file = "0063828_Vasile_challenge2_ADCNfeat.RData") # saving training features (models)
## SAVE ##

# pred_svmRadial <- prediction(as.numeric(y_pred_svmRadial), as.numeric(y_train)) # prediction object for ROCR lib
# f1_svmRadial <- performance(pred_svmRadial,"f")
# acc_svmRadial <- performance(pred_svmRadial,"acc")
# spec_svmRadial<- performance(pred_svmRadial,"spec")
# sens_svmRadial <- performance(pred_svmRadial, "sens")
# prec_svmRadial <- performance(pred_svmRadial,"prec")

############## SVM POLY ##

train.control <- trainControl(method = "cv", repeats = 5, number = 5,
                              classProbs =  TRUE)
svmPoly.fit <- train(Labels ~ .,
                       data = x_train,
                       method = "svmPoly",
                       # preProc = c("center", "scale"),
                       # tuneGrid = expand.grid(degree = seq(from = 5, to = 7, by = 1),
                       #                        scale = seq(from = 0.1, to = 0.2, by = 0.05),
                       #                        C = seq(from = 0.25, to = 1, by = 0.25)),
                       trControl = train.control)
print(svmPoly.fit)

y_pred_svmPoly <- predict(svmPoly.fit, x_train) # predicting on training data

roc_svmPoly <- roc(as.numeric(y_train), as.numeric(y_pred_svmPoly)) # builds a ROC curve and returns a “roc” object
plot(roc_svmPoly, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(roc_svmPoly$auc[[1]],2))) # plot AUC
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

mcc_svmPoly <- mcc(y_pred_svmPoly, y_train)


# pred_svmPoly <- prediction(as.numeric(y_pred_svmPoly), as.numeric(y_train)) # prediction object for ROCR lib
# f1_svmPoly <- performance(pred_svmPoly,"f")
# acc_svmPoly <- performance(pred_svmPoly,"acc")
# spec_svmPoly<- performance(pred_svmPoly,"spec")
# sens_svmPoly <- performance(pred_svmPoly, "sens")
# prec_svmPoly <- performance(pred_svmPoly,"prec")

############# svmLinear ##

train.control <- trainControl(method = "cv", number = 5,
                              classProbs = TRUE)
svmLinear.fit <- train(Labels ~ .,
                     data = x_train,
                     method = "svmLinear",
                     # preProc = c("center", "scale"),
                     # tuneGrid = expand.grid(C = seq(from = 0.001, to = 0.01, by = 0.0025)),
                     trControl = train.control)
print(svmLinear.fit)

y_pred_svmLinear <- predict(svmLinear.fit, x_train) # predicting on training data

roc_svmLinear <- roc(as.numeric(y_train), as.numeric(y_pred_svmLinear)) # builds a ROC curve and returns a “roc” object
plot(roc_svmLinear, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(roc_svmLinear$auc[[1]],2))) # plot AUC
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

mcc_svmLinear <- mcc(y_pred_svmLinear, y_train)

# pred_svmLinear <- prediction(as.numeric(y_pred_svmLinear), as.numeric(y_train)) # prediction object for ROCR lib
# f1_svmLinear <- performance(pred_svmLinear,"f")
# acc_svmLinear <- performance(pred_svmLinear,"acc")
# spec_svmLinear<- performance(pred_svmLinear,"spec")
# sens_svmLinear <- performance(pred_svmLinear, "sens")
# prec_svmLinear <- performance(pred_svmLinear,"prec")
#############################################################################

### TESTING SAVE ###

save(best_features, file = "0063828_Vasile_challenge2_ADCNfeat.RData") # saving training features (models)
save(best_features, file = "0063828_Vasile_challenge2_MCICNfeat.RData") # saving training features (models)
save(best_features, file = "0063828_Vasile_challenge2_ADMCIfeat.RData") # saving training features (models)
load("0063828_Vasile_challenge2_ADCNres.RData") # saving training features (models)
load("0063828_Vasile_challenge2_ADMCIfeat.RData") # saving training features (models)