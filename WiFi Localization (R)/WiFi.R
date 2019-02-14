# needed libraries
install.packages('doMC')
install.packages('ranger')
install.packages('e1071')
install.packages('kknn')
install.packages('kernlab')
library(kernlab)
library(e1071)
library(ranger)
library(caret)
library(doMC)
library(dplyr)
library(doParallel)
library(kknn)

# parrallel processing
detectCores()
registerDoMC(cores = 4)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)


# set working directory to location with data files
getwd()
setwd('/Users/sergiorobledo/Desktop/OneDrive/Workspace/WiFi_R')
dir()

### import data & preprocess ###
# convert labels to factors
# drop unecessary features
# define labels
learn <- read.csv('trainingData.csv', header = TRUE)
valid <- read.csv('validationData.csv', header = TRUE)
learn <- learn[-c(521:522,525:529)]
valid <- valid[-c(521:522,525:529)]
learn$FLOOR <- as.factor(learn$FLOOR)
learn$BUILDINGID <- as.factor(learn$BUILDINGID)
valid$FLOOR <- as.factor(valid$FLOOR)
valid$BUILDINGID <- as.factor(valid$BUILDINGID)
# combine building and floor into a single column
learn$building_floor <- as.factor(paste(learn$BUILDINGID, learn$FLOOR, sep = '.'))
valid$building_floor <- as.factor(paste(valid$BUILDINGID, valid$FLOOR, sep = '.'))
dim(learn)
str(learn[521:523])
dim(valid)
str(valid[521:523])


# normalizing WAPs signal: 0 = no signal, 105 = strongest signal
WAPs <- learn[1:520]
WAPs <- as.data.frame(apply(WAPs, 2, function(x) ifelse(x == 100, 0, x + 105)))
min(WAPs)
max(WAPs)

WAPs_test <- valid[1:520]
WAPs_test <- as.data.frame(apply(WAPs_test, 2, function(x) ifelse(x == 100, 0, x + 105)))
min(WAPs_test)
max(WAPs_test)

# training/validated data sets w/ updated WAPs signal values
learn[1:520] <- WAPs
min(learn[1:520])
max(learn[1:520])

valid[1:520] <- WAPs_test
min(valid[1:520])
max(valid[1:520])

# Principal Component Analysis (PCA): Reduce number of WAP features 
# 95% of Variance is reached at Principal Compoment 110
# 99% of Variance is reached at Principal Compoment 221
set.seed(123)
pca_values <- prcomp(learn[,1:520],
                     center = TRUE,
                     scale. = FALSE)
summary(pca_values)$importance[,1:110]
summary(pca_values)$importance[,1:221]
str(pca_values)
learn_pca <- as.data.frame(predict(pca_values, learn[,1:520]))
learn_pca$building_floor <- learn$building_floor
dim(learn_pca)
valid_pca <- as.data.frame(predict(pca_values, valid[,1:520]))
valid_pca$building_floor <- valid$building_floor
dim(valid_pca)


# sampling
set.seed(123)
learn_pca_250 <- learn_pca[sample(1:nrow(learn_pca), 250, replace = FALSE),]
learn_pca_500 <- learn_pca[sample(1:nrow(learn_pca), 500, replace = FALSE),]
learn_pca_1000 <- learn_pca[sample(1:nrow(learn_pca), 1000, replace = FALSE),]
learn_pca_2000 <- learn_pca[sample(1:nrow(learn_pca), 2000, replace = FALSE),]
learn_pca_4000 <- learn_pca[sample(1:nrow(learn_pca), 4000, replace = FALSE),]
learn_pca_8000 <- learn_pca[sample(1:nrow(learn_pca), 8000, replace = FALSE),]


###                                                ###
### 'out of the box' model building and evaluation ###
###                                                ###


# trainControl: repeated cross-validation
fit_control <- trainControl(method = 'repeatedcv',
                            number = 10,
                            repeats = 1)

# Random Forest Classifier
rf_tG <- expand.grid(.mtry = 110,
                     .min.node.size = 1,
                     .splitrule = 'extratrees')
set.seed(123)
start_time <- Sys.time()
rf_fit <- train(learn_pca_8000[,1:110], 
                learn_pca_8000$building_floor,
                trControl = fit_control,
                method = 'ranger',
                tuneGrid = rf_tG)
end_time <- Sys.time()
rf_fit_time <- end_time - start_time
rf_fit_time
rf_fit
rf_building_pred <- predict(rf_fit, valid_pca)
confusionMatrix(rf_building_pred, valid_pca$building_floor)

# K-Nearest Neighbor
knn_tG <- expand.grid(.kmax= c(1),
                      .distance = c(1),
                      .kernel = 'optimal')
set.seed(123)
start_time <- Sys.time()
knn_fit <- train(learn_pca_8000[,1:50], 
                 learn_pca_8000$building_floor,
                 trControl = fit_control,
                 method = 'kknn',
                 tuneGrid = knn_tG)
end_time <- Sys.time()
knn_fit_time <- end_time - start_time
knn_fit_time
knn_fit
knn_building_pred <- predict(knn_fit, valid_pca)
confusionMatrix(knn_building_pred, valid_pca$building_floor)

# Support Vector Machines w/ Linear Kernel
svm_tG <- expand.grid(.cost = 0.1)
set.seed(123)
start_time <- Sys.time()
svm_linear_fit <- train(learn_pca[,1:50], 
                        learn_pca$building_floor,
                        trControl = fit_control,
                        method = 'svmLinear2',
                        tuneGrid = svm_tG)
end_time <- Sys.time()
svm_linear_fit_time <- end_time - start_time
svm_linear_fit_time
svm_linear_fit
svm_linear_building_pred <- predict(svm_linear_fit, valid_pca)
confusionMatrix(svm_linear_building_pred, valid_pca$building_floor)

# resamples
# only problem is that these are the scores of the models on the training data,
# but resamples does not show the scores of the models on the testing data
# testing data scores were much lower than the training data scores; which is
# a probable indication of overfitting the training data
ModelData <- resamples(list(RF = rf_fit, KNN = knn_fit, SVM = svm_linear_fit))
summary(ModelData)

# Best Model: Support Vector Machine w/ Linear Kernel
# Samples Used: 19,937
# Top 50 Principal Components
# Cost: 0.1
# Training Time ~ 1 min
# Accuracy Score ~ 92.26%
# Kappa Score ~ 0.9134

Best_Model <- svm_linear_fit
predictions <- predict(Best_Model, valid_pca)
confusionMatrix(predictions,valid_pca$building_floor)
