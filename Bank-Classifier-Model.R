options(java.parameters = "-Xmx1000m")
setwd("C:/Users/vaish/Dropbox/CSU/Data Mining/Assignment")
library(caret)
library(plyr)
library(dplyr)
library(C50)
library(kernlab)
#Load input data 
bank_full <- read.csv("bank-additional-full.csv", header=TRUE, sep=";")
str(bank_full)
class(bank_full)
nomvars <- c(2:10,15,21)
str(nomvars)
library(mlbench)
# how many missing values.
sum(is.na(bank_full))
# Lets check correlations , see - nomvars to exclude nominal variables for dimensionality reduction,
# not looking for factors but only nominal attributes
corrMatrix <- cor(bank_full[,-nomvars], use = "pairwise.complete.obs")
# Are there NA's ?
table(is.na(corrMatrix))
# replace NA's
corrMatrix <- replace(corrMatrix, which(is.na(corrMatrix)), 0)
corrMatrix
# Column indexes to remove
#where corelation is >0.9
rmcols <- findCorrelation(corrMatrix, cutoff = 0.9, verbose = TRUE)
# Column names to remove
colnames(corrMatrix[,rmcols])

# New dataset with columns removed
bank_full2 <- bank_full[,-which(names(bank_full) %in% colnames(corrMatrix[,rmcols]))]

# Factor levels-- we can do it in carrot much more easily
faclen <- bank_full2 %>% Filter(f = is.factor) %>% sapply(levels) %>% sapply(length) %>% (function(x) x[x<2])# these are all factors of length 1
facnames <-names(bank_full2) %in% names(faclen)
bank_full3 <- bank_full2[!facnames]

# Zero variance
# nearzerovar is from carrot package, it gives me data which has variance near to 0
nzv <- nearZeroVar(bank_full3)
nzv
bank_full4 <- bank_full3[,-nzv]
bank_full4
View(bank_full4)

# Lets do stratified sampling. Select rows to based on Class variable as strata
#40% are used to create training model, rest 60 % used in test dataset
# based on computing infrastructure, time, acceptable level of accuracy for 
#project to be success we choose the % of training data
TrainingDataIndex <- createDataPartition(bank_full4$y, p=0.40, list = FALSE)

# Create Training Data as subset 
trainingData <- bank_full4[TrainingDataIndex,]

# Everything else not in training is test data.
testData <- bank_full4[-TrainingDataIndex,]

# setting training control parameters
NoTrainingParameters <- trainControl(method = "none")# cross validation or boot straping or repeatedcv
TrainingParameters <- trainControl(method = "cv", number = 10)

######## Neural Networks########
set.seed(245)# use same set of random numbers to get same results again & again.
NNModel <- train(y ~.,data=trainingData, # nnet is neural network
                 method = "nnet",#or rf for random forest, put name of algorithm u want to use
                 trControl= TrainingParameters,
                 tuneGrid = data.frame(size = 5,
                                       decay = 0
                 ))
NNPredictions <-predict(NNModel, testData)
# See predictions
NNPredictions
# Create confusion matrix
cmNN <-confusionMatrix(NNPredictions, testData$y)
cmNN$overall
cmNN$byClass

######## SVM############
SVModel <- train(y ~ ., data = trainingData,
                 method = "svmPoly",
                 trControl= TrainingParameters,
                 tuneGrid = data.frame(degree = 1,
                                       scale = 1,
                                       C = 1
                 ))
SVMPredictions <-predict(SVModel, testData)
# See predictions
SVMPredictions
# Create confusion matrix
cmSVM <-confusionMatrix(SVMPredictions, testData$y)
cmSVM$overall
cmSVM$byClass

######### Naive bayes############
# train model with NB
NBModel <- train(y ~ .,data=trainingData,
              method = "nb",
              trControl = trainControl(method="none"),
              tuneGrid = data.frame(fL=0, usekernel=FALSE ,adjust=FALSE))
pred3 <- predict(NBModel, testData, type="raw")
#Warnings that probability is 0 for some cases
NBPredictions <-predict(NBModel, testData)
cmnb <-confusionMatrix(NBModel, testData$y)
cmnb$overall
cmnb$byClass

###decision tree#####
DecTreeModel <- train(y ~., data=trainingData, 
                      method = "C5.0",# is the decision tree
                      trControl= TrainingParameters,
                      na.action = na.omit# if missing values omit them
)

DecTreeModel
#Predict
DTPredictions <-predict(DecTreeModel, testData, na.action = na.pass)
# See predictions
DTPredictions
# Create confusion matrix
cm <-confusionMatrix(DTPredictions, testData$y)
cm$overall
cm$byClass


######## With PCA #######
#######
set.seed(245)# use same set of random numbers to get same results again & again.
NNModel_pca <- train(y ~.,data=trainingData, # nnet is neural network
                 method = "nnet",#or rf for random forest, put name of algorithm u want to use
                 trControl= NoTrainingParameters,
                 preProcess = c("pca"),
                 tuneGrid = data.frame(size = 5,
                                       decay = 0
                 ))
NNPredictions_pca <-predict(NNModel_pca, testData)
# See predictions
NNPredictions_pca
# Create confusion matrix
cmNN <-confusionMatrix(NNPredictions_pca, testData$y)
cmNN$overall
cmNN$byClass

######## SVM############
SVModel <- train(y ~ ., data = trainingData,
                 method = "svmPoly",
                 preProcess = c("pca"),
                 trControl= NoTrainingParameters,
                 tuneGrid = data.frame(degree = 1,
                                       scale = 1,
                                       C = 1
                 ))
SVMPredictions <-predict(SVModel, testData)
# See predictions
SVMPredictions
# Create confusion matrix
cmSVM <-confusionMatrix(SVMPredictions, testData$y)
cmSVM$overall
cmSVM$byClass

######### Naive bayes############
# train model with NB
NBModel <- train(y ~ .,data=trainingData,
                 method = "nb",
                 trControl =NoTrainingParameters ,
                 preProcess = c("pca"),
                 tuneGrid = data.frame(fL=0, usekernel=FALSE ,adjust=FALSE))
pred3 <- predict(NBModel, testData, type="raw")
#Warnings that probability is 0 for some cases
NBPredictions <-predict(NBModel, testData)
cmnb <-confusionMatrix(NBModel, testData$y)
cmnb$overall
cmnb$byClass

###decision tree#####
DecTreeModel <- train(y ~., data=trainingData, 
                      method = "C5.0",# is the decision tree
                      trControl= TrainingParameters,
                      preProcess = c("pca"),
                      na.action = na.omit# if missing values omit them
)

DecTreeModel
#Predict
DTPredictions <-predict(DecTreeModel, testData, na.action = na.pass)
# See predictions
DTPredictions
# Create confusion matrix
cm <-confusionMatrix(DTPredictions, testData$y)
cm$overall
cm$byClass

