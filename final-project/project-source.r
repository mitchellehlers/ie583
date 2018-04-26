library(data.table)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)
library(lubridate)
library(DMwR)
library(randomForest)
library(Boruta)
library(e1071)
library(RWeka)

#### Read Data In ####
# train_sample.csv per Kaggle.com contains 100K random instances from the full training data.
train_sbset <- read.csv("C:/bench/iowastate/datasets/ie583/final-project/train_sample.csv", stringsAsFactors = F, nrows=10000)

################################
####### Exploration ############
################################

# brief look and dataset structure
glimpse(train_sbset)
str(train_sbset)

# Check for missing values
colSums(is.na(train_sbset))
colSums(train_sbset == "")  ## there are 99,7773 instances where a column has a blank value "" populated in a cell

#make a table of the "Class value column" to understand the distibution
table(train_sbset$is_attributed) #there are only 227 instances that populated the value with a 1


###################################
###### Add A Class Attribute 
##################################

# rename the 'Is_Attributed' column to 'Class'
colnames(train_sbset)[colnames(train_sbset) == 'is_attributed'] <- 'Class'
str(train_sbset)

#change 'Class' column to an factor
train_sbset$Class <- as.factor(train_sbset$Class)
levels(train_sbset$Class) <- c("zeros", "ones")
head(train_sbset)

###################################
####  Construct Derived Attirbutes
###################################

train_sbset$click_time <- as.POSIXct(as.character(train_sbset$click_time))
train_sbset <- train_sbset %>% 
  mutate(hour = hour(click_time),
         minute = minute(click_time),
         am_pm = ifelse(hour(click_time) >= 12, "AM", "PM"),
         ip_device_os_channel_app = paste(ip,'-',device,'-',os,'-',channel,'-',app),
         ip_device_os = paste(ip,'-',device,'-',os),
         ip_device = paste(ip,'-',device),
         ip_channel_app = paste(ip,'-',channel,'-',app),
         ip_channel = paste(ip,'-',channel),
         ip_app = paste(ip,'-',app),
         channel_app = paste(channel,'-',app),
         channel_app_os_device = paste(channel,'-',app,'-',os,'-',device),
         channel_os_device = paste(channel,'-',os,'-',device)
  )    
head(train_sbset)

#factors
train_sbset$Class <- factor(train_sbset$Class)
train_sbset$am_pm <- factor(train_sbset$am_pm)
train_sbset$ip_device_os_channel_app <- factor(train_sbset$ip_device_os_channel_app)
train_sbset$ip_device_os <- factor(train_sbset$ip_device_os)
train_sbset$ip_device <- factor(train_sbset$ip_device)
train_sbset$ip_channel_app <- factor(train_sbset$ip_channel_app)
train_sbset$ip_channel <- factor(train_sbset$ip_channel)
train_sbset$ip_app <- factor(train_sbset$ip_app)
train_sbset$channel_app <- factor(train_sbset$channel_app)
train_sbset$channel_app_os_device <- factor(train_sbset$channel_app_os_device)
train_sbset$channel_os_device <- factor(train_sbset$channel_os_device)
head(train_sbset)

#########################################
#### Variable Importance Measurment #####
#########################################

set.seed(123)
boruta.train <- Boruta(Class ~ ., data = train_sbset, doTrace = 2)
print(boruta.train)
attStats(boruta.train)
plot(boruta.train, cex.axis=.7, las=2, xlab="", main="Variable Importance") 

#removing attributed time as this is the time the app was downloaded
projTrainFS<-train_sbset[, c(1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20)]
head(projTrainFS)

# Split off only the important attributes
projTrainFS<-train_sbset[, c(1,2,18,19,20,8)]
head(projTrainFS)

######################################
#### Instance Sampling #####
######################################

# how imbalenced is the data?
table(projTrainFS$Class)
prop.table(table(projTrainFS$Class))

# Will try SMOTE, Undersampling, and Oversampling below in Predicitive Modeling

#########################################
#### Predictive Modeling  #####
#########################################

# Attempt 1a) Random Forest with SMOTE
cvCount = 3
ctrl_cv = trainControl(method = "cv", number = cvCount, savePred = T, classProb = T, sampling="smote")
cols = ncol(projTrainFS)
tunegrid <- expand.grid(.mtry=c(1:cols))
folds = split(sample(nrow(projTrainFS), nrow(projTrainFS),replace=FALSE), as.factor(1:cvCount))
train.accuracy.estimate.rf = NULL
fold.accuracy.estimate.rf = NULL
prediction.rf <- data.frame()
for(f in 1:cvCount){
  testData = projTrainFS[folds[[f]],]
  trainingData = projTrainFS[-folds[[f]],]
  trainingData2 = SMOTE(Class~., data=trainingData)   
  RF_model <- train(Class~., data=trainingData2, method="rf", tuneGrid=tunegrid, trControl=ctrl_cv)
  best<-as.numeric(RF_model$bestTune)
  show(RF_model)
  tempPredict <- predict(RF_model,testData)
  prediction.rf <- rbind(prediction.rf, as.data.frame(tempPredict))
  train.accuracy.estimate.rf[f] = as.numeric(RF_model$results[best,3])
  fold.accuracy.estimate.rf[f] = (table(tempPredict,testData$Class)[1,1]+table(tempPredict,testData$Class)[2,2])/length(testData$Class)
}
mean(train.accuracy.estimate.rf)
mean(fold.accuracy.estimate.rf)

result.rf <- cbind(prediction.rf, projTrainFS[, 6])
names(result.rf) <- c("Predicted", "Actual")

# Attempt 1b) Random Forest with Undersampling
cvCount = 3
ctrl_cv = trainControl(method = "cv", number = cvCount, savePred = T, classProb = T, sampling="down")
cols = ncol(projTrainFS)
tunegrid <- expand.grid(.mtry=c(1:cols))
folds = split(sample(nrow(projTrainFS), nrow(projTrainFS),replace=FALSE), as.factor(1:cvCount))
train.accuracy.estimate.rf.down = NULL
fold.accuracy.estimate.rf.down = NULL
prediction.rf.down <- data.frame()
for(f in 1:cvCount){
  testData = projTrainFS[folds[[f]],]
  trainingData = projTrainFS[-folds[[f]],]
  trainingData2 = downSample(x = trainingData[,-ncol(trainingData)],
                             y = trainingData$Class)    
  RF_model_down <- train(Class~., data=trainingData2, method="rf", tuneGrid=tunegrid, trControl=ctrl_cv)
  best<-as.numeric(RF_model_down$bestTune)
  show(RF_model_down)
  tempPredict <- predict(RF_model,testData)
  prediction.rf.down <- rbind(prediction.rf.down, as.data.frame(tempPredict))
  train.accuracy.estimate.rf.down[f] = as.numeric(RF_model_down$results[best,3])
  fold.accuracy.estimate.rf.down[f] = (table(tempPredict,testData$Class)[1,1]+table(tempPredict,testData$Class)[2,2])/length(testData$Class)
}
mean(train.accuracy.estimate.rf.down)
mean(fold.accuracy.estimate.rf.down)

result.rf.down <- cbind(prediction.rf.down, projTrainFS[, 6])
names(result.rf.down) <- c("Predicted", "Actual")

# Attempt 1c) Random Forest with Oversampling - NOTE: Took 3hours to run on 16gb RAM machine
cvCount = 3
ctrl_cv = trainControl(method = "cv", number = cvCount, savePred = T, classProb = T, sampling="up")
cols = ncol(projTrainFS)
tunegrid <- expand.grid(.mtry=c(1:cols))
folds = split(sample(nrow(projTrainFS), nrow(projTrainFS),replace=FALSE), as.factor(1:cvCount))
train.accuracy.estimate.rf.up = NULL
fold.accuracy.estimate.rf.up = NULL
prediction.rf.up <- data.frame()
for(f in 1:cvCount){
  testData = projTrainFS[folds[[f]],]
  trainingData = projTrainFS[-folds[[f]],]
  trainingData2 = upSample(x = trainingData[,-ncol(trainingData)],
                           y = trainingData$Class)    
  RF_model_up <- train(Class~., data=trainingData2, method="rf", tuneGrid=tunegrid, trControl=ctrl_cv)
  best<-as.numeric(RF_model_up$bestTune)
  show(RF_model_up)
  tempPredict <- predict(RF_model,testData)
  prediction.rf.up <- rbind(prediction.rf.down, as.data.frame(tempPredict))
  train.accuracy.estimate.rf.up[f] = as.numeric(RF_model_up$results[best,3])
  fold.accuracy.estimate.rf.up[f] = (table(tempPredict,testData$Class)[1,1]+table(tempPredict,testData$Class)[2,2])/length(testData$Class)
}
mean(train.accuracy.estimate.rf.up)
mean(fold.accuracy.estimate.rf.up)

result.rf.up <- cbind(prediction.rf.up, projTrainFS[, 6])
names(result.rf.up) <- c("Predicted", "Actual")

# Attempt 2) Decision Tree with SMOTE
cvCount = 3
ctrl_cv = trainControl(method = "cv", number = cvCount, savePred = T, classProb = T, sampling="smote")
treeGrid_dectree = expand.grid(C=(1:3)*0.1, M=(1:3))
folds = split(sample(nrow(projTrainFS), nrow(projTrainFS),replace=FALSE), as.factor(1:cvCount))
train.accuracy.estimate.j48 = NULL
fold.accuracy.estimate.j48 = NULL
prediction.j48 <- data.frame()
for(f in 1:cvCount){
  testData = projTrainFS[folds[[f]],]
  trainingData = projTrainFS[-folds[[f]],]
  trainingData2 = SMOTE(Class~., data=trainingData)  
  J48_model <- train(Class~., data=trainingData2, method="J48", trControl=ctrl_cv, tuneGrid=treeGrid_dectree)
  best<-as.numeric(J48_model$bestTune)
  show(J48_model)
  tempPredict <- predict(J48_model,testData)
  prediction.j48 <- rbind(prediction.j48, as.data.frame(tempPredict))
  train.accuracy.estimate.j48[f] = as.numeric(J48_model$results[best,3])
  fold.accuracy.estimate.j48[f] = (table(tempPredict,testData$Class)[1,1]+table(tempPredict,testData$Class)[2,2])/length(testData$Class)
}
mean(train.accuracy.estimate.j48)
mean(fold.accuracy.estimate.j48)

result.j48 <- cbind(prediction.j48, projTrainFS[, 6])
names(result.j48) <- c("Predicted", "Actual")

# Attempt 3) naiveBayes with SMOTE
cvCount = 3
ctrl_cv = trainControl(method = "cv", number = cvCount, savePred = T, classProb = T, sampling="smote")
cols = ncol(projTrainFS)
folds = split(sample(nrow(projTrainFS), nrow(projTrainFS),replace=FALSE), as.factor(1:cvCount))
train.accuracy.estimate.nb = NULL
fold.accuracy.estimate.nb = NULL
prediction.nb <- data.frame()
for(f in 1:cvCount){
  testData = projTrainFS[folds[[f]],]
  trainingData = projTrainFS[-folds[[f]],]
  trainingData2 = SMOTE(Class~., data=trainingData)  
  NB_model <- train(Class~., data=trainingData2, method="nb", trControl=ctrl_cv)
  best<-as.numeric(NB_model$bestTune)
  show(NB_model)
  tempPredict <- predict(NB_model,testData)
  prediction.nb <- rbind(prediction.nb, as.data.frame(tempPredict))
  train.accuracy.estimate.nb[f] = as.numeric(NB_model$results[best,3])
  fold.accuracy.estimate.nb[f] = (table(tempPredict,testData$Class)[1,1]+table(tempPredict,testData$Class)[2,2])/length(testData$Class)
}
mean(train.accuracy.estimate.nb)
mean(fold.accuracy.estimate.nb)

result.nb <- cbind(prediction.nb, projTrainFS[, 6])
names(result.nb) <- c("Predicted", "Actual")


####################################################
##### Compare Models with ConfustionMatrix #########
####################################################

# Attempt 1a) Random Forest with SMOTE
confusionMatrix(data = result.rf$Predicted,
                reference = result.rf$Actual)

# Attempt 1b) Random Forest with Undersampling
confusionMatrix(data = result.rf.down$Predicted,
                reference = result.rf.down$Actual)

# Attempt 1c) Random Forest with Oversampling - NOTE: Took 3hours to run on 16gb RAM machine
confusionMatrix(data = result.rf.up$Predicted,
                reference = result.rf.up$Actual)

# Attempt 2) Decision Tree with SMOTE
confusionMatrix(data = result.j48$Predicted,
                reference = result.j48$Actual)

# Attempt 3) naiveBayes with SMOTE
confusionMatrix(data = result.nb$Predicted,
                reference = result.nb$Actual)
