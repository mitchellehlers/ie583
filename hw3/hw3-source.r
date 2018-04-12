library(RWeka)
library(e1071)
library(caret)
library(Boruta)
library(DMwR)
library(FactoMineR)

creditData <- read.table("/Users/mitchellehlers/bench/ie583/hw3/credit-g.csv",header=T,sep=",",quote="\"")
creditData[] = lapply(creditData[],as.factor)

#### 1) Attribute selection ####

# ranking attributes by importance
set.seed(1234)
control <- trainControl(method="cv", number = 10, savePred=T, classProb=T)
treeGrid <- expand.grid(C=(1:3)*0.09, M=5)
model <- train(class ~ . , data=creditData,method="J48",trControl=control,tuneGrid=treeGrid)
importance <- varImp(model, scale=FALSE)
plot(importance)
print(importance)

#### 2) Attribute construction #### 
#Added a ratio of two of the numeric attributes.
creditDataConstruct <- read.table("/Users/mitchellehlers/bench/ie583/hw3/credit-g.csv",header=T,sep=",",quote="\"")
creditDataConstruct$creditAgeRatio <- creditDataConstruct$credit_amount / creditDataConstruct$age
creditDataConstruct[] = lapply(creditDataConstruct[],as.factor)
selectedCreditData <- creditDataConstruct[c("checking_status", "duration", "savings_status", "creditAgeRatio", "class")]

#### 3) Biased random sampling to improve the minority class prediction ("bad" instances) #### 
# up-sampling
set.seed(9560)
oversample <- upSample(x = selectedCreditData[, -ncol(selectedCreditData)],
                     y = selectedCreditData$class)  
table(oversample$Class)

#down-sampling
set.seed(9560)
undersample <- downSample(x = selectedCreditData[, -ncol(selectedCreditData)],
                         y = selectedCreditData$class)
table(undersample$Class)

#J48 Tree cross validation
ctrl<-trainControl(method="cv", number = 10, savePred=T,classProb=T)
treeGrid <- expand.grid(C=(1:3)*0.09, M=5)

upsample_model <- train(Class ~ . , data=oversample, method="J48",trControl=ctrl,tuneGrid=treeGrid)
confusionMatrix(upsample_model)
plot(upsample_model)

downsample_model <- train(Class ~ . , data=undersample, method="J48",trControl=ctrl,tuneGrid=treeGrid)
confusionMatrix(downsample_model)
plot(downsample_model)

#### 4) SMOTE #### 
set.seed(9560)
smote_train <- SMOTE(class~., data  = selectedCreditData)                         
table(smote_train$class)

#J48 Tree cross validation
smote_model <- train(class ~ . , data=smote_train, method="J48",trControl=ctrl,tuneGrid=treeGrid)
confusionMatrix(smote_model)
plot(smote_model)

#Boruta
set.seed(123)
boruta.train <- Boruta(class ~ ., data = creditData, doTrace = 2)
print(boruta.train)

borutaCreditData <- creditData[c("checking_status", "duration", "age", "credit_amount", "credit_history", "class")]
smote_model <- train(class ~ . , data=borutaCreditData, method="J48",trControl=ctrl,tuneGrid=treeGrid)
confusionMatrix(smote_model)
plot(smote_model)
