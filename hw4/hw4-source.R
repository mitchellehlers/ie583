library(caret)
library(randomForest)

creditData <- read.csv("~/bench/ie583/hw4/credit-g.csv")

##### Random Forest #####

set.seed(5)

#Baseline Performance
#default value for classification for mtry, number of variables ramdomly sampled as candidates at each split
mtry <- sqrt(ncol(creditData))

rf_tunegrid = expand.grid(.mtry=mtry)

ctrl = trainControl(method="cv",
                    number=10,
                    savePred=T,
                    classProb=T,
                    sampling="up")

rf_upsample_default = train(class~.,
                            data=creditData,
                            method="rf",
                            trControl=ctrl,
                            tuneGrid=rf_tunegrid,
                            metric="Accuracy")

print(rf_upsample_default)

#Grid Search
rf_tunegrid = expand.grid(.mtry=c(1:15))

ctrl = trainControl(method="cv",
                    number=10,
                    savePred=T,
                    classProb=T,
                    sampling="up")

rf_upsample_1 = train(class~.,
                            data=creditData,
                            method="rf",
                            trControl=ctrl,
                            tuneGrid=rf_tunegrid,
                            metric="Accuracy")

print(rf_upsample_1)
plot(rf_upsample_1)

#Tune RF to find optimal value for mtry, randomForest package in R provides tuneRF to try to find optimal value
x <- creditData[,1:20]
y <- creditData[,21]
optimalMtry <- tuneRF(x, y, stepFactor=1.5, improve=1e-10)

print(optimalMtry)

rf_tunegrid = expand.grid(.mtry=optimalMtry)

ctrl = trainControl(method="cv",
                    number=10,
                    savePred=T,
                    classProb=T,
                    sampling="up")

rf_upsample_2 = train(class~.,
                      data=creditData,
                      method="rf",
                      trControl=ctrl,
                      tuneGrid=rf_tunegrid,
                      metric="Accuracy")

print(rf_upsample_2)

##### Support Vector Machine ##### 

set.seed(56789)

#svm linear
svm_linear_tunegrid = expand.grid(C = c(0, 0.01, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0))

ctrl = trainControl(method="cv",
                    number=10,
                    savePred=T,
                    classProb=T,
                    sampling="up")

svm_linear = train(class~.,
                   data=creditData,
                   method="svmLinear",
                   trControl=ctrl,
                   tuneGrid=svm_linear_tunegrid,
                   metric="Accuracy")

print(svm_linear)
plot(svm_linear)

#svm radial
svm_radial_tunegrid = expand.grid(C=(1:5))

ctrl = trainControl(method="cv",
                    number=10,
                    savePred=T,
                    classProb=T,
                    sampling="up")

svm_radial = train(class~.,
                   data=creditData,
                   method="svmRadial",
                   trControl=ctrl,
                   metric="Accuracy")

print(svm_radial)
plot(svm_radial)

#svm poly
svm_poly_tunegrid = expand.grid(degree = (1:20), scale=0.01, C=(1:5))

ctrl = trainControl(method="cv",
                    number=10,
                    savePred=T,
                    classProb=T,
                    sampling="up")

svm_poly = train(class~.,
                   data=creditData,
                   method="svmPoly",
                   trControl=ctrl,
                   metric="Accuracy")

print(svm_poly)
plot(svm_poly)
