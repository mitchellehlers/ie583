library(caret)

creditData <- read.csv("/bench/iowastate/ie583/hw4/credit-g.csv")

metric <- "Accuracy"
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
                            metric=metric)

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
                            metric=metric)

print(rf_upsample_1)




