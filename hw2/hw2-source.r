library(RWeka)
library(caret)
library(arules)  
library(arulesViz)

voteData <- read.table("/Users/mitchellehlers/bench/ie583/hw2/vote.csv",header=T,sep=",",quote="\"")
voteData[] = lapply(voteData[],factor)

rules = apriori(voteData)
summary(rules)
plot(rules)

##################
good_rules = rules[quality(rules)$confidence>0.9] 
plot(good_rules)

good_rules = good_rules[quality(good_rules)$support>0.35] 
plot(good_rules)

good_rules = good_rules[quality(good_rules)$lift>1.8] 
plot(good_rules)

plot(good_rules, method="graph", itemLabels=FALSE)
##################

######Best Association Rule######
inspect(head(rules, n=10, by = "confidence"))
inspect(head(rules, n=10, by = "support"))
inspect(head(rules, n=10, by = "lift"))

plot(head(rules, n=10, by = "lift"), method="graph", itemLabels=FALSE)

general_rules = subset(rules, subset = lift > 2.0)
inspect(head(general_rules, n=10, by = "support"))
plot(head(general_rules, n=10, by = "support"))
plot(head(general_rules, n=10, by = "support"), method="graph", itemLabels=FALSE)

elsalvadoraid_rules = subset(rules, subset = rhs %in% "el_salvador_aid=y") 
inspect(head(elsalvadoraid_rules, n=10, by = "support"))
inspect(head(elsalvadoraid_rules, n=10, by = "confidence"))
inspect(head(elsalvadoraid_rules, n=10, by = "lift"))

#Independent Test Set
set.seed(1234)
trainIndex <- createDataPartition(voteData$el_salvador_aid,p=.67,list=FALSE,times=1)
voteTrain <- voteData[trainIndex,]
voteTest<-voteData[-trainIndex,]

#Decision Tree
DTmodel<-J48(el_salvador_aid ~ .,data=voteTrain)
prediction<-predict(DTmodel,voteTest)
confusionMatrix(prediction,voteTest$el_salvador_aid)
plot(DTmodel)

#####################################

######Best Rules Class=democrat######
Dem_rules = subset(rules, subset = rhs %in% "Class=democrat")
Dem_rules_sorted = subset(Dem_rules, subset = support < 0.3)
inspect(head(Dem_rules_sorted, n=10, by = "support"))

plot(Dem_rules)

inspect(head(Dem_rules_sorted, n=10, by = "confidence"))
inspect(head(Dem_rules_sorted, n=10, by = "support"))
inspect(head(Dem_rules_sorted, n=10, by = "lift"))

inspect(head(Dem_rules, n=10, by = "confidence"))
inspect(head(Dem_rules, n=10, by = "support"))
inspect(head(Dem_rules, n=10, by = "lift"))

plot(head(Dem_rules, n=10, by = "support"))
plot(head(Dem_rules, n=10, by = "support"), method="graph", itemLabels=FALSE)

#####################################

######Best Rules Class=republican######
Rep_rules = subset(rules, subset = rhs %in% "Class=republican") 
inspect(head(Rep_rules, n=10, by = "support"))
plot(Rep_rules)

inspect(head(Rep_rules, n=10, by = "confidence"))
inspect(head(Rep_rules, n=10, by = "support"))
inspect(head(Rep_rules, n=10, by = "lift"))

Rep_rules_sorted = subset(Rep_rules, subset = support < 0.2)
inspect(head(Rep_rules_sorted, n=10, by = "confidence"))
inspect(head(Rep_rules_sorted, n=10, by = "support"))
inspect(head(Rep_rules_sorted, n=10, by = "lift"))

plot(head(Rep_rules_sorted, n=10, by = "confidence"))
plot(head(Rep_rules_sorted, n=10, by = "support"))
plot(head(Rep_rules_sorted, n=10, by = "lift"))

#Independent Test Set
set.seed(1234)
trainIndex <- createDataPartition(voteData$Class,p=.67,list=FALSE,times=1)
voteTrain <- voteData[trainIndex,]
voteTest<-voteData[-trainIndex,]

#Decision Tree
DTmodel<-J48(Class ~ .,data=voteTrain)
prediction<-predict(DTmodel,voteTest)
confusionMatrix(prediction,voteTest$Class)
plot(DTmodel)

###########################################

### Cross Validation for Decision Tree ###
ctrl<-trainControl(method="cv", number = 10, savePred=T,classProb=T)
treeGrid <- expand.grid(C=(1:5)*0.01, M=5)
DTmodel5 <- train(Class ~ . , data=voteData,method="J48",trControl=ctrl,tuneGrid=treeGrid)
confusionMatrix(DTmodel5)