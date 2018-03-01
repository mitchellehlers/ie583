library(RWeka)
library(caret)
library(arules)  
library(arulesViz)

voteData <- read.table("/Users/mitchellehlers/bench/ie583/hw2/vote.csv",header=T,sep=",",quote="\"")
voteData[] = lapply(voteData[],factor)

rules = apriori(voteData)
summary(rules)
plot(rules)

good_rules = rules[quality(rules)$confidence>0.9] 
plot(good_rules)

good_rules = good_rules[quality(good_rules)$support>0.35] 
plot(good_rules)

good_rules = good_rules[quality(good_rules)$lift>1.8] 
plot(good_rules)

plot(good_rules, method="graph", itemLabels=FALSE)

#Best Association Rule
inspect(head(rules, n=10, by = "confidence"))
inspect(head(rules, n=10, by = "support"))
inspect(head(rules, n=10, by = "lift"))

plot(head(rules, n=10, by = "lift"), method="graph", itemLabels=FALSE)

general_rules = subset(rules, subset = lift > 2.0) 
inspect(head(general_rules, n=10, by = "confidence"))
inspect(head(general_rules, n=10, by = "support"))
inspect(head(general_rules, n=10, by = "lift"))

#Best Rules Class=democrat
Dem_rules = subset(rules, subset = rhs %in% "Class=democrat") 
plot(Dem_rules)

inspect(head(Dem_rules, n=10, by = "confidence"))
inspect(head(Dem_rules, n=10, by = "support"))
inspect(head(Dem_rules, n=10, by = "lift"))

#Best Rules Class=republican
Rep_rules = subset(rules, subset = rhs %in% "Class=republican") 
plot(Rep_rules)

inspect(head(Rep_rules, n=10, by = "confidence"))
inspect(head(Rep_rules, n=10, by = "support"))
inspect(head(Rep_rules, n=10, by = "lift"))

plot(head(Rep_rules, n=10, by = "confidence"))
plot(head(Rep_rules, n=10, by = "support"))
plot(head(Rep_rules, n=10, by = "lift"))

#Independent Test Set
set.seed(1234)
trainIndex <- createDataPartition(voteData$adoption_of_the_budget_resolution,p=.67,list=FALSE,times=1)
voteTrain <- voteData[trainIndex,]
voteTest<-voteData[-trainIndex,]

#Decision Tree
DTmodel<-J48(adoption_of_the_budget_resolution ~ .,data=voteTrain)
prediction<-predict(DTmodel,voteTest)
confusionMatrix(prediction,voteTest$adoption_of_the_budget_resolution)
plot(DTmodel)

ctrl<-trainControl(method="cv", number = 10, savePred=T,classProb=T)
treeGrid <- expand.grid(C=(1:5)*0.01, M=5)
DTmodel5 <- train(Class ~ . , data=voteData,method="J48",trControl=ctrl,tuneGrid=treeGrid)
confusionMatrix(DTmodel5)