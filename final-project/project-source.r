library(data.table)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)
library(lubridate)
library(DMwR)
library(randomForest)
library(Boruta)

#### Upload data to Console###
train_sbset <- read.csv("C:/bench/iowastate/datasets/ie583/final-project/train_sample.csv", stringsAsFactors = F)
test <- read.csv("C:/bench/iowastate/datasets/ie583/final-project/test.csv", stringsAsFactors = F)

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

### Manipulate Date-Time columns to create dervied attributes
##Add Date and Time individual columns for 'click_time'

#Change 'click_time' column from 'CHR' format to POSIXct, in order to use 
train_sbset$click_time <- as.POSIXct(as.character(train_sbset$click_time))

#create day,hr,minute and time_since fields.  
train_sbset <- train_sbset %>% 
  mutate(day = day(click_time),
         hour = hour(click_time),
         minute = minute(click_time),
         Sec_since_start = as.integer(click_time - as.POSIXct("2017-11-07 00:00:00")))

head(train_sbset)

################################
###### Instance selection Data Preprocessing
################################

# Create train subset of data based off feature engineering
# spliting the data into 8th and taking the first 8th
num_split <- nrow(train_sbset) / 8
num_rows <- nrow(train_sbset)

train_sbset_split_list <- split(train_sbset, rep(1:ceiling(num_rows/num_split), each=num_split, length.out=num_rows))

### train selection data frames
train_sbset_1 <- as.data.frame(train_sbset_split_list[1])
names(train_sbset_1) <- substring(names(train_sbset_1), 4)
str(train_sbset_1)

train_sbset_2 <- as.data.frame(train_sbset_split_list[2])
names(train_sbset_2) <- substring(names(train_sbset_2), 4)
str(train_sbset_2)

# Create test subsets, split into 16th
test_num_split <- nrow(test) / 16
test_num_rows <- nrow(test)

test_sbset_split_list <- split(train_sbset, rep(1:ceiling(test_num_rows/test_num_split), each=test_num_split, length.out=test_num_rows))

test_sbset_1 <- as.data.frame(test_sbset_split_list[1])
names(test_sbset_1) <- substring(names(test_sbset_1), 4)
str(test_sbset_1)


######################################
#### Instance Sampling #####
######################################

# how imbalenced is the data? VERY
table(train_sbset$Class)
prop.table(table(train_sbset$Class))

# OVERSAMPLING - oversamples minority class instances with replacement to equal out class imbalance
# see - http://topepo.github.io/caret/subsampling-for-class-imbalances.html
set.seed(1234)
oversampled <- upSample(x = train_sbset_1[,-ncol(train_sbset_1)],
                       y = train_sbset_1$Class)  
table(oversampled$Class)

mtry <- sqrt(ncol(train_sbset_1))
rf_tunegrid = expand.grid(.mtry=mtry)

ctrl = trainControl(method="cv",
                    number=10,
                    savePred=T,
                    classProb=T,
                    sampling="up")

oversample_inside = train(Class~.,
                          data=train_sbset_1,
                          method="rf",
                          trControl=ctrl,
                          tuneGrid=rf_tunegrid,
                          metric="Accuracy")

print(oversample_inside)

# UNDERSAMPLING - Leave the mintory class untouched and select instances of majority class via random sampling
set.seed(1234)

undersampled <- downSample(x = train_sbset_1[, -ncol(train_sbset_1)],
                                y = train_sbset_1$Class)
table(undersampled$Class)

### SMOTE with Undersampling

#Smote data preprocessing, taking same data frame used for sampling but SMOTE requires char and POSIXct attributes to be factor
smot_sel_df <- train_sbset_1
smot_sel_df$attributed_time <- sub("^$", "Na", smot_sel_df$attributed_time) #doing this so we can factor this
smot_sel_df$attributed_time <- as.factor(smot_sel_df$attributed_time)
smot_sel_df$click_time <- as.factor(smot_sel_df$click_time)

set.seed(1234)
smote_train <- SMOTE(Class~., data = smot_sel_df)                         
table(smote_train$Class)

undersample_smote <- downSample(x = smote_train[, -ncol(smote_train)],
                          y = smote_train$Class)
table(undersample_smote$Class)
prop.table(table(undersample_smote$Class))

#########################################
#### Variable Importance Measurment #####
#########################################

set.seed(123)
boruta.train <- Boruta(Class ~ ., data = undersample_smote, doTrace = 2)
print(boruta.train)
plot(boruta.train, cex.axis=.7, las=2, xlab="", main="Variable Importance") 

# Split off only the top 5 important attributes
#TODO

#########################################
#### Predictive Modeling  #####
#########################################

# Attempt 1) Random Forest with SMOTE and Undersampling
cols <- ncol(train_sbset_1)
mtry <- c(1:cols)#sqrt(ncol(train_sbset_1))

x <- train_sbset_1[,1:(cols-1)]
y <- train_sbset_1[,cols]
optimalMtry <- tuneRF(x, y, stepFactor=1.5, improve=1e-10)

print(optimalMtry)

rf_tunegrid = expand.grid(.mtry=mtry)

ctrl = trainControl(method="cv",
                    number=5,
                    savePred=T,
                    classProb=T)

rf_model = train(Class~.,
                           data=undersample_smote,
                           method="rf",
                           trControl=ctrl,
                           tuneGrid=rf_tunegrid,
                           metric="Accuracy")

confusionMatrix(rf_model)

rf_model_1 <- randomForest(Class ~ attributed_time + app + ip,
             data=undersample_smote, 
             importance=TRUE, 
             ntree=2000)

prediction <- predict(rf_model_1, test_sbset_1)

# Attempt 2) kNN with SMOTE and Undersampling
ctrl<-trainControl(method="cv",
                   number = 5,
                   savePred=T,
                   classProb=T)

knnGrid <-expand.grid(k=(1:10))

kNN_model <- train(Class ~ . ,
                   data=oversampled,
                   method="knn",
                   trControl=ctrl,
                   tuneGrid = knnGrid)

confusionMatrix(kNN_model)

# Attempt 3) Decision Tree with Undersampling
ctrl<-trainControl(method="cv",
                   number = 5,
                   savePred=T,
                   classProb=T)

treeGrid <- expand.grid(C=(1:3)*0.1, M=5)

J48_model <- train(Class ~ . ,
                  data=undersampled,
                  method="J48",
                  trControl=ctrl,
                  tuneGrid=treeGrid)

confusionMatrix(J48_model)
