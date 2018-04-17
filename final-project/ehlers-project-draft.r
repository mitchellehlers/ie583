library(data.table)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)
library(lubridate)
library(DMwR)

#### Upload data to Console###
train_sbset <- read.csv("C:/bench/iowastate/datasets/ie583/final-project/train_sample.csv", stringsAsFactors = F)

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

### Unable to run with these dervied fields R crashed on local system.
## Franz's dervied fields.  Needs additional testing
#train_sbset <- train_sbset %>% 
#  mutate(hour = hour(click_time),
#         ip_device_os_channel_app = paste(ip,'-',device,'-',os,'-',channel,'-',app),
#         ip_device_os = paste(ip,'-',device,'-',os),
#         ip_device = paste(ip,'-',device),
#         ip_channel_app = paste(ip,'-',channel,'-',app),
#         ip_channel = paste(ip,'-',channel),
#         ip_app = paste(ip,'-',app),
#         channel_app = paste(channel,'-',app),
#         channel_app_os_device = paste(channel,'-',app,'-',os,'-',device),
#         channel_os_device = paste(channel,'-',os,'-',device)#,
#         #x = if(hour > 12) return ('PM')
#  )
#head(train_sbset)

################################
###### Instance selection Data Preprocessing
################################
### Create subset of data based off feature engineering
#spliting the data into 8th and taking the first 8th
num_split <- nrow(train_sbset) / 8
num_rows <- nrow(train_sbset)

train_sbset_split_list <- split(train_sbset, rep(1:ceiling(num_rows/num_split), each=num_split, length.out=num_rows))

### instance selection data frame
inst_sel_df <- as.data.frame(train_sbset_split_list[1])
names(inst_sel_df) <- substring(names(inst_sel_df), 4)

str(inst_sel_df)

######################################
#### Instance Selection/Sampling #####
######################################

# OVERSAMPLING - oversamples minority class instances with replacement to equal out class imbalance
set.seed(1234)
oversample <- upSample(x = inst_sel_df[,-ncol(inst_sel_df)],
                       y = inst_sel_df$Class)  
table(oversample$Class)

# Oversampling durning resampling - http://topepo.github.io/caret/subsampling-for-class-imbalances.html
mtry <- sqrt(ncol(inst_sel_df))
rf_tunegrid = expand.grid(.mtry=mtry)

ctrl = trainControl(method="cv",
                    number=10,
                    savePred=T,
                    classProb=T,
                    sampling="up")

oversample_inside = train(Class~.,
                          data=inst_sel_df,
                          method="rf",
                          trControl=ctrl,
                          tuneGrid=rf_tunegrid,
                          metric="Accuracy")

print(oversample_inside)

# UNDERSAMPLING - Leave the mintory class untouched and select instances of majority class via random sampling
set.seed(1234)
undersample <- downSample(x = inst_sel_df[, -ncol(inst_sel_df)],
                          y = inst_sel_df$Class)
table(undersample$Class)

# Understample durning resampling - http://topepo.github.io/caret/subsampling-for-class-imbalances.html
ctrl = trainControl(method="cv",
                    number=10,
                    savePred=T,
                    classProb=T,
                    sampling="down")

undersample_inside = train(Class~.,
                           data=inst_sel_df,
                           method="rf",
                           trControl=ctrl,
                           tuneGrid=rf_tunegrid,
                           metric="Accuracy")

print(undersample_inside)

#Cross validation  ######################33 WHY J48 CROSS VALIDATION
ctrl<-trainControl(method="cv", number=10, savePred=T,classProb=T)  #####3 WHY THESE VALUES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
treeGrid <- expand.grid(C=(1:3)*0.1, M=5)  ### WHY THESE VALUES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

upsample_model <- train(Class ~ . , data=oversample, method="J48",trControl=ctrl,tuneGrid=treeGrid)
confusionMatrix(upsample_model)
plot(upsample_model)

downsample_model <- train(Class ~ . , data=undersample, method="J48",trControl=ctrl,tuneGrid=treeGrid)
confusionMatrix(downsample_model)
plot(downsample_model)

str(inst_sel_df)

### SMOTE

#Smote data preprocessing, taking same data frame used for sampling but SMOTE requires char and POSIXct attributes to be factor
smot_sel_df <- inst_sel_df
smot_sel_df$attributed_time <- sub("^$", "Na", smot_sel_df$attributed_time) #doing this so we can factor this
smot_sel_df$attributed_time <- as.factor(smot_sel_df$attributed_time)
smot_sel_df$click_time <- as.factor(smot_sel_df$click_time)

set.seed(1234)
smote_train <- SMOTE(Class~., data = smot_sel_df)                         
table(smote_train$Class)


#J48 Tree cross validation
smote_model <- train(class ~ . , data=smote_train, method="J48",trControl=ctrl,tuneGrid=treeGrid)
confusionMatrix(smote_model)
plot(smote_model)

#########################################
#### Variable Importance Measurment #####
#########################################

