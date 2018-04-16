library(data.table)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)
library(lubridate)
library(plyr)
library(ggbiplot)
library(devtools)
install_github("ggbiplot", "vqv")

#### Upload data to Console###
train_sbset <- read.csv('/bench/iowastate/datasets/ie583/final-project/train_sample.csv', stringsAsFactors = F)
full_train <- read.csv('/bench/iowastate/datasets/ie583/final-project/train.csv', stringsAsFactors = F)

################################
####### Exploration 
#################################


###################################
###### Add A Class Attribute 
##################################


###################################
####  Construct Derived Attributes
###################################


####################################
####  Initial Modeling
###################################


####################################
####  Principal Component Analysis
###################################

## Ref = https://tgmstat.wordpress.com/2013/11/28/computing-and-visualizing-pca-in-r/
## https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/
##PCA works on numeric variables
#checking for variables other than numeric
str(train_sbset)

#Have a couple non numeric removing them, I don't think we need anyway
pca_types <- train_sbset[,c("Class")]
new_data_pca <- train_sbset %>% select(-click_time, -attributed_time, -Class)
str(new_data_pca)

pca <- prcomp(new_data_pca,
              center = TRUE,
              scale. = TRUE)

print(pca)
plot(pca, type = "l")
summary(pca)

biplot(pca)

#found a nicer plotting tool than biplot on github called ggbiplot to visualize PCA.
g <- ggbiplot(pca, obs.scale = 1, var.scale = 1, 
              groups = pca_types, ellipse = TRUE, 
              circle = TRUE)
g <- g + theme(legend.direction = 'horizontal', 
               legend.position = 'top')
print(g)

#############################
#### Instance Selection
#############################

