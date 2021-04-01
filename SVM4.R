####################################################
# Support Vector Machines (SVM) - Otto Dahlin
####################################################


library(caret)
library(tidyverse)
library(ggplot2)
library(lattice)
library(reshape2)
library(DataExplorer)
library(doParallel)
library(corrplot)

data <- read.csv("Churn Modeling.csv", sep=",")
str(data)
summary(data)

# Target Variable: Exited (Binary)

sum(is.na(data)) # 0
sum(complete.cases(data))

str(data)

# deletion:
# 1) RowNumber
# 2) CustomerID
# 3) Surname

# Subsetat data
data2 <- subset(data, select= - c(RowNumber, CustomerId, Surname))
str(data2)


# factor
data2$Gender <-as.factor(data2$Gender)
data2$Geography <-as.factor(data2$Geography)

# numeric
data2$CreditScore <-as.numeric(data2$CreditScore)
data2$Age <-as.numeric(data2$Age)
data2$Tenure <-as.numeric(data2$Tenure)
data2$Balance <-as.numeric(data2$Balance)
data2$NumOfProducts <-as.numeric(data2$NumOfProducts)
data2$EstimatedSalary <-as.numeric(data2$EstimatedSalary)
data2$HasCrCard<-as.factor(data2$HasCrCard)
data2$IsActiveMember <-as.factor(data2$IsActiveMember)
data2$Exited <-as.factor(data2$Exited)
str(data2)


#############################################

# Training 70% and Test 30%
set.seed(123)
train.index <- createDataPartition(data2$Exited, p=0.7, list=FALSE)
train.index


train.data2 <- data2[drop(train.index),]
str(train.data2)
test.data2 <- data2[-drop(train.index),]
str(test.data2)


# repeated 5-fold cross-validation, 3 repeats.
fitControl <- trainControl(method="repeatedcv", number=5, repeats=3)


#########################################################
# SVM linear classifier (SVM1), c=1 default

# using 3 cores out of total 4
library(doParallel)
cl <- makePSOCKcluster(3)
registerDoParallel(cl)


# SVM 1, linear.
svm1 <- train(Exited ~., data = data2, method = "svmLinear", trControl = fitControl, preProcess = c("center","scale"))
# default cost = 1
svm1
svm1Saved <- saveRDS(svm1, "svm1Saved")
readRDS("svm1Saved.RDS")

res1<-as_tibble(svm1$results[which.min(svm1$results[,2]),])
res1


###############################################################################################
# SVM 2 Linear Classification with optimized C parameter

# Hyperparameter optimization:

# It's possible to automatically compute SVM for different values of C and to 
# choose the optimal one that maximize the model cross-validation accuracy

library(doParallel)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)


# SVM 2, Linear classification with hyperparameter optimization with tuneGrid
svm2 <- train(Exited ~., data = data2, method = "svmLinear", trControl = fitControl,  
              preProcess = c("center","scale"), tuneGrid = expand.grid(C = seq(0, 50, length = 5)))

read.

svm2 # c= 0.1052632

# Print the best tuning parameter C that
# maximizes model accuracy
svm2$bestTune

# sparar resultat för svm2
res2<-as_tibble(svm2$results[which.min(svm2$results[,2]),])
res2


###############################################################################################
# SVM with RBF
library(doParallel)
cl <- makePSOCKcluster(3)
registerDoParallel(cl)


# SVM med Radiala Basfunktioner
set.seed(123)
# SVM 3
# Fit the model:
svm3 <- train(Exited ~., data = data2, method = "svmRadial", 
              trControl = fitControl, preProcess = c("center","scale"), tuneLength = 10)
svm3 # C=16 och sigma = 0.057

confusionMatrix.train(svm3)

#svm3Saved <- saveRDS(svm3, "svm3Saved.RDS")
#svm3Saved
readRDS("svm3Saved.RDS")


plot(svm3, main="SVM Radiala Basfunktioner") # cost vs accuracy

# sparar resultat för svm2
res3 <-as_tibble(svm3$results[which.min(svm3$results[,2]),])
res3

###############################################################################################
# SVM Polynomial:

library(doParallel)
cl <- makePSOCKcluster(3)
registerDoParallel(cl)

svm4 <- train(Exited ~., data = data2, method = "svmPoly",
              trControl = fitControl, preProcess = c("center","scale"), tuneLength = 5)
svm4

svm4$results[23,]
#save the results for later
#res4<-as_tibble(svm4$results[which.min(svm4$results[,2]),])
#res4 <- 

res4 <- as_tibble(svm4$results[23,])
res4

######################################################
# SVM RBF prediction:
Predicted.svm3 <- predict(svm3, test.data2)


Predicted.svm3 == test.data2$Exited
mean(Predicted.svm3 == test.data2$Exited) 
# 88% out-of-sample prediction accuracy.
######################################################