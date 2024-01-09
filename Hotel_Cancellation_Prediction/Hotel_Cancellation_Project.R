rm(list = ls())

set.seed(8)
library(ggplot2)
library(ggcorrplot)
library(dplyr)
library(caTools)
library(ROCR)
library(e1071)
library(class)

hotel <- read.csv("C:\\Users\\iholt\\Desktop\\data\\Hotel Reservations.csv")

hotel <- sample_frac(hotel[,-1], 0.1)

hotel$booking_status[hotel$booking_status == "Not_Canceled"] <- 0 
hotel$booking_status[hotel$booking_status == "Canceled"] <- 1

hotel$booking_status <- as.numeric(hotel$booking_status)
hotel$type_of_meal_plan <- as.factor(hotel$type_of_meal_plan)
hotel$room_type_reserved <- as.factor(hotel$room_type_reserved)
hotel$market_segment_type <- as.factor(hotel$market_segment_type)
hotel$arrival_year <- as.factor(hotel$arrival_year)
hotel$arrival_month <- as.factor(hotel$arrival_month)
hotel$required_car_parking_space <- as.factor(hotel$required_car_parking_space)



head(hotel)
summary(hotel)

split <- sample.split(hotel, SplitRatio = 0.75)

train <- subset(hotel, split == "TRUE")
test <- subset(hotel, split == "FALSE")


ggplot(data = hotel, aes(x = as.factor(booking_status), y = lead_time))+
  geom_boxplot()+ ggtitle("Number of Days Reservation Was Made in Advance")+
  ylab("Days")+xlab("0 == Not Canceled, 1 == Canceled")

ggplot(data = hotel, aes(x = as.factor(booking_status), y = no_of_special_requests))+
  geom_boxplot()+ggtitle("Number of Special Requests Asked for by Guest")+
  xlab("0 == Not Canceled, 1 == Canceled") + ylab("Special Requests")

ggplot(data = hotel, aes(x = as.factor(booking_status), y = arrival_month))+
  geom_boxplot()

###########boosting
library(gbm)

gbm_model <- gbm(booking_status ~ .,data=train,
                 distribution = 'bernoulli',
                 n.trees = 2500, 
                 shrinkage = 0.1, 
                 interaction.depth = 3,
                 cv.folds = 10)

gbm_model
summary(gbm_model)


perf_gbm = gbm.perf(gbm_model, method="cv") 
perf_gbm
## Training error
pred_gbm <- predict(gbm_model,
                    newdata = train,
                    n.trees=perf_gbm, 
                    type="response")

pred <- ifelse(pred_gbm < 0.5, 0, 1)
gbm_train_error <- sum(pred != train$booking_status)/length(train$booking_status) 


## Testing Error

pred_test <- predict(gbm_model,
                     newdata = test[,-18],
                     n.trees = 2500,
                     shrinkage = 0.1,
                     type = "response")

pred_test_new <- ifelse(pred_test < 0.5, 0, 1)

gbm_test_error <- sum(pred_test_new != test$booking_status)/length(test$booking_status)
gbm_test_error
######### GBM NUMBER TREES ################
tree_num <- c(500, 1000, 1500, 2000, 2500, 3000, 3500)
gbm_train_error <- c()
opt_trees <- c()
for (i in tree_num){
  gbm_model <- gbm(booking_status ~ .,data=train,
                   distribution = 'bernoulli',
                   n.trees = i, 
                   shrinkage = 0.1, 
                   interaction.depth = 3,
                   cv.folds = 10)
  perf_gbm <-  gbm.perf(gbm_model, method="cv") 
  opt_trees <- c(opt_trees, perf_gbm)
  
  pred_gbm <- predict(gbm_model,
                      newdata = train,
                      n.trees=perf_gbm, 
                      type="response")
  
  pred <- ifelse(pred_gbm < 0.5, 0, 1)
  gbm_temp_train_error <- sum(pred != train$booking_status)/length(train$booking_status) 
  gbm_train_error <- c(gbm_train_error, gbm_temp_train_error); 

}
gbm_train_error
opt_trees

gbm_tree_effect <- as.matrix(cbind(tree_num, gbm_train_error, opt_trees))
gbm_tree_effect

######### GBM SHRINKAGE ################
shrink <- c(0.1, 0.05, 0.01)
gbm_shrink_train_error <- c()
for (i in shrink){
  gbm_model <- gbm(booking_status ~ .,data=train,
                   distribution = 'bernoulli',
                   n.trees = 1000, 
                   shrinkage = i, 
                   interaction.depth = 3,
                   cv.folds = 10)
  perf_gbm <-  gbm.perf(gbm_model, method="cv") 
  
  pred_gbm <- predict(gbm_model,
                      newdata = train,
                      n.trees=perf_gbm, 
                      type="response")
  
  pred <- ifelse(pred_gbm < 0.5, 0, 1)
  gbm_temp_train_error <- sum(pred != train$booking_status)/length(train$booking_status) 
  gbm_shrink_train_error <- c(gbm_shrink_train_error, gbm_temp_train_error); 
  
}
gbm_shrink_train_error

shrinkage_effect <- as.matrix(cbind(shrink, gbm_shrink_train_error))
shrinkage_effect

#### FINAL GBM MODEL ############
final_gbm_model <- gbm(booking_status ~ .,data=train,
                 distribution = 'bernoulli',
                 n.trees = 1000, 
                 shrinkage = 0.05, 
                 interaction.depth = 3,
                 cv.folds = 10)

final_perf_gbm = gbm.perf(final_gbm_model, method="cv") 

final_pred_test <- predict(final_gbm_model,
                     newdata = test[,-18],
                     n.trees = final_perf_gbm,
                     shrinkage = 0.05,
                     type = "response")

final_pred_test_new <- ifelse(pred_test < 0.5, 0, 1)

gbm_test_error <- sum(final_pred_test_new != test$booking_status)/length(test$booking_status)
gbm_test_error

summary(final_gbm_model)

table(final_pred_test_new, test$booking_status)

##################random forest
library(randomForest)

rf_default_train <- randomForest(booking_status ~., 
                                 data=train, 
                                 importance=TRUE)

summary(rf_default_train)

## Check Important variables
importance(rf_default_train)
varImpPlot(rf_default_train)

## Prediction on the training data set
rf_pred_train = predict(rf_default_train, train[,-18], type='class')
pred_train_rf <- ifelse(rf_pred_train < 0.5, 0, 1)


rf_train_error <- sum(pred_train_rf != train$booking_status)/length(train$booking_status)
rf_train_error

## Prediction on the testing data set
rf_pred = predict(rf_default_train, test[,-18], type='class')
pred_test_rf <- ifelse(rf_pred < 0.5, 0, 1)


rf_test_error_default <- sum(pred_test_rf != test$booking_status)/length(test$booking_status)
rf_test_error_default

table(pred_test_rf, test$booking_status)

############# RANDOM FOREST MTRY NUM ######################
rf_mtry_train_error <- c()
mtry_num <- c(3, 4, 5, 6, 7, 8)
for (i in mtry_num){
  rf_default_train <- randomForest(booking_status ~., 
                                   data=train,
                                   ntree = 500,
                                   mtry = mtry_num,
                                   importance=TRUE)
  
  rf_pred_train = predict(rf_default_train, train[,-18], type='class')
  pred_train_rf <- ifelse(rf_pred_train < 0.5, 0, 1)
  
  rf_temp_train_error <- sum(pred_train_rf != train$booking_status)/length(train$booking_status) 
  rf_mtry_train_error <- c(rf_mtry_train_error, rf_temp_train_error); 
  
}
rf_mtry_train_error


rf_mtry_effect <- as.matrix(cbind(mtry_num, rf_mtry_train_error))
rf_mtry_effect



#########RANDOM FOREST TESTING ERROR ##############
rf_test_model <- randomForest(booking_status ~., 
                                 data=train,
                                 ntree = 500,
                                 mtry = 5,
                                 importance=TRUE)

rf_pred_test = predict(rf_default_train, test[,-18], type='class')
pred_test_rf <- ifelse(rf_pred_test < 0.5, 0, 1)

rf_test_error <- sum(pred_test_rf != test$booking_status)/length(test$booking_status) 
rf_test_error

importance(rf_test_model)
varImpPlot(rf_test_model)

table(pred_test_rf, test$booking_status)


########### OTHER MODELS ############

modA <- step(glm(booking_status ~ ., data = train));
pred_A <- ifelse(predict(modA, test[,-18], type="response" ) < 0.5, 0, 1)
modA_test_error <- sum(pred_A != test$booking_status)/length(test$booking_status) 
modA_test_error

table(pred_A, test$booking_status)


#Naive Bayes
library(e1071)
modC <- naiveBayes(as.factor(train[,18]) ~. , data = train)
y2hatC <- predict(modC, test, type = 'class')
modC_test_error <- mean( y2hatC != test$booking_status) 
modC_test_error

table(y2hatC, test$booking)

####logistic regression
log_model <- glm(booking_status~., data = train, family = "binomial")
summary(log_model)

pred_model <- predict(log_model, test, type = "response")
pred_model_new <- ifelse(pred_model > 0.6, 1, 0)
log_test_error <- mean(pred_model_new != test$booking_status)
log_test_error

probs <- seq(0.1, 0.95, by = 0.05)
log_test_error <- c()
for (i in probs){
  pred_model <- predict(log_model, test, type = "response")
  pred_model_new <- ifelse(pred_model > i, 1, 0)
  temp_test_error <- mean(pred_model_new != test$booking_status)
  log_test_error <- c(log_test_error, temp_test_error)
}

table(pred_model_new, test$booking)

log_threshold <- data.frame(as.matrix(cbind(probs, log_test_error)))

ggplot(data = log_threshold, aes(x = probs, y = log_test_error))+
  geom_point()+ ggtitle("Logistic Regression Testing Error Using Different Classification Threshold") +
  scale_x_continuous(limits = c(0, 1)) + xlab("Probability Threshold") + ylab("Testing Error")
  
  

#############SVM

svm_model <- svm(booking_status ~ ., data = hotel)

## The "SVM" prediction of the testing subset 
Yhat_svm <- predict(svm_model, test, type="class")
pred_model_svm <- ifelse(Yhat_svm > 0.5, 1, 0)
modsvm_test_error <- mean( pred_model_svm != test$booking_status) 
modsvm_test_error
table(pred_model_svm, test$booking_status)

models<- c("Boosting", "Random Forest", "SVM", "Naive Bayes", "Logistic Regression")
accuracy <- c(1 - 0.1693, 1 - 0.1406, 1 - 0.1871, 1-0.0267, 1 - 0.1970)
specificity <- c(0.8856, 0.9296, 0.4970, 0.9619, 0.5457)
sensitivity <- c(0.7165, 0.7134, 0.9648, 0.9970, 0.9267)

results_table <- data.frame(cbind(models, accuracy, specificity, sensitivity))
results_table

ggplot(data = results_table, aes(x = as.factor(models), y = accuracy))+
  geom_point(color = "red", size = 3)+
  ggtitle("Accuracy of Models") + ylab("Accuracy") +xlab("Model")
  
ggplot(data = results_table, aes(x = as.factor(models), y = specificity))+
  geom_point(color = "blue", size = 3)+
  ggtitle("Specificity of Models") + ylab("Specificity") +xlab("Model")

ggplot(data = results_table, aes(x = as.factor(models), y = sensitivity))+
  geom_point(color = "darkgreen", size = 3)+
  ggtitle("Sensitivity of Models") + ylab("Sensitivity") +xlab("Model")
  
  geom_point(aes(y = accuracy))+
  geom_point(aes(y = specificity))+
  geom_point(aes(y = sensitivity))+
  scale_color_discrete(breaks = c("accuracy", "sensitivity", "specificity"),
                      labels = c("Accuracy", "Sensitivity", "Specificity"))
