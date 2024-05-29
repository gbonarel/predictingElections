#Importing Raw Data
data_2016 <- read.csv("2016Data.csv")
data_2020 <- read.csv("2020Data.csv")


#Dropping Na rows
data_2016 <- na.omit(data_2016)
data_2020 <- na.omit(data_2020)

#Random Forest
library(randomForest)
train.rf <- randomForest(as.factor(Outcome) ~., data = data_2016, importance = TRUE, proximity = TRUE)
print(train.rf)

#Bagging 
library(tidyr)
library(dplyr)
set.seed(8675309)
train.bag <- randomForest(as.factor(Outcome) ~ ., data = data_2016,
                            mtry = 20, ntree = 500, oob_score = TRUE)
print(train.bag)
importance(train.bag) %>% as_tibble() %>%
  mutate(var = rownames(importance(train.bag))) %>% 
  select(2, 1)

#Training predictions: RF
data_2016 %>% mutate(pred = train.rf$predicted) %>%
  select(Outcome, pred) %>% 
  mutate(pred = as.numeric(pred)-1) %>%
  mutate(category = "Training: Random Forest") -> predTrainRF

# Training predictions: Bag
data_2016 %>% mutate(pred = train.bag$predicted) %>%
  select(Outcome, pred) %>% 
  mutate(pred = as.numeric(pred)-1) %>%
  mutate(category = "Training: Bag") -> predTrainBag


bind_rows(predTrainBag, predTrainRF) %>%
  group_by(category, Outcome, pred) %>%
  count() %>%
  group_by(category) %>%
  mutate(totaln = sum(n), pred = ifelse(pred == 2, "R", "D"),
         correct = ifelse(Outcome == pred, 1, 0)) %>%
  filter(correct == 1) %>%
  summarize(accuracy = sum(n)/first(totaln)) %>%
  separate(category, c("Data", "Model"), sep=": ") -> accuracyTT

#Support Vector Machines
#Logistic Regression

#Support Vector Machines
#1 : Polynomial
library(e1071)
set.seed(1)
numerical_2016 <- data_2016[,c(-1,-2)]
numerical_2020 <- data_2020[,c(-1,-2)]
tune.out.poly <- tune(svm, as.factor(Outcome) ~ ., data = numerical_2016, 
                 kernel = "polynomial",
                 na.rm = TRUE,
                 ranges = list(
                   cost = c(0.1, 1, 10, 100, 1000),
                   degree = c(0.1, 0.5, 1, 2, 3, 4, 5)
                 )
)
pred.poly = predict(tune.out.poly$best.model, newdata = numerical_2016)
table(pred.poly, numerical_2016$Outcome)
misclassification_error_train_poly <- mean(pred.poly != numerical_2016$Outcome)
print(paste("Misclassification Error:", misclassification_error_train_poly))
#Misclassification Error: 0.0328502415458937
#Best Model:
# cost: 10
# degree: 2
poly.svm <- svm(as.factor(Outcome) ~., data = numerical_2016, cost=10, degree=2, probability = TRUE)
poly.svm.prob <- predict(poly.svm, type="prob", newdata = numerical_2020, probability = TRUE)

poly.svm.preds <- predict(poly.svm, newdata = numerical_2020)

pred.poly.valid = predict(tune.out.poly$best.model, newdata = numerical_2020)
poly.svm.predictions <- ifelse(poly.svm.prob == 2, "D", "R")
mean(pred.poly.valid == observed.classes)
table(pred.poly.valid, observed.classes)
library(ROCR)
predicted.classes <- ifelse(pred.poly.valid == "R", 1, 0)
actual <- numerical_2020$Outcome
actual <- ifelse(actual == "R", 1, 0)
poly.svm.prob.rocr <- prediction(predicted.classes, actual)
poly.svm.perf <- performance(poly.svm.prob.rocr, "tpr","fpr")

#2: Radial
tune.out.radial <- tune(svm, as.factor(Outcome) ~ ., data = numerical_2016, 
                 kernel = "radial",
                 na.rm = TRUE,
                 ranges = list(
                   cost = c(0.1, 1, 10, 100, 1000),
                   gamma = c(0.25, 0.5, 2, 3)
                 )
)
pred.radial = predict(tune.out.radial$best.model, newdata = numerical_2016)
table(pred.radial, numerical_2016$Outcome)
misclassification_error_train_radial <- mean(pred.radial != numerical_2016$Outcome)
print(paste("Misclassification Error:", misclassification_error_train_radial))
#Misclassification Error: 0
#cost: 10
#gamma: 0.25
pred.radial.valid = predict(tune.out.radial$best.model, newdata = numerical_2020)
radial.svm <- svm(as.factor(Outcome) ~., data = numerical_2016, cost=10, gamma=0.25, probability = TRUE)
radial.svm.prob <- predict(radial.svm, type="prob", newdata = numerical_2020, probability = TRUE)

radial.svm.predictions <- ifelse(radial.svm.prob == 2, "D", "R")
table(pred.radial.valid, observed.classes)

predicted.classes <- ifelse(pred.radial.valid == "R", 1, 0)
actual <- numerical_2020$Outcome
actual <- ifelse(actual == "R", 1, 0)
radial.svm.prob.rocr <- prediction(predicted.classes, actual)
radial.svm.perf <- performance(radial.svm.prob.rocr, "tpr","fpr")

#Lasso Regression
library(glmnet)
x <- model.matrix(Outcome ~ ., numerical_2016)[,-1]
y <- numerical_2016$Outcome
x.validation <- model.matrix(Outcome ~ ., numerical_2020)[,-1]
y.validation <- numerical_2020$Outcome

library(glmnet)
library(tidyverse)
library(caret)
# Find the best lambda using cross-validation
set.seed(123) 
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
# Fit the final model on the training data
model <- glmnet(x, y, alpha = 1, family = "binomial",
                lambda = cv.lasso$lambda.min)
# Display regression coefficients
coef(model)
# Make predictions on the test data
x.test <- model.matrix(Outcome ~., numerical_2020)[,-1]
probabilities <- model %>% predict(newx = x.validation)
pred.lasso.valid <- ifelse(probabilities > 0.5, "R", "D")
# Model accuracy
observed.classes <- numerical_2020$Outcome
table(pred.lasso.valid, observed.classes)

#92.4% Acuracy
library(ROCR)
predicted.classes <- ifelse(predicted.classes == "R", 1, 0)
actual <- numerical_2020$Outcome
actual <- ifelse(actual == "R", 1, 0)
lasso.prob.rocr <- prediction(predicted.classes, actual)
lasso.perf <- performance(lasso.prob.rocr, "tpr","fpr")

#Ridge Regression
lambda_seq <- 10^seq(2, -2, by = -.1)
# Using glmnet function to build the ridge regression in r
fit <- glmnet(x, y, alpha = 0, lambda  = lambda_seq, family = "binomial")
# Checking the model
summary(fit)

ridge_cv <- cv.glmnet(x, y, alpha = 0, lambda = lambda_seq, family = "binomial")
# Best lambda value
best_lambda <- ridge_cv$lambda.min
best_lambda

best_fit <- ridge_cv$glmnet.fit
head(best_fit)

best_ridge <- glmnet(x, y, alpha = 0, lambda = 0.01, family = "binomial")

pred <- predict(best_ridge, s = best_lambda, newx = x.validation)

pred.ridge.valid <- ifelse(pred > 0.5, "R", "D")
# Model accuracy
observed.classes <- numerical_2020$Outcome
table(pred.ridge.valid, observed.classes)

#Accuracy 92.72%
predicted.classes <- ifelse(predicted.classes == "R", 1, 0)
actual <- numerical_2020$Outcome
actual <- ifelse(actual == "R", 1, 0)
ridge.prob.rocr <- prediction(predicted.classes, actual)
ridge.perf <- performance(lasso.prob.rocr, "tpr","fpr")

#ROC curves of different models
plot(poly.svm.perf, col = 1, main="Performance of Tested Models")
plot(radial.svm.perf,col = 2,  add = TRUE)
plot(lasso.perf, col = 3, add = TRUE)
plot(ridge.perf, col = 4, add = TRUE)
legend(0.6, 0.6, c('Polynomial SVM', 'Radial SVM', 'Lasso Regression', 'Ridge Regression'), 1:4)

#Final Prediction Data Wrangling
pred.poly.valid.new <- ifelse(pred.poly.valid == 1, "D", "R")
pred.poly.valid.new <- as.data.frame(pred.poly.valid)
prediction_set_2020 <- cbind(data_2020$Total.Population, 100 - data_2020$Persons.under.18.years,
                             data_2020$White.only,
                             data_2020$Black.or.African.american.only,
                             data_2020$Asian.only,
                             data_2020$Hispanic.or.Latino,
                             data_2020$Outcome, 
                             pred.poly.valid.new, 
                             pred.ridge.valid )
colnames(prediction_set_2020) <- c("Population", "Perc_Over_18", 
                                   "Perc_White",
                                   "Perc_Black",
                                   "Perc_Asian",
                                   "Perc_Latino",
                                   "Actual_Outcome", 
                                   "Predicted_Outcome_POLY", "Predicted_Outcome_Ridge")
prediction_set_2020 <- as.data.frame(prediction_set_2020)

prediction_set_2020$voting_pop <- (as.numeric(prediction_set_2020$Perc_Over_18) / 100) * as.numeric(prediction_set_2020$Population)
#Calculating white votes with turnout
prediction_set_2020$white_votes <- (as.numeric(prediction_set_2020$Perc_White) / 100) *
                                    prediction_set_2020$voting_pop * 0.71
#Calculating black votes with turnout
prediction_set_2020$black_votes <- (as.numeric(prediction_set_2020$Perc_Black) / 100) *
  prediction_set_2020$voting_pop * 0.63
#Calculating asian votes with turnout
prediction_set_2020$asian_votes <- (as.numeric(prediction_set_2020$Perc_Asian) / 100) *
  prediction_set_2020$voting_pop * 0.60
#Calculating latino votes with turnout
prediction_set_2020$latino_votes <- (as.numeric(prediction_set_2020$Perc_Latino) / 100) *
  prediction_set_2020$voting_pop * 0.54

#Calculating Total Votes
prediction_set_2020$total_votes_adj <- prediction_set_2020$white_votes +
  prediction_set_2020$black_votes +
  prediction_set_2020$asian_votes +
  prediction_set_2020$latino_votes

prediction_set_2020$Predicted_Outcome_POLY <- pred.poly.valid.new



prediction_set_2020$democrat <- ifelse(prediction_set_2020$Actual_Outcome == "D", 1, 0)
prediction_set_2020$republican <- ifelse(prediction_set_2020$Actual_Outcome == "R", 1, 0)
prediction_set_2020$votes_democrat <- prediction_set_2020$total_votes_adj * prediction_set_2020$democrat
prediction_set_2020$votes_republican <- prediction_set_2020$total_votes_adj * prediction_set_2020$republican

prediction_set_2020$democrat_pred_ridge <- ifelse(prediction_set_2020$Predicted_Outcome_Ridge == "D", 1, 0)
prediction_set_2020$republican_pred_ridge <- ifelse(prediction_set_2020$Predicted_Outcome_Ridge == "R", 1, 0)
prediction_set_2020$votes_democrat_ridge <- prediction_set_2020$total_votes_adj * prediction_set_2020$democrat_pred_ridge
prediction_set_2020$votes_republican_ridge <- prediction_set_2020$total_votes_adj * prediction_set_2020$republican_pred_ridge

prediction_set_2020$democrat_pred_poly <- ifelse(prediction_set_2020$Predicted_Outcome_POLY == "D", 1, 0)
prediction_set_2020$republican_pred_poly <- ifelse(prediction_set_2020$Predicted_Outcome_POLY == "R", 1, 0)
prediction_set_2020$votes_democrat_poly <- prediction_set_2020$total_votes_adj * prediction_set_2020$democrat_pred_poly
prediction_set_2020$votes_republican_poly <- prediction_set_2020$total_votes_adj * prediction_set_2020$republican_pred_poly

#Correlation Matrix
numerical_2016_df <- as.data.frame(numerical_2016)
mydata.cor = cor(cor(numerical_2016[sapply(numerical_2016, is.numeric)]), method = "spearman")
palette = colorRampPalette(c("green", "white", "red")) (35)
heatmap(x = mydata.cor, col = palette, cexRow = 0.5,cexCol = 0.3, symm = TRUE)

library(corrplot)
corrplot(mydata.cor)
png(height=1200, width=1500, pointsize=15, file="ROCplot.png")

#2024 prediction 

data_2024 <- read.csv("2024Data.csv")
numerical_2024 <- data_2024[,c(-1,-2)]

pred.poly.test = predict(tune.out.poly$best.model, newdata = numerical_2024)


x.test <- model.matrix(~ ., numerical_2024)[,-1]
pred <- predict(best_ridge, s = best_lambda, newx = x.test)
pred.ridge.test <- ifelse(pred > 0.5, "R", "D")


pred.prob <- predict(best_ridge, s = best_lambda, newx = x.test, "response")

pred.poly.test.prob = predict(tune.out.poly$best.model, newdata = numerical_2024, type = "probability")

pred.ridge.test <- ifelse(pred > 0.5, "R", "D")




pred.poly.test.new <- as.data.frame(pred.poly.test)

prediction_set_2024 <- cbind(data_2024$Total.Population, 100 - data_2024$Persons.under.18.years,
                             data_2024$White.only,
                             data_2024$Black.or.African.american.only,
                             data_2024$Asian.only,
                             data_2024$Hispanic.or.Latino,
                             pred.poly.test.new, 
                             pred.ridge.test,
                             pred.prob)
colnames(prediction_set_2024) <- c("Population", "Perc_Over_18", 
                                   "Perc_White",
                                   "Perc_Black",
                                   "Perc_Asian",
                                   "Perc_Latino",
                                   "Predicted_Outcome_POLY", 
                                   "Predicted_Outcome_Ridge",
                                   "Probability")
prediction_set_2024 <- as.data.frame(prediction_set_2024)

prediction_set_2024$voting_pop <- (as.numeric(prediction_set_2024$Perc_Over_18) / 100) * as.numeric(prediction_set_2024$Population)
#Calculating white votes with turnout
prediction_set_2024$white_votes <- (as.numeric(prediction_set_2024$Perc_White) / 100) *
  prediction_set_2024$voting_pop * 0.71
#Calculating black votes with turnout
prediction_set_2024$black_votes <- (as.numeric(prediction_set_2024$Perc_Black) / 100) *
  prediction_set_2024$voting_pop * 0.63
#Calculating asian votes with turnout
prediction_set_2024$asian_votes <- (as.numeric(prediction_set_2024$Perc_Asian) / 100) *
  prediction_set_2024$voting_pop * 0.60
#Calculating latino votes with turnout
prediction_set_2024$latino_votes <- (as.numeric(prediction_set_2024$Perc_Latino) / 100) *
  prediction_set_2024$voting_pop * 0.54

#Calculating Total Votes
prediction_set_2024$total_votes_adj <- prediction_set_2024$white_votes +
  prediction_set_2024$black_votes +
  prediction_set_2024$asian_votes +
  prediction_set_2024$latino_votes



prediction_set_2024$democrat_pred_ridge <- ifelse(prediction_set_2024$Predicted_Outcome_Ridge == "D", 1, 0)
prediction_set_2024$republican_pred_ridge <- ifelse(prediction_set_2024$Predicted_Outcome_Ridge == "R", 1, 0)
prediction_set_2024$votes_democrat_ridge <- prediction_set_2024$total_votes_adj * prediction_set_2024$democrat_pred_ridge
prediction_set_2024$votes_republican_ridge <- prediction_set_2024$total_votes_adj * prediction_set_2024$republican_pred_ridge

prediction_set_2024$democrat_pred_poly <- ifelse(prediction_set_2024$Predicted_Outcome_POLY == "D", 1, 0)
prediction_set_2024$republican_pred_poly <- ifelse(prediction_set_2024$Predicted_Outcome_POLY == "R", 1, 0)
prediction_set_2024$votes_democrat_poly <- prediction_set_2024$total_votes_adj * prediction_set_2024$democrat_pred_poly
prediction_set_2024$votes_republican_poly <- prediction_set_2024$total_votes_adj * prediction_set_2024$republican_pred_poly

prediction_set_2024$weighted_votes_republican <- prediction_set_2024$Probability * prediction_set_2024$total_votes_adj

prediction_set_2024$weighted_votes_democrat <- (1 - prediction_set_2024$Probability) * prediction_set_2024$total_votes_adj


x.test <- model.matrix(~ ., numerical_2024)[,-1]
pred <- predict(best_ridge, s = best_lambda, newx = x.test)
table(pred, )