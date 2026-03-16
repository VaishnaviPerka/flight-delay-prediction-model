##### Loading the csv file #####
path <- "C:\\Users\\vaish\\Desktop\\Pred Modelling\\Major Project\\main_dataset.csv"
data <- read.csv(path)
dim(data)
data[1:5, 1:10]

colnames(data)

# Pre-processing Steps
##### 1. Sample Size #####
## As the Data set is huge
##considering only top 5 busiest airports
library(dplyr)
busiest_airports <- data %>%
  count(Origin, sort = TRUE) %>%
  top_n(5) %>%
  pull(Origin)


filtered_flights <- data %>%
  filter(Origin %in% busiest_airports)

## final dataset

dim(filtered_flights)
str(filtered_flights)

##### 2. Dropping #####
## drop unnecessary columns
filtered_flights <- subset(filtered_flights, select = -c(DepTime,WheelsOff,TaxiOut,WheelsOn,ArrTime,TaxiIn,CRSArrTime,ArrDel15,ArrivalDelayGroups,
                                                         DepartureDelayGroups,ArrTimeBlk,
                                                         DivAirportLandings,ActualElapsedTime,
                                                         DayofMonth,
                                                         FlightDate,Year,DOT_ID_Marketing_Airline, DOT_ID_Operating_Airline, OriginAirportID,
                                                         OriginAirportSeqID, OriginCityMarketID, DestAirportID, DestAirportSeqID,
                                                         DestCityMarketID, Flight_Number_Marketing_Airline, Flight_Number_Operating_Airline,
                                                         IATA_Code_Marketing_Airline, IATA_Code_Operating_Airline, Tail_Number,
                                                         Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners,
                                                         OriginCityName, OriginState, DestCityName, DestState,
                                                         OriginStateName,OriginStateFips,OriginWac
                                                         ,DestStateName,DestStateFips,DestWac,Operating_Airline,DepDelayMinutes,ArrDelayMinutes,DepDelay,ArrDelay))



dim(filtered_flights)
str(filtered_flights)
colnames(filtered_flights)

###### 3. Handle time format #####
time_columns <- c("CRSDepTime")

for (col in time_columns) {
  hour   <- filtered_flights[[col]] %/% 100
  minute <- filtered_flights[[col]] %% 100
  total_minutes <- hour * 60 + minute
  
  filtered_flights[[paste0(col, "_sin")]] <- sin(2 * pi * total_minutes / (24 * 60))
  filtered_flights[[paste0(col, "_cos")]] <- cos(2 * pi * total_minutes / (24 * 60))
  
  # drop original column
  filtered_flights[[col]] <- NULL
}

colnames(filtered_flights)
# Adding new variables
str(filtered_flights)

###### 4. Missing Values ######
colSums(is.na(filtered_flights))
# Remove
filtered_flights <- na.omit(filtered_flights)
dim(filtered_flights)
##### 5. Categorical to Numerical conversion #####
library(recipes)
library(dplyr)

library(embed)
filtered_flights$Cancelled <- as.numeric(filtered_flights$Cancelled)
filtered_flights$Diverted  <- as.numeric(filtered_flights$Diverted)

filtered_flights$DepTimeBlk_num <- as.numeric(substr(filtered_flights$DepTimeBlk, 1, 2))
filtered_flights <- filtered_flights %>% select(-DepTimeBlk)

library(tidymodels)

rec <- recipe(DepDel15 ~ ., data = filtered_flights) %>%
  
  
  step_mutate(DepDel15 = factor(DepDel15)) %>%
  
 
  step_mutate(
    Airline = factor(Airline),
    Origin = factor(Origin),
    Dest = factor(Dest),
    Quarter = factor(Quarter),
    Month = factor(Month),
    DayOfWeek = factor(DayOfWeek)
  ) %>%
  
  
  step_dummy(all_of(c("Quarter", "Month", "DayOfWeek"))) %>%
  
  
  step_lencode_mixed(all_nominal_predictors(), outcome = "DepDel15")

prep_rec <- prep(rec, verbose = TRUE)
filtered_flights <- bake(prep_rec, new_data = NULL)
str(filtered_flights)
###### 6. Near zero variance ######
library(caret)
nzv <- nearZeroVar(filtered_flights)
length(nzv)
nzv_vars <- colnames(filtered_flights)[nzv]
nzv_counts <- sapply(filtered_flights[nzv_vars], function(x) length(unique(x)))
barplot(
  nzv_counts,
  main = "Near Zero Variance Variables",
  ylab = "Number of Unique Values",
  xlab = "Variables",
  las = 2,
  cex.names = 0.8
)

filtered_flights <- filtered_flights[, -nzv]
str(filtered_flights)
-------------------------------------------------------------------
##### SPLIT #####
trainIndex <- createDataPartition(filtered_flights$DepDel15, p = 0.8, list = FALSE)
train_data <- filtered_flights[trainIndex, ]
test_data  <- filtered_flights[-trainIndex, ]
target_var <- "DepDel15"   

# Outcome (y)
y_train <- train_data[[target_var]]
y_test  <- test_data[[target_var]]

y_train <- factor(y_train, levels = c(0,1), labels = c("NoDelay","Delay"))
y_test  <- factor(y_test,  levels = c(0,1), labels = c("NoDelay","Delay"))

# Predictors (X)
x_train <- train_data[, setdiff(names(train_data), target_var), drop = FALSE]
x_test  <- test_data[,  setdiff(names(test_data),  target_var), drop = FALSE]

str(x_train)

###### 8. Box Cox Transformation (ONLY) #######  

x_train <- x_train %>% 
  mutate(
    AirTime = as.numeric(AirTime),
    CRSElapsedTime = as.numeric(CRSElapsedTime),
    Distance = as.numeric(Distance)
  )

library(caret)

cont_vars <- c("AirTime", "CRSElapsedTime", "Distance")
x_train_only_boxcox <- x_train

preproc_bc <- preProcess(
  x_train[ , cont_vars] %>% as.data.frame(),
  method = "BoxCox"
)

x_train_only_boxcox[ , cont_vars] <- predict(preproc_bc, x_train[ , cont_vars] %>% as.data.frame())

x_test_only_boxcox <- x_test
x_test_only_boxcox[ , cont_vars] <- predict(preproc_bc, x_test[ , cont_vars] %>% as.data.frame())

###### 9.(A) Remove Highly Correlated variables (and box-cox) ######
num_cols_bc <- names(x_train_only_boxcox)[sapply(x_train_only_boxcox, is.numeric)]

cor_mat_bc <- cor(x_train_only_boxcox[, num_cols_bc])
highCorr_bc <- findCorrelation(cor_mat_bc, cutoff = 0.90)
remove_corr_bc <- num_cols_bc[highCorr_bc]

x_train_both_box_RHC <- x_train_only_boxcox[, !(names(x_train_only_boxcox) %in% remove_corr_bc)]
x_test_both_box_RHC  <- x_test_only_boxcox[,  !(names(x_test_only_boxcox)  %in% remove_corr_bc)] # Logistic, LDA, QDA, MDA

###### 9.(B) Remove Highly Correlated variables (no box cox) ######
num_cols_orig <- names(x_train)[sapply(x_train, is.numeric)]

cor_mat_orig <- cor(x_train[, num_cols_orig])
highCorr_orig <- findCorrelation(cor_mat_orig, cutoff = 0.90)

remove_corr_orig <- num_cols_orig[highCorr_orig]

x_train_only_RHC <- x_train[, !(names(x_train) %in% remove_corr_orig)]
x_test_only_RHC  <- x_test[,  !(names(x_test)  %in% remove_corr_orig)] # NN, NB, FDA

#### NO Box-cox No RHC ####
x_train_no_box_no_RHC <- x_train
x_test_no_box_no_RHC  <- x_test #SVM, KNN

------------------------------------------------------------------------------------------------------------------------------
######## MODEL BUILDING #########
# BOTH Box-Cox and Removing Highly correlated (Logistic, LDA, QDA, MDA)

library(caret)
library(MASS)  # for lda, qda
library(mda)   # for mda

set.seed(123)  # for reproducibility

ctrl_both_box <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = defaultSummary,  # gives Accuracy & Kappa
  savePredictions = "final"
)

# Logistic
set.seed(123)
fit_logit <- train(
  x = x_train_both_box_RHC,
  y = y_train,
  method = "glm",
  family = binomial,
  metric = "Kappa",
  preProcess = c("center","scale"),
  trControl = ctrl_both_box
)

# LDA
set.seed(123)
fit_lda <- train(
  x = x_train_both_box_RHC,
  y = y_train,
  method = "lda",
  metric = "Kappa",
  preProcess = c("center","scale"),
  trControl = ctrl_both_box
)

# QDA
set.seed(123)
fit_qda <- train(
  x = x_train_both_box_RHC,
  y = y_train,
  method = "qda",
  metric = "Kappa",
  preProcess = c("center","scale"),
  trControl = ctrl_both_box
)

#MDA
# Define tuning grid for MDA: number of subclasses
mda_grid <- expand.grid(subclasses = 1:5)

set.seed(123)
fit_mda <- train(
  x = x_train_both_box_RHC,
  y = y_train,
  method = "mda",
  metric = "Kappa",
  tuneGrid = mda_grid,
  preProcess = c("center","scale"),
  trControl = ctrl_both_box
)

# ONLY Box-Cox (PLS_DA, Penalized, RDA)
library(caret)
library(glmnet)  # for penalized (via method = "glmnet")
library(klaR)   # for RDA (via method = "rda")

set.seed(123)

ctrl_only_box <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = defaultSummary,  # Accuracy & Kappa
  savePredictions = "final"
)

# PLS_DA
# Tuning grid for number of PLS components
pls_grid <- expand.grid(ncomp = 1:10)

set.seed(123)
fit_plsda <- train(
  x = x_train_only_boxcox,
  y = y_train,
  method = "pls",
  metric = "Kappa",
  tuneGrid = pls_grid,
  preProcess = c("center","scale"),
  trControl = ctrl_only_box
)

# Penalized 
penal_grid <- expand.grid(
  alpha  = seq(0, 1, by = 0.25),
  lambda = 10^seq(-3, 1, length = 10)
)

set.seed(123)
fit_penal <- train(
  x = x_train_only_boxcox,
  y = y_train,
  method = "glmnet",
  metric = "Kappa",
  tuneGrid = penal_grid,
  preProcess = c("center","scale"),
  trControl = ctrl_only_box
)

# RDA
# Tuning grid for RDA
rda_grid <- expand.grid(
  gamma  = seq(0, 1, by = 0.25),
  lambda = seq(0, 1, by = 0.25)
)

set.seed(123)
fit_rda <- train(
  x = x_train_only_boxcox,
  y = y_train,
  method = "rda",
  metric = "Kappa",
  tuneGrid = rda_grid,
  preProcess = c("center","scale"),
  trControl = ctrl_only_box
)

# ONLY Removing Highly Correlated (Neural Network, Naive Bayes, FDA)
library(caret)
library(klaR)     # Naive Bayes
library(nnet)     # Neural Net
library(mda)      # for FDA (method = "fda")

set.seed(123)

ctrl_only_RHC <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = defaultSummary,   # Accuracy & Kappa
  savePredictions = "final"
)

# Neural Network
nn_grid <- expand.grid(
  size = c(1, 3, 5, 7, 9),
  decay = c(0, 0.001, 0.01, 0.1)
)

set.seed(123)
fit_nn <- train(
  x = x_train_only_RHC,
  y = y_train,
  method = "nnet",
  trace = FALSE,
  metric = "Kappa",
  tuneGrid = nn_grid,
  preProcess = c("center","scale","spatialSign"),
  trControl = ctrl_only_RHC
)

# NAive Bayes
nb_grid <- expand.grid(
  fL = c(0, 0.5, 1),
  usekernel = c(TRUE, FALSE),
  adjust = c(0.5, 1, 2)
)

set.seed(123)
fit_nb <- train(
  x = x_train_only_RHC,
  y = y_train,
  method = "nb",
  metric = "Kappa",
  tuneGrid = nb_grid,
  preProcess = c("center","scale"),
  trControl = ctrl_only_RHC
)

# FDA
fda_grid <- expand.grid(
  degree = 1:2,
  nprune = seq(5, 30, by = 5)
)

set.seed(123)
fit_fda <- train(
  x = x_train_only_RHC,
  y = y_train,
  method = "fda",
  metric = "Kappa",
  tuneGrid = fda_grid,
  preProcess = c("center","scale"),
  trControl = ctrl_only_RHC
)

# No boxcox no RHC (SVM,KNN)
library(caret)
library(kernlab)   # for svmRadial

set.seed(123)

ctrl_no_box_no_RHC <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = defaultSummary,   # Accuracy & Kappa
  savePredictions = "final"
)

# SVM
sigma_vals <- sigest(as.matrix(x_train_no_box_no_RHC))  # from kernlab, gives a range
svm_grid <- expand.grid(
  sigma = as.numeric(sigma_vals),
  C     = c(0.25, 0.5, 1, 2, 4)
 )

 set.seed(123)
 fit_svm <- train(
  x = x_train_no_box_no_RHC,
   y = y_train,
  method = "svmRadial",
  metric = "Kappa",
  tuneGrid = svm_grid,
  preProcess = c("center","scale"),
  trControl = ctrl_no_box_no_RHC
)
 
#KNN
knn_grid <- expand.grid(
   k = seq(1, 25, by = 2)   # odd ks from 1 to 25
 )
 
set.seed(123)
fit_knn <- train(
   x = x_train_no_box_no_RHC,
   y = y_train,
   method = "knn",
   metric = "Kappa",
   tuneGrid = knn_grid,
   preProcess = c("center","scale","spatialSign"),
   trControl = ctrl_no_box_no_RHC
)

# Tuning Plots
plot(fit_logit, main = "Logistic Regression – (No tuning grid)")
plot(fit_lda,   main = "LDA – (No tuning grid)")
plot(fit_qda,   main = "QDA – (No tuning grid)")
plot(fit_mda,   main = "MDA – Tuning (subclasses vs Kappa)")
plot(fit_plsda, main = "PLS-DA – Tuning (ncomp vs Kappa)")
plot(fit_penal, main = "Penalized (glmnet) – Tuning (alpha, lambda)")
plot(fit_rda,   main = "RDA – Tuning (gamma, lambda)")
plot(fit_nn,  main = "Neural Network – Tuning (size, decay)")
plot(fit_nb,  main = "Naive Bayes – Tuning (fL, usekernel, adjust)")
plot(fit_fda, main = "FDA – Tuning (degree, nprune)")
plot(fit_nn,  main = "Neural Network – Tuning (size, decay)")
plot(fit_nb,  main = "Naive Bayes – Tuning (fL, usekernel, adjust)")
plot(fit_fda, main = "FDA – Tuning (degree, nprune)")

# SUmmary Table
###############################################################
### MASTER SUMMARY TABLE FOR ALL MODELS (TRAINING PERFORMANCE)
###############################################################

all_models <- list(
  Logistic      = fit_logit,
  LDA           = fit_lda,
  QDA           = fit_qda,
  MDA           = fit_mda,
  PLS_DA        = fit_plsda,
  Penalized     = fit_penal,
  RDA           = fit_rda,
  NeuralNet     = fit_nn,
  NaiveBayes    = fit_nb,
  FDA           = fit_fda,
  SVM_Radial    = fit_svm,
  KNN           = fit_knn
)

# ---- 1. Extract max Kappa for each model ----
train_kappa_all <- sapply(all_models, function(m) {
  max(m$results$Kappa, na.rm = TRUE)
})

# ---- 2. Extract max Accuracy ----
train_acc_all <- sapply(all_models, function(m) {
  max(m$results$Accuracy, na.rm = TRUE)
})

# ---- 3. Extract BEST tuning parameter(s) as text ----
best_tune_all <- sapply(all_models, function(m) {
  if (!is.null(m$bestTune) && ncol(m$bestTune) > 0) {
    paste0(
      apply(m$bestTune, 1, function(row)
        paste(names(row), "=", row, collapse = ", ")
      ),
      collapse = "; "
    )
  } else {
    "None"
  }
})

# ---- 4. Build final summary table ----
all_model_summary <- data.frame(
  Model              = names(train_kappa_all),
  Training_Kappa     = as.numeric(train_kappa_all),
  Training_Accuracy  = as.numeric(train_acc_all),
  Best_Tuning        = best_tune_all,
  row.names = NULL
)

# ---- 5. Sort by Training Kappa (descending) ----
all_model_summary <- all_model_summary[order(-all_model_summary$Training_Kappa), ]

all_model_summary





#### Best models ####
# NAIVE BAYES
pred_nb <- predict(fit_nb, newdata = x_test_only_RHC)

# Confusion Matrix
cm_nb <- confusionMatrix(pred_nb, Y_test)
cm_nb

# Extract metrics
nb_accuracy <- cm_nb$overall['Accuracy']
nb_kappa    <- cm_nb$overall['Kappa']

nb_accuracy
nb_kappa

# QDA
pred_qda <- predict(fit_qda, newdata = x_test_both_box_RHC)

# Confusion Matrix
cm_qda <- confusionMatrix(pred_qda, Y_test)
cm_qda

# Extract metrics
qda_accuracy <- cm_qda$overall['Accuracy']
qda_kappa    <- cm_qda$overall['Kappa']

qda_accuracy
qda_kappa
