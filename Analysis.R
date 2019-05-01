setwd("/Users/manonbreuvart/Documents/R studio/DataScienceMachineLearning/")
library(caret); library(dplyr)

data <- read.csv("pml-training.csv",header = TRUE, stringsAsFactors = FALSE)
## preparing the data, removing columns for which the sets are incomplete
## a look at the data will show that the case where 'new_window' is no has a lot of missing data
toKeep <- (apply(data[data$new_window=="no",], 2, function(x) {mean(!is.na(x) && !(x==""))}) ==1)
data <- data[,toKeep]

## remove data that does not seem meaningful: row number, timestamps and window
## User name is also removed here as I assume the model will be used with other people
data <- data[,-c(1:7)]
data$classe <- as.factor(data$classe)

# reduction of features
correlationMatrix <- cor(data[,1:52])
highcorrel <- findCorrelation(correlationMatrix, cutoff=.8)
data <- data[,-highcorrel]

## load and process the testing data
testing <- read.csv("pml-testing.csv",header = TRUE, stringsAsFactors = FALSE)
testing <- testing[,toKeep]
testing <- testing[,-c(1:7)]
testing <- testing[,-highcorrel]

## separate training / validation
set.seed(1)
inTrain <- createDataPartition(data$classe, p = .7, list = FALSE)
validation <- data[-inTrain, ]
training <- data[inTrain,]



## set up parallel computing for faster compute
library(doParallel); library(parallel)


train_control <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

train_Model <- function(model, tc) {
    set.seed(123)
    cluster <- makeCluster(detectCores() -1)
    registerDoParallel(cluster)
    start_time <- Sys.time()
    fit <- train(classe ~ ., data = training, method = model, metric = "Accuracy", trControl = tc)
    duration <- as.numeric(Sys.time() - start_time)
    print(duration)
    stopCluster(cluster)
    registerDoSEQ()
    return(fit)
}

validate_Model <- function(model){
    confM <- table(predict(model,newdata=validation),validation$classe)
    return((sum(confM )-sum(diag(confM )))/sum(confM))
}

train_control <- trainControl(method = "cv", number = 2, allowParallel = TRUE)
models <- c("treebag","gbm","rf","knn","lvq","svmRadial","xgbTree","nnet","avNNet")
models <- c("treebag","gbm")

fits <- lapply(models, function(x) {train_Model(x,train_control)})
accuracies <- sapply(fits, function(x) {1- validate_Model(x)})
results <- as.data.frame(cbind(models,accuracy = round(as.numeric(accuracies*100),digits = 2),
                               error = round(100- as.numeric(accuracies*100),digits = 2)))
arrange(results, desc(accuracy))

train_control <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

randomForest <- train_Model("rf", train_control)
xgbTree <- train_Model("xgbTree", train_control)  


validate_Model(randomForest)*100
validate_Model(xgbTree)*100

randomForestImp <- varImp(randomForest, scale=FALSE) 



#adaboost
start_time <- Sys.time()
ada <- train(classe ~ ., data = training, method = "ada",control=rpart.control(maxdepth=30, cp=0.010000, minsplit=20, 
                                                                               xval=10), iter=500)
Sys.time() - start_time
ada.validation.predict <- predict(ada, newdata = validation)
tb <- table(ada.validation.predict, validation$classe)
# error rate validation
tb <- table(ada.validation.predict, validation$classe)
(sum(tb)-sum(diag(tb)))/sum(tb)



consensus <- cbind(as.data.frame(gbm.validation.predict),as.data.frame(treebag.validation.predict),as.data.frame(randomForest.validation.predict))
consensus <- tbl_df(consensus)
consensus$score <- ifelse(consensus$randomForest.validation.predict == consensus$treebag.validation.predict, consensus$randomForest.validation.predict,
                          ifelse(consensus$randomForest.validation.predict == consensus$gbm.validation.predict, consensus$randomForest.validation.predict,
                                 ifelse(consensus$treebag.validation.predict == consensus$gbm.validation.predict,consensus$treebag.validation.predict,
                                        consensus$randomForest.validation.predict)))

consensus$score <- as.factor(consensus$score)
levels(consensus$score) <- levels(consensus$gbm.validation.predict)

tb <- table(consensus$score, validation$classe)
(sum(tb)-sum(diag(tb)))/sum(tb)

## some plots pairwise - no super obvious relationships
library(ggplot2) ; library(GGally)
ggpairs(data, columns=5:9, aes(color=classe, alpha=.0001))
ggpairs(filter(training, classe %in%c("A","B")), columns=5:9, aes(color=classe, alpha=.1))
ggpairs(filter(training, classe %in%c("C","D")), columns=5:12, aes(color=classe, alpha=.1))
ggpairs(filter(training, classe %in%c("E")), columns=5:12, aes(color=classe, alpha=.1))

# some otherway to plot pairwise
library(AppliedPredictiveModeling)
transparentTheme(trans = .3)
featurePlot(x = data[, 1:10], 
                y = data$classe, 
                plot = "ellipse",
                ## Add a key at the top
                auto.key = list(columns = 3))