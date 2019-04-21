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

## separate training / validation and testing // now using validation==testing
set.seed(1)
inTrain <- createDataPartition(data$classe, p = .7, list = FALSE)
validation <- data[-inTrain, ]
training <- data[inTrain,]

## set up parallel computing for faster compute
library(doParallel); library(parallel)
cluster <- makeCluster(detectCores() -1)
registerDoParallel(cluster)

train_control <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

testModel <- function(model, tc) {
    set.seed(123)
    start_time <- Sys.time()
    fit <- train(classe ~ ., data = training, method = model, metric = "Accuracy", trControl = tc)
    print(Sys.time() - start_time)
    confM <- table(predict(fit,newdata=validation),validation$classe)
    print((sum(confM )-sum(diag(confM )))/sum(confM))
}

treebag <- testModel("treebag", train_control)
gbm <- testModel("gbm", train_control)
randomForest <- testModel("rf", train_control)
knn <- testModel("knn", train_control)
lvq <- testModel("lvq", train_control)
svmRadial <- testModel("svmRadial", train_control)
    

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

stopCluster(cluster)
registerDoSEQ()

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