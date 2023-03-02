rm(list = ls())
library(dplyr)
library(TTR)
library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)
library(caret)
library(randomForest)
#==================================
ma <- function(x, n = 5){stats::filter(x, rep(1 / n, n), sides = 1)}

mse <- function(obs, pred) {
  mean((obs-pred)^2)
}
#=================================
df=read.csv(".../Bime.Saman.Co..csv") 
df$name=rep('Saman',dim(df)[1])
df$class=rep(NA,dim(df)[1])
for (i in 2:dim(df)[1]){
  df$class[i]=ifelse(df$X.CLOSE.[i]>df$X.CLOSE.[i-1],'UP','DOWN')
}

df$MACD=MACD(df$X.CLOSE.)[,1]
df$RSI=RSI(df$X.CLOSE.)
df$stochK <- stoch(df[,c(4,5,6)])[,"fastK"]
df$ADX<- ADX(df[,c(4,5,6)])[,4]
df$lag1<-c(NA,lags(df$X.CLOSE.,n=1)[,1])

df=na.omit(df)

df1=data.frame(df$X.OPEN.,df$lag1,df$ADX,df$MACD,df$RSI,df$stochK,df$class)
colnames(df1) <- c('OPEN','Lag1','ADX','MACD','RSI','stochK','CLASS')

dat.d <- sample(1:nrow(df1),size=nrow(df1)*0.8,replace = FALSE) #random selection of 80% data.

train.X=scale(df1[dat.d,-7])
test.X=scale(df1[-dat.d,-7])

train.Y=df1$CLASS[dat.d]

#Creating seperate dataframe for 'class' feature which is our target.
train.labels <- df1[dat.d,7]
test.labels <-df1[-dat.d,7]
#=================================================

ggplot(df.plot) +
  geom_bar(aes(x=name,y=value),stat="identity", fill="skyblue", alpha=0.5)+
  ggtitle(main) +
  xlab('Class') +
  ylab('Freq')
#KNN===================================================
accuracy.knn.test.k=c()
for(k in 1:35){
  knn.pred=class::knn(train.X,test.X,train.labels ,k=k)
  acc.knn.test=table(knn.pred ,test.labels)
  accuracy.knn.test.k[k]=(acc.knn.test[1,1]+acc.knn.test[2,2])/sum(acc.knn.test)
}

plot(accuracy.knn.test.k,type='b',col="blue",ylab="Accuracy",xlab="K",
     main="",cex.lab=.7,cex.axis=.7,cex.main=.7)

knn.pred=class::knn(train.X,test.X,train.labels ,k=which.max(accuracy.knn.test.k))
(acc.knn.test=table(knn.pred ,test.labels))
(accuracy.knn.test=(acc.knn.test[1,1]+acc.knn.test[2,2])/sum(acc.knn.test))
mse(as.numeric(knn.pred) ,as.numeric(as.factor(test.labels)))

#=====================================
#=======================================================
#=================Decision Tree=========================
#=======================================================



data_train=df1[dat.d,]
data_test=df1[-dat.d,]

fit <- rpart::rpart(as.factor(CLASS)~., data = data_train, method = 'class')

fit$variable.importance

printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results rsquare
#summary(fit) # detailed summary of splits





# prune the tree using the lowest xerror
pfit<- prune(fit, cp=   fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])

# plot the pruned tree

#fancyRpartPlot(pfit,cex = 0.4, xpd = TRUE)

rpart.plot(pfit,extra=106,cex=.4, xpd = TRUE,type=0,main="",cex.main=.7)



#prediction
predict_unseen <-predict(pfit, data_test, type = 'class')



#table_mat <- table(data_test$CLASS, predict_unseen)
#table_mat

#accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
#print(paste('Accuracy for test', round(accuracy_Test,4)))

head(data.frame(data_test,predict_unseen))

confusionMatrix(as.factor(data_test$CLASS),predict_unseen)

mse(as.numeric(as.factor(data_test$CLASS)),as.numeric(predict_unseen))

#=======================================================
#=================Random Forest=========================
#=======================================================



#data_train=as.data.frame(cbind(Standardized.X[dat.d,],df1$CLASS[dat.d]))
#data_test=as.data.frame(cbind(Standardized.X[-dat.d,]))
#data_train=df1[dat.d,-5]
data_test=data_test[,-7]


customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

# train model
metric <- "Accuracy"
control <- trainControl(method="repeatedcv", number=10, repeats=3)
tunegrid <- expand.grid(.mtry=c(1:6), .ntree=c(100, 150, 200, 250, 500, 1000))
set.seed(12345)
custom <- train(as.factor(CLASS)~., data=data_train, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
#summary(custom)

par()  
# view current settings
opar <- par()      # make a copy of current settings
par(cex.lab=.4,cex.main=.5,mex=.8) # red x and y labels
#560-270
plot(custom,ylab='Accuracy',main="",cex.lab=.4,cex.main=.5)   # create a plot with these new settings
par(opar)          # restore original settings


(mtry=custom$finalModel$mtry)
(ntree=custom$finalModel$ntree)

#use the best values finding above
tunegrid <- expand.grid(.mtry=c(1:6))
fit_rf <- train(as.factor(CLASS)~.,
                data_train,
                method = "rf",
                metric=metric, 
                tuneGrid=tunegrid, 
                trControl=control,
                importance = TRUE,
                nodesize = 14,
                ntree = 250,
                maxnodes = 2)

prediction <-predict(fit_rf, data_test)

prediction.tr <-predict(fit_rf, data_train)

confusionMatrix(prediction, as.factor(df1$CLASS[-dat.d]))

confusionMatrix(prediction.tr, as.factor(df1$CLASS[dat.d]))

mse(as.numeric(prediction), as.numeric(as.factor(df1$CLASS[-dat.d])))
#prediction <-predict(fit_rf, data_train)

#confusionMatrix(prediction, as.factor(df1$CLASS[dat.d]))

#varImpPlot(fit_rf)

varImp(fit_rf)
#============================================
#Roc curve and AUC
#============================================
ConMat <- table(test.y,pred.y)
TN <- ConMat[1,1]
FP <- ConMat[1,2]
FN <- ConMat[2,1]
TP <- ConMat[2,2]

(Acc <- (TP + TN)/(TP + TN + FP + FN))
(pres <- TP/(TP + FP))
(recall <- TP/(TP + FN))
(f.score <- 2*(pres*recall/(pres+recall)))
#============
library(pROC)

set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(as.factor(CLASS) ~ ., data = data_train, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)


knn_prediction <- predict(knnFit, data_test, type = "prob")
dt_prediction <- predict(pfit, data_test, type = "prob")
rf_prediction <- predict(fit_rf, data_test, type = "prob")

roc.knn = roc(true.test.class ~ knn_prediction[,2], plot = TRUE, print.auc = TRUE)
roc.dt = roc(as.numeric(as.factor(data_test$CLASS)) ~ dt_prediction[,2], plot = TRUE, print.auc = TRUE)
roc.rf = roc(as.numeric(as.factor(df1$CLASS[-dat.d])) ~ rf_prediction[,2], plot = TRUE, print.auc = TRUE)


auc(roc.knn)
auc(roc.dt)
auc(roc.rf)

g.list <- ggroc(list(roc.knn, roc.dt,roc.rf),aes=c("linetype", "color"),show.legend = T)
g.list + scale_colour_manual(labels = c( "KNN","DT","RF"),values = c( "red", "blue","green"))+
  ggtitle("ROC curve")+
  theme(plot.title = element_text(size = 8, face = "bold"),legend.text = element_text(size=8))+
  guides(color=guide_legend("Model"),linetype=F)+
  annotate("text", x=.25, y=.55, label= "AUCs:",size=3) + 
  #annotate("text", x=.25, y=.55, label= "Logistic=0.803",size=2) + 
  annotate("text", x = .25, y=.45, label = "KNN=0.788",size=3)+
  annotate("text", x=.25, y=.35, label= "DT=0.764",size=3) + 
  annotate("text", x = .25, y=.25, label = "RF=0.743",size=3)



