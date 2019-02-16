#############
# # K N N # #
#############


ata("iris")
str(iris)

library(caret) #R에서 데이터 분석 할 때 많이 쓰는 패키지
library(class) #KNN 패키지

#데이터 7:3으로 나누기
idx = createDataPartition(y=iris$Species, p=0.7, list = F)
train = iris[idx,]
test = iris[-idx,]

#KNN 써보자(비교를 위해 아무것도 안함), 임의로 K는 5로 지정
model.knn <- knn(train[-5],test[-5], train$Species, k=5)  #5번째 변수가 종속변수
confusionMatrix(model.knn, test$Species) #Accuracy : 0.9556  #분류할 때 모델의 정확도 -- 모델이 예측한 분류와 실제 분류 비교



#######다듬고 KNN 써보자!
## min-max 스케일링
normalize <- function(x){
  return( (x-min(x))/(max(x)-min(x)) )
}

iris_normal <- as.data.frame(lapply(iris[-5], normalize))
summary(iris_normal) #모두다 0~1값

iris_normal$Species <- iris$Species 

#다시 나누기
train_n = iris_normal[idx,]
test_n = iris_normal[-idx,]

#최적의 k 찾기
# ks = 실험할 k, 일부러 홀수로 지정하였다. leave one out 방법
install.packages("kknn"); 
library(kknn)
cv <- train.kknn(Species ~., train_n, 
                      ks = seq(1, 50, by=2), scale = F); cv
best_k <- cv$best.parameters$k; best_k #5

pred_cv <- kknn(Species ~., train = train_n, test = test_n, k = best_k, scale = F)
pred_cv <- pred_cv$fitted.values

confusionMatrix(pred_cv, test_n$Species) #0.9556

#그리드 서치 cv
cv <- trainControl(method = "cv", number = 5, verbose = T)

knn.grid = expand.grid(
  .k = c(1,3,5,7,9)
)

train.knn <- train(Species~.,train_n, method = "knn",trControl = cv,
                   tuneGrid = knn.grid)
train.knn$results
train.knn$bestTune #9 #최적의 k는 9
predict.knn <- predict(train.knn, test_n)
confusionMatrix(predict.knn, test_n$Species) #0.9778 #이전의 knn보다 성능 상승 
