#####################################
##부동산 경매 데이터 회귀분석##
#####################################


#데이터 불러오기
df = read.csv("C:/Users/Kim Yuum/Desktop/Auction_master_train.csv", stringsAsFactors = T, fileEncoding="utf-8")
head(df)

#데이터 확인해보기
head(df)
str(df)
colSums(is.na(df))  #결측치 개수 확인

#필요없는 열 제거
##결측치가 1000개 이상인 컬럼, 공백인 컬럼, key컬럼 삭제, 회사 데이터는 plot을 그려봤을 때 매우 상관성이 없으므로 제거
df2 = subset(df, select = -c(road_bunji1,Final_result,Auction_key,addr_li,
                             road_bunji2,addr_bunji2,Appraisal_company,Creditor,
                             addr_si,addr_dong,addr_san,addr_bunji1,addr_etc,Specific, Close_result))  

#날짜 데이터를 시계 데이터로 변환
df2$First_auction_date = as.Date(df2$First_auction_date)   
df2$Final_auction_date = as.Date(df2$Final_auction_date)
df2$Preserve_regist_date = as.Date(df2$Preserve_regist_date)
df2$Appraisal_date = as.Date(df2$Appraisal_date)
df2$Close_date = as.Date(df2$Close_date)
data$auction_date = as.numeric(data$auction_date)
head(data$auction_date)

#파생변수 만들기
df2$auction_date = df2$Final_auction_date - df2$First_auction_date  #총 경매일 = 최종경매일 - 최초경매일 

lalon = cbind(df2$point.x,df2$point.y)    ##위도 경도 데이터 kmeans clustering -> 비슷한 위치끼리 군집화
Data_scaled <- scale(lalon, center = TRUE, scale = TRUE)  #먼저 위도 경도 데이터 스케일링
km = kmeans(Data_scaled,6)    #스케일링한 위경도 데이터 군집화
df2$point_cluster = km$cluster   #군집화된 결과를 파생변수로 추가

df2$area = df2$Total_land_auction_area / df2$Total_land_gross_area #경매된 토지 면적 비율 #df2$Total_land_gross_area이 0일 때 결측치나 Inf 값 도출
df2$area[which(df2$Total_land_gross_area==0)] = 0  #area가 NA나 Inf 값이 나올 경우 모두 0 처리 

df2$AC.dummy[df2$Auction_class=="임의"] <- 0  #강제경매와 임의경매를 더미변수 처리 
df2$AC.dummy[df2$Auction_class=="강제"] <- 1
df2$AC.dummy <- as.factor(df2$AC.dummy)

df2$BC.dummy[df2$Bid_class=="일반"] <- 0  #입찰구분을 더미변수 처리 
df2$BC.dummy[df2$Bid_class=="개별"] <- 1
df2$BC.dummy[df2$Bid_class=="일괄"] <- 2
df2$BC.dummy <- as.factor(df2$BC.dummy)

df2$addr_dummy[df2$addr_do=="서울"] <- 0  #주소 구분을 더미변수 처리 
df2$addr_dummy[df2$addr_do=="부산"] <- 1
df2$addr_dummy <- as.factor(df2$addr_dummy)

df2$usage_dummy[df2$Apartment_usage=="아파트"] <- 0  #건물 용도를 더미변수 처리 
df2$usage_dummy[df2$Apartment_usage=="주상복합"] <- 1
df2$usage_dummy <- as.factor(df2$usage_dummy)

df2$share_dummy[df2$Share_auction_YorN=="N"] <- 0  #지분경매 여부를 더미변수 처리 
df2$share_dummy[df2$Share_auction_YorN=="Y"] <- 1
df2$share_dummy <- as.factor(df2$share_dummy)

data = subset(df2, select = -c(Auction_class, Bid_class,Share_auction_YorN,Apartment_usage,addr_do, road_name))  #더미변수화한 변수 삭제
str(data)

#Hammer_price 상위 50%는 1, 하위 50%는 2로 정렬
library(dplyr)
data = arrange(data, desc(Hammer_price))  #Hammer_price 크기 순으로 데이터 정렬

nrow(data)/2 #median 행 번호 알기

for(i in 1:nrow(data)){   #상위 50%는 1, 하위 50%는 2 부여
  if(data$Hammer_price[i] >= data$Hammer_price[996]){
    data$Hammer_level[i] = 1
  }else{
    data$Hammer_level[i] = 2
  }
}
data$Hammer_level = as.factor(data$Hammer_level)

#레이블과 변수의 분포 시각화
##낙찰가와 유의한 연관이 있을 것이라고 예상되는 변수와 레이블과의 plot살펴보기
plot(data$Total_land_gross_area, data$Hammer_level)  #총토지전체면적이 작을 때 낙찰가도 낮음 #이상치가 2개 있음 
plot(data$Total_appraisal_price, data$Hammer_level)  #총 감정가가 높을수록 낙찰가도 높음  
plot(df2$Minimum_sales_price, df2$Hammer_price)  #최저 매각 가격이 높을수록 낙찰가도 높음  

###1. 로지스틱 회귀###
#데이터 train, test 분리 7:3
# idx 설정 (7 : 3)
idx = sample(x = c("train", "test"),
              size = nrow(data),
              replace = TRUE,
              prob = c(7, 3))

# idx에 따라 데이터 나누기
train = data[idx == "train", ]
test = data[idx == "test", ]

fit = glm(Hammer_level~., family = "binomial", data = train) #로지스틱 회귀 모델 적합
summary(fit)  #각 변수의 p-value를 볼 때 모두 유의한 변수임을 알 수 있음

pred1 = predict(fit, newdata = test[-29], type = 'response') ##예측
pred_label = ifelse(pred1 >= 0.5, 2, 1)  #예측값 임계치 0.5로 적용
table(test$Hammer_level, pred_label) #정확도 89.4%
test$label = pred_label  #예측한 컬럼 추가 
View(test)


#5-fold cross validation
library(caret)
cv <- trainControl(method = "cv", number = 5, verbose = T) 
train.lm <- train(Hammer_level~.,train, method = "lm", trControl =cv)  
train.lm  #RMSE : 0.3472277  Rsquared : 0.5248626


###2.나이브 베이즈###
#데이터 train, test 분리 7:3
intrain = createDataPartition(y=data$Hammer_level, p=0.7, list=FALSE) 
train_nb = data[intrain, ]
test_nb = data[-intrain, ]


#나이브베이즈 사용!
nb_model = naiveBayes(Hammer_level~.,data = train_nb)

#예측
str(test_nb) #Hammer_level index파악
nbpred <- predict(nb_model, test_nb[,-29], type='class')
confusionMatrix(nbpred, test_nb$Hammer_level)  #정확도 : 0.9344

#laplace smoothing
lapl <- naiveBayes(Hammer_level~.,data = train_nb, laplace = 3)
laplpred <- predict(lapl, test_nb[,-29], type='class')
confusionMatrix(laplpred, test_nb$Hammer_level) #정확도 : 0.9396

#5-fold cross validation
cv <- trainControl(method = "cv", number = 5, verbose = T) 
train.nb <- train(Hammer_level~.,train, method = "nb", trControl =cv)  
summary(train.nb) 



###3. SVM###
library(e1071)
#데이터 train, test 분리 7:3
intrain = createDataPartition(y=data$Hammer_level, p=0.7, list=FALSE) 
train_svm = data[intrain, ]
test_svm = data[-intrain, ]


#SVM training
svm_data = svm(Hammer_level~., data = train_svm) #data개수에 비해 피쳐가 적으므로 커널없는 svm적용
summary(svm_data)

#grid search
gs = tune(svm, Hammer_level~., data = data, ranges = list(gamma = 2^(-5:1), cost = 2^(0:4)))  #gamma와 c를 조절해가며 grid search
summary(gs)
plot(gs)

#예측
pred_tuned = predict(gs$best.model, test_svm)
confusionMatrix(pred_tuned,test_svm$Hammer_level)  #Accuracy : 0.9914



###4.KNN###
## min-max 스케일링
normalize <- function(x){
  return( (x-mean(x))/sd(x))
}
#날짜 데이터는 스케일링이 불가능 => 시간의 전후가 중요한 데이터이므로 as.numeric 후 scaling
dataforknn = subset(data, select = -c(First_auction_date, Final_auction_date)) #auction_date라는 변수로 앞서 적용했으므로 제거 
dataforknn$Appraisal_date = as.numeric(data$Appraisal_date) 
dataforknn$Preserve_regist_date = as.numeric(data$Preserve_regist_date)
dataforknn$Close_date = as.numeric(data$Close_date)

data_normal = as.data.frame(lapply(dataforknn[-c(22:27)], normalize))  #더미변수를 제외한 연속형 변수 scaling
dataforknn = cbind(data_normal, dataforknn[c(22:27)])  #scaling한 데이터 + 기존 더미변수 데이터

#데이터 train, test 분리 7:3
intrain = createDataPartition(y=dataforknn$Hammer_level, p=0.7, list=FALSE) 
train_knn = dataforknn[intrain, ]
test_knn = dataforknn[-intrain, ]

#최적의 k 찾기
library(kknn)
cv = train.kknn(Hammer_level ~., train_knn, 
                 ks = seq(1, 50, by=2), scale = F)
best_k = cv$best.parameters$k; best_k #최적의 k: 37

pred_cv = kknn(Hammer_level ~., train_knn, test = test_knn, k = best_k, scale = F)
pred_cv = pred_cv$fitted.values

confusionMatrix(pred_cv, test_knn$Hammer_level) #Accuracy: 0.905

#grid search cv
cv = trainControl(method = "cv", number = 5, verbose = T)

knn.grid = expand.grid(
  .k = c(31,33,35,37,39)
)

train.knn = train(Hammer_level ~., train_knn, method = "knn",trControl = cv,
                   tuneGrid = knn.grid)
train.knn$results
train.knn$bestTune #31 #최적의 k는 31
predict.knn <- predict(train.knn, test_knn)
confusionMatrix(predict.knn, test_knn$Hammer_level) #Accuracy: 0.8929 

library(class)
dataforknn_acc<-numeric()
for(i in 1:50){
  predict<-knn(train_knn[,-27],test_knn[,-27],
               train_knn$Hammer_level,k=i)
  dataforknn_acc<-c(dataforknn_acc,
              mean(predict==test_knn$Hammer_level))
}
#Plot k= 1 through 50
plot(1-dataforknn_acc,type="l",ylab="Error Rate",
     xlab="K",main="Error Rate for Data With Varying K")


###4가지 모델의 정확도를 비교했을 때 svm기법으로 분류했을 때 정확도가 99.14%로 가장 높음!