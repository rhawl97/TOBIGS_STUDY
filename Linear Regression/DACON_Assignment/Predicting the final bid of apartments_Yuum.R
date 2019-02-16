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
                             addr_si,addr_dong,addr_san,addr_bunji1,addr_etc,Specific))  

#날짜 데이터를 시계 데이터로 변환
df2$First_auction_date = as.Date(df2$First_auction_date)   
df2$Final_auction_date = as.Date(df2$Final_auction_date)
df2$Preserve_regist_date = as.Date(df2$Preserve_regist_date)
df2$Appraisal_date = as.Date(df2$Appraisal_date)
df2$Close_date = as.Date(df2$Close_date)


#파생변수 만들기
df2$auction_date = df2$Final_auction_date - df2$First_auction_date  #총 경매일 = 최종경매일 - 최초경매일 

lalon = cbind(df2$point.x,df2$point.y)    ##위도 경도 데이터 kmeans clustering -> 비슷한 위치끼리 군집화
Data_scaled <- scale(lalon, center = TRUE, scale = TRUE)  #먼저 위도 경도 데이터 스케일링
km = kmeans(Data_scaled,6)    #스케일링한 위경도 데이터 군집화
df2$point_cluster = km$cluster   #군집화된 결과를 파생변수로 추가

df2$area = df2$Total_land_auction_area / df2$Total_land_gross_area #경매된 토지 면적 비율 #df2$Total_land_gross_area이 0일 때 결측치나 Inf 값 도출
df2$area[which(df2$Total_land_gross_area==0)] = 0  #area가 NA나 Inf 값이 나올 경우 모두 0 처리 


#레이블과 변수의 분포 시각화
##낙찰가와 유의한 연관이 있을 것이라고 예상되는 변수와 레이블과의 plot살펴보기
plot(df2$Total_land_gross_area, df2$Hammer_price)  #총토지전체면적이 작을 때 낙찰가도 낮음 #이상치가 2개 있음 
plot(df2$Total_appraisal_price, df2$Hammer_price)  #총 감정가가 높을수록 낙찰가도 높음  
plot(df2$Minimum_sales_price, df2$Hammer_price)  #최저 매각 가격이 높을수록 낙찰가도 높음  


#데이터 train, test 분리 7:3
install.packages("caret")   
library(caret)
idx <- createDataPartition(y = df2$Hammer_price, p = 0.7, list =FALSE)  
train<- df2[idx,]
test <- df2[-idx,]


#전진선택법, 후진제거법, 단계적회귀
#설명변수를 넣지않은 모델
fit.con <- lm(Hammer_price ~ 1, train) 
#다 적합한 모델
fit.full <- lm(Hammer_price~., train)

# 전진선택법
fit.forward <- step(fit.con, list(lower=fit.con, upper=fit.full), direction = "forward")
# 후진제거법
fit.backward <- step(fit.full, list(lower=fit.con, upper = fit.full), direction = "backward")
# 단계적회귀방법(stepwise)
fit.both <- step(fit.con, list(lower=fit.con, upper=fit.full), direction = "both")

summary(fit.forward) # Multiple R-squared:  0.9915,	Adjusted R-squared:  0.9914   #모두 설명력이 매우 좋음!
summary(fit.backward) # Multiple R-squared:  0.9916,	Adjusted R-squared:  0.9915  #후진제거법이 가장 높은 설명력을 가짐
summary(fit.both) # Multiple R-squared:  0.9915,	Adjusted R-squared:  0.9914 

#이상치 확인
library(car)
#outlierTest(model, ...)
outlierTest(fit.backward, order=TRUE)

#################################test에 대입
predicted <- predict(fit.backward, test[ ,-27]) #설명력이 가장 높은 fit.backward 사용
y <- test[,27]
RMSE(predicted,y) #47675414원의 RMSE를 가짐!


#r^2값
1 - sum((y-predicted)^2)/sum((y-mean(y))^2) #0.9872496  #높은 설명력 


######### cross validation
library(caret)
cv <- trainControl(method = "cv", number = 5, verbose = T)  #5개의 폴드 

######### lm
train.lm <- train(Hammer_price~.,train, method = "lm", trControl =cv)


