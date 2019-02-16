
#install.packages("e1071")
library(e1071) #나이브베이즈 뿐만 아니라 svm도 제공합니다:)는

#데이터 불러오기
df<-read.csv('http://www-bcf.usc.edu/~gareth/ISL/Heart.csv')

#또는
setwd("")
df <- read.csv('Heart.csv')

#데이터 확인
str(df)


#Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터입니다. 
#각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 
#각 환자들이 심장병이 있는지 여부가 기록되어 있습니다. 

library(caret)
set.seed(1234) 
intrain<-createDataPartition(y=df$AHD, p=0.7, list=FALSE) 
train<-df[intrain, ]
test<-df[-intrain, ]

#나이브베이즈 사용!
nb_model <- naiveBayes(AHD~.,data = train)
nb_model

#예측
nbpred <- predict(nb_model, test[,-15], type='class')
confusionMatrix(nbpred, test$AHD)
#Accuracy : 0.8444

####범주형일때
data(HouseVotes84, package = "mlbench")
model <- naiveBayes(Class ~ ., data = HouseVotes84)
model

#laplace
model <- naiveBayes(Class ~ ., data = HouseVotes84, laplace = 3)
pred <- predict(model, HouseVotes84[,-1])
confusionMatrix(pred, HouseVotes84$Class)
# Accuracy : 0.9034 