rm(list=ls())

# install.packages("cluster")
library(cluster)
# install.packages("NbClust")
library(NbClust)
# install.packages("flexclust")
library(flexclust)
# install.packages("fpc")
library(fpc)
# 거리계산함수 dist
x <- matrix(rnorm(100),nrow=5)
x
dist(x) # 대칭이기 때문에 대칭값은 안나오고 자기자신과의 거리는 0이므로 안나옴
        # 유클리드 거리가 default 값이다.

# method를 통해 거리유형 선택가능(default=euclidean)
dist(x,method ="manhattan") 
dist(x,method ="maximum")

#################################################################################################################
# 1. 데이터 전처리
set.seed(802) # k-means 군집 분석을 할 때 무작위로 K개의 행을 선택하므로 실행할 때마다 결과가 달라지기 때문
data <- iris
str(data)
iris
data <- iris
data1 <- data[-5] # label 제거, clustering에서는 라벨을 제거해야 한다
str(data1)

# 2. k means 함수 적용
library(cluster)

# 3의 의미는 cluster를 3개로 해줄거야 이말이다.
kmeans1 <- kmeans(data1, 3, nstart = 10) 
# nstart : 여러가지 초기 구성을 시도하고 가장 좋은 것을 선택
# 예를 들어, nstart = 10인 경우 10 개의 초기 무작위 중심을 생성하고 알고리즘에 가장 적합한 것을 선택한다

# 할당된 군집 번호
kmeans1$cluster
# J의 값을 말함, 각 군집에서
kmeans1$withinss
# 3개 합친 값, 전체의 J값을 말함
kmeans1$tot.withinss

#  cluster : 각 개체별 할당된 군집 번호, 1부터 k번까지 군집 숫자 
#  centers : 군집의 중심 (centroid) 좌표 행렬
#  totss : 제곱합의 총합
#  withinss : 군집 내 군집과 개체간 거리의 제곱합 벡터.
#  tot.withinss : 군집 내 군집과 개체간 거리의 제곱합의 총합, 즉, sum(withinss)
#  betweenss : 군집과 군집 간 중심의 거리 제곱합
#  size : 각 군집의 개체의 개수
#  iter : 반복 회수

result_table <- table(kmeans1$cluster, data$Species);result_table  #과제 평가 지표 
sum(50, 48, 36) / sum(result_table) * 100

# Adjusted Rand Index는 Type과 cluster의 일치 정도를 정량화하는데 사용할 수 있다. 
# -1은 전혀 일치하지 않는 것이고 1은 완벽하게 일치하는 것
library(flexclust)
# 일치정도를 나타냄
randIndex(result_table) 
?randIndex

# 시각화 & 실루엣을 통해 군집화가 얼마나 잘되었는지 확인
windows()
plot(data1[,1:2], col = kmeans1$cluster) 

plot(silhouette(kmeans1$cluster, dist = dist(data1), col =1:3))

### 변수마다 스케일의 차이로 인해 생긴 문제일 수 있으니, scale함수를 통해 단위를 맞춰주자.
scaled_data <- scale(data1)
kmeans_scaled <- kmeans(scaled_data, 3, nstart=10)
result_table <- table(kmeans_scaled$cluster, data$Species)
result_table
sum(50, 39, 36) / sum(result_table) * 100

randIndex(result_table) 
plot(scaled_data[,1:2], col = kmeans_scaled$cluster) 
plot(silhouette(kmeans_scaled $cluster, dist = dist(scaled_data), col =1:3))
# 표준화를 해줬는데 더 안좋아짐 -> 표준화를 할 때는 변수마다 단위가 다를 때 해야 효과가 있다
### 최적의 군집 수 k 결정

# NbClust 이용, Best.nc가 최적의 클러스터 숫자를 찾아준다,
# barplot이 높은것이 좋은것이다.
library(NbClust)
nc <- NbClust(data1,min.nc=2,max.nc=15,method="kmeans")
nc$Best.nc
table(nc$Best.n[1,])
barplot(table(nc$Best.n[1,]),xlab="Number of Clusters",ylab="Number of Criteria") # 2, 3일 때 높다

# elbow point 기법 이용 k 값 정하기
# 군집의 개수 k = 2부터 k = 6인 경우까지 확인
visual <- NULL
for ( i in 2:6 ){
  result <- kmeans(data1, i, 10)
  visual[i] <- result$tot.withinss #  tot.withinss : 군집 내 군집과 개체간 거리의 제곱합의 총합 
}

# visual에는 sse값이 들어간다, 오차와 실제와의 차이
plot(visual, type="l", xlab="k") 
abline(v=3, col="red")

# 위를 함수로 만들면
wssplot=function(data,nc=15,plot=TRUE){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for( i in 2:nc){
    set.seed(802) 
    wss[i]<- sum(kmeans(data,centers=i,10)$withinss)
  }
  if(plot) plot(1:nc,wss,type="b",xlab="Number pf Clusters",ylab="Within group sum of squares")
  wss
}
wssplot(data1)

# 3. H-Clust 함수 적용
data_hclust <- data1
data_hclust
distance <- round(dist(data_hclust), 2) # 소숫점 둘째 자리까지만

# 덴드로그램 그려보기
plot(h <- hclust(distance, method="single")) # method="single" 최단연결법
plot(h <- hclust(distance, method="complete")) # method="complete" 최장연결법
plot(h <- hclust(distance, method="average")) # method="average" 평균연결법
plot(h <- hclust(distance, method="centroid")) # method="centroid" 중심연결법

# 중심연결법을 이용한 Hclust 예시
data_x <- data_hclust
distance <- round(dist(data_x), 2)
hc <- hclust(distance, method = "centroid")
windows()
plot(hc)
# 박스를 그려주어서 3개로 구분해준다
rect.hclust(hc, k = 3, border = "red")
# 박스를 그려주어서 6개로 구분해준다
rect.hclust(hc, k = 6, border = "red")
result <- cutree(hc, k = 6) # 군집 k개로
result
table(result)

# 각 군집 별 데이터 확인
data_x[result == 2, ]
data_x[result == 4, ]
data_x[result == 6, ]

# 2,4,6 번째 군집의 수가 너무 적어서 제거해보자!
remove <- which(result %in% c(2,4,6))
data_r <- data[-remove,]
data_x <- data_x[-remove,]
distance <- round(dist(data_x), 2)

hc <- hclust(distance, method = "centroid")
plot(hc)
rect.hclust(hc, k = 3, border = "red")
result <- cutree(hc, k = 3)
result
table(result)

# 각 군집 별 centroid 계산
center <- NULL
for ( i in 1:3 ) center <- rbind(center, colMeans(data_x[result == i, ]))
center

# 확인
result_table <- table(data_r$Species, result)
sum(diag(result_table)) / sum(result_table)

# 4. DBSCAN 적용
# 위의 scaled_data 이용
library(fpc)
# eps = 원의 반지름, minpts = 한 원에 몇개의 자료가 있으면 군집으로 볼 것인가.
db <- dbscan(scaled_data,eps=0.5, MinPts = 3) # eps = 밀도측정 반지름 , MinPts = 반지름 epsilon내에 있는 최소 데이터 갯수
plot(scaled_data[,1:2], col = db$cluster)

db <- dbscan(scaled_data,eps=1, MinPts = 3) 
plot(scaled_data[,1:2], col = db$cluster)

db <- dbscan(scaled_data,eps=2, MinPts = 3) 
plot(scaled_data[,1:2], col = db$cluster)

