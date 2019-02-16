 
k_means = function(data, centers, iter.max = 50){
  dist_fun = function(x, y)   #유사도를 판단하는 거리 #맨해튼 거리 등으로 변경 가능 
  {
    return(colSums((x - y)^2))
  }
  
  data_mat = as.matrix(data)
  n = nrow(data_mat)  #초기 데이터의 행과 열 개수 
  p = ncol(data_mat)
  
  center_mat = matrix(nrow = centers, ncol = p)  #빈 행렬 생성
  cluster = sample(1:centers, n, replace = TRUE)   #군집의 초기 중심 좌표 랜덤 생성
  
  for (i in 1:centers) {   
    center_mat[i, ] = colMeans(data_mat[cluster == i, , drop = FALSE])  #군집에 해당하는 데이터의 평균을 구해 새로운 중심 지정
  }
  
  dist = matrix(nrow = n, ncol = centers)  #빈 행렬 생성 
  
  iter = 0
  
  center_mat_old = center_mat   #초기 중심좌표 행렬 
  
  while (((iter = iter + 1) < iter.max)) {   #input으로 지정해 놓은 최대 루프 횟수 전까지
    
     
    for (i in 1:centers) {   
      dist[, i] = dist_fun(t(data_mat), center_mat[i,])    #군집과 중심 간의 거리(유사도)구하기
    }
    
    cluster <- apply(dist, 1, which.min)  #거리가 최소일 경우의 중심 찾기
    
    
    for(i in 1:centers){
      center_mat[i, ] = colMeans(data_mat[cluster == i, , drop = FALSE])  #구한 중심들로 중심좌표행렬 업데이트 
    } 
    
    if (sqrt(sum((center_mat_old - center_mat)^2)) < 1e-10) {    #중심좌표행렬이 계속 비슷하게 다다를 경우 멈추기 
      break  
    } else {
      center_mat_old = center_mat   #그렇지 않을 경우 새로 다시 반복
    }
  }
  result = cluster  #최종으로 구해진 군집들 반환 
  return(result)
}

#iris데이터로 성능 검증
kresult = k_means(iris[-5],3)
result_table = table(kresult, iris$Species)
result_table  #과제 평가 지표 
sum(50, 48, 36) / sum(result_table) * 100  #Accuracy:89.3333

#kaggle 문제에 변수로 적용!
df = read.csv("C:/Users/Kim Yuum/Desktop/투빅스/3주차/forkmeans.csv")
head(df)
df["cluster"] = k_means(df, 5)
write.csv(df, file = "C:/Users/Kim Yuum/Desktop/투빅스/3주차/newdf.csv") 
