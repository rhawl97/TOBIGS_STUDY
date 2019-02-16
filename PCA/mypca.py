import numpy as np
import numpy.linalg as lin
from sklearn.preprocessing import StandardScaler


class mypca(object):
    '''
    k : component 수 주성분 개수
    n : 원래 차원 열
    components : 고유벡터 저장소 shape (k,n)
    explain_values : 고유값 shape (k,)
    '''
    
    k = None
    components = None
    explain_values= None
    
    def __init__(self, k=None, X_train=None):
        '''
        k의 값이 initial에 없으면 None으로 유지
        '''
        if k is not None :
            self.k = k       
        if X_train is not None:
            self.fit(X_train)
            
    def fit(self,X_train=None):
        if X_train is None:
            print('Input is nothing!')
            return
        if self.k is None:
            self.k = min(X_train.shape[0],X_train.shape[1])
            
        k = self.k
        components = [0]*k
        import pandas as pd
        X_train = pd.DataFrame(X_train)
        cov_mat = X_train.cov()
        for i in range(k):
            components[i] = lin.eig(cov_mat)[1][:,i]
        explain_values = lin.eig(cov_mat)[0]
        self.components = np.asarray(components)
        
        #############################################
        # TO DO                                     #
        # 인풋 데이터의 공분산행렬을 이용해         #
        # components와 explain_values 완성          # 
        #############################################
        
        
        
        #############################################
        # END CODE                                  #
        #############################################
        
        return
    
    def transform(self,X=None):
        if X is None:
            print('Input is nothing!')
            return
        
        result = None
        '''
        N : X의 행 수
        result의 shape : (N, k)
        '''
        k = self.k
        components = self.components
        result_t = [0]*k

        for i in range(k):
            result_t[i] = components[i].dot(X.T)
        result_t = np.array(result_t)
        result = result_t.T
        #############################################
        # TO DO                                     #
        # components를 이용해 변환결과인            #
        # result 계산                               #
        #############################################
        
        
        
        #############################################
        # END CODE                                  #
        #############################################       
        return result
    
    def fit_transform(self,X=None):
        if X is None:
            print('Input is nothing!')
            return
        self.fit(X)
        return self.transform(X)