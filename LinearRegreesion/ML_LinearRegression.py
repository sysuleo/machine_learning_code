# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:26:21 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class LinearRegression:
    def __inti__(self):
        self.coef=None
        self.intercept=None
        self._theta=None
        
    def fit_normal(self,X_train,y_train):
        """根据训练集训练linear regression模型"""
        assert X_train.shape[0]==y_train.shpae[0]
        "the size of X_train must be equal to the size of Y_train"
            
        X_b=np.hstack([np.ones((len(X_train),1)),X_train])
        self._theta=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.coef=None
        self.intercept=self._theta[0]
        self._theta=self._theta[1:]
    
        
    def fit_gd(self,X_train,y_train,eta=0.01,n_iters=1e4):
        """用梯度下降法，根据训练集训练linear regression模型"""
        assert X_train.shape[0]==y_train.shape[0]
        "the size of X_train must be equal to the size of Y_train"
        def J(theta,X_b,y):
            try:
                return np.sum(y-X_b.dot(theta)**2)/len(X_b)
            except:
                return float('inf')
    
        def dj(theta,X_b,y):
            #res=np.empty(len(theta))
            #res[0]=np.sum(X_b.dot(theta)-y)
            #for i in range (1,len(theta)):
            #    res[i]=np.sum((X_b.dot(theta)-y).dot(X_b[:,i])) #取第i列
            # return res*2/len(X_b)
            return X_b.T.dot(X_b.dot(theta)-y)*2/len(y)
        
        def gradient_descent(X_b,y,initial_theta,eta,n_iters=1e4,epsilon=1e-8):
            theta=initial_theta
            i_iter=0

            while i_iter<n_iters:   #防止进入死循环
                gradient=dj(theta,X_b,y)
                last_theta=theta
                theta=theta-eta*gradient
    
                if(abs(J(theta,X_b,y)-J(last_theta,X_b,y))<epsilon):
                    break
                i_iter+=1
            return theta
        X_b=np.hstack([np.ones((len(X_train),1)),X_train])  
        #————————————————————————————注意注意ones((a,b))两层括号！！！！！！！！！！！！！！！！！！
        initial_theta=np.zeros(X_b.shape[1]) #shape【1】取列数
        self._theta=gradient_descent(X_b,y_train,initial_theta,eta,n_iters)
        self.intercept=self._theta[0]
        self.coef=self._theta[1:]
        
        return self
    
    def fit_sgd(self,X_train,y_train,n_iters=5): 
        """用随机梯度下降法，根据训练集训练linear regression模型"""
        assert X_train.shape[0]==y_train.shape[0]
        "the size of X_train must be equal to the size of Y_train"
        
        def dj_sgd(theta,X_b_i,y_i):
            return X_b_i.T.dot(X_b_i.dot(theta)-y_i)*2.
    
        def sgd(X_b,y,initial_theta,n_iters,t0=5,t1=50):
          
            def learning_rate(t):
                return t0/(t+t1)
            
            theta=initial_theta
            m=len(X_b)
            
             #for cur_iter in range (n_iters):
                 #rand_i=np.random.randint(len(X_b))
                 #gradient=dj_sgd(theta,X_b[rand_i],y[rand_i])
                 #theta=theta-learning_rate(cur_iter)*gradient
            #return theta
            
            
            for cur_iter in range (n_iters):
                #优化保证每个数据都被便利n_iters遍,上面是随机的，不能保证
                indexes=np.random.permutation(m)
                X_b_new=X_b[indexes]
                y_new=y[indexes]
                for i in range(m):
                    gradient=dj_sgd(theta,X_b_new[i],y_new[i])
                    theta=theta-learning_rate(cur_iter*m+i)*gradient
            return theta
        
        X_b=np.hstack([np.ones((len(X_train),1)),X_train])  
        initial_theta=np.random.randn(X_b.shape[1]) #shape【1】取列数
        self._theta=sgd(X_b,y_train,initial_theta,n_iters)
        self.intercept=self._theta[0]
        self.coef=self._theta[1:]
    
    
    def predict(self,X_predict):
         
        assert self.intercept is not None and self.coef is not None,\
        "must fit before predict"
        assert X_predict.shape[1]==len(self.coef),\
        "the feacture number of X_predict must be equal X_train"
        X_b=np.hstack([np.ones((len(X_predict),1)),X_predict])
        return X_b.dot(self._theta)
    
    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)