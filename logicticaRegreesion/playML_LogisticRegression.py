# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:20:06 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __inti__(self):
        self.coef_=None
        self.intercept_=None
        self._theta=None
        
    def _sigmoid(self,t):
        return 1/(1+np.exp(-t))

    def fit(self,X_train,y,eta=0.01,n_iters=1e4):
        #根据训练集X_trian和Y_tian，使用梯度下降法训练
        assert X_train.shape[0]==y.shape[0]
        "the size of X_train must be equal to the size of Y_train"
        
        def J(theta,X_b,y):
            y_hat=self._sigmoid(X_b.dot(theta))
            try:
                return np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))/len(y)
            except:
                return float('inf') #正无穷
        
        def dj(theta,X_b,y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta))-y)/len(X_b)
        def gradient_descent(X_b,initial_theta,eta,n_iters=1e4,epsilon=1e-8):
            theta=initial_theta
            cur_iter=0
            while cur_iter<n_iters:
                gradient=dj(theta,X_b,y)
                last_theta=theta
                theta=theta-eta*gradient
                if(abs(J(theta,X_b,y)-J(last_theta,X_b,y))<epsilon):
                    break #跳出循环 
                cur_iter+=1
            return theta
            
        X_b=np.hstack([np.ones((len(X_train),1)),X_train])
        initial_theta=np.zeros(X_b.shape[1])
        #生成全零数组，shape【1】表示列，shape【0】表示行
        self._theta=gradient_descent(X_b,initial_theta,eta,n_iters)
        self.intercept_=self._theta[0] #截距
        self.coef_=self._theta[1:] #权重 
        
    def predict_proba(self,X_predict):
        #给定待预测数据集X_predict，返回表示X_predict的结果概率向量
        assert self.intercept_ is not None and self.coef_ is not None,\
        "must fit before predict"
        assert X_predict.shape[1]==len(self.coef_),\
        "the feacture number of X_predict must be equal X_train"
        X_b=np.hstack([np.ones((len(X_predict),1)),X_predict]) 
        #ones创建len*1 值为一的矩阵   hhastck为水平合成矩阵
        return self._sigmoid(X_b.dot(self._theta)) 
        #a.dot(b) 与 np.dot(a,b)效果相同
        
        
    def predict(self,X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        
        assert self.intercept_ is not None and self.coef_ is not None,\
        "must fit before predict"
        assert X_predict.shape[1]==len(self.coef_),\
        "the feacture number of X_predict must be equal X_train"
        proba=self.predict_proba(X_predict)
        return np.array(proba>=0.5,dtype='int')
    
    
    def score(self,X_test,y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        
        y_predict=self.predict(X_test)
        return accuracy_score(y_test,y_predict)
    
    def __repr__(self):
        return "LogisticRegression"

#x=np.linspace(-10,10,500)   #均分计算指令，区间为【-10，10】取500个数
#同样还有logspace（a，b，N） 生成10^a到10^b内n个数
#y=sigmoid(x)
#plt.plot(x,y)
#plt.show()
    


