# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:28:38 2018

@author: linh
"""

import numpy as np

class PCA:
    def __init__(self,n_components):
        assert n_components >=1, "n_components must be valid"
        self.n_components=n_components #用户传来的主成分
        self.components=None #计算出来的主成分值
    
    def fit(self,X,eta=0.01,n_iters=1e4):
        #获得数据集前n个主成分
        assert self.n_components <=X.shape[1], \
        "n_components must not be greater than the feacture number of X"
        
        def demean(X):  #相当于平移了坐标，是均值在远点
            return X-np.mean(X,axis=0) 
        #axis=0，那么输出矩阵是1行，求每一列的平均，即每个特征的平均值     1*n的向量

        def f(w,X):
            return np.sum((X.dot(w)**2))/len(X)

        def df(w,X):
            return X.T.dot(X.dot(w))*2./len(X)

        def df_debug(w,X,epsilon=0.0001):
            res=np.empty(len(w))
            for i in range (len(w)):
                w_1=w.copy()
                w_1[i]+=epsilon
                w_2=w.copy()
                w_2[i]-=epsilon
                res[i]=(f(w_1,X)-f(w_2,X))/(2*epsilon)
            return res

        def direction(w):
       #使其为单位向量
            return w/np.linalg.norm(w) #linalg.norm求矩阵的模

        def first_component(X,initial_w,eta,n_iters=1e4,epsilon=1e-8):
    
            w=direction(initial_w)
            i_iter=0

            while i_iter<n_iters:   #防止进入死循环
                gradient=df(w,X)
                last_w=w
                w=w+eta*gradient
                w=direction(w)
                if(abs((f(w,X)-f(last_w,X)))<epsilon):
                    break
                i_iter+=1
            return w
        X_pca=demean(X)
        self.components=np.empty(shape=(self.n_components,X.shape[1]))
        for i in range (self.n_components):
            initial_w=np.random.random(X_pca.shape[1])
            w=first_component(X_pca,initial_w,eta)
            self.components[i,:]=w
            
            X_pca=X_pca-X_pca.dot(w).reshape(-1,1)*w
            
        return self
    
    def transform(self,X):
        assert X.shape[1]==self.components.shape[1]
        return X.dot(self.components.T)
    
    def inverse_transform(self,X):
        assert X.shape[1]==self.components.shape[0]
        return X.dot(self.components)
        
    def __repr__(self):
        return "PCA(n_components=%d)" %self.n_components
    