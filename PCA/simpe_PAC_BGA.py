# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:28:57 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt

X=np.empty((100,2))
X[:,0]=np.random.uniform(0.,100.,size=100) #第一维
X[:,1]=0.75*X[:,0]+np.random.normal(0.,10.,size=100)

plt.scatter(X[:,0],X[:,1])
plt.show()

def demean(X):  #相当于平移了坐标，是均值在远点
    return X-np.mean(X,axis=0) 
    #axis=0，那么输出矩阵是1行，求每一列的平均，即每个特征的平均值     1*n的向量
X_demean=demean(X)
plt.scatter(X_demean[:,0],X_demean[:,1])
plt.show()

def f(w,X):
    return np.sum((X.dot(w)**2))/len(X)

def df_math(w,X):
    return X.T.dot(X.dot(w))*2./len(X)

def df_debug(w,X,epsilon=0.0001):
    res=np.empty(len(w))
    for i in range (len(w)):
        w_1=w.copy()
        w_1[i]+=epsilon
        w_2=w.copy()
        w_2[i]-=epsilon
        res[i]=((f(w_1,X)-f(w_2,X)))/(2*epsilon)
    return res

def direction(w):
    #使其为单位向量
    return w/np.linalg.norm(w) #linalg.norm求矩阵的模

def gradient_asscent(df,X,initial_w,eta,n_iters=1e4,epsilon=1e-8):
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

initial_w=np.random.random(X.shape[1]) #注意： 不能从零向量开始
print(w)
w=gradient_asscent(df_math,X_demean,initial_w,eta=0.001)
print(w)
e=gradient_asscent(df_debug,X_demean,initial_w,eta=0.001)
print(e)

plt.scatter(X_demean[:,0],X_demean[:,1])
plt.plot([0,w[0]*100],[0,w[1]*100],color='r')