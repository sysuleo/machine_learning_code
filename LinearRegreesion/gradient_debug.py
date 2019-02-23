# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:49:46 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
X=np.random.random(size=(1000,10))#1000个维度为十的随机数

true_theta=np.arange(1,12,dtype=float) 
#从1到11依次生成11个数，x有十个特征，但还有个截距
X_b=np.hstack([np.ones((len(X),1)),X])
y=X_b.dot(true_theta)+np.random.normal(size=1000)
print(true_theta)

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

def dj_debug(theta,X_b,y,epsilon=0.01):
    res=np.empty(len(theta))
    for i in range (len(theta)):
        theta_1=theta.copy()
        theta_1[i]+=epsilon
        theta_2=theta.copy()
        theta_2[i]+=epsilon
        res[i]=(J(theta_1,X_b,y)-J(theta_2,X_b,y))/(2*epsilon)
    return res

def gradient_descent(dJ,X_b,y,initial_theta,eta,n_iters=1e4,epsilon=1e-8):
    theta=initial_theta
    i_iter=0

    while i_iter<n_iters:   #防止进入死循环
        gradient=dJ(theta,X_b,y)
        last_theta=theta
        theta=theta-eta*gradient
    
        if(abs(J(theta,X_b,y)-J(last_theta,X_b,y))<epsilon):
            break
        i_iter+=1
    return theta
    
X_b=np.hstack([np.ones((len(X),1)),X])  
        #————————————————————————————注意注意ones((a,b))两层括号！！！！！！！！！！！！！！！！！！
initial_theta=np.zeros(X_b.shape[1]) #shape【1】取列数
eta=0.01
theta=gradient_descent(dj_debug,X_b,y,initial_theta,eta)
print(theta)