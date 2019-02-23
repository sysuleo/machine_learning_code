# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:05:28 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(666) #每次随机生成的数一样
#随机数种子对后面的结果一直有影响。
#同时，加了随机数种子以后，后面的随机数组都是按一定的顺序生成的
m=10000

x=np.random.normal(size=m)
y=x*4.+3.+np.random.normal(size=m) #正态分布
X=x.reshape(-1,1)#不知道z的shape属性是多少，但是想让x变成只有1列，行数不知道多少

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
    
X_b=np.hstack([np.ones((len(X),1)),X])  
#————————————————————————————注意注意ones((a,b))两层括号！！！！！！！！！！！！！！！！！！
initial_theta=np.zeros(X_b.shape[1])
eta=0.01
theta=gradient_descent(X_b,y,initial_theta,eta)
print(theta)

#-------------------------一下用随机梯度下降--------------

def dj_sgd(theta,X_b_i,y_i):
    return X_b_i.T.dot(X_b_i.dot(theta)-y_i)*2.
    
def sgd(X_b,y,initial_theta,n_iters):
    t0=5
    t1=50
        
    def learning_rate(t):
        return t0/(t+t1)
    theta=initial_theta
    for cur_iter in range (n_iters):
        rand_i=np.random.randint(len(X_b))
        gradient=dj_sgd(theta,X_b[rand_i],y[rand_i])
        theta=theta-learning_rate(cur_iter)*gradient
    return theta

X_b=np.hstack([np.ones((len(X),1)),X]) 
initial_theta=np.zeros(X_b.shape[1])
theta=sgd(X_b,y,initial_theta,n_iters=len(X_b)//3)
print(theta)
        