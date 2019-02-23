# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:19:54 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(666) #每次随机生成的数一样
#随机数种子对后面的结果一直有影响。
#同时，加了随机数种子以后，后面的随机数组都是按一定的顺序生成的
x=2*np.random.random(size=100)
y=x*3.+4.+np.random.normal(size=100) #正态分布
X=x.reshape(-1,1)#不知道z的shape属性是多少，但是想让x变成只有1列，行数不知道多少
print(X.shape)

plt.scatter(x,y) #散点图绘制
plt.show()

def J(theta,X_b,y):
    try:
        return np.sum(y-X_b.dot(theta)**2)/len(X_b)
    except:
        return float('inf')
    
def dj(theta,X_b,y):
    res=np.empty(len(theta))
    res[0]=np.sum(X_b.dot(theta)-y)
    for i in range (1,len(theta)):
        res[i]=np.sum((X_b.dot(theta)-y).dot(X_b[:,i])) #取第i列
    return res*2/len(X_b)

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

X_b=np.hstack([np.ones((len(X),1)),X.reshape(-1,1)])  
#————————————————————————————注意注意ones((a,b))两层括号！！！！！！！！！！！！！！！！！！
initial_theta=np.zeros(X_b.shape[1]) #shape【1】取列数
eta=0.01

from ML_LinearRegression import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit_gd(X,y)
print(lin_reg.coef)
print(lin_reg.intercept_) 


