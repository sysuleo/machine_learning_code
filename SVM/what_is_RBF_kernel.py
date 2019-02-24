# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:09:39 2018

@author: liuw
"""
import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-4,5,1)

y=np.array((x>=-2)&(x<=2),dtype='int')
print(y)

plt.scatter(x[y==0],[0]*len(x[y==0]))
#x轴取y=0的点，y轴都取0长度为len（X【y==0】）
plt.scatter(x[y==1],[0]*len(x[y==1]))
plt.show()

#高斯核函数升维度
def gaussian(x,l):
    gamma=1.0
    return np.exp(-gamma*(x-l)**2)

l1,l2=-1,1
X_new=np.empty((len(x),2))
for i ,data in enumerate(x):
    X_new[i,0]=gaussian(data,l1)
    X_new[i,1]=gaussian(data,l2)
    
plt.scatter(X_new[y==0,0],X_new[y==0,1])
#x轴取y=0的点，y轴都取0长度为len（X【y==0】）
plt.scatter(X_new[y==1,0],X_new[y==1,1])
plt.show()