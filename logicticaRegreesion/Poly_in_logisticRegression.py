# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:15:58 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
X=np.random.normal(0,1,size=(200,2))
y=np.array(X[:,0]**2+X[:,1]**2<1.5,dtype='int')

plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()

#_---------------------使用逻辑回归------------------
from playML_LogisticRegression import LogisticRegression

log_reg=LogisticRegression()
log_reg.fit(X,y)
print(log_reg.score(X,y))


def plot_descision_boundary(model,axis):
    x0,x1=np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
        #x轴右边界减去左边界
        np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1)
    )
    X_new=np.c_[x0.ravel(),x1.ravel()]
    y_predict=model.predict(X_new)
    zz=y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_map=ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])  
    plt.contourf(x0,x1,zz,linewidth=5,camp=custom_map)
    
plot_descision_boundary(log_reg,axis=[-4,4,-4,4])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('std_scaler',StandardScaler()),
        ('log_reg',LogisticRegression())
    ])
    
poly_log_reg=PolynomialLogisticRegression(8)
poly_log_reg.fit(X,y)
print(poly_log_reg.score(X,y))

plot_descision_boundary(poly_log_reg,axis=[-4,4,-4,4])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()


