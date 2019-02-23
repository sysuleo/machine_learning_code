# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:32:01 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris =datasets.load_iris()

x=iris.data
y=iris.target

x=x[y<2,:2]
y=y[y<2]
print(x.shape )

#--------------------------绘制
plt.scatter(x[y==0,0],x[y==0,1],color="red")
plt.scatter(x[y==1,0],x[y==1,1],color="blue")
plt.show()

#--------------------------逻辑回归
from playML_LogisticRegression import LogisticRegression

log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
log_reg.score(X_test,y_test)
log_reg.coef
log_reg.intercept_

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
    plt.contourf(x0,x1,zz,linewidth=5,camp=custom_camp)

plot_descision_boundary(log.reg,axis=[4,7.5,1.5,4.5])
plt.scatter(x[y==0,0],x[y==0,1],color="red")
plt.scatter(x[y==1,0],x[y==1,1],color="blue")
plt.show()



    