# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:39:18 2018

@author: liuw
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris=datasets.load_iris()
X=iris.data
y=iris.target

X=X[y<2,:2]
##限定y=0或y=1，只取前两个特征
y=y[y<2]

print(y.shape)

plt.scatter(X[y==0,0],X[y==0,1],color='red')
plt.scatter(X[y==1,0],X[y==1,1],color='black')
plt.show()

#-------------------归一化------------------------------
from sklearn.preprocessing import StandardScaler

standardScaler=StandardScaler()
standardScaler.fit(X)
X_standard=standardScaler.transform(X)


plt.scatter(X_standard[y==0,0],X_standard[y==0,1],color='red')
plt.scatter(X_standard[y==1,0],X_standard[y==1,1],color='black')
plt.show()

from sklearn.svm import LinearSVC

svc=LinearSVC(C=1e9)
svc.fit(X_standard,y)


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


plot_descision_boundary(svc,axis=[-3,3,-3,3])
plt.scatter(X_standard[y==0,0],X_standard[y==0,1],color='red')
plt.scatter(X_standard[y==1,0],X_standard[y==1,1],color='black')
plt.show()

svc2=LinearSVC(C=0.01)
svc2.fit(X_standard,y)
plot_descision_boundary(svc2,axis=[-3,3,-3,3])
plt.scatter(X_standard[y==0,0],X_standard[y==0,1],color='red')
plt.scatter(X_standard[y==1,0],X_standard[y==1,1],color='black')
plt.show()

print(svc.coef_)
print(svc.intercept_)

def plot_svc_descision_boundary(model,axis):
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
    
    w=model.coef_[0]
    b=model.intercept_[0]
    
    #w0*x0+w1*w1+b=0
    #x1=-w0/w1*x0-b/w1
    plot_x=np.linspace(axis[0],axis[1],200)
    up_y=-w[0]/w[1]*plot_x-b/w[1]+1/w[1]
    #w0*x0+w1*w1+b=1
    down_y=-w[0]/w[1]*plot_x-b/w[1]-1/w[1]
    #w0*x0+w1*w1+b=-1
    
    up_index=(up_y>=axis[2])&(up_y<=axis[3]) #布尔数组
    down_index=(down_y>=axis[2])&(down_y<=axis[3])
    
    plt.plot(plot_x[up_index],up_y[up_index],color='blue')
    plt.plot(plot_x[down_index],down_y[up_index],color='blue')
    
plot_svc_descision_boundary(svc,axis=[-3,3,-3,3])
plt.scatter(X_standard[y==0,0],X_standard[y==0,1],color='red')
plt.scatter(X_standard[y==1,0],X_standard[y==1,1],color='black')
plt.show() 