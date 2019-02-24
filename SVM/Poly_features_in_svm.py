# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 14:17:27 2018

@author: liuw
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

X,y=datasets.make_moons()
print(X.shape)

plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()

X,y=datasets.make_moons(noise=0.15,random_state=666) #使标准差增大，随机化
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()

###使用多项式特征的SVM
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

def PolynomialSVC(degree,C):
    return Pipeline([
            ('poly',PolynomialFeatures(degree=degree)),
            ('scaler',StandardScaler()),
            ('LinearSVC',LinearSVC(C=C))
            ])
    
poly_svc=PolynomialSVC(degree=3,C=1.0)
poly_svc.fit(X,y)

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
    
plot_descision_boundary(poly_svc,axis=[-1.5,2.5,-1.0,1.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()


from sklearn.svm import SVC
def polynomialKernelSVC(degree,C=1.0):
    return Pipeline([
            "std_scaler",StandardScaler(),
            "kernelSVC",SVC(kernel="poly")
            ])