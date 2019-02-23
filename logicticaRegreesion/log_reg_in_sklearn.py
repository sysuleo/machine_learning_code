# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:02:12 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
X=np.random.normal(0,1,size=(200,2))
y=np.array(X[:,0]**2+X[:,1]<1.5,dtype='int')
#--------------布尔向量的数组
for _ in range(20):
    y[np.random.randint(200)]=1
    
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=666)

from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
print(log_reg.score(X_train,y_train))
print(log_reg.score(X_test,y_test))

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
    
poly_log_reg=PolynomialLogisticRegression(10)
poly_log_reg.fit(X_train,y_train)
print(poly_log_reg.score(X_trian,y_trian))
print(poly_log_reg.score(X_test,y_test))

plot_descision_boundary(poly_log_reg,axis=[-4,4,-4,4])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()

#------------------正则化----------------
def PolynomialLogisticRegression(degree,C):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('std_scaler',StandardScaler()),
        ('log_reg',LogisticRegression(C=C))
    ])
poly_log_reg2= PolynomialLogisticRegression(degree=10,C=0.1)  
poly_log_reg2.fit(X_train,y_train)
print(poly_log_reg2.score(X_trian,y_trian))
print(poly_log_reg2.score(X_test,y_test))

plot_descision_boundary(poly_log_reg2,axis=[-4,4,-4,4])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()


#---------------------------L1正则化------------------------
def PolynomialLogisticRegression(degree,C,penalty='l2'):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('std_scaler',StandardScaler()),
        ('log_reg',LogisticRegression(C=C,penalty=penalty))
    ])
    
poly_log_reg3= PolynomialLogisticRegression(degree=10,C=0.1,penalty='l1')  
poly_log_reg3.fit(X_train,y_train)
print(poly_log_reg3.score(X_trian,y_trian))
print(poly_log_reg3.score(X_test,y_test))

plot_descision_boundary(poly_log_reg3,axis=[-4,4,-4,4])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()




