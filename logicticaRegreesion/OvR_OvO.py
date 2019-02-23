# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:33:57 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris=datasets.load_iris()
X=iris.data[:,:2] #只取前两个特征
y=iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=666)

from sklearn.linear_model import LogisticRegression 

log_reg=LogisticRegression()
"""默认支持ovr"""
log_reg.fit(X_train,y_train)

print(log_reg.score(X_test,y_test)) 
"""不够优化是因为我们只取了前两个特征值"""

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
    
print(log_reg.score(X_train,y_train))
print(log_reg.score(X_test,y_test))

plot_descision_boundary(log_reg,axis=[-4,8.5,1.5,4.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.scatter(X[y==2,0],X[y==2,1])
plt.show()


log_reg2=LogisticRegression(multi_class="multinomial",solver="newton-cg")
"""调用ovo"""
log_reg2.fit(X_train,y_train)

print(log_reg2.score(X_train,y_train))
print(log_reg2.score(X_test,y_test))

plot_descision_boundary(log_reg2,axis=[-4,8.5,1.5,4.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.scatter(X[y==2,0],X[y==2,1])
plt.show()


#-------------------使用所有数据-------------------
#----------------使用ovr----------------------
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=666)
log_reg=LogisticRegression()

log_reg.fit(X_train,y_train)

print(log_reg.score(X_test,y_test))

#----------------使用0v0----------------------
log_reg2=LogisticRegression(multi_class="multinomial",solver="newton-cg")
"""调用ovo"""
log_reg2.fit(X_train,y_train)

print('Log_reg2.score is ',log_reg2.score(X_train,y_train))
print('Log_reg2.score is ',log_reg2.score(X_test,y_test))


#----------------------sklearn中ovr，ovo--------------
from sklearn.multiclass import OneVsRestClassifier

ovr=OneVsRestClassifier(log_reg)
ovr.fit(X_train,y_train)
print(ovr.score(X_test,y_test))

from sklearn.multiclass import OneVsOneClassifier
ovo=OneVsOneClassifier(log_reg)
ovo.fit(X_train,y_train)
print(ovo.score(X_test,y_test))

