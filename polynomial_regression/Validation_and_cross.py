# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:38:03 2018

@author: linh
"""

import numpy as np
from sklearn import datasets

digits=datasets.load_digits()
X=digits.data
y=digits.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)

from sklearn.neighbors import KNeighborsClassifier

best_score,best_p,best_k=0,0,0
for k in range(2,11):
    for p in range(1,6):
        knn_clf=KNeighborsClassifier(weights='distance',n_neighbors=k,p=p)
        knn_clf.fit(X_train,y_train)
        score=knn_clf.score(X_test,y_test)
        if score>best_score:
            best_score,best_p,best_k=score,p,k
print("Best K=",best_k)
print("Best p=",best_p)
print("Best Score=",best_score)


#------------------------交叉验证--------------------------
from sklearn.model_selection import cross_val_score #交叉验证

knn_clf=KNeighborsClassifier()
cross_val_score(knn_clf,X_train,y_train,cv=5)#默认是3，可以令CV=5

best_score,best_p,best_k=0,0,0
for k in range(2,11):
    for p in range(1,6):
        knn_clf=KNeighborsClassifier(weights='distance',n_neighbors=k,p=p)
        scores=cross_val_score(knn_clf,X_train,y_train)
        score=np.mean(scores) 
        if score>best_score:
            best_score,best_p,best_k=score,p,k
print("Best K=",best_k)
print("Best p=",best_p)
print("Best Score=",best_score)