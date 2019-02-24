# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:32:21 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits=datasets.load_digits()  #手写识别数据
X=digits.data
y=digits.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=666)
#顺序不要搞乱了

from sklearn.neighbors import KNeighborsClassifier

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_train)

a=knn_clf.score(X_test,y_test)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(X_train)
X_train_reduction=pca.transform(X_train)
X_test_reduction=pca.transform(X_test)

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train_reduction,y_train)

b=knn_clf.score(X_test_reduction,y_test)
print(a)
print(b)

print(pca.explained_variance_ratio_) 
#解释原来方差的百分之几，这里丢失了70%左右

pca=PCA(n_components=X_train.shape[1])
pca.fit(X_train)
print(pca.explained_variance_ratio_)
#从大到小排列，每个主成分能解释的方差是多少，即重要程度

plt.plot([i for i in range(X_train.shape[1])],[np.sum(pca.explained_variance_ratio_[:i+1])for i in range(X_train.shape[1])])
plt.show()  

pca=PCA(0.95) #指定方差大小
pca.fit(X_train)
pca.n_components_

X_train_reduction=pca.transform(X_train)
X_test_reduction=pca.transform(X_test)

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train_reduction,y_train)

c=knn_clf.score(X_test_reduction,y_test)
print(c)
