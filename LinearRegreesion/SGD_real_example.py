# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 10:02:53 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston=datasets.load_boston()
X=boston.data
y=boston.target

#print(y)

X=X[y<50.0]
y=y[y<50.0]

from model_selection import train_test_split
#划分测试和训练
X_train,X_test,y_train,y_test=train_test_split(X,y,seed=666)


from sklearn.preprocessing import StandardScaler
#归一化处理
standardScaler=StandardScaler()
standardScaler.fit(X_train)
X_train_standard=standardScaler.transform(X_train)
X_test_standard=standardScaler.transform(X_test)

from ML_LinearRegression import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit_sgd(X_train_standard,y_train,n_iters=100)
print(lin_reg.score(X_test_standard,y_test))
#theta=lin_reg._theta
#print(theta)