# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:36:17 2018

@author: linh
"""
from sklearn import datasets
import matplotlib.pyplot as plt

digits=datasets.load_digits()
X=digits.data
y=digits.target

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def PolynomialRegression(degree):
    return Pipeline([
            ("poly",PolynomialFeatures(degree=degree)),
            ("std_scaler",StandardScaler()),
            ("lin_reg",LinearRegression())
    ])
    
from sklearn.metrics import mean_squared_error

poly100_reg=PolynomialRegression(degree=100)
poly100_reg.fit(X,y)

y100_predict=poly100_reg.predict(X)
print(mean_squared_error(y,y100_predict))

X_plot=np.linspace(-3,3,100).reshape(100,1)
y_plot=poly100_reg.predict(X_plot)

plt.scatter(x_plot,y_plot)
plt.plot(X_plot[:,0],y_plot,color='r')
plt.axis([-3,3,-1,10]) #限制显示大小
plt.show()