# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:08:21 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.random.uniform(-3,3,size=100)
X=x.reshape(-1,1)

y=0.5*x**2+x+2+np.random.normal(0,1,size=100)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression


poly_reg=Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
])

poly_reg.fit(X,y)
y_predict=poly_reg.predict(X)

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict[np.argsort(x)],color='r')
plt.show()