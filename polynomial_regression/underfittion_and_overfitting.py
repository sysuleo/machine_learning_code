# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:19:21 2018

@author: linh
"""
import numpy as np
import matplotlib.pyplot as plt

x=np.random.uniform(-3,3,size=100)
X=x.reshape(-1,1)

y=0.5*x**2+x+2+np.random.normal(0,1,size=100)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import LinearRegression

poly_reg=Pipeline([
        ("poly",PolynomialFeatures(degree=2)),
        ("std_scaler",StandardScaler()),
        ("lin_reg",LinearRegression())
])

poly_reg.fit(X,y)
y_predict=poly_reg.predict(X)

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict[np.argsort(x)],color='r')
plt.show()

from sklearn.metrics import mean_squared_error
#——————————————————————————求均分误差————————————————————————
y_predict=poly_reg.predict(X)
print(mean_squared_error(y,y_predict))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

def polynomialRegreesion(degree):
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std_scaler",StandardScaler()),
        ("lin_reg",LinearRegression())
])

poly2_reg=polynomialRegreesion(degree=100)
poly2_reg.fit(X,y)
y2_predict=poly2_reg.predict(X)
print(mean_squared_error(y,y2_predict))

plt.scatter(x,y)
plt.plot(np.sort(x),y2_predict[np.argsort(x)],color='r')
plt.show()

X_plot=np.linspace(-3,3,100).reshape(100,1)
y_plot=poly2_reg.predict(X_plot)

plt.scatter(x,y)
plt.plot(X_plot[:,0],y_plot,color='r')
plt.axis([-3,3,-1,10]) #限制显示大小
plt.show()
