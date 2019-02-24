# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:09:03 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x=np.random.uniform(-3.0,3.0,size=100)
X=x.reshape(-1,1)
y=0.5*x+3+np.random.normal(0,1,size=100)

plt.scatter(X,y)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def PolynomialRegression(degree):
    return Pipeline([
            ("Poly",PolynomialFeatures(degree=degree)),
            ("std_scaler",StandardScaler()),
            ("lin_reg",LinearRegression())
            ])
    
from sklearn.model_selection import train_test_split
np.random.seed(666)
X_train,X_test,y_train,y_test=train_test_split(X,y)

from sklearn.metrics import mean_squared_error

poly_reg=PolynomialRegression(20)
poly_reg.fit(X_train,y_train)
y_predict=poly_reg.predict(X_test)
print(mean_squared_error(y_test,y_predict))

def plot_model(model):
    X_plot=np.linspace(-3,3,100).reshape(100,1)
    y_plot=model.predict(X_plot)

    plt.scatter(x,y)
    plt.plot(X_plot[:,0],y_plot,color='r')
    plt.axis([-3,3,0,6])
    plt.show()
    
plot_model(poly_reg)


#------------------------使用岭回归-------------------------------------
from sklearn.linear_model import Ridge
def RidgeRegression(degree,alpha):
    return Pipeline([
            ("Poly",PolynomialFeatures(degree=degree)),
            ("std_scaler",StandardScaler()),
            ("ridge_reg",Ridge(alpha=alpha))
            ])
    
    
ridge_reg=RidgeRegression(20,100)
ridge_reg.fit(X_train,y_train)
y_predict=ridge_reg.predict(X_test)
print(mean_squared_error(y_test,y_predict))
plot_model(ridge_reg)