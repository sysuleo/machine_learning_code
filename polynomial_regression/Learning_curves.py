# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:49:01 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
x=np.random.uniform(-3,3,size=100)
X=x.reshape(-1,1)

y=0.5*x**2+x+2+np.random.normal(0,1,size=100)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

plt.scatter(x,y)
plt.show()

def plot_learning_curve(algo,X_train,X_test,y_train,y_test):
        train_score=[]
        test_score=[]
        for i in range(1,len(X_train)+1):
            algo.fit(X_train[:i],y_train[:i])
    
            y_train_predict=algo.predict(X_train[:i])
            train_score.append(mean_squared_error(y_train[:i],y_train_predict))
    
            y_test_predict=algo.predict(X_test)
            test_score.append(mean_squared_error(y_test,y_test_predict))
            
        plt.plot([i for i in range(1,len(X_train)+1)],np.sqrt(train_score),label="train")
        plt.plot([i for i in range(1,len(X_train)+1)],np.sqrt(test_score),label="test")
        plt.legend()
        plt.axis([0,len(X_train)+1,0,4])
        plt.show()

plot_learning_curve(LinearRegression(),X_train,X_test,y_train,y_test)


def polynomialRegression(degree):
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std_scaler",StandardScaler()),
        ("lin_reg",LinearRegression())
])
    
poly2_reg=polynomialRegression(degree=2)
plot_learning_curve(poly2_reg,X_train,X_test,y_train,y_test)