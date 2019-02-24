# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:07:14 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import train_test_split

class learning:
    def polynomialRegreesion(degree):
        return Pipeline([
            ("poly",PolynomialFeatures(degree=degree)),
            ("std_scaler",StandardScaler()),
            ("lin_reg",LinearRegression())
    ])
    
    def plot_learning_curve(algo,X_train,X_test,y_trian,y_test):
        train_score=[]
        test_score=[]
        for i in range(1,len(X_train)+1):
            algo.fit(X_train[:i],y_train[:i])
    
            y_train_predict=algo.predict(X_train[:i])
            train_score.append(mean_squared_error(y_train[:i],y_train_predict))
    
            y_test_predict=lin_algo.predict(X_test)
            test_score.append(mean_squared_error(y_test,y_test_predict))
            
        plt.plot([i for i in range(1,len(X_train)+1)],np.sqrt(train_score),label="train")
        plt.plot([i for i in range(1,len(X_train)+1)],np.sqrt(test_score),label="test")
        plt.legend()
        plt.axis([0,len(X_train)+1,0,4])
        plt.show()
        

    