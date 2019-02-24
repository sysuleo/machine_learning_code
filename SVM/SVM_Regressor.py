# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:47:13 2018

@author: liuw
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

boston=datasets.load_boston()
X=boston.data
y=boston.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=666)

from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def StandardLinearSVR(epsilon=0.1):
    return Pipeline([
            ("std_scaler",StandardScaler),
            ("linearSVR",LinearSVR(epsilon=epsilon))
            ])
    
svr=StandardLinearSVR()
svr.fit(X_train,y_train)

print(svr.score(X_test,y_test))