# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:41:41 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.random.uniform(-3,3,size=100)
X=x.reshape(-1,1)

y=0.5*x**2+x+2+np.random.normal(0,1,size=100)

from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=2) #代表几次方
poly.fit(X)
X2=poly.transform(X)
print(X2)

from sklearn.linear_model import LinearRegression
lin_reg2=LinearRegression()
lin_reg2.fit(X2,y)
y_predict2=lin_reg2.predict(X2)

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r')
plt.show()

print(lin_reg2.coef_)

#-----------------x

X=np.arange(1,11).reshape(-1,2)
print(X)
poly=PolynomialFeatures(degree=2)
poly.fit(X)
X3=poly.transform(X)
print(X3)