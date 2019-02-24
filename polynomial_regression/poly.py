# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:42:27 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.random.uniform(-3,3,size=100)
X=x.reshape(-1,1)

y=0.5*x**2+x+2+np.random.normal(0,1,size=100)

plt.scatter(x,y)
plt.show()

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(X,y)
y_predict=lin_reg.predict(X)

plt.scatter(x,y)
plt.plot(x,y_predict,color='r')
plt.show()

#——————————————————————————解决方案——————————————————————

X2=np.hstack([X,X**2])
X2.shape
lin_reg2=LinearRegression()
lin_reg2.fit(X2,y)
y_predict2=lin_reg2.predict(X2)

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r')
plt.show()

a=lin_reg2.coef_
print(a)