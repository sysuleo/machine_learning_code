import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from scipy.optimize import leastsq

def real_func(x):
    return np.sin(2*np.pi*x)

def fit_func(p,x):
    f=np.poly1d(p)
    return f(x)

def residuals_func(p,x,y):
    return fit_func(p,x)-y

x=np.linspace(0,1,10)
x_points=np.linspace(0,1,1000)
y_=real_func(x)
y=[np.random.normal(0,0.1)+y1 for y1 in y_]

def fitting(M=0):
    """
        M    为 多项式的次数
    """
    # 随机初始化多项式参数
    p_init=np.random.rand(M+1) # w的维数
# 最小二乘法
    p_lsq=leastsq(residuals_func,p_init,args=(x,y))
    #p_lsq: <class 'tuple'>,p_lsq[0]是求出的w向量，p_lsq[1]=1
    print('Fitting Parameters:', p_lsq[0])

    plt.plot(x_points,real_func(x_points),label='real')
    plt.plot(x_points,fit_func(p_lsq[0],x_points),label='fitting curve')
    plt.plot(x,y,'bo',label='noise')
    plt.show()

#for i in range(10):
    #fitting(i)




"""回归问题中，损失函数是平方损失，正则化可以是参数向量的L2范数,也可以是L1范数。

    L1: regularization*abs(p)

    L2: 0.5 * regularization * np.square(p)

结果显示过拟合， 引入正则化项(regularizer)，降低过拟合
"""
regularization = 0.0001
def residuals_func_with_regularization(p,x,y):
    ret=fit_func(p,x)-y
    ret=np.append(ret,np.sqrt(0.5*regularization*np.square(p)))# L2范数作为正则化项
    #numpy.append(arr,values,axis=None) 将values插入到目标arr的最后。
    return ret

def fitting_with_regularization(M=0):
    """
        M    为 多项式的次数
    """
    # 随机初始化多项式参数
    p_init=np.random.rand(M+1) # w的维数
# 最小二乘法
    p_lsq=leastsq(residuals_func_with_regularization,p_init,args=(x,y))
    #p_lsq: <class 'tuple'>,p_lsq[0]是求出的w向量，p_lsq[1]=1
    print('Fitting Parameters:', p_lsq[0])

    plt.plot(x_points,real_func(x_points),label='real')
    plt.plot(x_points,fit_func(p_lsq[0],x_points),label='fitting curve')
    plt.plot(x,y,'bo',label='noise')
    plt.show()

for i in range(8,10):
    fitting(i)
    fitting_with_regularization(i)