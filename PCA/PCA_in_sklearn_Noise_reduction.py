# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:27:22 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt

X=np.empty((100,2))
X[:,0]=np.random.uniform(0.,100.,size=100) #第一维
X[:,1]=0.75*X[:,0]+3.+np.random.normal(0.,10.,size=100)

from sklearn.decomposition import PCA

pca=PCA(n_components=1)
pca.fit(X)
X_reduction=pca.transform(X)
print(X_reduction.shape)
X_restore=pca.inverse_transform(X_reduction)
print(X_restore.shape)

plt.scatter(X[:,0],X[:,1],color='b',alpha=0.5)
plt.scatter(X_restore[:,0],X_restore[:,1],color='r',alpha=0.5)
plt.show()

from sklearn import datasets

digits=datasets.load_digits()
X=digits.data
y=digits.target
noisy_digits=X+np.random.normal(0,4,size=X.shape)
 #随机的正太分布的噪音，均值为0，方差为4
example_digits=noisy_digits[y==0,:][:10]
for num in range(1,10):
    X_num=noisy_digits[y==num,:][:10]
    example_digits=np.vstack([example_digits,X_num])
    
def plot_digits(data):
    fig,axes=plt.subplots(10,10,figsize=(10,10),
                          subplot_kw={'xticks':[],'yticks':[]},
    gridspec_kw=dict(hspace=0.1,wspace=0.1))
    for i,ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),
                  cmap='binary',interpolation='nearest',
                  clim=(0,16))
    plt.show()
    
plot_digits(example_digits)

pca=PCA(0.6)
pca.fit(noisy_digits)
components=pca.transform(example_digits)
filtered_digits=pca.inverse_transform(components)
plot_digits(filtered_digits)