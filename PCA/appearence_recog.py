# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:03:01 2018

@author: linh
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
import time

start = time.time()
faces=fetch_lfw_people()
print(faces)

faces.keys()
print(faces.data.shape)
faces.data.shape
faces.images.shape

random_indexes=np.random.permutation(len(faces.data))
X=faces.data[random_indexes]

example_faces=X[:36,:] #取出前36张
example_faces.shape

def plot_faces(faces):
    
    fig,axes=plt.subplots(6,6,figsize=(10,10),
                          subplot_kw={'xticks':[],'yticks':[]},
    gridspec_kw=dict(hspace=0.1,wspace=0.1))
    for i,ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62,47),cmap='bone')
    plt.show()
    
plot_faces(example_faces)
plt.savefig('fig.png',bbox_inches='tight')
faces.target_names

from sklearn.decomposition import PCA
pca=PCA(svd_solver='randomized')
pca.fit(X)

plot_faces(pca.components_[:36,:])  #特征脸


end = time.time()
print('Running time: %s Seconds'%(end-start))