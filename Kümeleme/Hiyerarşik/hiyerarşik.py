# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:46:31 2019

@author: lenovoz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
veriler = pd.read_csv('musteriler.csv')
x = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,init='k-means++')
kmeans.fit(x)
print(kmeans.cluster_centers_)
sonuclar=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=100)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)
plt.show()

kmeans = KMeans(n_clusters=4,init='k-means++',random_state=100)
y_pred2=kmeans.fit_predict(x)
print(y_pred2)
plt.scatter(x[y_pred2==0,0],x[y_pred2==0,1],s=100,c='red')
plt.scatter(x[y_pred2==1,0],x[y_pred2==1,1],s=100,c='blue')
plt.scatter(x[y_pred2==2,0],x[y_pred2==2,1],s=100,c='green')
plt.scatter(x[y_pred2==3,0],x[y_pred2==3,1],s=100,c='yellow')
plt.title('k-means')
plt.show()

#affinity dogrusallık oluyor mesafeler arasındaki farka bakacağız
#linkagle iki küme arasındaki farka bakacagız
from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters=3 ,affinity='euclidean',linkage='ward')
y_pred = agc.fit_predict(x)
print(y_pred)

plt.scatter(x[y_pred==0,0],x[y_pred==0,1],s=100,c='red')
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],s=100,c='blue')
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],s=100,c='green')
plt.title('HC')

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.show()
  













