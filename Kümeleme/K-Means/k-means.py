# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:34:54 2019

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
#merkez noktaları veriyor
print(kmeans.cluster_centers_)
sonuclar=[]
#burada random_state onemli eger biz random_state bir deger girmezsek her seferinde
#merkezi random atar bunu engellemek için random stateyi dolduruyoruz.
#inertia_=WCSS degerlerini verir yani basarısını alır
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=100)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)
plt.show()