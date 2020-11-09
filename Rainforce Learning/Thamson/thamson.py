# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:53:59 2019

@author: lenovoz
"""

import pandas as pd
import matplotlib.pyplot as plt
import random

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

N=10000
d=10
toplam = 0 
secilenler=[]
birler=[0]*d
sifirlar=[0]*d
for i in range(0,N):
    ad = 0
    max_th = 0
    for j in range(0,d):
        rastbeta=random.betavariate(birler[j]+1,sifirlar[j]+1)
        if rastbeta > max_th:
            max_th = rastbeta
            ad=j
    secilenler.append(ad)
    odul = veriler.values[i,ad]
    if odul ==1:
        birler[ad]+=1
    else:
        sifirlar[ad]+=1
    toplam +=odul
    
print('Toplam Odul:')
print(toplam)
plt.hist(secilenler,color='red')
plt.show()