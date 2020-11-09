# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:07:07 2019

@author: lenovoz
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import random
import math

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')
'''
N=10000
d=10
toplam = 0
secilenler=[]
for i in range(0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[i,ad]
    toplam +=odul
    
plt.hist(secilenler)
plt.show()
'''

N=10000 #10000 tıklanma
d=10 #10 tane reklam
#Ri(n)
oduller=[0]*d #tum ilanları bır dızının ıcınde odullerını tutuyoruz
#Ni(n)
tiklamalar = [0]*d #o ana kadarki tıklamalar
toplam = 0 #toplam odul
secilenler=[]
for i in range(0,N):
    ad = 0
    max_ucb = 0
    for j in range(0,d):
        if(tiklamalar[j] > 0):
            ortalama = oduller[j]/tiklamalar[j]
            delta = math.sqrt(3/2*math.log(i)/tiklamalar[j])
            ucb = ortalama+delta
        else:
            ucb=N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad=j
    secilenler.append(ad)
    tiklamalar[ad]+=1
    odul = veriler.values[i,ad]
    oduller[ad]+=odul
    toplam +=odul
    
print('Toplam Odul:')
print(toplam)
plt.hist(secilenler)
plt.show()
    
    
    
    