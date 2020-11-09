# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:38:52 2019

@author: lenovoz
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import random

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')
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