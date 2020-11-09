# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:38:28 2019

@author: lenovoz
"""

import pandas as pd

veriler = pd.read_csv('eksikveriler.csv')
ulke = veriler.iloc[:,0:1].values
print(ulke)

#LabelEncoder = verilen degerleri birebir sayıya çeviriyor
#OneHotEncoder = her bir etiketi kolon bazlı olarak ceviriyor
#                eger kolon baslıgında ıse 1 degılse 0 gırıyor

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all' )
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)