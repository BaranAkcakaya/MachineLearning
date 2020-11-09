# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:19:21 2019

@author: lenovoz
"""
import pandas as pd

veriler = pd.read_csv('veriler.csv')

ulke = veriler.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all' )
ulke = ohe.fit_transform(ulke).toarray()

cinsiyet = veriler.iloc[:,-1:].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cinsiyet[:,0] = le.fit_transform(cinsiyet[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all' )
cinsiyet = ohe.fit_transform(cinsiyet).toarray()

sonuc = pd.DataFrame(data = ulke,index = range(22),columns=['fr','tr','us'])
print(sonuc)
#cinsiyet = veriler.iloc[:,-1:].values
sonuc3 = pd.DataFrame(data = cinsiyet[:,:1],index = range(22),columns=['cinsiyet'])
print(sonuc3)

s2 = pd.concat([sonuc,sonuc3],axis=1)
print(s2)