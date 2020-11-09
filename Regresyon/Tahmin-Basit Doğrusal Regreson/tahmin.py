# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:40:33 2019

@author: lenovoz
"""
import pandas as pd

veriler = pd.read_csv('satislar.csv')
aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)
'''
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#buranda x ve y train bilgilerini kullanarak bil model insa et diyoruz
lr.fit(x_train,y_train)
#burada ise predict komutu ile tahmin etmesini istiyoruz
tahmin = lr.predict(x_test)