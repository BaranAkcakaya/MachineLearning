# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:46:54 2019

@author: lenovoz
"""

import pandas as pd
import matplotlib.pyplot as plt

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
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)
x_train = x_train.sort_index()
y_train = y_train.sort_index()
#bunlar bizim verilerimizin grefiksel hali
plt.plot(x_train,y_train)
x_test = x_test.sort_index()
#bu ise bizim linear tahmin modelimiz
plt.plot(x_test,lr.predict(x_test))
plt.title('Aylara Göre Satışlar')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')