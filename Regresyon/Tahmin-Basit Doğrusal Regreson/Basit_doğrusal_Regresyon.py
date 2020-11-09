# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:37:55 2019

@author: lenovoz
"""

import pandas as pd

veriler = pd.read_csv('satislar.csv')
aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]

#buranda x_train kısmı bagımsız degisken
#x_test kısmı ise bagımlı degiskendir .Ona göre yerlestirme yapılır
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)

#from komutu ile ilgili kütüphaneyi ekliyoruz
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#buranda x ve y train bilgilerini kullanarak bil model insa et diyoruz
lr.fit(x_train,y_train)
