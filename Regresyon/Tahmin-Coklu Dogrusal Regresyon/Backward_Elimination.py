# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:18:43 2019

@author: lenovoz
"""
import pandas as pd
import numpy as np

veriler = pd.read_csv('veriler.csv')

Yas = veriler.iloc[:,1:4].values

ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

c = veriler.iloc[:,-1:].values
print(c)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
print(c)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)

sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1] , index=range(22), columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
solb = s2.iloc[:,:3]
sagb = s2.iloc[:,4:]

veri = pd.concat([solb,sagb],axis=1)
x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)
r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)

import statsmodels.formula.api as sm
#ones birlerden olsan bir dizi lusturur
#astype matrisin tipini veriyor
#OLS istatisliksel verileri cıkarmamıza yarıyor
#endog degeri digerleri ile baglatısı kurulacak deger(bu deger bagımlı degisken)
#exog diger diziler(bu degisken ise bagımsız degerleri iceren dizi)
x = np.append(arr = np.ones((22,1)).astype(int), values = veri,axis=1)
x_l = veri.iloc[:,[0,1,2,3,4,5]].values
reg = sm.OLS(endog = boy,exog = x_l)
regf = reg.fit()
print(regf.summary())

x_l = veri.iloc[:,[0,1,2,3,5]].values
reg = sm.OLS(endog = boy,exog = x_l)
regf = reg.fit()
print(regf.summary())

x_l = veri.iloc[:,[0,1,2,3]].values
reg = sm.OLS(endog = boy,exog = x_l)
regf = reg.fit()
print(regf.summary())


