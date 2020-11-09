# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:14:23 2019

@author: lenovoz
"""

import pandas as pd
import numpy as np

veriler = pd.read_csv('odev_tenis.csv')
#apply komutu labelEncoder objesisini tum kolonlara uyguluyor
from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)

wdy = veriler2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
wdy = ohe.fit_transform(wdy).toarray()

havadurumu = pd.DataFrame(data = wdy,index = range(14),columns = ['O','R','S'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler],axis = 1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
import statsmodels.formula.api as sm

x = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler.iloc[:,:-1],axis=1)
x_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
reg = sm.OLS(endog = sonveriler.iloc[:,-1:],exog = x_l)
regf = reg.fit()
print(regf.summary())

sonveriler = sonveriler.iloc[:,1:]

x = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler.iloc[:,:-1],axis=1)
x_l = sonveriler.iloc[:,[0,1,2,3,4]].values
reg = sm.OLS(endog = sonveriler.iloc[:,-1:],exog = x_l)
regf = reg.fit()
print(regf.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

