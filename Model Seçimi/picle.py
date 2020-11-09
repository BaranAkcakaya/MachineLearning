# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:42:06 2020

@author: lenovoz
"""

import pandas as pd
import pickle

veriler=pd.read_csv('satislar.csv')
x=veriler.iloc[:,0:1].values
y=veriler.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)

dosya="model_kayit"
pickle.dump(lr,open(dosya,'wb'))
yükle=pickle.load(open(dosya,'rb'))
print(yükle.predict(x_test))                  
                  