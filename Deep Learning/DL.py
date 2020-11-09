# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:45:55 2019

@author: lenovoz
"""

import pandas as pd
import numpy as np
import keras 

#Veri Ön İşleme
veriler = pd.read_csv('Churn_Modelling.csv')
x=veriler.iloc[:,3:13].values
y=veriler.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,1]=le.fit_transform(x[:,1])
le2=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe=ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
x=ohe.fit_transform(x)
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
#Yapay Sinşr Ağı
from keras.models import Sequential#ardışık#nbu komutla bir yapay sinir ağı oluşturuyoruz
from keras.layers import Dense#katmanları olusturuyor
classifier=Sequential()#Suan bir YSA oluşturduk ve içi boş
#activation kullanacagımız YPA fonksiyonu
#init=initializer ilklendir : input_dim giriş katmanı
classifier.add(Dense(8,kernel_initializer='uniform',activation='relu',input_dim=11))
classifier.add(Dense(8,init='uniform',activation='relu'))
classifier.add(Dense(1,init='uniform',activation='relu'))

#optimizer='adam' = skolastik gradyan alcalısı yontemi ile aynı
#buradaki loss sistemin feedback degerin belirliyor.Bu c formulu isini goruyor.
#yani gercek degerden ne kadar saptıgımızı gösteriyor
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,epochs=50)
y_pred=classifier.predict(x_test)
y_pred=(y_pred >0.5)
from sklearn.metrics import confusion_matrix,explained_variance_score
cm=confusion_matrix(y_test,y_pred)
vs=explained_variance_score(y_pred,y_test)
print(cm)





