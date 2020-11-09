# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:53:39 2019

@author: lenovoz
"""
import pandas as pd
import numpy as np

dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values
 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
x[:,1]=le.fit_transform(x[:,1])
le2=LabelEncoder()
x[:, 2]=le2.fit_transform(x[:, 2])
ohe=OneHotEncoder(categorical_features=[1])
x=ohe.fit_transform(x).toarray()
x=x[:,1:]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
print(cm)