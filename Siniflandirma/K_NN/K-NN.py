# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:50:29 2019

@author: lenovoz
"""
import pandas as pd

veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.transform(x_test)
#fit egitme
#transform ugulama,kullanma

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)