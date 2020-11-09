# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:09:42 2019

@author: lenovoz
"""
import pandas as pd

veriler = pd.read_excel('iris.xls')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print('LR')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('KNN')
print(cm)

from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)
svc = SVC(kernel = 'poly')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion ='entropy')
dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('DTC GİNİ')
print(cm)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)
y_proba = rfc.predict_proba(x_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)
print(y_proba)
print(y_test)

from sklearn import metrics
fpr,tpr,thd = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)