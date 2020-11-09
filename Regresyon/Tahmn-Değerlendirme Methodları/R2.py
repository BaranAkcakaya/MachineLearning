# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:21:48 2019

@author: lenovoz
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar.csv')

egts = veriler.iloc[:,1:2]
maas = veriler.iloc[:,2:]

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(egts,maas)
plt.scatter(egts,maas)
plt.plot(egts,lr.predict(egts))
plt.show()
from sklearn.metrics import  r2_score
r2 = r2_score(maas,lr.predict(egts))
print("Linear R2 Değeri")
print(r2)

from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 2) 
x_poly = pr.fit_transform(egts)
lr2 = LinearRegression()
lr2 .fit(x_poly,maas)
plt.scatter(egts,maas)
plt.plot(egts,lr2.predict(pr.fit_transform(egts)))
plt.show()

from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 4) 
x_poly = pr.fit_transform(egts)
lr2 = LinearRegression()
lr2 .fit(x_poly,maas)
plt.scatter(egts,maas)
plt.plot(egts,lr2.predict(pr.fit_transform(egts)))
plt.show()
r2 = r2_score(maas,lr2.predict(x_poly))
print("Polynomial R2 Değeri")
print(r2)

print(lr.predict([[11]]))
print(lr.predict([[6.6]]))
print(lr2.predict(pr.fit_transform([[11]])))
print(lr2.predict(pr.fit_transform([[6.6]])))

from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_train = sc1.fit_transform(egts)
sc2 = StandardScaler()
y_train = sc2.fit_transform(maas)

from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(x_train,y_train)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,svr.predict(x_train),color='yellow')
plt.show()
print(svr.predict([[11]]))
print(svr.predict([[6.6]]))
r2 = r2_score(y_train,svr.predict(x_train))
print("SVR R2 Değeri")
print(r2)

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(egts,maas)
plt.scatter(egts,maas,color='blue')
plt.plot(egts,dtr.predict(egts),color='black')
plt.show()
print(dtr.predict([[11]]))
print(dtr.predict([[6.6]]))
r2 = r2_score(maas,dtr.predict(egts))
print("Decision Tree R2 Değeri")
print(r2)

#ensemble birden fazla kisides,görüsten olusan bir grup denilebilir
#n_estimators kac tane decisiontree kullanacagını soyluyoruz
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=0,n_estimators=10)
rfr.fit(egts,maas)
plt.scatter(egts,maas,color='blue')
plt.plot(egts,rfr.predict(egts),color='black')
plt.show()
print(rfr.predict([[6.6]]))
print(rfr.predict([[11]]))
r2 = r2_score(maas,rfr.predict(egts))
print("Random Forest R2 Değeri")
print(r2)