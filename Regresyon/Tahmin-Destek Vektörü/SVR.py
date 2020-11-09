    # -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:29:34 2019

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

plt.scatter(x_train,y_train)
plt.plot(x_train,svr.predict(x_train))














