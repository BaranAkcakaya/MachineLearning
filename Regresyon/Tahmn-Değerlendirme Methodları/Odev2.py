# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 18:08:22 2019

@author: lenovoz
"""
import pandas as pd
import numpy  as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import  r2_score

veriler = pd.read_csv('maaslar_yeni.csv')

x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values

print(veriler.corr())

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)
model = sm.OLS(lr.predict(x),x)
print(model.fit().summary())
r2 = r2_score(Y,lr.predict(X))
print("Linear R2 Değeri")
print(r2)

from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 2) 
x_poly = pr.fit_transform(X)
lr2 = LinearRegression()
lr2 .fit(x_poly,Y)

from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 4) 
x_poly = pr.fit_transform(X)
lr2 = LinearRegression()
lr2 .fit(x_poly,Y)

print('Poly OLS')
model2 = sm.OLS(lr2.predict(pr.fit_transform(X)),X)
print(model2.fit().summary())

r2 = r2_score(Y,lr2.predict(x_poly))
print("Polynomial R2 Değeri")
print(r2)

from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_train = sc1.fit_transform(X)
sc2 = StandardScaler()
y_train = sc2.fit_transform(Y)

from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(x_train,y_train)
r2 = r2_score(y_train,svr.predict(x_train))
print('SVR OLS')
model3 = sm.OLS(svr.predict(x_train),x_train)
print(model3.fit().summary())
print("SVR R2 Değeri")
print(r2)

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X,Y)
r2 = r2_score(Y,dtr.predict(X))
print('RDT OLS')
model4 = sm.OLS(dtr.predict(X),X)
print(model4.fit().summary())
print("Decision Tree R2 Değeri")
print(r2)

#ensemble birden fazla kisides,görüsten olusan bir grup denilebilir
#n_estimators kac tane decisiontree kullanacagını soyluyoruz
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=0,n_estimators=10)
rfr.fit(X,Y)
r2 = r2_score(Y,rfr.predict(X))
print('RFR OLS')
model5 = sm.OLS(rfr.predict(X),X)
print(model5.fit().summary())
print("Random Forest R2 Değeri")
print(r2)