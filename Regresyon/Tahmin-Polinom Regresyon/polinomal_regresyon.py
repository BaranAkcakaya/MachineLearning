    # -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:12:33 2019

@author: lenovoz
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar.csv')

egts = veriler.iloc[:,1:2]
maas = veriler.iloc[:,2:]
#linear regresyon ile
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(egts,maas)
plt.scatter(egts,maas,color = 'red')
plt.plot(egts,lr.predict(egts))
plt.show()

#polinomal regresyon ile
#polynomialFeatures bize herhangi bir sayıyı polinomal olarak ifade etmemizi sagliyor
#degree = derece
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 2) 
x_poly = pr.fit_transform(egts)
lr2 = LinearRegression()
lr2 .fit(x_poly,maas)
plt.scatter(egts,maas,color = 'red')
plt.plot(egts,lr2.predict(pr.fit_transform(egts)))
plt.show()

from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 4) 
x_poly = pr.fit_transform(egts)
lr2 = LinearRegression()
lr2 .fit(x_poly,maas)
plt.scatter(egts,maas,color = 'red')
plt.plot(egts,lr2.predict(pr.fit_transform(egts)))
plt.show()

#ornekle gorelim
print(lr.predict([[11]]))
print(lr.predict([[6.6]]))

print(lr2.predict(pr.fit_transform([[11]])))
print(lr2.predict(pr.fit_transform([[6.6]])))
