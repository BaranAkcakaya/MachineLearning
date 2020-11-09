# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:56:30 2019

@author: lenovoz
"""

import pandas as pd
import numpy as np

#Veri Ön İşleme
veriler = pd.read_csv('Wine.csv')
x=veriler.iloc[:,0:13].values
y=veriler.iloc[:,13].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#PCA
from sklearn.decomposition import PCA
#n_components kac boyuta indirgencegini belirliyoruz
pca=PCA(n_components=2)
x_train2=pca.fit_transform(x_train)
x_test2=pca.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

classifier2 = LogisticRegression(random_state=0)
classifier2.fit(x_train2,y_train)

y_pred=classifier.predict(x_test)
y_pred2=classifier2.predict(x_test2)

from sklearn.metrics import confusion_matrix
print("GERCEK/PCASIZ:")
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("GERCEK/PCA ILE:")
cm2=confusion_matrix(y_test,y_pred2)
print(cm2)
print("PCASIZI/PCAILE:")
cm3=confusion_matrix(y_pred,y_pred2)
print(cm3)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=2)
x_train_lda=lda.fit_transform(x_train,y_train)
x_test_lda=lda.transform(x_test)

classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(x_train_lda,y_train)

y_pred_lda=classifier_lda.predict(x_test_lda)
print("LDA/GERCEK:")
cm4=confusion_matrix(y_pred,y_pred_lda)
print(cm4)




