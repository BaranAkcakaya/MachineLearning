# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:42:29 2019

@author: lenovoz
"""
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

yorumlar = pd.read_csv('Restaurant_Reviews.csv')
nltk.download('stopwords')
#PorterStemmer kelimeyi koklerine ayırıyor
ps=PorterStemmer()
derlem=[]
#preprocessing(ön işleme)
for i in range(0,1000):
    #♦Düzenli İfadeler (Regular Expressions)
    #burada sub=substitute degistir anlamına geliyor
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    #stop word temizleme
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)
#CountVectorizer bir kelime vektoru olusturuyor.max_features ise en cok gecen kelimeleri getiriyor
cv=CountVectorizer(max_features=500)
x = cv.fit_transform(derlem).toarray() #bagımsız degisken
y = yorumlar.iloc[:,1].values #bagımlı degisken

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20)
#feature extraction(özitelik cıkarımı)
from sklearn.naive_bayes import GaussianNB
gnb= GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm) #K%72,5 acurracy
