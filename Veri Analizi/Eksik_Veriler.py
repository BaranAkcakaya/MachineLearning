# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:43:26 2019

@author: lenovoz
"""

import pandas as pd
import numpy as np

veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

# =============================================================================
# sci-kit-learn = sklearn
# iloc = hangi kolonları alcagımızı belirtiyoruz
# transform = veriyi degistiriyoruz
#imputer eksik verileri degistirmemize yarıyor
# =============================================================================
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)

