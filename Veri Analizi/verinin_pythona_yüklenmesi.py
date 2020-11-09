import pandas as pd

veriler = pd.read_csv('veriler.csv')
print(veriler)
boy = veriler[['boy']]
print(boy)
boykilo = veriler[["boy","kilo"]]
print(boykilo)
x=10