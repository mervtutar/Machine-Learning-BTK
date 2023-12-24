#kütüphaneler

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
#veri ön işleme

veriler = pd.read_csv('satislar.csv')#veri yükleme
print(veriler)

aylar = veriler[{'Aylar'}]#bağımsız değişken
print(aylar)

satislar=veriler[["Satislar"]]#bağımlı değişken
print(satislar)

satislar2=veriler.iloc[:,:1].values#iloc[:,0:1] aynı şey
print(satislar2)



#verileri bölmek
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)
'''
#öznitelik ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test= sc.fit_transform(y_test)
print(X_train)
print(X_test)'''

#model inşası(linear regression)
from sklearn.linear_model import LinearRegression
lr=LinearRegression() #lr obje LinearRegression()=constructor
lr.fit(x_train, y_train)

tahmin=lr.predict(x_test)
print(tahmin)
print('---')

print(x_train)
print('---')
print(x_test)
print('---')

print(y_train)
print('---')

print(y_test)
print('---')




