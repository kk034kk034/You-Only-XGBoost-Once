# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:52:32 2022

@author: HOME
"""

#%%
## 匯入基本庫
import numpy as np 
import pandas as pd
## 繪圖函數庫
import matplotlib.pyplot as plt
import seaborn as sns
## sklearn & xgboost
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import XGBRegressor

#%%
## 匯入檔案
df = pd.read_csv("Hunya_pchome.csv")
print(df)
"""
## 挑選特徵與要預測的東西
X = 
y = 

## 資料前處理

## 切資料
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=33)

## 建模之前確認資料
print(f'X_train:\n{X_train}\ny_train:\n{y_train}')

## 建模 & 推論 & 預測
# xgbc = XGBClassifier(n_estimators=100,learning_rate=0.3)
xgbc = XGBRegressor(n_estimators=100, learning_rate=0.2)
xgbc.fit(X_train, y_train)
print('train acc:', xgbc.score(X_train, y_train))
print('test acc:', xgbc.score(X_test, y_test))
predictions = xgbc.predict(X_test)
print("predictions:",predictions)
"""