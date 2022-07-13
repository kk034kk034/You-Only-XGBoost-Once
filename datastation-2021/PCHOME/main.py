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
df = pd.read_excel("PChome_日用品銷售_樣本.xlsx", sheet_name='工作表2')
#print(df)

## 挑選特徵與要預測的東西
# [Day 18 : 模型前的資料處理 (2)](https://ithelp.ithome.com.tw/articles/10272964?sc=hot)
# 原始資料是有序離散值的話 => Label Encoding
# 原始資料是無序離散值的話 => One Hot Encoding (Dummies) 但我這邊不知為啥我array有問題
# 還需加入處理:'消費者編號(member_id) ','訂單日期(date_cd)','訂單時間(time_cd)'
X = df[['郵遞區號(postal_cd)','商品類別(department)','訂單商品售價(price)','Prime卡會員(prime)']]
# 是否無法處理純文字? => df['商品名稱(goods)']
# y = df['商品編號(prod_id)']
# y = pd.get_dummies(df['商品編號(prod_id)'])
# y = OneHotEncoder().fit_transform(df['商品編號(prod_id)']).toarray()
# y = LabelBinarizer().fit_transform(df['商品編號(prod_id)']).toarray()
y = LabelEncoder().fit_transform(df['商品編號(prod_id)'])
# print("here is y:\n",y)

## 資料前處理
X['Prime卡會員(prime)'] = X['Prime卡會員(prime)'].replace({'否':0, '是':1})
X['商品類別(department)'] = X['商品類別(department)'].replace({'食品':0, '日用':1})

## 切資料
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=33)
#print(X_train)

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
