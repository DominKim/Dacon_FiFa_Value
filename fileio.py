# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import  LinearSVR
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv("c:/itwill/4_python-ii/data/fifa_train.csv")
train.info()
train.shape # (8932, 12)
train.isnull().sum() # 결측치 확인
'''
 0   id                8932 non-null   int64  
 1   name              8932 non-null   object 
 2   age               8932 non-null   int64  
 3   continent         8932 non-null   object 
 4   contract_until    8932 non-null   object 
 5   position          8932 non-null   object 
 6   prefer_foot       8932 non-null   object 
 7   reputation        8932 non-null   float64
 8   stat_overall      8932 non-null   int64  
 9   stat_potential    8932 non-null   int64  
 10  stat_skill_moves  8932 non-null   float64
 11  value             8932 non-null   float64
'''
train.head()

# x, y split
idx = list(train.columns)
train.id.dtype == "int64"
x_idx = [i for i in idx[1:11] if train[i].dtype == "int64" or train[i].dtype == "float64"]
X = train[x_idx]
y = train["value"]

# 정규화
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X)
X_train_scaled
# train, test split
train_x, test_x, train_y, test_y = train_test_split(X_train_scaled, y, test_size = 0.3)


# 선형회귀 model 생성
obj = LinearRegression()
model = obj.fit(X = train_x, y = train_y)
y_pred = model.predict(test_x)

# mse, score
model.score(train_x, train_y) # 0.6483753991528678
model.score(test_x, test_y)   # 0.6483753991528678
model.coef_
mse = mean_squared_error(test_y, y_pred)
mse 
score = r2_score(test_y, y_pred)
score # 0.6250441446025428
y_true = np.array(test_y)

df = pd.DataFrame({"y_true":test_y, "y_pred":y_pred})
df.corr() # 0.79


# 선형 svm 생성
lsvm = LinearSVR()
model_svm = lsvm.fit(train_x, train_y)
y_pred_svm = model_svm.predict(test_x)

# mse, score
model_svm.score(train_x, train_y) # 0.6483753991528678
model_svm.score(test_x, test_y)   # 0.6483753991528678

mse = mean_squared_error(test_y, y_pred)
mse 
score = r2_score(test_y, y_pred)
score # 0.6250441446025428
y_true = np.array(test_y)
y_true

df = pd.DataFrame({"y_true":test_y, "y_pred":y_pred_svm})
df.corr()


test = pd.read_csv("c:/itwill/4_python-ii/data/fifa_test.csv")
test.info()
test.shape # (3828, 11)
test.isnull().sum() # 결측치 확인
'''
 0   id                3828 non-null   int64  
 1   name              3828 non-null   object 
 2   age               3828 non-null   int64  
 3   continent         3828 non-null   object 
 4   contract_until    3828 non-null   object 
 5   position          3828 non-null   object 
 6   prefer_foot       3828 non-null   object 
 7   reputation        3828 non-null   float64
 8   stat_overall      3828 non-null   int64  
 9   stat_potential    3828 non-null   int64  
 10  stat_skill_moves  3828 non-null   float64
'''
X = test[x_idx]
y_pred = model.predict(X)
y_pred

submit = pd.read_csv("c:/itwill/4_python-ii/data/submission.csv")
submit.info()
submit["value"] = y_pred
submit.head()
submit.to_csv("c:/itwill/4_python-ii/data/submission.csv", index = None, encoding = "utf-8")
'''
 0   id      3828 non-null   int64
 1   value   3828 non-null   int64
'''
