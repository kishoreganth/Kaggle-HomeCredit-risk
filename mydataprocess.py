# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 09:31:08 2018

@author: kisho
"""

# import libs 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# importing the dataset 
app_train = pd.read_csv('application_train.csv', index_col = "SK_ID_CURR")
app_test = pd.read_csv('application_test.csv', index_col = "SK_ID_CURR")


features = [x for x in app_train.columns if x != "TARGET"]

app_train['TARGET'].value_counts()

data_all = app_train.append(app_test)

cat_cols=['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','NAME_TYPE_SUITE',
             'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','FLAG_MOBIL',
             'FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','OCCUPATION_TYPE',
             'WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION',
             'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY',
             'REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','ORGANIZATION_TYPE','FONDKAPREMONT_MODE',
             'HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE',
             'FLAG_DOCUMENT_2',
             'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
             'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12',
             'FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17',
             'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']


def add_one_hot_feature(data, cat_col, train_indicate = True):
    for col in cat_col:
        print(col)
        temp = pd.get_dummies(data[col], drop_first = train_indicate , prefix = col)
        temp.index = data.index 
        data = pd.concat([data, temp], axis = 1)
        print(data.shape)
    data  = data.drop(cat_col , axis = 1)
    print(data.shape)
    return data 

data_all_onehot = add_one_hot_feature(data_all,cat_cols, train_indicate = True)

data_train_onehot = data_all_onehot.loc[app_train.index]
data_test_onehot = data_all_onehot.loc[app_test.index]



data_train_onehot.to_csv('train_onehot.csv')
data_test_onehot.to_csv('test_onehot.csv')

data_train_onehot.isnull().sum()

# handling a missing data 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(data_test_onehot)
data_test_onehot_miss = imputer.transform(data_test_onehot)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(data_train_onehot)
data_train_onehot_miss = imputer.transform(data_train_onehot)

complete_train = pd.DataFrame(data_train_onehot_miss)
complete_train.to_csv('complete_train.csv')

complete_test = pd.DataFrame(data_test_onehot_miss)
complete_test.to_csv('complete_test.csv')

# logistic regression 
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression()
classifier.fit(data_train_onehot, app_train['TARGET'])





