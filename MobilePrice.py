# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:15:13 2019

@author: ADMIN
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
'''from sklearn.metrics import r2_score'''
from sklearn.metrics import accuracy_score



data = pd.read_csv('dataset.csv')
#print(data.head())
data['clock_speed'] = data['clock_speed'].astype('int64')
data['m_dep'] = data['m_dep'].astype('int64')
real_x = data.iloc[:,[0,2,3,4,5,6,8,9,11,12,13,14,15,17,18,19,20]].values
print(real_x)
real_y = data.iloc[:,20].values
print(real_y)
training_x,test_x,training_y,test_y=train_test_split(real_x,real_y,test_size=0.3,random_state=0)
ss = StandardScaler()
training_x = ss.fit_transform(training_x)
test_x = ss.fit_transform(test_x)
cls_LR = LogisticRegression(random_state=0)
cls_LR.fit(training_x,training_y)
pred_y = cls_LR.predict(test_x)

print('------------Predicted Output---------')
pred_y=pred_y.astype('int64')
print(pred_y)

print ("Accuracy is ", accuracy_score(test_y,pred_y)*100)