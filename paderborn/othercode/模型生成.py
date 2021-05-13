# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:29:53 2021

@author: 12480
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import joblib

class Result:
    precision = 0
    recall = 0
    accuracy = 0
    rocX = []
    rocY = []
    featureImportances = []
params = {}
params['n_estimators'] = 100
params['max_depth'] =None
params['max_features'] = 'auto'
params['min_samples_split'] = 2
params['min_samples_leaf'] = 1
params['train'] = 'D:/数据集/111/train5.csv'
params['test'] = 'D:/数据集/111/test5.csv'
params['store'] = {}

try:
    
        train = np.array(pd.read_csv(params['train']))
        train_y = train[:, -1]
    
        train_x = train[:, :-1]
        print(train_x)
        test = pd.read_csv(params['test'])
        test = np.array(test.drop(['label'], axis=1))
        
    
    

        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                             max_features=params['max_features'],
                             max_depth=params['max_depth'],
                             min_samples_split=params['min_samples_split'],
                             min_samples_leaf=params['min_samples_leaf']).fit(train_x, train_y)
        clf.fit(train_x,train_y)
        predict = clf.predict(test)
    
        
        predict = pd.DataFrame(predict,columns=['label'])
        predict.to_csv('D:/数据集/111/predict3.csv',index=False)
        joblib.dump(clf,'D:/数据集/111/paderborn.model')
    
except Exception as e:
    print(e)