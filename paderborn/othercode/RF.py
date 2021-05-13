# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:10 2021

@author: 12480
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import json

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
params['min_samples_split'] = 3
params['min_samples_leaf'] = 2
params['train'] = 'D:/数据集/paderborn-train/train_7.csv'
params['test'] = 'D:/数据集/paderborn-train/test_7.csv'
argvs = sys.argv
try:
    for i in range(len(argvs)):
        if i < 1:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])

    train = np.array(pd.read_csv(params['train']))
    train_y = train[:, -1]
    train_x = train[:, :-1]

    test = np.array(pd.read_csv(params['test']))
    test_y = test[:, -1]
    test_x = test[:, :-1]

    clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                             max_features=params['max_features'],
                             max_depth=params['max_depth'],
                             min_samples_split=params['min_samples_split'],
                             min_samples_leaf=params['min_samples_leaf']).fit(train_x, train_y)
    clf.fit(train_x,train_y)
    predict = clf.predict(test_x)
    
    print(predict)
    print(clf.predict_proba(test_x))
    precision = precision_score(test_y, predict,average='micro')
    recall = recall_score(test_y, predict,average='micro')
    accuracy = accuracy_score(test_y,predict,normalize=False)
    roc_area = roc_auc_score(test_y,clf.predict_proba(test_x),multi_class='ovo')
    print(precision)
    print(recall)
    print(accuracy)
    print(roc_area)
except Exception as e:
    print(e)