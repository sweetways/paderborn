# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:39:01 2021

@author: 12480
"""
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import json
from sklearn.tree import DecisionTreeClassifier
class Result:
    precision = 0
    recall = 0
    accuracy = 0
    rocX = []
    rocY = []
    featureImportances = []
params = {}
params['learning_rate'] = 1.
params['n_estimators'] = 50
params['base_estimator'] = 'DecisionTreeClassifier'
params['algorithm'] = 'SAMME'
params['random_state'] = None
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

    lf=DecisionTreeClassifier(criterion='gini',max_depth=None).fit(train_x, train_y)

    clf = AdaBoostClassifier(base_estimator=lf,
                         learning_rate=params['learning_rate'],
                         algorithm=params['algorithm'],
                         n_estimators=params['n_estimators'],
                         random_state=params['random_state']).fit(train_x, train_y)
    
    
    predict = clf.predict(test_x)
    predicta = clf.predict_proba(test_x)
    precision = precision_score(test_y, predict,average='micro')
    recall = recall_score(test_y, predict,average='micro')
    accuracy = accuracy_score(test_y, predict)
    f1 = f1_score(test_y, predict,average='micro')
    roc_area = roc_auc_score(test_y,predicta,multi_class='ovo')
    print(precision)
    print(recall)
    print(accuracy)
    print(f1)
    print(roc_area)
    
except Exception as e:
    print(e)