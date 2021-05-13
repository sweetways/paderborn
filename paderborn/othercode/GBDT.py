# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:17:06 2021

@author: 12480
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:39:01 2021

@author: 12480
"""
from sklearn.ensemble import GradientBoostingClassifier
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
params['n_estimators'] = 100
params['learning_rate'] = 1
params['subsample'] = 1.0
params['loss'] = 'deviance'
params['max_depth'] = 3
params['min_samples_split'] = 2
params['min_samples_leaf'] = 1
params['min_weight_fraction_leaf'] = 0
params['max_features'] = None
params['max_leaf_nodes'] = None
params['min_impurity_split'] = None
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

    clf = GradientBoostingClassifier(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'],subsample=params['subsample'],loss=params['loss'],max_depth=params['max_depth'],min_samples_split=params['min_samples_split'],min_weight_fraction_leaf=params['min_weight_fraction_leaf'],min_samples_leaf=params['min_samples_leaf'],max_features=params['max_features'],max_leaf_nodes=params['max_leaf_nodes'],min_impurity_split=params['min_impurity_split']).fit(train_x, train_y)
    
    
    predict = clf.predict(test_x)
    
    precision = precision_score(test_y, predict,average='micro')
    recall = recall_score(test_y, predict,average='micro')
    accuracy = accuracy_score(test_y, predict)
    f1 = f1_score(test_y, predict,average='micro')
    
    print(precision)
    print(recall)
    print(accuracy)
    print(f1)
    
    
except Exception as e:
    print(e)