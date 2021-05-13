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
from sklearn import svm
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
params['C'] = 1.0
params['gamma'] ='auto'
params['kernel'] ='rbf'
params['degree'] = 3
params['coef0'] = 0.0
params['tol'] = 0.001
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

    clf = svm.SVC(gamma=params['gamma'],
              C=params['C'],
              kernel=params['kernel'],
              degree=params['degree'],
              coef0=params['coef0'],
              tol=params['tol']).fit(train_x, train_y)
    
    
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