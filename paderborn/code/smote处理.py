# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:19:09 2021

@author: 12480
"""
from imblearn.over_sampling import SMOTE
import csv
import pandas as pd
import sys
import numpy as np
from collections import Counter

params = {}
params['feature_range'] = (0,1)
params['path'] = 'D:/数据集/paderborn-train/traindata_N15_M01_F10.csv'
params['opath'] ='D:/数据集/paderborn-train/train_6.csv'
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
   
        
        
        
        
       
    with open(params['path'],'r') as f:
    #1.创建阅读器对象
        reader = csv.reader(f)
    #2.读取文件第一行数据
        head_row=next(reader)
    data_attribute = []
    for item in head_row:
        data_attribute.append(item)
    
    x = pd.read_csv(params['path']) 
    y = x.loc[:,'label']
    
    print(Counter(y))
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_resample(x, y)
    print(Counter(y_smo))
    out = X_smo.sort_values(by='label',ascending = True)
    result = pd.DataFrame(out,columns = data_attribute)
    result.to_csv(params['opath'], sep=',', header=True, index=False)
    
    
except Exception as e:
    print(e)
    
    
