# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 21:30:00 2021

@author: 12480
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 21:16:39 2021

@author: 12480
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import csv
import sys

params = {}
params['feature_range'] = (0,1)
params['path'] = 'D:/数据集/111/train4.csv'
params['opath'] ='D:/数据集/111/train5.csv'
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
    
    
    df = pd.read_csv(params['path'])
    delete_features = ['freq_rms','time_var'] #需要删除的列名自行加到数组里
    df = df.drop(delete_features, axis=1) #特征选择之后的数据
    df.to_csv(params['opath'],index=False)
except Exception as e:
    print(e)
