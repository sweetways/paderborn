# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 16:20:08 2021

@author: 12480
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import csv
import sys

params = {}
params['feature_range'] = (0,1)
params['path'] = 'D:/数据集/paderborn-train/train_7.csv'
params['opath'] ='D:/数据集/paderborn-train/ooo.csv'
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
    train = np.array(x)
    train_y = train[:,-1]
    train_y = np.array(train_y)
    x= x.drop("label",axis = 1)

    sel = VarianceThreshold(threshold=(.5))
    x1 = sel.fit_transform(train)
  #  out=np.column_stack((x1,train_y))

    csvfile2 = open(params['opath'],'w',newline='')
    writer = csv.writer(csvfile2)

    m = len(x1)

    for i in range(m):
        writer.writerow(x1[i])
except Exception as e:
    print(e)

