# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 08:56:50 2021

@author: 12480
"""
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import sys
import csv

#指定算法参数和备选参数值
params = {}
params['n_estimators'] = 100
params['max_samples'] ='auto'
params['contamination'] = 0.1
params['max_features'] = 1.0
params['path'] = 'D:/数据集/paderborn-train/traindata_N15_M07_F10.csv'
params['opath'] ='D:/数据集/paderborn-train/test_1.csv'
argvs = sys.argv

try:
    #用sys.argv读取命令行中传递过来的参数
    for i in range(len(argvs)):
        if i < 1:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])

    #1.创建阅读器对象
    with open(params['path'],'r') as f:
        reader = csv.reader(f)
    #2.读取文件第一行数据
        head_row=next(reader)
    data_attribute = []
    #3.保存属性名
    for item in head_row:
        data_attribute.append(item)

 #读取数据并删除最后一列标签
    tn = pd.read_csv(params['path']) 
    tn.dropna(inplace=True)
    train = np.array(tn)
    train_x = train[:, :-1]

 #存标签
    train_y = train[:,-1]
    train_y = np.array(train_y)
    train_x = np.array(train_x)

    #对所有数据行进行异常检测
    clf = IsolationForest(n_estimators=params['n_estimators'], 
                      max_samples=params['max_samples'], 
                      contamination=params['contamination'], 
                      max_features=params['max_features'], 
                      bootstrap=False, n_jobs=1, random_state=None, 
                      verbose=0).fit(train_x)

    #pred存入的是每一行数据的预测值，是1或者-1
    pred = clf.predict(train_x)
    normal = train_x[pred == 1]
    abnormal = train_x[pred == -1]

    #删除pred为-1的行数据
    df = pd.DataFrame(pd.read_csv(params['path']))[0:pred. 
         size]
    df['pred']=pred
    df2 = df[-df.pred.isin([-1])]
    df2 = df2.drop(['pred'],axis=1)

    #将清洗之后的数据存入opath指定位置的csv文件
    data_out = df2.iloc[:,:].values
    csvfile2 = open(params['opath'],'w',newline='')
    writer = csv.writer(csvfile2)
    writer.writerow(data_attribute)   #存属性
    m = len(data_out)
    print(m)
    for i in range(m):
        writer.writerow(data_out[i])
except Exception as e:
    print(e)