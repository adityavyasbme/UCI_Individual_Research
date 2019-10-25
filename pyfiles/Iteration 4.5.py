#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:03:00 2019

@author: adityavyas
"""

#IMPORTING LIBRARIES
import global_functions as gf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dataset = gf.get_dataset()
[X_train, X_test, y_train, y_test] = gf.divide(dataset)


def augmenter(X_train,y_train,sd,col):

    dataset = X_train.copy()
    length= len(dataset)
    new = pd.DataFrame()

    for i in range(length):
        row = dataset.iloc[[i]]
        row.insert(0, 'y',y_train.iloc[i])
        ans = gf.augment_column_byvalue_single(row,col_index=col,times=5,std=sd,sample_size=1000,add_initial=True)
        new=new.append(ans)
    X_train = new.iloc[:,1:]
    y_train = new.iloc[:,0:1].values.ravel()
    return X_train,y_train

#print(gf.run_xgboost(X_train,y_train,X_test,y_test))

data  = pd.DataFrame()
for i in range(150):
    print(i)
    Xt,yt= augmenter(X_train.copy(),y_train.copy(),0.2,i+1)
    d = gf.run_xgboost(Xt,yt,X_test,y_test)
    data= data.append(d,ignore_index=True)
    
data.to_csv("data/data2.csv")
    
    


