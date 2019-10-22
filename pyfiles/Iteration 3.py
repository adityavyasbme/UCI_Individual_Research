#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:33:48 2019

@author: adityavyas
"""

import global_functions as gf

import pandas as pd 

[X_train, X_test, y_train, y_test] = gf.get_dataset()

def run150(df):
    for index in range(150):
        df=gf.add_gaussian_index(df,index)
    return df
    


data = pd.DataFrame()

runnable = X_train.copy()
runnabletrain = y_train.copy()

for i in range(1,10):
    df = X_train.copy()
    for j in range(i):    
        df = run150(df)
    runnable=pd.concat([runnable,df])
    runnabletrain = pd.concat([runnabletrain,y_train.copy()])
    frame = gf.run_xgboost(runnable,runnabletrain,X_test,y_test)
    data=data.append(frame,ignore_index=True)
    print(frame)
    
    
    
        
        