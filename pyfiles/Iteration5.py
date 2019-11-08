#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:28:23 2019

@author: adityavyas
"""

import pandas as pd 
import global_functions as gf
import itertools

dataset = gf.get_dataset()
[X_train, X_test, y_train, y_test] = gf.divide(dataset)

data = pd.read_excel("Observation/Iteration45.xlsx",header=0)
data["accuracy"].describe()

print(gf.get_top_n(data["accuracy"],3))


def augmenter(X_train,y_train,sd,col):

    dataset = X_train.copy()
    length= len(dataset)
    new = pd.DataFrame()

    for i in range(length):
        row = dataset.iloc[[i]]
        row.insert(0, 'y',y_train.iloc[i])
        ans = gf.augment_column_byvalue_single(row,col_index=col+1,times=5,std=sd,sample_size=1000,add_initial=False)
        new=new.append(ans)
    X_train = new.iloc[:,1:]
    y_train = new.iloc[:,0:1].values.ravel()
    return X_train,y_train

#a,b=augmenter(X_train,y_train,sd=0.1,col=1)



def get_index_combo(data,num):
    lis=gf.get_top_n(data["accuracy"],num)
    temp = gf.subsets(lis)
    return temp

index_list = get_index_combo(data,7)


data1  = pd.DataFrame()
for i in index_list:
    Xt,yt = X_train.copy(),y_train.copy()
    print("Currently augmenting")
    print(i)
    for j in i:
        Xtr,ytr=augmenter(X_train.copy(),y_train.copy(),sd=0.1,col=j) #creating augmented values
        Xt = Xt.append(Xtr)
        yt = yt.append(pd.Series(ytr))
    d = gf.run_xgboost(Xt,yt,X_test,y_test)
    d["combination"]=i
    data1= data1.append(d,ignore_index=True)
    print(d["accuracy"])
    print("-------")

data1.to_csv("data/data3.csv")

        


    
