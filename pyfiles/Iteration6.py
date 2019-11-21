#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:05:58 2019

@author: adityavyas
"""

import pandas as pd 
import global_functions as gf

data = pd.read_csv("data/data3.csv",header=0)


print(gf.get_top_n(data["accuracy"],1))
print("Final combination of data is : ")
print(data["combination"].iat[108])
temp = data["combination"].iat[108] 

#COnvert the string tuple into list
temp = gf.convertstrtolist(temp)

#dataset = gf.get_dataset()
#[X_train, X_test, y_train, y_test] = gf.divide(dataset)

##SAVING THE DATA FOR FUTURE USE
#gf.savefile(X_train,"Xtr")
#gf.savefile(X_test,"Xte")
#gf.savefile(y_train,"ytr")
#gf.savefile(y_test,"yte")


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

Xtr = gf.loadfile("Xtr")
Xte = gf.loadfile("Xte")
ytr = gf.loadfile("ytr")
yte = gf.loadfile("yte")
temp = gf.convertstrtolist(temp)

Xt,yt = Xtr.copy(),ytr.copy()
print("Currently augmenting")
for j in temp:
    Xtr1,ytr1=augmenter(Xtr.copy(),ytr.copy(),sd=0.1,col=j) #creating augmented values
    Xt = Xt.append(Xtr1)
    yt = yt.append(pd.Series(ytr1))
print("Augmenting Finished")

d = gf.run_xgboost(Xt,yt,Xte,yte)
print(d)

"""
{'accuracy': 0.882377508399508, 'std': 0.012872278905955601, 'AUC': 0.7476636185875316}
"""

#%%

data  = pd.DataFrame()
for i in range(150):
    print(i)
    Xt1,yt1= augmenter(Xt.copy(),yt.copy(),0.2,i)
    d = gf.run_xgboost(Xt1,yt1,Xte,yte)
    data= data.append(d,ignore_index=True)
    
data.to_csv("data/iteration6.csv")

#%%


gf.savefile(Xt,"Xtrain_updated")
gf.savefile(yt,"Ytrain_updated")


