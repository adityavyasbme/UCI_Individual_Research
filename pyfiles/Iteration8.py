#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:42:38 2019

@author: adityavyas
"""






import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vecstack import stacking
import global_functions as gf    
import sklearn.metrics as metrics   

Xtr = gf.loadfile("Xtr")
Xte = gf.loadfile("Xte")
ytr = gf.loadfile("ytr")
yte = gf.loadfile("yte")
#
#%%
Xtr = gf.loadfile("Xtrain_updated")
ytr = gf.loadfile("Ytrain_updated")
#%%
Xtr[np.isnan(Xtr)] = 0
Xte[np.isnan(Xte)] = 0


#%%
#STACKING
models = [    RandomForestClassifier(random_state=50, n_jobs=-1, 
                           n_estimators=800)
,
        
    RandomForestClassifier(criterion = 'entropy', n_jobs=-1,random_state = 100,max_depth=1000,
                                        min_samples_leaf=60,min_impurity_decrease=0.0009),
        
    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
                  n_estimators=100, max_depth=3)
]

S_train, S_test = stacking(models,                   
                           Xtr, ytr, Xte,   
                           regression=False, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=accuracy_score, 
    
                           n_folds=10, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)

model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
                      n_estimators=100, max_depth=3)
    
model = model.fit(S_train, ytr)


y_pred = model.predict_proba(S_test)[:,1]

fpr, tpr, threshold = metrics.roc_curve(yte, y_pred)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)

#%%
#BAGGING

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1),n_jobs=-1)
model.fit(Xtr, ytr)
print(model.score(Xte,yte))
    
    
y_pred = model.predict_proba(Xte)[:,1]

fpr, tpr, threshold = metrics.roc_curve(yte, y_pred)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)

#%%
#ADABOOST

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(Xtr, ytr)
print(model.score(Xte,yte))
    
    
y_pred = model.predict_proba(Xte)[:,1]

fpr, tpr, threshold = metrics.roc_curve(yte, y_pred)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)

#%%

from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)

model.fit(Xtr, ytr)
print(model.score(Xte,yte))
    
    
y_pred = model.predict_proba(Xte)[:,1]

fpr, tpr, threshold = metrics.roc_curve(yte, y_pred)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)


#%%

import lightgbm as lgb
train_data=lgb.Dataset(Xtr,label=ytr)
#define parameters
params = {'learning_rate':0.001}
model= lgb.train(params, train_data, 100) 

y_pred = model.predict_proba(Xte)[:,1]

fpr, tpr, threshold = metrics.roc_curve(yte, y_pred)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)









