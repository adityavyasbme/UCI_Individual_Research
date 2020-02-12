#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:19:34 2019

@author: adityavyas
"""

import pandas as pd 
import global_functions as gf
import numpy as np
from scipy import stats

Xtr = gf.loadfile("Xtr")
Xte = gf.loadfile("Xte")
ytr = gf.loadfile("ytr")
yte = gf.loadfile("yte")

Xtr = gf.loadfile("Xtrain_updated")
ytr = gf.loadfile("Ytrain_updated")

Xtr[np.isnan(Xtr)] = 0
Xte[np.isnan(Xte)] = 0

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


classifier1 = RandomForestClassifier(n_estimators = 800, criterion = 'entropy', random_state = 0)
classifier1.fit(Xtr, ytr)
  

classifier2 = XGBClassifier()
classifier2.fit(Xtr, ytr)


classifier3 = DecisionTreeClassifier(criterion = 'entropy', random_state = 100,max_depth=1000,
                                        min_samples_leaf=60,min_impurity_decrease=0.0009)
classifier3.fit(Xtr, ytr)


pred1 = classifier1.predict_proba(Xte)[:,1] #Random Forest 
pred2 = classifier2.predict_proba(Xte)[:,1] #XGBOOST
pred3 = classifier3.predict_proba(Xte)[:,1] #Decision Tree Classifier


final_pred = np.array([])
for i in range(0,len(Xte)):
    final_pred = np.append(final_pred, stats.mode([pred1[i], pred2[i], pred3[i]])[0])     


final_pred = np.array([])
for i in range(0,len(Xte)):
    final_pred = np.append(final_pred, np.mean([pred1[i], pred2[i], pred3[i]]))     



final_pred = np.array([])
for i in range(0,len(Xte)):
    t = (pred1[i]*0.3)+(pred3[i]*0.3)+(pred2[i]*0.4)
    final_pred = np.append(final_pred, t )     
    
    
    
import sklearn.metrics as metrics    
fpr, tpr, threshold = metrics.roc_curve(yte, final_pred)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)

    
    
