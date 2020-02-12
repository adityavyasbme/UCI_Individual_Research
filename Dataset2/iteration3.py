#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:43:40 2020

@author: adityavyas
"""


import global_functions2 as gf
import numpy as np
import random
import pandas as pd 

dataset = gf.get_dataset()
[X_train, X_test, y_train, y_test] = gf.divide(dataset)

row = X_train.iloc[0]
import matplotlib.pyplot as plt
plt.plot(row,'.')
plt.title('Normal Data')
plt.ylabel('Value')
plt.xlabel('Features')
plt.show()

def choose_random(lis):
    return random.choice(lis)

def gaussiansample(mu,sigma,vals): 
    s = np.random.normal(mu, sigma, 1000)
    return s

def augment_row_byvalue_all(data,ignore_index,times,std,sample_size,add_initial=False,verbose=0):
    a = data.copy()
    temp = a.copy()
    for _ in range(times):
        ans=0
        for i in range(len(data.columns)):
            if i>ignore_index:
                if not np.isnan(data.iat[0,i]):
                    ans+=1
                    s = gaussiansample(data.iat[0,i],std,sample_size)
                    val = choose_random(s)
                    temp.iat[0,i]=val
        data=data.append(temp)
    if verbose!=0:
        print("Found {} values in the row and augmented {} values".format(ans,ans*times))
    #Include the first row i.e. the original data            
    if add_initial:
        return data
    else:
        return data.iloc[1:]    


row2 = augment_row_byvalue_all(X_train.iloc[[0]],ignore_index=-1,times=1,std=0.001,sample_size=1000,add_initial=True)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(row2.iloc[0],'.' , label='original')
plt.legend(loc='upper left');
plt.title('Normal Data')
plt.ylabel('Value')
plt.xlabel('Features')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(row2.iloc[0],'.' , label='original')
ax1.plot(row2.iloc[1],'.', label='first')
plt.legend(loc='upper left');
plt.title('Augmented Data')
plt.ylabel('Value')
plt.xlabel('Features')
plt.show()

row = df.iloc[[2]]
row2 = augment_row_byvalue_all(row,ignore_index=0,times=1,std=0.001,sample_size=1000,add_initial=True)

def augment(X_train,y_train,tim):
    if len(X_train) != len(y_train):
        print("Check your data")
        return False
    df = pd.concat([y_train,X_train],axis=1, sort =False)
    new_data = pd.DataFrame()
    for i in range(len(df)):
        row = df.iloc[[i]]
        row2 = augment_row_byvalue_all(row,ignore_index=0,times=tim,std=0.001,sample_size=1000,add_initial=True)
        new_data = new_data.append(row2,ignore_index=True)
    print("Previous data length {} ".format(len(df)))
    print("New data length {} ".format(len(new_data)))
    return new_data
        
df = augment(X_train,y_train,tim=9)
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

"""
DATA AUGMENTATION COMPLETE.
"""    

from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

estimator = XGBClassifier(
    objective= 'binary:logistic',
    seed=42
)

parameters = {
    'max_depth': range (8, 15, 1),
    'n_estimators': range(160, 260, 40),
    'learning_rate': [0.1]
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 10,
    verbose=True
)
grid_search.fit(X, y)

grid_search.best_estimator_

def accuracy(y_test,x_test,cls):
    y_pred = cls.predict(x_test)
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    acc= diagonal_sum / sum_of_all_elements 

    #CALCULATING AUC
    import sklearn.metrics as metrics
    p = cls.predict_proba(x_test)
    preds = p[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return {"Accuracy":acc,"AUC":roc_auc}

accuracy(y_test,X_test,grid_search)
