#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:02:54 2019

@author: adityavyas
"""

import pandas as pd 
import global_functions as gf
import numpy as np

Xtr = gf.loadfile("Xtr")
Xte = gf.loadfile("Xte")
ytr = gf.loadfile("ytr")
yte = gf.loadfile("yte")

Xtr[np.isnan(Xtr)] = 0
Xte[np.isnan(Xte)] = 0

Xtr = gf.loadfile("Xtrain_updated")
ytr = gf.loadfile("Ytrain_updated")

#%%

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

def randomforestclassifier(Xtr,ytr,Xte,yte,n):
    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = n, criterion = 'entropy', random_state = 0)
    classifier.fit(Xtr, ytr)
    y_pred = classifier.predict(Xte)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte, y_pred)

    #CALCULATING AUC
    import sklearn.metrics as metrics
    p = classifier.predict_proba(Xte)
    preds = p[:,1]
    fpr, tpr, threshold = metrics.roc_curve(yte, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return {"Accuracy":accuracy(cm),"AUC":roc_auc}


print(randomforestclassifier(Xtr,ytr,Xte,yte,n=800))

#%%

def naivebayesclassifier(Xtr,ytr,Xte,yte):
    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(Xtr, ytr)
    
    # Predicting the Test set results
    y_pred = classifier.predict(Xte)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte, y_pred)
    
    #CALCULATING AUC
    import sklearn.metrics as metrics
    p = classifier.predict_proba(Xte)
    preds = p[:,1]
    fpr, tpr, threshold = metrics.roc_curve(yte, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return {"Accuracy":accuracy(cm),"AUC":roc_auc}

    
print(naivebayesclassifier(Xtr,ytr,Xte,yte))

#%%

def decisiontreeclassifier(Xtr,ytr,Xte,yte):
    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 100,max_depth=1000,
                                        min_samples_leaf=60,min_impurity_decrease=0.0009)
    classifier.fit(Xtr, ytr)
    
    # Predicting the Test set results
    y_pred = classifier.predict(Xte)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte, y_pred)
    
    #CALCULATING AUC
    import sklearn.metrics as metrics
    p = classifier.predict_proba(Xte)
    preds = p[:,1]
    fpr, tpr, threshold = metrics.roc_curve(yte, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return {"Accuracy":accuracy(cm),"AUC":roc_auc}

print(decisiontreeclassifier(Xtr,ytr,Xte,yte))    

#%%

def knnclassifier(Xtr,ytr,Xte,yte):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(Xtr, ytr)
    
    # Predicting the Test set results
    y_pred = classifier.predict(Xte)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte, y_pred)
    
    #CALCULATING AUC
    import sklearn.metrics as metrics
    p = classifier.predict_proba(Xte)
    preds = p[:,1]
    fpr, tpr, threshold = metrics.roc_curve(yte, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return {"Accuracy":accuracy(cm),"AUC":roc_auc}

print(knnclassifier(Xtr,ytr,Xte,yte))    

    