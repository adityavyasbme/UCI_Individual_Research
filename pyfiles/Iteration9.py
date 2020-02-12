#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:18:13 2019

@author: adityavyas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import global_functions as gf

def get_dataset():
    
    # Importing the dataset
    dataset = pd.read_csv('data/TCGA_data.csv')
    
    #DATA Cleaning
    dataset= dataset.drop(columns="Unnamed: 0") #CLEANING
    dataset= dataset.drop(columns="sample_barcode") #CLEANING
    
    dataset.dropna(subset=['project_name'], how='all', inplace = True)
    
#    dataset.drop(dataset.loc[dataset['gender']=="MALE"].index, inplace=True)
    
    dataset["project_name"] = np.where(dataset["project_name"]=="TCGA-HNSC",1,0)
    dataset["gender"] = np.where(dataset["gender"]=="MALE",1,0)
    dataset[np.isnan(dataset)] = 0

    return dataset

def divide(dataset):
     #Dividing the dataset
    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
    
    return [X_train, X_test, y_train, y_test]

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

dataset = get_dataset()
[X_train, X_test, y_train, y_test] = divide(dataset)


print(gf.run_xgboost(X_train,y_train,X_test,y_test))

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 800, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
final_pred = classifier.predict_proba(X_test)[:,1]
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Accuracy {}".format(accuracy(cm)))
import sklearn.metrics as metrics    
fpr, tpr, threshold = metrics.roc_curve(y_test, final_pred)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
final_pred = classifier.predict_proba(X_test)[:,1]
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Accuracy {}".format(accuracy(cm)))
import sklearn.metrics as metrics    
fpr, tpr, threshold = metrics.roc_curve(y_test, final_pred)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 100,max_depth=1000,
                                    min_samples_leaf=60,min_impurity_decrease=0.0009)
classifier.fit(X_train, y_train)
final_pred = classifier.predict_proba(X_test)[:,1]
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Accuracy {}".format(accuracy(cm)))
import sklearn.metrics as metrics    
fpr, tpr, threshold = metrics.roc_curve(y_test, final_pred)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
final_pred = classifier.predict_proba(X_test)[:,1]
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Accuracy {}".format(accuracy(cm)))
import sklearn.metrics as metrics    
fpr, tpr, threshold = metrics.roc_curve(y_test, final_pred)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)



   
