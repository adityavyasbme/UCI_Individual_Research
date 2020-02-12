#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:18:58 2020

@author: adityavyas
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_dataset():
    dataset = pd.read_csv('trainingdataBC_UKB.csv')
    
    #DATA Cleaning
    dataset= dataset.drop(columns="Unnamed: 0") #CLEANING
    dataset= dataset.drop(columns="PatientIDs") #CLEANING
    #dataset['sex'].describe()
    dataset= dataset.drop(columns="sex") #removing sex as all are females
    
    dataset["label"] = np.where(dataset["label"]=="Normal",0,1)
    
    dataset['cancer_selfreported'].unique()
    
    return dataset

    
def divide(dataset):
     #Dividing the dataset
    X = dataset.iloc[:, 5:]
    y = dataset.iloc[:, 0]
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1000)
    
    return [X_train, X_test, y_train, y_test]

def run_xgboost(X_train,y_train,X_test,y_test):
    # Fitting XGBoost to the Training set
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
#    # Making the Confusion Matrix
#    from sklearn.metrics import confusion_matrix
#    cm = confusion_matrix(y_test, y_pred)
    
    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#    accuracies.mean()
#    accuracies.std()
    
    
    import sklearn.metrics as metrics
    # calculate the fpr and tpr for all thresholds of the classification
    probs =  classifier.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    return {"accuracy":accuracies.mean(),"std":accuracies.std(),"AUC":roc_auc}

def gaussiansample(mu,sigma,vals): 
    s = np.random.normal(mu, sigma, 1000)
    return s

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

#run_xgboost(X_train,y_train,X_test,y_test)
