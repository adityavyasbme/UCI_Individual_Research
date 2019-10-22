#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:56:17 2019

@author: adityavyas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/TCGA_data.csv')

#DATA Cleaning
dataset= dataset.drop(columns="Unnamed: 0") #CLEANING
dataset= dataset.drop(columns="sample_barcode") #CLEANING
dataset.dropna(subset=['project_name'], how='all', inplace = True)
dataset["gender"] = np.where(dataset["gender"]=="MALE",1,0) #Setting males to 1 and females to 1 
#Choosing TCGA-BRCA
dataset["project_name"] = np.where(dataset["project_name"]=="TCGA-BRCA",1,0)
#%%

#Dividing the dataset
X = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()


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

#%%
#Setting nan to zero

dataset = dataset.fillna(0)

#%%
#setting nan to mean

dataset = dataset.fillna(dataset.mean())

#%%

dataset = dataset.fillna(dataset.median())


