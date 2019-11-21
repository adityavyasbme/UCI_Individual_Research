#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:30:48 2019

@author: adityavyas
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle

def get_dataset():
    
    # Importing the dataset
    dataset = pd.read_csv('data/TCGA_data.csv')
    
    #DATA Cleaning
    dataset= dataset.drop(columns="Unnamed: 0") #CLEANING
    dataset= dataset.drop(columns="sample_barcode") #CLEANING
    
    dataset.dropna(subset=['project_name'], how='all', inplace = True)
    
    dataset.drop(dataset.loc[dataset['gender']=="MALE"].index, inplace=True)
    
    dataset["project_name"] = np.where(dataset["project_name"]=="TCGA-BRCA",1,0)
    return dataset

#    dataset["project_name"].describe()

def divide(dataset):
     #Dividing the dataset
    X = dataset.iloc[:, 2:]
    y = dataset.iloc[:, 0]
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
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
    
#    # method I: plt
#    import matplotlib.pyplot as plt
#    plt.title('Receiver Operating Characteristic')
#    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
#    plt.legend(loc = 'lower right')
#    plt.plot([0, 1], [0, 1],'r--')
#    plt.xlim([0, 1])
#    plt.ylim([0, 1])
#    plt.ylabel('True Positive Rate')
#    plt.xlabel('False Positive Rate')
#    plt.show()
    
    return {"accuracy":accuracies.mean(),"std":accuracies.std(),"AUC":roc_auc}
 
#FROM ITERATION 3   
def add_gaussian_index(df,index):
    import math
    mean= df.mean()[index]
    std = df.std()[index]
    var = std**2
    denom = var*((2**math.pi)**0.5)
    new = df.iloc[:,index]
    new = new.fillna(0)

    new=new.apply(lambda x: ((2.7182**(-((x-mean)**2)/(2*var)))/denom))
    df.iloc[:,index]=new
    return df


#FROM ITERATION 4
    
"""
Input : 
    takes array of values
Output : 
    plot the function
"""
def plot(s):
    count, bins, ignored = plt.hist(s, 30, normed=True)
    plt.show()

"""
Input : 
    mu - mean
    sigma - standard deviation
    vals - how many values you want in that range
Output :
    return normally distributed sample
"""

def gaussiansample(mu,sigma,vals): 
    s = np.random.normal(mu, sigma, 1000)
    return s

def choose_random(lis):
    return random.choice(lis)

def augment_row_byvalue_single(data,ignore_index,times,std,sample_size,add_initial=False):
    a = data.copy()
    i = 0
    ans=0
    for i in range(len(data.columns)):
        if i>ignore_index:
            if not np.isnan(data.iat[0,i]):
                ans+=1
                temp2 = 0 
                s = gaussiansample(data.iat[0,i],std,sample_size)
                while temp2<times:
                    temp = a.copy()
                    val = choose_random(s)
                    temp.iat[0,i]=val
                    data=data.append(temp)
                    temp2+=1
#                break
    
    print("Found {} values in the row and augmented data {} times".format(ans,ans*times))
    #Include the first row i.e. the original data            
    if add_initial:
        return data
    else:
        return data.iloc[1:]    

def augment_row_byvalue_all(data,ignore_index,times,std,sample_size,add_initial=False):
    a = data.copy()
    i = 0
    ans=0
    temp = a.copy()
    for i in range(len(data.columns)):
        if i>ignore_index:
            if not np.isnan(data.iat[0,i]):
                ans+=1
                s = gaussiansample(data.iat[0,i],std,sample_size)
                val = choose_random(s)
                temp.iat[0,i]=val
    data=data.append(temp)

    print("Found {} values in the row and augmented {} values".format(ans,ans*times))
    #Include the first row i.e. the original data            
    if add_initial:
        return data
    else:
        return data.iloc[1:]    
    
def augment_column_byvalue_single(data,col_index,times,std,sample_size,add_initial=False):
    new = pd.DataFrame()
    if add_initial:
        new = new.append(data,ignore_index=True)
    if not np.isnan(data.iat[0,col_index]):
        s = gaussiansample(data.iat[0,col_index],std,sample_size)
        for i in range(times):
            temp = data.copy()
            val = choose_random(s)
            temp.iat[0,col_index]=val
            new = new.append(temp,ignore_index=True)
    return new    

#Return top n indexs based on value
def get_top_n(lis,num):
    return sorted(range(len(lis)), key=lambda i: lis[i], reverse=True)[:num]

#Return all the subsets of list except the blank one and itself
def subsets(lis):
    from itertools import chain, combinations
    def all_subsets(ss):
        return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))
    comb = []
    for subset in all_subsets(lis):
        if len(subset)!=0 and len(subset)!=1:
            comb.append(subset)
    return comb

#Takes tuple of string and convert them into list
def convertstrtolist(a):
    a=a.replace("(","")
    a=a.replace(")","")
    a=a.split(",")
    lis=[]
    for i in a:
        lis.append(int(i))
    return lis
    

#DUMPING THE DATA INTO DATA FOLDER
def savefile(data,name):
    string = "data/"+name
    # open a file, where you ant to store the data
    file = open(string, 'wb')
    # dump information to that file
    pickle.dump(data, file)
    # close the file
    file.close()
    print("Success")
    
def loadfile(name):
    string = 'data/'+name
    try:
        # open a file, where you stored the pickled data
        file = open(string, 'rb')
        # dump information to that file
        data = pickle.load(file)
        # close the file
        file.close()
        return data
    except:
        print("File not found")


    
