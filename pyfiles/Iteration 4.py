#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:49:41 2019

@author: adityavyas
"""
import global_functions as gf
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('data/TCGA_data.csv')

#DATA Cleaning
dataset= dataset.drop(columns="Unnamed: 0") #CLEANING
dataset= dataset.drop(columns="sample_barcode") #CLEANING

dataset.dropna(subset=['project_name'], how='all', inplace = True)

dataset.drop(dataset.loc[dataset['gender']=="MALE"].index, inplace=True)

dataset["project_name"] = np.where(dataset["project_name"]=="TCGA-BRCA",1,0)

dataset["project_name"].describe()

#%%

 #Dividing the dataset
X = dataset.iloc[:, 2:]
y = dataset.iloc[:, 0]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#%%

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

#%%
#s = gaussiansample(1.5,0.1,100)
#plot(s)
#print(choose_random(s))    
a = dataset.iloc[[0]]    

    
#%%
    
def augment(data,ignore_index,times,std,sample_size,add_initial=False):
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


ans = augment(a,ignore_index=1,times=1,std=0.1,sample_size=1000,add_initial=False)   
 
#%%

#dataset = gf.get_dataset()
dataset = X_train.copy()
length= len(dataset)
new = pd.DataFrame()

for i in range(length):
    row = dataset.iloc[[i]]
    row.insert(0, 'y',y_train.iloc[i])
    ans = augment(row,ignore_index=0,times=1,std=0.1,sample_size=1000,add_initial=False)
    new=new.append(ans)
#%%
dataset= dataset.append(new)
[X_train, X_test, y_train, y_test] = gf.divide(dataset)
print(gf.run_xgboost(X_train,y_train,X_test,y_test))
