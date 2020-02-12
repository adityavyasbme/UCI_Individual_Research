#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 08:38:06 2020

@author: adityavyas
"""



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

print(randomforestclassifier(X_train,y_train,X_test,y_test,n=500))

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

    
print(naivebayesclassifier(X_train,y_train,X_test,y_test))
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

print(decisiontreeclassifier(X_train,y_train,X_test,y_test))    

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

print(knnclassifier(X_train,y_train,X_test,y_test))  

#%%

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1),n_jobs=-1)
model.fit(Xtr, ytr)
print(model.score(Xte,yte))

y_pred1 = model.predict(Xte)

y_pred = model.predict_proba(Xte)[:,1]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yte, y_pred1)

import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(yte, y_pred)
roc_auc = metrics.auc(fpr, tpr)
print({"Accuracy":accuracy(cm),"AUC":roc_auc})

#%%

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(Xtr, ytr)
print(model.score(Xte,yte))
    
y_pred1 = model.predict(Xte)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yte, y_pred1)
    
y_pred = model.predict_proba(Xte)[:,1]

fpr, tpr, threshold = metrics.roc_curve(yte, y_pred)
roc_auc = metrics.auc(fpr, tpr)
print({"Accuracy":accuracy(cm),"AUC":roc_auc})

#%%

from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)

model.fit(Xtr, ytr)
print(model.score(Xte,yte))

y_pred1 = model.predict(Xte)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yte, y_pred1)
    
y_pred = model.predict_proba(Xte)[:,1]

fpr, tpr, threshold = metrics.roc_curve(yte, y_pred)
roc_auc = metrics.auc(fpr, tpr)
print({"Accuracy":accuracy(cm),"AUC":roc_auc})

