# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:00:17 2022

@author: Shiva Sheelvanth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('SalaryData_Train(1).csv')
train

test = pd.read_csv('SalaryData_Test(1).csv')
test

train.head()
test.head()

train.info()
test.info()

train.shape
test.shape

train.describe()
test.describe()

train.corr()
test.corr()

# Data Preprocessing
lb = LabelEncoder()

train["workclass"] = lb.fit_transform(train["workclass"])
train["education"] = lb.fit_transform(train["education"])
train["maritalstatus"] = lb.fit_transform(train["maritalstatus"])
train["occupation"] = lb.fit_transform(train["occupation"])
train["relationship"] = lb.fit_transform(train["relationship"])
train["race"] = lb.fit_transform(train["race"])
train["sex"] = lb.fit_transform(train["sex"])
train["native"] = lb.fit_transform(train["native"])
train["Salary"] = lb.fit_transform(train["Salary"])

test["workclass"] = lb.fit_transform(test["workclass"])
test["education"] = lb.fit_transform(test["education"])
test["maritalstatus"] = lb.fit_transform(test["maritalstatus"])
test["occupation"] = lb.fit_transform(test["occupation"])
test["relationship"] = lb.fit_transform(test["relationship"])
test["race"] = lb.fit_transform(test["race"])
test["sex"] = lb.fit_transform(test["sex"])
test["native"] = lb.fit_transform(test["native"])
test["Salary"] = lb.fit_transform(test["Salary"])

# EDA
train = train.iloc[: 2000, :]
train.info()

test = test.iloc[: 1300, :]
test.info()

for i in train.describe().columns[:-2]:
    train.plot.scatter(i,'workclass',grid=True)

for i in train.describe().columns[:-2]:
    train.plot.scatter(i,'occupation',grid=True)

for i in train.describe().columns[:-2]:
    train.plot.scatter(i,'education',grid=True)

train.corr()
test.corr()

sns.pairplot(test)

# Data Splitting
X_train=train.iloc[:,:-1]
X_train

y_train=train.iloc[:,-1]
y_train

X_test=test.iloc[:,:-1]
X_test

y_test = test.iloc[:,-1]
y_test

X_train.shape, y_train.shape, X_test.shape, y_test.shape

#SVM Model
model = SVC()

model.fit(X_train, y_train)

# Predicting model
y_pred = model.predict(X_test)
y_pred

#Model Evaluation
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

'''
Model is not predicting well, so we will improve the model by hyperparameter tunning using grid search method.
'''

#Improving Model using Grid Search CV
param_grid = {'C' : [1, 5, 10, 15, 20], 'gamma' : [1, 0.1, 0.01, 0.001, 0.0001], 'kernel' : ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,  cv = 5)
grid.fit(X_train, y_train)

grid.best_params_

grid_pred = grid.predict(X_test)

#Model Evaluation
grid_pred

print(confusion_matrix(y_test, grid_pred))

print(classification_report(y_test, grid_pred))

'''
Using grid search method,model accuracy improved from 0.79 to 0.83

'''


















