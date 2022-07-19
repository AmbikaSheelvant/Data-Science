# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:31:22 2022

@author: Shiva Sheelvanth
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('forestfires.csv')
data

data.info()
data.corr()
data.shape
data.describe()

#Plots
plt.figure(figsize=(10,6))
sns.countplot(x = 'month', hue = 'size_category', data = data, palette='Set1')

plt.figure(figsize=(10,6))
sns.countplot(x = 'day', hue = 'size_category', data = data, palette='Set1')

'''
The forest caught fire mostly on sunday and friday.
Most of the times the count of small burn area of forest is almost twice that of large burn area of forest
'''

# Data Preprocessing

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

data["month"] = label_encoder.fit_transform(data["month"])
data["day"] = label_encoder.fit_transform(data["day"])
data["size_category"] = label_encoder.fit_transform(data["size_category"])

for i in data.describe().columns[:-2]:
    data.plot.scatter(i,'area',grid=True)
    
X = data.iloc[:,:11]
X    

y=data["size_category"]
y

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# SVM Model
model = SVC()

model.fit(X_train, y_train)

# Predicting Model
y_pred = model.predict(X_test)
y_pred


# Model Evaluation
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))


'''Inference
Model is not predicting well, so we will improve the model by hyperparameter tunning using grid search method.
'''

# Improving Model using Grid Search CV
param_grid = {'C' : [0.1, 1, 10, 100, 1000], 'gamma' : [1, 0.1, 0.01, 0.001, 0.0001], 'kernel' : ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,  cv = 5)
grid.fit(X_train, y_train)

grid.best_params_

grid.best_estimator_

grid_pred = grid.predict(X_test)

# Evaluate Improved Model
grid_pred

print(confusion_matrix(y_test, grid_pred))

print(classification_report(y_test, grid_pred))

'''
Accuracy improved to 95% using Grid Search method
'''

