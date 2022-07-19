# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 18:36:05 2022

@author: Shiva Sheelvanth
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
import warnings
warnings.filterwarnings

df1=pd.read_csv('SalaryData_Train.csv')
df1
df1.head()
df1.shape

df2=pd.read_csv('SalaryData_Test.csv')
df2
df2.head()
df1.shape

df1.info()
df2.info()

df1.describe()
df2.describe()

#Finding special characters
df1.isin(['?']).sum(axis=0)
df2.isin(['?']).sum(axis=0)

print(df1[0:5])

#Explore categorical variables
categorical = [var for var in df1.columns if df1[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

df1[categorical].head()

# check missing values in categorical variables
df1[categorical].isnull().sum()

# view frequency counts of values in categorical variables

for var in categorical: 
    
    print(df1[var].value_counts())

# view frequency distribution of categorical variables

for var in categorical: 
    
    print(df1[var].value_counts()/np.float(len(df1)))


# check labels in workclass variable

df1.workclass.unique()

# check labels in occupation variable

df1.occupation.unique()

# check frequency distribution of values in occupation variable

df1.occupation.value_counts()

# check labels in native_country variable

df1.native.unique()

# check frequency distribution of values in native_country variable

df1.native.value_counts()

# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df1[var].unique()), ' labels')

numerical = [var for var in df1.columns if df1[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)

df1[numerical].head()

#checking missing values
df1[numerical].isnull().sum()

X = df1.drop(['Salary'], axis=1)

Y = df1['Salary']

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# check the shape of X_train and X_test

X_train.shape, X_test.shape

X_train.dtypes
X_test.dtypes

# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical

# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()

# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))

for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native'].fillna(X_train['native'].mode()[0], inplace=True)  

X_train[categorical].isnull().sum()
X_test[categorical].isnull().sum()


X_train.isnull().sum()
X_test.isnull().sum()

# print categorical variables

categorical
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train[categorical].head()

# import category encoders

import category_encoders as ce

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 
                                 'race', 'sex', 'native'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

X_train.head()
X_test.head()

cols = X_train.columns

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB

# instantiate the model
gnb = GaussianNB()
# fit the model
gnb.fit(X_train, Y_train)

Y_pred = gnb.predict(X_test)

Y_pred

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(Y_test, Y_pred)))

Y_pred_train = gnb.predict(X_train)

Y_pred_train

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(Y_train, Y_pred_train)))

# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train, Y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, Y_test)))

Y_test.value_counts()

null_accuracy = (7407/(7407+2362))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

# print precision score

precision = TP / float(TP + FP)

print('Precision : {0:0.4f}'.format(precision))

#Sensitivity
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))

#TPR
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

#FPR
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

#Specificity
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))


Y_pred_prob = gnb.predict_proba(X_test)[0:10]
Y_pred_prob

Y_pred_prob_df = pd.DataFrame(data=Y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])
Y_pred_prob_df

gnb.predict_proba(X_test)[0:10, 1]

Y_pred1 = gnb.predict_proba(X_test)[:, 1]

plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(Y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of salaries >50K')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(Y_test, Y_pred1, pos_label = '>50K')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Salaries')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(Y_test, Y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, Y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb, X_train, Y_train, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))

print('Average cross-validation score: {:.4f}'.format(scores.mean()))















