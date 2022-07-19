# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:58:47 2022

@author: Shiva Sheelvanth
"""

import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv('bank-full.csv')

df.columns

#selecting the columns
columns=['age' , 'balance' , 'duration' , 'campaign' , 'y']
df2=df[columns]

pd.crosstab(df2.age, df2.y).plot(kind="line")
# the graph shows that people between 20 to 60 age group have more anumber of applications rejected and people between age group 60 to 90 have applications rejected daily

sns.boxplot(data=df2, orient = "v") 

df2['outcome'] = df2.y.map({'no':0, 'yes':1})
df2.tail(5)

df2.boxplot(column='age', by='outcome')

feature_col=['age','balance','duration','campaign']
output_target=['outcome']

X = df2[feature_col]
Y = df2[output_target]

from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(X,Y)

model.coef_
model.predict_proba(X)

Y_pred=model.predict(X)
Y_pred

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,Y_pred)
confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')






