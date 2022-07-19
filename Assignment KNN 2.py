# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:22:51 2022

@author: Shiva Sheelvanth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

df=pd.read_csv('glass.csv')
df
df.head()

df.tail()

df.Type.value_counts()

#Data exploration and visualizaion
# correlation matrix 
cor = df.corr()
sns.heatmap(cor)       #we observe Ca and K values don't affect Type that much. and using RI is enough

# Scatter plot of two features, and pairwise plot
sns.scatterplot(df['RI'],df['Na'],hue=df['Type'])
# From the above plot, We first calculate the nearest neighbors from the new data point to be calculated.

#pairwise plot of all the features
sns.pairplot(df,hue='Type')
plt.show()
#The pairplot shows that the data is not linear and KNN can be applied to get nearest neighbors and classify the glass types

scaler = StandardScaler()
scaler.fit(df.drop('Type',axis=1))

StandardScaler(copy=True, with_mean=True, with_std=True)

#perform transformation
scaled_features = scaler.transform(df.drop('Type',axis=1))
scaled_features

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

#Applying KNN
dff = df_feat.drop(['Ca','K'],axis=1) #Removing features - Ca and K 

X_train,X_test,y_train,y_test  = train_test_split(dff,df['Type'],test_size=0.3,random_state=45) 
#setting random state ensures split is same eveytime, so that the results are comparable

knn = KNeighborsClassifier(n_neighbors=4,metric='manhattan')
knn.fit(X_train,y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=4, p=2,
                     weights='uniform')

y_pred = knn.predict(X_test)
print(classification_report(y_test,y_pred))

accuracy_score(y_test,y_pred)        #With this setup, We found the accuracy to be 73.84%

k_range = range(1,25)
k_scores = []
error_rate =[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #kscores - accuracy
    scores = cross_val_score(knn,dff,df['Type'],cv=5,scoring='accuracy')
    k_scores.append(scores.mean())
    
    #error rate
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred!=y_test))

#plot k vs accuracy
plt.plot(k_range,k_scores)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Cross validated accuracy score')
plt.show()

#plot k vs error rate
plt.plot(k_range,error_rate)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Error rate')
plt.show()

# we can see that k=4 produces the most accurate results













