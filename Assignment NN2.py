# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:18:28 2022

@author: Shiva Sheelvanth
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as sns
%matplotlib inline

df=pd.read_csv('gas_turbines.csv')
df

df.head()
df.describe()
df.columns

df['TEY'].value_counts()
len(df['TEY'].unique())
df.isnull().sum()

df.info()
df.corr()

# Plots and Visualizations
import matplotlib.pyplot as plt
plt.scatter(x="CDP",y="TEY",data=df) #which is linerly co-related

plt.scatter(x="CDP",y="GTEP",data=df)#which is linerly co-related

plt.scatter(x="GTEP",y="TEY",data=df)#which is linerly co-related

plt.scatter(x="AP",y="AT",data=df)# negtively corelated

import seaborn as sns
sns.barplot(data=df)

sns.boxplot(data=df)

plt.hist(df)

plt.hist(df['TEY'])
plt.hist(df['AP'])
plt.hist(df['TIT'])

plt.hist(df['TAT'])

features = df.columns.tolist()
features.remove('TEY')
features

X=df.drop(columns =['TEY'])
y=df['TEY']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size =0.3,random_state =42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

n_features =X.shape[1]
n_features

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from numpy.random import seed
import tensorflow

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
import tensorflow
import tensorflow as tf
from tensorflow import keras

optmizer =RMSprop(0.03)#0.01 is a learning rate
model=keras.Sequential([
    keras.layers.Dense(10,input_dim =(n_features),activation='relu'),
    keras.layers.Dense(8,activation ='relu')
])
model.compile(optimizer =optmizer,loss= 'mean_squared_error',metrics=['accuracy'])

seed_value =42;
import random
tensorflow.random.set_seed(seed_value)
model.fit(X_train, y_train, epochs=5, batch_size=30, verbose = 1)

model.evaluate(X_test,y_test)

optmizer =RMSprop(0.5)#0.01 is a learning rate
model=keras.Sequential([
    keras.layers.Dense(10,input_dim =(n_features),activation='relu'),
    keras.layers.Dense(8,activation ='relu')
])
model.compile(optimizer =optmizer,loss= 'mean_squared_error',metrics=['accuracy'])

seed_value =42;
import random
tensorflow.random.set_seed(seed_value)
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose = 1)

model.evaluate(X_test,y_test)
































