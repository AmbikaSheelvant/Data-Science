# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:10:39 2022

@author: Shiva Sheelvanth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns 
import warnings 
warnings.filterwarnings("ignore")
import os
%matplotlib inline

animal = pd.read_csv('Zoo.csv')
animal.head(10)

animal.info()

# Data Processing
animal.isnull().sum()

#check if there are duplicates in animal_name
duplicates = animal.animal_name.value_counts()
duplicates[duplicates > 1]

#select these duplicates frog
frog = animal.loc[animal['animal_name'] == 'frog']
frog

# observation: find that one frog is venomous and another one is not 
#              change the venomous one into frog2 to seperate 2 kinds of frog 
animal['animal_name'][(animal.venomous == 1 )& (animal.animal_name == 'frog')] = "frog2"

# finding Unique value of hair
color_list = [("red" if i == 1 else "blue" if i == 0 else "yellow" ) for i in animal.hair]
unique_color = list(set(color_list))
unique_color

# scatter matrix to observe relationship between every colomn attribute. 
pd.plotting.scatter_matrix(animal.iloc[:,:7],
                                       c=color_list,
                                       figsize= [20,20],
                                       diagonal='hist',
                                       alpha=1,
                                       s = 300,
                                       marker = '.',
                                       edgecolor= "black")
plt.show()

sns.countplot(x="hair", data=animal)
plt.xlabel("Hair")
plt.ylabel("Count")
plt.show()
animal.loc[:,'hair'].value_counts()



# Join animal table and class table to show actual class names
ani_class = pd.read_csv('class.csv')
df = pd.merge(animal,ani_class,how='left',left_on='class_type',right_on='Class_Number')
df.head()

# finding unique valure of class_type
type_list=[i for i in df.class_type]
unique_type=list(set(type_list))
unique_type

# use seaplot to plot the count of each 7 class_type
sns.factorplot('Class_Type', data=df, kind="count", size=5. aspect=2)

#splitting in train and test data
from sklearn.model_selection import train_test_split
X=animal.iloc[:,1:17]
y=animal.iloc[:,17]
X_train,X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=42)

#Training and Testing the data
#KNN
from sklearn.neighbors import KNeighborsClassifier
#Declare the model
clf=KNeighborsClassifier(n_neighbors=3)

#Train the model
clf.fit(X_train, y_train)
y_pred_KNeighborsClassifier=clf.predict(X_test)

scrs=[]

#Get Accuracy Score
score=accuracy_score(y_pred_KNeighborsClassifier, y_test)
scrs.append(score)

#use cross validation score of KNN
from sklearn.model_selection import cross_val_score
cv_scores=[]
store_knn=cross_val_score(clf, X, y, cv=10)
print("K-Nearest Neighbors Accuracy: %0.2f) with k value equals to 3" % (score_knn.mean(), score_knn.std() *2))

k_values=np.arrange(1, 20)
train_accuracy=[]
test_accuracy=[]

for i,k in enumerate (k_values):
    knn=KNeighborClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)      #fit the model
    train_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

plt.figure(figsize=[13,8])
plt.plot(k_values, test_accuracy, label='Testing Accuracy')
plt.plot(k_values, train_accuracy, label='Training Accuracy')
plt.legend
plt.title('K-values VS Accuracy Graph representation')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_values)

plt.show()
print("Best accuracy is {} with K= {}.format(np.max(test_accuracy), 1+test_accuracy.index(np.max(test_accuracy"))))

cv_scores.append(np.max(test_accuracy))    













































