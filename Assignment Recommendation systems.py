# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 15:03:01 2022

@author: Shiva Sheelvanth
"""


import pandas as pd
import numpy as np
books_df = pd.read_csv("book.csv", encoding="latin-1")
books_df.shape
list(books_df)
books_df.head()
list(books_df)


books_df['User.ID']
books_df['Book.Rating']
books_df['Book.Title']

books_df.drop(books_df.columns[[0]],axis=1,inplace=True)

books_df.sort_values('User.ID')

#number of unique users in the dataset
len(books_df)
len(books_df['User.ID'].unique())
len(books_df['Book.Title'].unique())

books_df['Book.Rating'].value_counts()
books_df['Book.Rating'].hist()



list(books_df)
books_df.shape


user_books_df=books_df.pivot_table(index ='User.ID',columns ='Book.Title',values ='Book.Rating')
pd.crosstab(books_df['User.ID'],books_df['Book.Title'])
user_books_df
user_books_df.iloc[200]
user_books_df.iloc[500]
list(user_books_df)

#Impute those NaNs with 0 values
user_books_df.fillna(0, inplace=True)

user_books_df

# from scipy.spatial.distance import cosine correlation
#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances( user_books_df.values,metric='cosine')

#user_sim = 1 - pairwise_distances( user_movies_df.values,metric='correlation')

user_sim.shape

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)

#Set the index and column names to user ids 
user_sim_df.index   = books_df['User.ID'].unique()
user_sim_df.columns = books_df['User.ID'].unique()

user_sim_df.iloc[0:100, 0:100]

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:100, 0:100]

#Most Similar Users
user_sim_df.max()

user_sim_df.idxmax(axis=1)[0:100]

books_df[(books_df['User.ID']==278390) | (books_df['User.ID']==278396)]

user_276729=books_df[books_df['User.ID']==276729]

user_276726=books_df[books_df['User.ID']==276726]


user_276726=books_df[books_df['User.ID']==276837]
user_276736=books_df[books_df['User.ID']==278582]


pd.merge(user_276726,user_276736,on='User.ID',how='outer')

'''conclusion:
     According  to data set collaberation is difficult because high rate of NaN values
     there low rate of reading multiple book maxmium users read one book and they give rating '''