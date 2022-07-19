# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:18:08 2022

@author: Shiva Sheelvanth
"""

import pandas as pd
import numpy as np

df=pd.read_csv('wine.csv')
df

df.describe()
df.head()
df['Type'].value_counts()
Wine=df.iloc[:,1:]

#converting data to numpy
df_ary=Wine.values

#normalizing data
from sklearn.preprocessing import scale
df_norm=scale(df_ary)
df_norm

# IMPLEMENTING PCA
from sklearn.decomposition import PCA
pca=PCA()
pca_values = pca.fit_transform(df_norm)
pca_values

pca.components_

var = pca.explained_variance_ratio_
var

Var = np.cumsum(np.round(var,decimals= 4)*100)
Var

import matplotlib.pyplot as plt
plt.plot(Var);

final_df=pd.concat([df['Type'],pd.DataFrame(pca_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df

import seaborn as sns
sns.scatterplot(data=final_df, x='PC1', y='PC2', hue='Type');

pca_values[: ,0:1]

x= pca_values[:,0:1]
y= pca_values[:,1:2]
plt.scatter(x,y);

# HIERARCHIAL CLUSTERING
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(df_norm,'complete'))

hclusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
hclusters

y=pd.DataFrame(hclusters.fit_predict(df_norm),columns=['clustersid'])
y['clustersid'].value_counts()

df2=df.copy()
df2['clustersid']=hclusters.labels_
df2

#K-means
from sklearn.cluster import KMeans
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS');


#Cluster using K=3
C3=KMeans(3,random_state=30).fit(df_norm)
C3

C3.labels_


df3=df.copy()
df3['clusters3id']=C3.labels_
df3


df3['clusters3id'].value_counts()










