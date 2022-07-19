# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:29:45 2022

@author: Shiva Sheelvanth
"""

import pandas as pd
import re
from textblob import TextBlob
data=pd.read_csv('Reviews.csv',encoding='latin1')
pd.set_option('display.max_colwidth', -1)
data.shape
list(data)
data.head
list(data)
data.isnull().sum()
data

import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')

#Defining a dcitionary containing all the emojis and their meanings
emojis={':)':'smile',':-)':'smile',';d':'wink',':-E':'vampire',':(':'sad',
        ':-(':'sad',':-<':'sad',':P':'raspberry',':O':'surprised',
        ':-@':'shocked',':@':'shocked',':-$':'confused',':\\':'annoyed',
        ':#':'mute',':X':'mute',':^)':'smile',':-&':'confused','$_$':'greedy',
        '@@':'eyeroll',':-!':'confused',':-D':'smile',':-0':'yell','O.o':'confused',
        '<(-_-)>':'robot','d[-_-]b':'dj',":'-)":'sadsmile',';)':'wink',
        ';-)':'wink','O:-)':'angel','O*-)':'angel','(:-D':'gossip','=^.^=':'cat'}

#Defining a function to clean the data
def clean_text(kit):
    kit=str(kit).lower()
    kit=re.sub(r"@\S+",r'',kit)
    
    for i in emojis.keys():
        kit=kit.replace(i,emojis[i])
    kit=re.sub("\s+",' ',kit)
    kit=re.sub("\n",' ',kit)
    letters=re.sub('[^a-zA-Z]',' ',kit)
    return letters

#Defining a function to remove the stop words        
def stops_words(words):
    filter_words=[]
    for w in words:
        if w not in stop_words:
            filter_words.append(w)
    return filter_words

#Defining a function for sentiment analysis
def getSubjectivity(tex):
    return TextBlob(tex).sentiment.subjectivity

def getPolarity(tex):
    return TextBlob(tex).sentiment.polarity

def getAnalysis(score):
    if int(score)<0:
        return 'Negative'
    elif int(score)==0:
        return 'Neutral'
    elif int(score)>0:
        return 'Positive'

#Cleaning the data
data['Text']=data['Text'].apply(lambda x:clean_text(x))

#Removing stop words
data['Text']=data['Text'].apply(lambda x:x.split(" "))
data['Text']=data['Text'].apply(lambda x:stops_words(x))

#Stemming
from nltk.stem import PorterStemmer
stem=PorterStemmer()
data['Text']=data['Text'].apply(lambda x: [stem.stem(k) for k in x])

#Lemmatization
from nltk.stem import WordNetLemmatizer
lemm=WordNetLemmatizer()
data['Text']=data['Text'].apply(lambda x: [lemm.lemmatize(j) for j in x])

data['Text']=data['Text'].apply(lambda x: ' '.join(x))

#Preparing a target variable which shows the sentiment i.e, Subjectivity and Polarity
data['sentiment_subj']=data['Text'].apply(lambda x:getSubjectivity(x))
data['sentiment_subj'].describe()    

data['sentiment_pol']=data['Text'].apply(lambda x:getPolarity(x))
data['sentiment_pol'].describe()

sentiment=[]
for i in range(0,1999,1):
    if data['sentiment_pol'].iloc[i,] < 0:
        sentiment.append('Negative')
    elif data['sentiment_pol'].iloc[i,] == 0:
        sentiment.append('Neutral')
    else:
        sentiment.append('Positive')
sentiment
Sentiment=pd.DataFrame(sentiment)
Sentiment.set_axis(['sentiment'],axis='columns',inplace=True)
data_new=pd.concat([data,Sentiment],axis=1)
data_new.shape
list(data_new)

import seaborn as sns
sns.distplot(data_new['sentiment_subj'])
sns.distplot(data_new['sentiment_pol'])
sns.countplot(data_new['sentiment'])

from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
word_cloud = WordCloud().generate('data_new[sentiment]')
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Splitting into train and test
from sklearn.model_selection import train_test_split
data_train,data_test=train_test_split(data,test_size=0.25,random_state=41)

data_train_clean=data_train['Text']
data_test_clean=data_test['Text']

#Vectorize the data
from sklearn.feature_extraction.text import TfidfVectorizer
vector=TfidfVectorizer(use_idf=True)
data_train_clean=vector.fit_transform(data_train_clean)
data_test_clean=vector.transform(data_test_clean)

data_train_clean.toarray()
data_train_clean.toarray().shape

vector.get_feature_names()
X=data['Text']
list(X)
X.head()
X.shape
import nltk
import re

HANDLE = '@\w+'
LINK = 'https?://t\.co/\w+'
SPECIAL_CHARS = '&lt;|&lt;|&amp;|#;|);|:;|!;|<;|>;|_;|*;|~;|-;'

def clean(zen):
    zen = re.sub(HANDLE, ' ', zen)
    zen = re.sub(LINK, ' ', zen)
    zen = re.sub(SPECIAL_CHARS, ' ', zen)
    return zen

for i in range(0,1999,1):
    data.iloc[i,1]=data.iloc[i,1]

X.iloc[5,]=X.iloc[5,]

#pip install gensim
from gensim.utils import simple_preprocess
from textblob import TextBlob
'''
#X1=re.sub(r'\W','',X)
#X2=re.sub(r'\s+[a-zA-Z]\s+',' ',X1])
#X3=re.sub(r'^b\s+','',X2)
'''
#To remove all special characters
for i in range(0,1999,1):
    X.iloc[i,]=re.sub(r'\W',' ',X.iloc[i,])
X  
X.shape
type(X)

#To remove all the single character and which are left as a result of removing special characters
for i in range(0,1999,1):
    X.iloc[i,]=re.sub(r'\s+[a-zA-Z]\s+',' ',X.iloc[i,])
X

#To replace all multiple spaces with single space
for i in range(0,1999,1):
    X.iloc[i,]=re.sub(r'^b\s+','',X.iloc[i,])
X.shape
X
list(X)
X['0'] = X['0'].str.replace('@','')

#To remove all URLs from dataset, Create a function to clean the text
def text_cleaner(texts):
    #Removing URL
    new_texts=[re.sub(r"http\S+","",str(i)) for i in texts]
    #Removing Emotions
    new_texts=[re.sub("@\S+","",i) for i in new_texts]
    #Further cleaning using gensim
    new_texts=[simple_preprocess(i,deacc=True) for i in new_texts]
    new_text_list=[''.join(i) for i in new_texts]
    return new_text_list

#Cleaned Words
new_words=[]
for i in range(0,1999,1):
    new_words=text_cleaner(X.iloc[i,])

print(new_words)

type(new_words)
new_words=pd.DataFrame(new_words)
type(new_words)
list(new_words)
for z in new_words.iloc[:,0]:
    print(z)

def sentiment_analysis(tweets):
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity














