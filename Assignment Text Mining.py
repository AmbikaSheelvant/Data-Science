# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:20:05 2022

@author: Shiva Sheelvanth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import spacy

from matplotlib.pyplot import imread


df=pd.read_csv('Elon_musk.csv', encoding='Latin1')
df

df.drop(['Unnamed: 0'],inplace=True,axis=1)
df

# Text Pre-processing
df=[Text.strip() for Text in df.Text] # remove both the leading and the trailing characters
df=[Text for Text in df if Text] # removes empty strings, because they are considered in Python as False
df[0:10]

# Joining the list into one string/text
df_text=' '.join(df)
df_text

# remove Twitter username handles from a given twitter text. (Removes @usernames)
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
df_tokens=tknzr.tokenize(df_text)
df_tokens

# Again Joining the list into one string/text
df_tokens_text=' '.join(df_tokens)
df_tokens_text

# Remove Punctuations 
no_punc_text=df_tokens_text.translate(str.maketrans('','',string.punctuation))
no_punc_text

# remove https or url within text
import re
no_url_text=re.sub(r'http\S+', '', no_punc_text)
no_url_text


from nltk.tokenize import word_tokenize
text_tokens=word_tokenize(no_url_text)
print(text_tokens)


import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Tokens count
len(text_tokens)

from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')

sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)

# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])

# Stemming (Optional)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])

# Lemmatization
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)

lemmas=[token.lemma_ for token in doc]
print(lemmas)

clean_df=' '.join(lemmas)
clean_df

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
dfcv=cv.fit_transform(lemmas)

print(cv.vocabulary_)

print(cv.get_feature_names()[100:200])

print(dfcv.toarray()[100:200])

print(dfcv.toarray().shape)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matix_ngram.toarray())

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud

nlp=spacy.load('en_core_web_sm')

one_block=clean_df
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


for token in doc_block[100:200]:
    print(token,token.pos_)  

# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])

# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results

# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');

from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(df))
sentences

sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df

# Emotion Lexicon - Affin

affin=pd.read_csv('Afinn.csv',sep=',',encoding='Latin-1')
affin

nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

affinity_scores=affin.set_index('word')['value'].to_dict()
affinity_scores

nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score

calculate_sentiment(text='great')

sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']


sent_df['word_count']=sent_df['sentence'].str.split().apply(len)
sent_df['word_count']

sent_df.sort_values(by='sentiment_value')


sent_df['sentiment_value'].describe()

sent_df[sent_df['sentiment_value']<=0]

# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]

sent_df['index']=range(0,len(sent_df))
sent_df

import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])

plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)

sent_df.plot.scatter(x='word_count',y='sentiment_value',figsize=(8,8),title='Sentence sentiment value to sentence word




