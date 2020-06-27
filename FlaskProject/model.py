import pandas as pd
import csv
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

data=pd.read_csv("train.csv")
pd.set_option('display.max_colwidth',1000)
data.head()

testdata=pd.read_csv("test.csv")
testdata.head()

import string
string.punctuation
import re
import nltk
stopwords=nltk.corpus.stopwords.words('english')
from nltk.stem import PorterStemmer 
ps=PorterStemmer()
def clean_text(txt):
    txt1="".join(c for c in txt if c not in string.punctuation)
    tokens=re.split('\W+',txt1)
    txt=" ".join([word for word in tokens if word not in stopwords])
    
    return txt

data['Statement']=data['Statement'].apply(lambda x:clean_text(x.lower()))
data.head()

testdata['Statement']=testdata['Statement'].apply(lambda x:clean_text(x.lower()))
testdata.head()


print(data.isnull().any())

x_train=data['Statement']
y_train=data['Label']
x_test=testdata['Statement']
y_test=testdata['Label']

from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer()
ccl=vect.fit_transform(x_train)
ccl2=vect.transform(x_test)


from sklearn.feature_extraction.text import TfidfTransformer 
tfidfV = TfidfTransformer()
train_tfidf = tfidfV.fit_transform(ccl)
test_tfidf=tfidfV.transform(ccl2)
test_tfidf.shape

tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,3),use_idf=True,smooth_idf=True)
tfidf_train=tfidf_ngram.fit_transform(x_train)
tfidf_test=tfidf_ngram.transform(x_test)

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline


nb_pipeline_passive = Pipeline([
        ('NBCV',tfidf_ngram),
        ('nb_clf',PassiveAggressiveClassifier())])

nb_pipeline_passive.fit(x_train,y_train)
predicted_passive = nb_pipeline_passive.predict(x_test)
np.mean(predicted_passive ==y_test)


import pickle
model_file_passive = 'final_model_passive.sav'
pickle.dump(nb_pipeline_passive,open(model_file_passive,'wb'))

from sklearn.ensemble import RandomForestClassifier
random_forest_ngram = Pipeline([
        ('rf_tfidf',tfidf_ngram),
        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3))
        ])
    
random_forest_ngram.fit(x_train,y_train)
predicted_rf_ngram = random_forest_ngram.predict(x_test)
np.mean(predicted_rf_ngram == y_test)

import pickle
model_file_random = 'final_model_randomClassifier.sav'
pickle.dump(random_forest_ngram,open(model_file_random,'wb'))

from sklearn.linear_model import  LogisticRegression
logR_pipeline_ngram  = Pipeline([
        ('LogR_tfidf',tfidf_ngram),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))])

logR_pipeline_ngram .fit(x_train,y_train)
predicted_logR_ngram  = logR_pipeline_ngram .predict(x_test)
np.mean(predicted_logR_ngram  == y_test)

import pickle
model_file_logR = 'final_model_logR.sav'
pickle.dump(logR_pipeline_ngram,open(model_file_logR,'wb'))

load_model_randomClassifier = pickle.load(open('final_model_randomClassifier.sav', 'rb'))

load_model_logR=pickle.load(open('final_model_logR.sav','rb'))
load_model_passive=pickle.load(open('final_model_passive.sav','rb'))
