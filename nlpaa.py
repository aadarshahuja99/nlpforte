import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import Word
import re
import pickle
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download('wordnet')

data = pd.read_csv('text_emotion.csv')

data = data.drop('author', axis=1)

data = data.drop(data[data.sentiment == 'anger'].index)
data = data.drop(data[data.sentiment == 'boredom'].index)
data = data.drop(data[data.sentiment == 'enthusiasm'].index)
data = data.drop(data[data.sentiment == 'empty'].index)
data = data.drop(data[data.sentiment == 'fun'].index)
data = data.drop(data[data.sentiment == 'relief'].index)
data = data.drop(data[data.sentiment == 'surprise'].index)
data = data.drop(data[data.sentiment == 'love'].index)
data = data.drop(data[data.sentiment == 'hate'].index)
data = data.drop(data[data.sentiment == 'neutral'].index)
data = data.drop(data[data.sentiment == 'worry'].index)

data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

data['content'] = data['content'].str.replace('[^\w\s]',' ')

stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

data['content'] = data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]

freq = list(freq.index)
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.sentiment.values)

X_train, X_val, y_train, y_val = train_test_split(data.content.values, y, stratify=y, random_state=42, test_size=0.2, shuffle=True)
print(X_train[0])


tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.fit_transform(X_val)

count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['content'])

pickle.dump(count_vect,open('transform.pkl','wb'))

X_train_count =  count_vect.transform(X_train)
X_val_count =  count_vect.transform(X_val)

lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
lsvm.fit(X_train_count, y_train)
y_pred = lsvm.predict(X_val_count)
print('lsvm using count vectors accuracy %s' % accuracy_score(y_pred, y_val))

tweets = pd.DataFrame(['I am very happy today! The atmosphere looks cheerful',
'Things are looking great. It was such a good day',
'Will GLADLY take a 1-1 draw with that crucial, crucial away goal. We all know our home form, we can be confident. But obviously with the suspension and injury tonight, the draw comes with a price.',
'Gutted. Really disappointed to lose. But overall, I don’t think we can blame any individual today. Everyone should take responsibility. Not just Setién, not just Messi. The team tried today, but we missed big chances, in big moments.',
'Success is right around the corner. Lets celebrate this victory',
'The players seem convinced they should’ve won today. They feel as though they’ve been really hard done by to lose 2-0. I just hope that for the rest of the season, we see them fight back then. If you felt like you should’ve won today, go and prove it every game from now.',
'Everything is more beautiful when you experience them with a smile!',
'Now this is my worst, okay? But I am gonna get better.',
'I am tired, boss. Tired of being on the road, lonely as a sparrow in the rain. I am tired of all the pain I feel',
'This is quite depressing. I am filled with sorrow',
'His death broke my heart. It was a sad day'])

tweets[0] = tweets[0].str.replace('[^\w\s]',' ')
from nltk.corpus import stopwords
stop = stopwords.words('english')
tweets[0] = tweets[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
from textblob import Word
tweets[0] = tweets[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

tweet_count = count_vect.transform(tweets[0])

tweet_pred = lsvm.predict(tweet_count)
print(tweet_pred)

pickle.dump(lsvm,open('nlpaa2.pkl','wb'))

