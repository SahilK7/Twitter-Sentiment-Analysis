#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener 
from tweepy import OAuthHandler
from tweepy import Stream

import tweepy
from textblob import TextBlob
import csv
import re
import sys
import pandas as pd
import numpy as np
from tweepy import Cursor
    
ACCESS_TOKEN_KEY = "133791853-AAmVHjrq7BoyvzLH4AhC6SavgPWIAOA39jvg4laG"
ACCESS_TOKEN_SECRET = "qvkfVWmI8hPS2Wjyh2LIda3IVgbIJ0kjHw7oOvb6zUhqK"
CONSUMER_KEY = "pc3gtY5H9MiTYnTdIJsShTklT"
CONSUMER_SECRET = "3sCDLXjpoaala3ynwh69PRZzGzSJZevZYCeYCh6bko9gmGkVKz"

auth=tweepy.OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN_KEY,ACCESS_TOKEN_SECRET)

api=tweepy.API(auth)
#api=tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
  
#topicname='Machine Learning'
#tweets=api.search(topicname)
tweets = api.user_timeline(screen_name="realDonaldTrump", count=2000)
#tweets1 = api.friends()



#tweets =tweepy.Cursor(api.user_timeline, max_id="realDonaldTrump").items(tweets)
#for tweet in tweepy.Cursor(api.user_timeline, id="<redacted>").items():
data=[]
for tweet in tweets:
    text=tweet.text
    textWords=text.split()
    #textWords
    cleanedTweet=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", " ", text).split())
    
    #TextBlob(cleanedTweet).tags
    analysis= TextBlob(cleanedTweet)
     
     
#   polarity = '1'
#   if(analysis.sentiment.polarity < 0):
#       polarity = '-1'
#   if(0<=analysis.sentiment.polarity ==0):
#       polarity = '0'  
    
    if(analysis.sentiment.polarity <= 0):
        polarity = '-1'
    if(analysis.sentiment.polarity >0):
        polarity = '1' 
    dic={}
    dic['Sentiment']=polarity
    dic['Tweets']=cleanedTweet
    data.append(dic)
df=pd.DataFrame(data)
    
dir(df)
df['id'] = np.array([tweet.id for tweet in tweets])
df['len'] = np.array([len(tweet.text) for tweet in tweets])
df['date'] = np.array([tweet.created_at for tweet in tweets])
df['source'] = np.array([tweet.source for tweet in tweets])
df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
   
df.to_csv('analysis1.csv')
     
    
    
#Returns information about the specified user.
#t = api.get_user('realdonaldtrump')

#Returns the authenticated user’s information.
#me = api.me()


#Visualization
df.Sentiment.value_counts().plot(kind='bar',title="sentiment analysis")

# Get average length over all tweets:
np.mean(df['len'])

# Get the number of likes for the most liked tweet:
np.max(df['likes'])

# Get the number of retweets for the most retweeted tweet:
np.max(df['retweets'])

# Time Series
time_likes = pd.Series(data=df['len'].values, index=df['date'])
time_likes.plot(figsize=(16, 4), color='r')


time_favs = pd.Series(data=df['likes'].values, index=df['date'])
time_favs.plot(figsize=(16, 4), color='r')


time_retweets = pd.Series(data=df['retweets'].values, index=df['date'])
time_retweets.plot(figsize=(16, 4), color='r')


# Layered Time Series:
time_likes = pd.Series(data=df['likes'].values, index=df['date'])
time_likes.plot(figsize=(16, 4), label="likes", legend=True)

time_retweets = pd.Series(data=df['retweets'].values, index=df['date'])
time_retweets.plot(figsize=(16, 4), label="retweets", legend=True)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 200):
    tweet = re.sub('[^a-zA-Z]', ' ', df['Tweets'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words())]
    tweet = ' '.join(tweet)
    corpus.append(tweet)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 800)
X = cv.fit_transform(corpus).toarray()
y = df['Sentiment']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)
classifier.score(y_test, y_pred)

classifier.predict_proba(X_test)









     
     