
# coding: utf-8

# In[5]:

import os
import sys
import tweepy
import requests
import numpy as np
import pandas as pd
import psycopg2
import json

from textblob import TextBlob


# In[6]:

with open('config.json') as data_file:
    configdata = json.load(data_file)


# In[7]:

consumer_key = configdata['consumerkey'] 
consumer_secret = configdata['consumersecret'] 
access_token = configdata['accesstoken'] 
access_token_secret = configdata['accesstokensecret'] 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
user = tweepy.API(auth)
google_username=configdata['googleusername'] 
google_password=configdata['googlepassword'] 



# In[8]:

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,test_size=0.25)
    #clf = neighbors.KNeighborsClassifier(k=25)
    clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                       ('knn',neighbors.KNeighborsClassifier()),
                        ('rfor',RandomForestClassifier())])
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:',Counter(predictions))


# In[9]:

def stock_sentiment(quote, num_tweets):
    list_of_tweets = user.search(quote, count=num_tweets)
    positive, null = 0, 0
    count = 0 
    for tweet in list_of_tweets:
#         print(tweet.text)
#         print("************************************************************************")
        blob = TextBlob(tweet.text).sentiment
#         print("subjectivity Score "+ str(blob.subjectivity))
#         print("Polarity Score "+ str(blob.polarity))
        if blob.subjectivity == 0:
            null += 1
            next
        if blob.polarity > 0:
            positive += 1
        count +=1 
    pt = str(float(float(positive)/float(len(list_of_tweets))*100))
    use = str(float(len(list_of_tweets)-null)/float(len(list_of_tweets))*100)
    tweetlen = str(len(list_of_tweets))
    
    print("Number of Positive Tweets % : "+ pt)
    print("Total Number of Tweet : "+ str(len(list_of_tweets)))
    print("Total Useful Subjectivity % : "+ use)
    return pt,use,tweetlen
#     if positive > ((num_tweets - null)/2):
#         return True


# In[10]:

def connectionString():
    con=psycopg2.connect(dbname= 'userTable', host= configdata['PostgreHost'], 
    port= configdata['PostgrePort'], user= configdata['postgreUser'], password= configdata['PostgrePassword'])
    con.autocommit = True
    return con
def connectionStringclose(con):
    con.close()


# In[11]:

def insertintopostgre(stock,pt,tweetlen,use):
    con=connectionString()
    curs = con.cursor()
    query =  "select * from twitter where Stock= '"+ stock + "';"
    curs.execute(query)
    records = curs.fetchall()
    df = pd.DataFrame(records)
    if df.empty:
        query =  " INSERT INTO twitter (Stock,positive,total,subject) VALUES (%s, %s, %s, %s);"
        data = (stock,pt,tweetlen,use)
        print("trying to print data")
        result =curs.execute(query, data)
        con.commit()
        curs.close()      
        connectionStringclose(con)
    else:
        query =  " UPDATE twitter SET positive='"+ pt +"',total='"+tweetlen+"',subject='"+use+"' WHERE Stock= '"+ stock + "';"
        print("trying to print data")
        result =curs.execute(query)
        con.commit()
        curs.close()      
        connectionStringclose(con)


# In[ ]:

pt,use,tweetlen = stock_sentiment("DOW",100)
insertintopostgre("DOW",pt,tweetlen,use)

pt,use,tweetlen = stock_sentiment("AAPL",100)
insertintopostgre("AAPL",pt,tweetlen,use)

pt,use,tweetlen = stock_sentiment("XON",100)
insertintopostgre("XON",pt,tweetlen,use)

pt,use,tweetlen = stock_sentiment("JNJ",100)
insertintopostgre("JNJ",pt,tweetlen,use)

pt,use,tweetlen = stock_sentiment("T",100)
insertintopostgre("T",pt,tweetlen,use)

pt,use,tweetlen = stock_sentiment("BAC",100)
insertintopostgre("BAC",pt,tweetlen,use)


# In[ ]:



