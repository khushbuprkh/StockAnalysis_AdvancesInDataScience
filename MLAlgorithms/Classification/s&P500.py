
# coding: utf-8

# In[1]:

import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader as web
import pickle
import requests
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
style.uhttp://localhost:8888/notebooks/Assignments/StockMarketAnalysis/s%26P500.ipynb#se("ggplot")


# In[2]:

def save_sp500_tickers():
    resp  = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table= soup.find('table', {'class': 'wikitable sortable'})
    tickers=[]
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers, f)
    return tickers


# In[3]:

save_sp500_tickers()


# In[4]:

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        print "yes"
        os.makedirs('stock_dfs')
    start= dt.datetime(2000,1,1)
    i = dt.datetime.now()
    end=dt.datetime(i.year,i.month,i.day)

    for ticker in tickers:
        # if ticker == "AMZN" or ticker == "AAPL" or ticker == "EBAY" or ticker == "MSFT" or ticker == "WMT":
        #if not os.path.exists("stock_dfs/{}.csv".format(str(ticker))):
        if not ticker =='BRK.B' and not ticker =='BF.B':
            df = web.DataReader(ticker,'yahoo',start,end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
       # else:
          #  print('Already have {}'.format(ticker))


# In[6]:

get_data_from_yahoo()


# In[7]:

def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)
    main_df=pd.DataFrame()
    
    for count,ticker in enumerate(tickers):
        #if ticker == "AMZN" or ticker == "AAPL" or ticker == "EBAY" or ticker == "MSFT" or ticker == "WMT":
        if not ticker =='BRK.B' and not ticker =='BF.B':              
            df= pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace= True)
            df.rename(columns= {'Adj Close' : ticker} , inplace=True)
            df.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

            if main_df.empty:
                main_df=df
            else:
                main_df= main_df.join(df, how='outer')

    print main_df.head()
    main_df.to_csv("Adj_CLose_All.csv")


# In[8]:

compile_data()


# In[9]:

def visualize_data():
    df=pd.read_csv("Adj_CLose_All.csv")
    df_corr=df.corr()
    df_corr.to_csv('sp500corr.csv')
    
    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    #plt.savefig("correlations.png", dpi = (300))
    plt.show()  


# In[10]:

visualize_data()


# In[11]:

def process_data_for_labels(ticker):
    hm_days = 1
    df = pd.read_csv('Adj_CLose_All.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    
    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        
    df.fillna(0, inplace=True)
    
    return tickers, df


# In[24]:

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.10
    for col in cols:
        if col >= requirement:
            return 1
    return -1


# In[25]:

from collections import Counter
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier 


# In[26]:

def extract_featuresets(t):
    tickers, df = process_data_for_labels(t)

    df['{}_target'.format(t)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(t)]))
    vals = df['{}_target'.format(t)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))
    vals = df['{}_target'.format(t)].values.tolist()
    str_vals = [str(i) for i in vals]
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    X = df_vals.values
    y = df['{}_target'.format(t)].values
    
    return X,y,df


# In[35]:

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


# In[36]:

do_ml('AMZN')


# In[ ]:




# In[ ]:



