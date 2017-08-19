
# coding: utf-8

# In[29]:

import bs4 as bs
import quandl
import datetime as dt
import time
import os
import pandas as pd
import pandas_datareader as web
import pickle
import requests
#import matplotlib.pyplot as plt
#import matplotlib.style as style
import numpy as np
import boto
import boto.s3
import sys
from boto.s3.key import Key
import glob
import boto3
import botocore
import json
#style.use("ggplot")


# In[30]:

# Create logfile.
def log_entry(s):
    #print('Date now: %s' % datetime.datetime.now())
    
    timestamp = '[%s] : ' % dt.datetime.now()
    log_line = timestamp + s + '\n'
    logfile.write(log_line)
    logfile.flush()
    
with open('config.json') as data_file:
    configdata = json.load(data_file)


# In[31]:

def save_sp500_tickers():
    resp  = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text)
    table= soup.find('table', {'class': 'wikitable sortable'})
    tickers=[]
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers, f)
    return tickers


# In[32]:

def get_data_from_yahoo(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,ts,reload_sp500=False):
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
            clean_world_data(df)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
    for ticker in tickers:
        if not ticker =='BRK.B' and not ticker =='BF.B':
            upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'{}.csv'.format(ticker),'stock_dfs_')
       # else:
          #  print('Already have {}'.format(ticker))


# In[33]:

def getStock(symbol, start, end):
    """
    Downloads Stock from Yahoo Finance.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    df =  web.get_data_yahoo(symbol, start, end)
    df.columns = df.columns + '_' + symbol
    df['Return_%s' %symbol] = df['Adj Close_%s' %symbol].pct_change()
    print(str(symbol)+ " Downloaded ")
    
    return df


# In[34]:

# Download data for world market 
def get_data_world_market(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,ts):
    if not os.path.exists('world_stock'):
        print "yes"
        os.makedirs('world_stock')
    start= dt.datetime(2000,1,1)
    i = dt.datetime.now()
    end=dt.datetime(i.year,i.month,i.day)
    nasdaq = getStock('^IXIC', start, end)
    nasdaq = clean_world_data(nasdaq)
    nasdaq.to_csv('world_stock/nasdaq.csv')
    upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'world_stock/{}.csv'.format("nasdaq"),'world_stock')
    dow = getStock('^DJI', start, end)
    dow.to_csv('world_stock/dow.csv')
    clean_world_data(dow)
    upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'world_stock/{}.csv'.format("dow"),'world_stock')
    sp500 = getStock('^GSPC',start,end)
    sp500.to_csv('world_stock/sp500.csv')
    clean_world_data(sp500)
    upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'world_stock/{}.csv'.format("sp500"),'world_stock')
    frankfurt = getStock('^GDAXI', start, end)
    frankfurt = clean_world_data(frankfurt)
    frankfurt.to_csv('world_stock/frankfurt.csv')
    upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'world_stock/{}.csv'.format("frankfurt"),'world_stock')
    london = getStock('^FTSE', start, end)
    london.to_csv('world_stock/london.csv')
    clean_world_data(london)
    upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'world_stock/{}.csv'.format("london"),'world_stock')
    paris = getStock('^FCHI', start, end)
    paris.to_csv('world_stock/paris.csv')
    clean_world_data(paris)
    upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'world_stock/{}.csv'.format("paris"),'world_stock')
    hkong = getStock('^HSI', start, end)
    hkong.to_csv('world_stock/hkong.csv')
    clean_world_data(hkong)
    upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'world_stock/{}.csv'.format("hkong"),'world_stock')
    nikkei = getStock('^N225', start, end)
    nikkei.to_csv('world_stock/nikkei.csv')
    clean_world_data(nikkei)
    upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'world_stock/{}.csv'.format("nikkei"),'world_stock')
    australia = getStock('^AXJO', start, end)
    australia.to_csv('world_stock/australia.csv')
    clean_world_data(australia)
    upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'world_stock/{}.csv'.format("australia"),'world_stock')
    india = getStock('^NSEI', start, end)
    india.to_csv('world_stock/india.csv')
    clean_world_data(india)
    upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'world_stock/{}.csv'.format("india"),'world_stock')


# In[35]:

def clean_world_data(ticker):
    ticker.dropna(inplace=True)
    ticker.isnull().sum().reset_index()
    df2=ticker.isnull().sum().reset_index()
    df2.columns=['column_name', 'missing_count']
    c=ticker.count()
    for i,v in df2.missing_count.iteritems():
        df2.missing_count[i]= v
    return ticker


# In[38]:

def upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,file_name,bucketName):
    bucket_name = bucketName
    #log_entry("S3 bucket has been successfully created.")
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
    bucket = conn.create_bucket(bucket_name, location=boto.s3.connection.Location.DEFAULT)
    s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    file = file_name
    exists = False
   
    try:
        s3.Object(bucket_name, file).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            exists = False
        else:
            raise
    else:
        exists = True

    if exists==False:
        print ('Uploading %s to Amazon S3 bucket %s' % (file, bucket_name))
        def percent_cb(complete, total):
            sys.stdout.write('.')
            sys.stdout.flush()
        k = Key(bucket)
        k.key = file
        k.set_contents_from_filename(file, cb=percent_cb, num_cb=10)
        #log_entry(file+" has been uploaded to "+bucket_name)
        print("File uploaded.")

    elif exists==True:
        print("File already exists.")
        bucket = conn.get_bucket(bucket_name) 
        k = Key(bucket,file)
        k.delete()
        upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,file_name,bucketName)
        #log_entry("File already exists.")
    


# In[39]:

def main():
    AWS_ACCESS_KEY_ID= configdata['AWSAccess'] #raw_input('Amazon Access Key')
    AWS_SECRET_ACCESS_KEY= configdata['AWSSecret']#raw_input('Amazon Secret Key')
    ts = time.time()
    st = dt.datetime.fromtimestamp(ts).strftime('%d%m%y%M%S')
    st1 = dt.datetime.fromtimestamp(ts).strftime('%d%m%y')
    logfile = open(st+".log", "a")
    save_sp500_tickers()
    get_data_world_market(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,st)
    #get_data_from_yahoo(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,st)
    


if __name__ == '__main__':
    main()       


# In[ ]:



