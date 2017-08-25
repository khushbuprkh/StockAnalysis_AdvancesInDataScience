
# coding: utf-8

# In[1]:

# !sudo pip install h5py


# In[ ]:

import copy
import datetime
import numpy as np
import time
import pandas as pd
import pandas_datareader.data as web
from sklearn.cluster import KMeans
from sklearn import preprocessing
import plotly
import itertools
from talib.abstract import *
import os
import sys
from keras.models import model_from_json
from keras.models import model_from_yaml
import requests
import numpy as np
import pandas as pd
import h5py
from keras.models import Sequential
from keras.layers import Dense
from sklearn.externals import joblib
print("Import Function for the Neural Network Done")
from multiprocessing import Pool


import pickle
fileDir = os.path.dirname(os.path.realpath('__file__'))
print(' Local Path defined')
print(fileDir)


# In[ ]:

def get_open_normalised_prices(df, start, end):
    
    print('************************************************************************')
    df1 = pd.DataFrame(df["High"]/df["Open"])
    df1.columns=["H/O"]
    df1["L/O"] = df["Low"]/df["Open"]
    df1["C/O"] = df["Close"]/df["Open"]
    
    return df1



def create_follow_cluster_matrix(data,k):
    
    print('************************************************************************')
    data["ClusterTomorrow"] = data["Cluster"].shift(-1)
    data.dropna(inplace=True)
    data["ClusterTomorrow"] = data["ClusterTomorrow"].apply(int)
#     print("Tesssssssssssttttttttttttttt")
#     print(zip(data["Cluster"], data["ClusterTomorrow"]))
    data["ClusterMatrix"] = list(zip(data["Cluster"], data["ClusterTomorrow"]))
    cmvc = data["ClusterMatrix"].value_counts()
    clust_mat = np.zeros( (k, k) )
    for row in cmvc.iteritems():
#         print(row[1]*100.0/len(data))
        clust_mat[row[0]] = row[1]*100.0/len(data)
    print("Cluster Follow-on Matrix:")
    print(clust_mat)
    return clust_mat,cmvc


# In[ ]:

def get_indicators(stocks, period):
#     for i in stocks:
        print("***********************************************************************")
        stocks.columns = [s.lower() for s in stocks.columns]
        print("Adding Features: Started  ")
        features = pd.DataFrame(SMA(stocks, timeperiod=10))
        features.columns = ['sma_10']
        features['NATR']=pd.DataFrame(NATR(stocks, timeperiod=14))
        features = pd.concat([features,STOCHF(stocks, fastk_period=14, fastd_period=3)], axis=1)
        features['willr'] = pd.DataFrame(WILLR(stocks, timeperiod=14))
        features['rsi'] = pd.DataFrame(RSI(stocks, timeperiod=14))
        features['wma_10'] = pd.DataFrame(WMA(stocks,10))
        features['T3'] =pd.DataFrame(T3(stocks, timeperiod=5, vfactor=0))
        features['closePrice']=pd.DataFrame(stocks['close'].shift(-period))
        features['return_pct_change'] = ROC(stocks, timeperiod=period)
        features['return_pct_change'] = features['return_pct_change'].shift(-period)
        features['pct_change'] = features['return_pct_change'].apply(lambda x: '1' if x > 0 else '0' if x <= 0 else np.nan)
        features = features.dropna()
        return features


# In[ ]:

def modellingNeuralNetwork(trainX,trainY,numbFeatures,i):
    print("Creating Neural Network .....")
    starttime= time.time()
    model = Sequential()
    model.add(Dense(5, input_dim=numbFeatures, activation='relu'))
    model.add(Dense(11, input_dim=numbFeatures, activation='relu'))
    model.add(Dense(11, input_dim=numbFeatures, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mape'])
    history1 = model.fit(trainX, trainY, epochs=20, batch_size=2, verbose=0)
    print("Time Taken to Model %s seconds ---" % (time.time() - starttime))
    print("*******************************************")
    print(history1)
    print("*******************************************")
    scores = model.evaluate(trainX,trainY,verbose = 0)
#     scores = model.evaluate(X, Y, verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[0], scores[1]*100))
    print(model.summary())
    # serialize model to JSON
    model_json = model.to_json()
    with open(str(stock)+"model"+str(i)+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    h5filename = str(stock)+'model'+str(i)+'.h5'
    model.save_weights(h5filename)
    print("Saved model to disk")
 

    return model,history1


# In[7]:

def create_dataset1(datasetReal,i):
    print(type(datasetReal))
    xdataset = datasetReal.copy()
    xdataset = xdataset[xdataset.columns[xdataset.columns !='closePrice']]
    dataset = np.array(xdataset)
    print("Total Number of features in trainig Dataset "+ str(xdataset.columns.size))
    dataX = [dataset[n] for n in range(len(dataset)-i)]
    return np.array(dataX),xdataset.columns.size




# In[9]:

def datadownload(start,end,stock):
    try:
        dow = web.DataReader(stock, "yahoo", start, end)
        dow.to_csv(fileDir+ '/stockMarketModels/'+str(stock)+'/'+"All"+stock+"data.csv")
    except (RuntimeError, TypeError, NameError,Exception):
                   print("Oops! Yahoo Data Service is not working ....")
                   print("Data Taken from last updated repository")
                   dow=pd.read_csv(fileDir+ '/stockMarketModels/'+str(stock)+'/'+"All"+stock+"data.csv")
        
    
    return dow
    


# In[10]:

def knn(dow_norm,k):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(dow_norm)
    labels = km.labels_
    return labels,km


# In[11]:

def getinformationCluster(cmvc,km):
 


# In[12]:

def predictCluster(km,dow_norm,cmvc,clust_mat):
	print("Please contanct maiti.t@husky.neu.edu")
    return nextClusterMatrix
    


# In[13]:

def getRightClusteredData(nextClusterMatrix,dowtrain):
	
    return inputClusterone
    



# frameshistory=[]
def testingModel(i):
#         for i in range(1,3):
            print("Training the Neural Network for next "+ str(i)+" Days Prediction")
            starttime= time.time() 
            print("Copying the input Dataset for "+str(i)+" Days Prediction")
            inputClusterone = joblib.load(fileDir+ '/stockMarketModels/'+str(stock)+'/'+str(stock)+'inputClusterone.pkl')
            
            inputdata = inputClusterone.copy()
            
            dataset1 = get_indicators(inputClusterone,i)
          
            
            #################Modelling Neural Network ##########################
            m,h = modellingNeuralNetwork(trainX,trainY,numbFeatures,i)
            
 
            
            print("All Models Saved ....")
            
            fullReport  = pd.DataFrame(h.history)
            fullReport.to_csv(fileDir+ '/stockMarketModels/'+str(stock)+'/'+"Day_"+ str(i)+str(stock)+" NN_PredictionModel.csv")
           


            print("Time Taken to Model %s seconds ---" % (time.time() - starttime))
            


# In[17]:

def runprogram(stock):
    # Obtain DOW 30  pricing data from Yahoo Finance
    print('************************************************************************')
    print("Step 1:Downloading the "+str(stock)+"  data from Yahoo Finance ")
    starttime= time.time()
    start = datetime.datetime(2005, 1, 1)
    end =  time.strftime("%Y-%m-%d")
#     dow = web.DataReader("^DJI", "yahoo", start, end)
    
    dow = datadownload(start,end,stock)
        

            
    print("Status ::Data Download Finish....")
    print('************************************************************************')
    print("Step 2:Normalising the Data with respect to Daily Open Stock Price ")
    
    dow_norm = get_open_normalised_prices(dow, start, end)
    print("Number of Chosen Cluster : 5 ")
    
    k = 5
    print('************************************************************************')
    print("Step 3: KNN Process Started ...")
    
    # Deleting the last row from the KNN Dataset 
    traindowNorm = dow_norm.copy()
    traindowNorm.drop(traindowNorm.index[len(traindowNorm)-1])
    
    # Reatining the latest Trade info as normalised one
    testDataLatestTrade = dow_norm.tail(1)
    
    labels,km = knn(traindowNorm,k)
    
    dowtrain = dow.copy()
    dowtrain.drop(dowtrain.index[len(dowtrain)-1])
    dowtrain["Cluster"] = labels
    traindowNorm["Cluster"] = labels
    
    print("Status : KNN Process Finished ...")
    # Create and output the cluster follow-on matrix
    clust_mat,cmvc = create_follow_cluster_matrix(dowtrain,k)
    
    print('************************************************************************')
    
    getinformationCluster(cmvc,km)
    
 
    p = Pool(20)
    #p.map(testingModel, range (1,8))
    print("Total Time Taken to Model %s seconds ---" % (time.time() - starttime))
    
 


# In[19]:

def makefiles(stocknames):
    for i in stocknames:
        if(not os.path.exists(fileDir+'/stockMarketModels/'+str(i))):
                os.makedirs(fileDir+'/stockMarketModels/'+str(i)) 

stocknames=["AAPL","T","BAC","JNJ","XOM"]
        
makefiles(stocknames)

for i in stocknames:
    stock = i  ## TECHNOLOGY   ## APPLE 
    runprogram(i)
    
# stock = "T"   ## TELECOMMUNICATION ## AT&T Inc.
# runprogram("T")
# stock = "BAC"    ## FINANCE BANK OF AMERICA
# runprogram("BAC")
# stock = "JNJ" ## HEALTH CARE - ## JOHNSON AND JHONSON
# runprogram("JNJ")
# stock = "XOM"  ## ENERGY - ## Exxon Mobil Corporation
# runprogram("XOM")



# In[ ]:



