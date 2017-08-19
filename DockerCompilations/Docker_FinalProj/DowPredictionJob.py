
# coding: utf-8

# In[2]:

#!sudo pip install h5py
#!pip install pandas_datareader
#!pip install plotly


# In[3]:

import copy
import datetime
import numpy as np
import time
import pandas as pd
import pandas_datareader.data as web
from sklearn.cluster import KMeans
from sklearn import preprocessing
from talib.abstract import *
import plotly
import itertools

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


# import plotly.plotly as py
# from plotly.graph_objs import *
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import plotly.graph_objs as go
# plotly.tools.set_credentials_file(username='maiti.t', api_key='km8Kdfszic1Cu6ZAWiZJ')
# plotly.offline.init_notebook_mode(connected=True)


# In[4]:

import pickle


# In[5]:

def get_open_normalised_prices(df, start, end):
    
    print('************************************************************************')
    df1 = pd.DataFrame(df["High"]/df["Open"])
    df1.columns=["H/O"]
    df1["L/O"] = df["Low"]/df["Open"]
    df1["C/O"] = df["Close"]/df["Open"]
    
    return df1


# # create_follow_cluster_matrix

# In[6]:

def create_follow_cluster_matrix(data):
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


# In[7]:

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


# In[8]:

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
    with open("model"+str(i)+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    h5filename = 'model'+str(i)+'.h5'
    model.save_weights(h5filename)
    print("Saved model to disk")


    return model,history1


# In[9]:

def create_dataset1(datasetReal,i):
    print(type(datasetReal))
    xdataset = datasetReal.copy()
    xdataset = xdataset[xdataset.columns[xdataset.columns !='closePrice']]
    dataset = np.array(xdataset)
    print("Total Number of features in trainig Dataset "+ str(xdataset.columns.size))
    dataX = [dataset[n] for n in range(len(dataset)-i)]
    return np.array(dataX),xdataset.columns.size


# In[10]:

predictedClosePrice1 = []
frameshistory1=[]



# In[11]:

def datadownload(start,end):
    try:
        dow = web.DataReader("^DJI", "yahoo", start, end)
        dow.to_csv("AllDJIdata.csv")
    except (RuntimeError, TypeError, NameError,Exception):
                   print("Oops! Yahoo Data Service is not working ....")
                   print("Data Taken from last updated repository")
                   dow=pd.read_csv("AllDJIdata.csv")
        
    
    return dow
    


# In[12]:

def knn(dow_norm,k):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(dow_norm)
    labels = km.labels_
    return labels,km


# In[13]:

def getinformationCluster(cmvc,km):
    print("Ranking of Clusters ");
    print(str(cmvc))
    print('************************************************************************')
    print("Centroid for the K Means Clusters ")
    kmean = pd.DataFrame(km.cluster_centers_)
    kmean.columns=['X','Y','Z']
    print(kmean)
    print('************************************************************************')
    


# In[14]:

def predictCluster(km,dow_norm,cmvc,clust_mat):
    print("Getting the Latest Trade OHLC Values for DOW 30 ") 
    testdoeN = dow_norm.copy()
    print("*****************************************8")
#     print(testdoeN.tail(5))
#     del testdoeN['Cluster']
#     print("Testtttttt")
    
    print(testdoeN.tail(1).values.reshape(-1,3))
    predictedCluster = km.predict(testdoeN.tail(1).values.reshape(-1,3))[0]
    print("Predicted Cluster of Last Trade OHLC Date "+ " "+ str(testdoeN.index[len(testdoeN)-1])+" "+ str(predictedCluster))
    predictedNextCluster = []
    for i,row in cmvc.iteritems():
        if(i[0]==predictedCluster):
            predictedNextCluster.append(i[1])
    print("Predicted Next Day Cluster would be : "+ str(predictedNextCluster))
    
    probabCluster=[]
    totalPrbablityfromPast=0
    for i in predictedNextCluster:
            totalPrbablityfromPast=totalPrbablityfromPast+clust_mat[predictedCluster][i]
    #         probabCluster.append(clust_mat[predictedCluster][i])
    print("Calulating the Relative Probablity ")
    for i in predictedNextCluster:
            probabCluster.append(clust_mat[predictedCluster][i]/totalPrbablityfromPast*100)
            
    nextClusterMatrix = pd.DataFrame(predictedNextCluster)
    nextClusterMatrix.columns=["Cluster"]
    nextClusterMatrix['Probability']=probabCluster
    
    print("Next Day CLuster and its Relative probablity from the Predicted Cluster : "+ str(predictedCluster))
    print(nextClusterMatrix)
    return nextClusterMatrix
    


# In[15]:

def getRightClusteredData(nextClusterMatrix,dowtrain):
    print("Getting the filtered Past data where the maximum predicted range of the next DAY OHLC would be.. ")
    inputClusterone = dowtrain[dowtrain['Cluster']==nextClusterMatrix['Cluster'][0]]
    return inputClusterone
    


# In[16]:

# import pandas_datareader.data as web
# start = datetime.datetime(2005, 1, 1)
# end =  time.strftime("%Y-%m-%d")
# dow = web.DataReader("^DJI", "yahoo", start, end)


# In[17]:

# dow=pd.read_csv("AllDJIdata.csv")


# In[18]:

# frameshistory=[]
def testingModel(i):
#         for i in range(1,3):
            print("Training the Neural Network for next "+ str(i)+" Days Prediction")
            starttime= time.time() 
            print("Copying the input Dataset for "+str(i)+" Days Prediction")
#             inputClusterone = joblib.load('inputClusterone.pkl')
            
            inputdata = inputClusterone.copy()
            
            dataset1 = get_indicators(inputClusterone,i)
 
            
            print("**********Checking Process")
            print(inputdata.tail(5))
            print("_________________________________________")
            print(dataset1.tail(5))

            print("Making The Training Dataset")
            traindataset = dataset1.head(len(dataset1)-2)

            trainY = np.array(traindataset['closePrice'])[i:]
            trainX, numbFeatures = create_dataset1(traindataset,i)
            
            #################Modelling Neural Network ##########################
            m,h = modellingNeuralNetwork(trainX,trainY,numbFeatures,i)
            
 
            
            print("All Models Saved ....")
            
            fullReport  = pd.DataFrame(h.history)
            fullReport.to_csv("Day_"+ str(i)+" NN_PredictionModel.csv")
  
 

 
            print("**********************************************************")
            print("Time Taken to Model %s seconds ---" % (time.time() - starttime))
            


# In[19]:

if __name__ == "__main__":
    # Obtain DOW 30  pricing data from Yahoo Finance
    print('************************************************************************')
    print("Step 1:Downloading the DOW 30 data from Yahoo Finance ")
    starttime= time.time()
    start = datetime.datetime(2005, 1, 1)
    end =  time.strftime("%Y-%m-%d")
#     dow = web.DataReader("^DJI", "yahoo", start, end)
    
    dow = datadownload(start,end)
        

            
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
    clust_mat,cmvc = create_follow_cluster_matrix(dowtrain)
    
    print('************************************************************************')
    
    getinformationCluster(cmvc,km)
    
    nextClusterMatrix = predictCluster(km,testDataLatestTrade,cmvc,clust_mat)
    
    inputClusterone = getRightClusteredData(nextClusterMatrix,dowtrain)
    
    inputClusterone.append(dow.tail(1))
    print("Input Data ::::::::::::::::::::::::::::::::::::::::::")
    print(inputClusterone.tail(1))
 
    
    p = Pool(20)
    p.map(testingModel, range (1,8))
#     testingModel1(inputClusterone)
    
    print("Total Time Taken to Model %s seconds ---" % (time.time() - starttime))
    
 



