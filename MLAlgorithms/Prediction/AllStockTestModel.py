
# coding: utf-8

# In[25]:

# !sudo pip install h5py


# In[22]:

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
import json
import boto
import boto.s3
import sys
from boto.s3.key import Key
import glob
import boto3
import botocore
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


# In[23]:

import pickle
fileDir = os.path.dirname(os.path.realpath('__file__'))
print(' Local Path defined')
print(fileDir)

with open('config.json') as data_file:
    configdata = json.load(data_file)


# In[24]:

def get_open_normalised_prices(df, start, end):
    
    print('************************************************************************')
    df1 = pd.DataFrame(df["High"]/df["Open"])
    df1.columns=["H/O"]
    df1["L/O"] = df["Low"]/df["Open"]
    df1["C/O"] = df["Close"]/df["Open"]
 
    return df1


# # create_follow_cluster_matrix

# In[25]:

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


# In[26]:

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


# In[27]:

def create_dataset1(datasetReal,i):
    print(type(datasetReal))
    xdataset = datasetReal.copy()
    xdataset = xdataset[xdataset.columns[xdataset.columns !='closePrice']]
    dataset = np.array(xdataset)
    print("Total Number of features in trainig Dataset "+ str(xdataset.columns.size))
    dataX = [dataset[n] for n in range(len(dataset)-i)]
    return np.array(dataX),xdataset.columns.size


# In[28]:

predictedClosePrice2 = []
frameshistory=[]
# model = []
def testingModel1(inputClusterone):
    
        starttime1= time.time()
              
        
        for i in range(1,8):
            print("Training the Neural Network for next "+ str(i)+" Prediction")
            starttime= time.time() 
            
            print("Copying the input Dataset ")
            inputdata = inputClusterone.copy()
            
            dataset1 = get_indicators(inputClusterone,i)
            if(i ==1):
                print("Getting the features for Test data ")
                testdata = dataset1.tail(1)
                joblib.dump(testdata, fileDir+ '/stockMarketModels/'+str(stock)+'/'+str(stock)+'testdata2.pkl')

            print("Making The Training Dataset")
            traindataset = dataset1.head(len(dataset1)-2)

            trainY = np.array(traindataset['closePrice'])[i:]
            trainX, numbFeatures = create_dataset1(traindataset,i)
            
            
            json_file = open(str(stock)+'model'+str(i)+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            modelname = str(stock)+'model'+str(i)+'.h5' 
            loaded_model.load_weights(modelname)
            print("Loaded model"+ str(i)+" from disk")

            # evaluate loaded model on test data
            loaded_model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mape'])
            

            print("Model Compiled")

            print("Testing on the Latest Trade Data ...")

#             trainfeature=traindataset.tail(len(traindataset)-1)
#             trainlabel= trainfeature['closePrice']

#             prediction = m.predict(trainX)
#             my_list = map(lambda x: x[0], prediction)
#             traintestlabel = pd.Series(my_list)

#             testdata = dataset1.tail(1)
            testdata = joblib.load(fileDir+ '/stockMarketModels/'+str(stock)+'/'+str(stock)+'testdata2.pkl')
#             testdata = testDataLatestTrade
#             testexpectedlabel = testdata['closePrice']

            xtest = testdata[testdata.columns[testdata.columns !='closePrice']]
            print("-----------------------------------------")
            print("Test record Given")
            print(xtest)
            print("----------------------------------")
            
            predictiontest = loaded_model.predict(np.array(xtest))
            
            
            temp_list = map(lambda x: x[0], predictiontest)
            testlabel = pd.Series(temp_list)
            print("**********************************************************")
#             print("Excpected Close Price "+ str( testexpectedlabel.values))
            print("Predicted Close Price  "+ str(testlabel[0]))
#             predictedClosePrice.append(testlabel[0])
            clf = joblib.load(fileDir+ '/stockMarketModels/'+str(stock)+'/'+str(stock)+'predictedClosePrice2.pkl') 
            clf.append(testlabel[0])
            joblib.dump(clf, fileDir+ '/stockMarketModels/'+str(stock)+'/'+str(stock)+'predictedClosePrice2.pkl')
            print("**********************************************************")
            print("Time Taken to Model %s seconds ---" % (time.time() - starttime))
            
        print("Testing the Model Finished ...")
        print("Total Time Taken to Model %s seconds ---" % (time.time() - starttime1))
#         allrecord = pd.concat(frameshistory)
#         allrecord.index = range(1,len(allrecord)+1)
#         print("Final Model Evaluation Scores ")
#         print(allrecord)


# In[29]:

def datadownload(start,end,stock):
    try:
        dow = web.DataReader(stock, "yahoo", start, end)
        dow.to_csv(fileDir+ '/stockMarketModels/'+str(stock)+'/'+"All"+str(stock)+"data.csv")
    except (RuntimeError, TypeError, NameError,Exception):
                   print("Oops! Yahoo Data Service is not working ....")
                   print("Data Taken from last updated repository")
                   dow=pd.read_csv(fileDir+ '/stockMarketModels/'+str(stock)+'/'+"All"+str(stock)+"data.csv")
        
    
    return dow
    


# In[30]:

def knn(dow_norm,k):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(dow_norm)
    labels = km.labels_
    return labels,km


# In[31]:

def getinformationCluster(cmvc,km):
    print("Ranking of Clusters ");
    print(str(cmvc))
    print('************************************************************************')
    print("Centroid for the K Means Clusters ")
    kmean = pd.DataFrame(km.cluster_centers_)
    kmean.columns=['X','Y','Z']
    print(kmean)
    print('************************************************************************')
    


# In[32]:

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
    


# In[33]:

def getRightClusteredData(nextClusterMatrix,dowtrain):
    print("Getting the filtered Past data where the maximum predicted range of the next DAY OHLC would be.. ")
    inputClusterone = dowtrain[dowtrain['Cluster']==nextClusterMatrix['Cluster'][0]]
    return inputClusterone
    


# In[34]:

def upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,file_name,bucketName):
    ts = time.time()
    st = dt.datetime.fromtimestamp(ts).strftime('%d%m%y%M%S')
    st1 = dt.datetime.fromtimestamp(ts).strftime('%d%m%y')
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


# In[35]:

def runprogram(stock):
    # Obtain DOW 30  pricing data from Yahoo Finance
    print('************************************************************************')
    print("Step 1:Downloading the"+str(stock)+" data from Yahoo Finance ")
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
    
    nextClusterMatrix = predictCluster(km,testDataLatestTrade,cmvc,clust_mat)
    
    inputClusterone = getRightClusteredData(nextClusterMatrix,dowtrain)
    
    inputClusterone.append(dow.tail(1))
    
    joblib.dump(predictedClosePrice2, str(stock)+'predictedClosePrice2.pkl')
#     joblib.dump(inputClusterone, str(stock)+'inputClusterone.pkl')
#     p = Pool(20)
# #     frameshistory=[]
#     print(p.map(testingModel, range (1,4)))
    testingModel1(inputClusterone)
#     print("Testing the Model Finished ...")
#     allrecord = pd.concat(frameshistory)
#     allrecord.index = range(1,len(allrecord)+1)
#     print("Final Model Evaluation Scores ")
#     print(allrecord)
    # load json and create model
    clf = joblib.load(str(stock)+'predictedClosePrice2.pkl')
    print("**************************************")
    print(clf)
    upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,str(stock)+'predictedClosePrice2.pkl','results_models')


# In[36]:

stocknames=["AAPL","T","BAC","JNJ","XOM"]
        
for i in stocknames:
    stock = i  ## TECHNOLOGY   ## APPLE 
    runprogram(i)



# stock = "AAPL"  ## TECHNOLOGY   ## APPLE 
# runprogram("AAPL")
# stock = "T"   ## TELECOMMUNICATION ## AT&T Inc.
# runprogram("T")
# stock = "BAC"    ## FINANCE BANK OF AMERICA
# runprogram("BAC")
# stock = "JNJ" ## HEALTH CARE - ## JOHNSON AND JHONSON
# runprogram("JNJ")
# stock = "XOM"  ## ENERGY - ## Exxon Mobil Corporation
# runprogram("XOM")


# In[50]:

clf = joblib.load(str(stock)+'predictedClosePrice2.pkl')


# In[35]:

clf 


# In[16]:



