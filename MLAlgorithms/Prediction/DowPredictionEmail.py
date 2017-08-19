
# coding: utf-8

# In[20]:

#!pip install psycopg2


# In[5]:

import copy
import datetime
import numpy as np
import time
import datetime as dt
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
import smtplib
import email.message
import psycopg2
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


# In[6]:

import pickle
filename = 'NeuralNetworkModel.pickle'

with open('config.json') as data_file:
    configdata = json.load(data_file)


# In[7]:

def get_open_normalised_prices(df, start, end):
    
    print('************************************************************************')
    df1 = pd.DataFrame(df["High"]/df["Open"])
    df1.columns=["H/O"]
    df1["L/O"] = df["Low"]/df["Open"]
    df1["C/O"] = df["Close"]/df["Open"]
 
    return df1


# # create_follow_cluster_matrix

# In[8]:

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


# In[9]:

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


# In[10]:

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
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    h5filename = 'model'+str(i)+'.h5'
    model.save_weights(h5filename)
    print("Saved model to disk")
#     picklefilename = 'model'+str(i)+'.pkl'
#     joblib.dump(model, picklefilename)
#     joblib.dump(model, 'model'+str(i)+'.pkl')

    return model,history1


# In[11]:

def create_dataset1(datasetReal,i):
    print(type(datasetReal))
    xdataset = datasetReal.copy()
    xdataset = xdataset[xdataset.columns[xdataset.columns !='closePrice']]
    dataset = np.array(xdataset)
    print("Total Number of features in trainig Dataset "+ str(xdataset.columns.size))
    dataX = [dataset[n] for n in range(len(dataset)-i)]
    return np.array(dataX),xdataset.columns.size


# In[12]:

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
                joblib.dump(testdata, 'testdata2.pkl')

            print("Making The Training Dataset")
            traindataset = dataset1.head(len(dataset1)-2)

            trainY = np.array(traindataset['closePrice'])[i:]
            trainX, numbFeatures = create_dataset1(traindataset,i)
            
#             m,h = modellingNeuralNetwork(trainX,trainY,numbFeatures,i)
            
            json_file = open('model'+str(i)+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            modelname = 'model'+str(i)+'.h5' 
            loaded_model.load_weights(modelname)
            print("Loaded model"+ str(i)+" from disk")

            # evaluate loaded model on test data
            loaded_model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mape'])
            
#             f={"models":m}
#             s = pickle.dumps( m, "model"+str(i)+".pkl" ) 
#             s = pickle.dumps(m) 
#             joblib.dump(s, 'model'+str(i)+'.pkl')
            print("Model Compiled")
#             joblib.dump(m, '"model"+str(i)+".pkl"') 
#             model.append(m)
            
#             fullReport  = pd.DataFrame(h.history)
#             fullReport.to_csv("Day_"+ str(i)+" NN_PredictionModel.csv")
#             finalMAPE=fullReport['mean_absolute_percentage_error'].tail(1)
#             finalLoss = fullReport['loss'].tail(1)
#             finalMAE = fullReport['mean_absolute_error'].tail(1)

#             finalreport = pd.DataFrame(finalMAPE)
#             finalreport.columns = ['MAPE']
#             finalreport['Loss']=finalLoss
#             finalreport['MAE']=finalMAE
#             finalreport.index = range(1,len(finalreport)+1)

#             frameshistory.append(finalreport)
#             print(finalreport)

            print("Testing on the Latest Trade Data ...")

#             trainfeature=traindataset.tail(len(traindataset)-1)
#             trainlabel= trainfeature['closePrice']

#             prediction = m.predict(trainX)
#             my_list = map(lambda x: x[0], prediction)
#             traintestlabel = pd.Series(my_list)

#             testdata = dataset1.tail(1)
            testdata = joblib.load('testdata2.pkl')
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
            clf = joblib.load('predictedClosePrice2.pkl') 
            clf.append(testlabel[0])
            joblib.dump(clf, 'predictedClosePrice2.pkl')
            print("**********************************************************")
            print("Time Taken to Model %s seconds ---" % (time.time() - starttime))
            
        print("Testing the Model Finished ...")
        print("Total Time Taken to Model %s seconds ---" % (time.time() - starttime1))
#         allrecord = pd.concat(frameshistory)
#         allrecord.index = range(1,len(allrecord)+1)
#         print("Final Model Evaluation Scores ")
#         print(allrecord)


# In[13]:

def datadownload(start,end):
    try:
        dow = web.DataReader("^DJI", "yahoo", start, end)
        dow.to_csv("AllDJIdata.csv")
    except (RuntimeError, TypeError, NameError,Exception):
                   print("Oops! Yahoo Data Service is not working ....")
                   print("Data Taken from last updated repository")
                   dow=pd.read_csv("AllDJIdata.csv")
        
    
    return dow
    


# In[14]:

def knn(dow_norm,k):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(dow_norm)
    labels = km.labels_
    return labels,km


# In[15]:

def getinformationCluster(cmvc,km):
    print("Ranking of Clusters ");
    print(str(cmvc))
    print('************************************************************************')
    print("Centroid for the K Means Clusters ")
    kmean = pd.DataFrame(km.cluster_centers_)
    kmean.columns=['X','Y','Z']
    print(kmean)
    print('************************************************************************')
    


# In[16]:

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
    


# In[17]:

def getRightClusteredData(nextClusterMatrix,dowtrain):
    print("Getting the filtered Past data where the maximum predicted range of the next DAY OHLC would be.. ")
    inputClusterone = dowtrain[dowtrain['Cluster']==nextClusterMatrix['Cluster'][0]]
    return inputClusterone
    


# In[18]:

# import pandas_datareader.data as web
# start = datetime.datetime(2005, 1, 1)
# end =  time.strftime("%Y-%m-%d")
# dow = web.DataReader("^DJI", "yahoo", start, end)


# In[19]:

# dow=pd.read_csv("AllDJIdata.csv")


# In[20]:

# frameshistory=[]
# def testingModel(i):
    
            
              
        
# #         for i in range(1,3):
#             print("Training the Neural Network for next "+ str(i)+" Prediction")
#             starttime= time.time() 
            
#             print("Copying the input Dataset ")
#             inputdata = inputClusterone.copy()
            
#             dataset1 = get_indicators(inputClusterone,i)

#             print("Making The Training Dataset")
#             traindataset = dataset1.head(len(dataset1)-2)

#             trainY = np.array(traindataset['closePrice'])[i:]
#             trainX, numbFeatures = create_dataset1(traindataset,i)
            
#             m,h = modellingNeuralNetwork(trainX,trainY,numbFeatures,i)
            
# #             f={"models":m}
# #             s = pickle.dumps( m, "model"+str(i)+".pkl" ) 
# #             s = pickle.dumps(m) 
            
#             print("Model Saved")
# #             joblib.dump(m, '"model"+str(i)+".pkl"') 
# #             model.append(m)
            
#             fullReport  = pd.DataFrame(h.history)
# #             fullReport.to_csv("Day_"+ str(i)+" NN_PredictionModel.csv")
#             finalMAPE=fullReport['mean_absolute_percentage_error'].tail(1)
#             finalLoss = fullReport['loss'].tail(1)
#             finalMAE = fullReport['mean_absolute_error'].tail(1)

#             finalreport = pd.DataFrame(finalMAPE)
#             finalreport.columns = ['MAPE']
#             finalreport['Loss']=finalLoss
#             finalreport['MAE']=finalMAE
#             finalreport.index = range(1,len(finalreport)+1)

#             frameshistory.append(finalreport)
#             print(finalreport)

#             print("Testing on the Latest Trade Data ...")

# #             trainfeature=traindataset.tail(len(traindataset)-1)
# #             trainlabel= trainfeature['closePrice']

# #             prediction = m.predict(trainX)
# #             my_list = map(lambda x: x[0], prediction)
# #             traintestlabel = pd.Series(my_list)

#             testdata = dataset1.tail(1)
# #             testdata = testDataLatestTrade
# #             testexpectedlabel = testdata['closePrice']

#             xtest = testdata[testdata.columns[testdata.columns !='closePrice']]
#             predictiontest = m.predict(np.array(xtest))
            
            
#             temp_list = map(lambda x: x[0], predictiontest)
#             testlabel = pd.Series(temp_list)
#             print("**********************************************************")
# #             print("Excpected Close Price "+ str( testexpectedlabel.values))
#             print("Predicted Close Price  "+ str(testlabel[0]))
    
#             clf = joblib.load('predictedClosePrice2.pkl') 
#             clf.append(testlabel[0])
#             joblib.dump(clf, 'predictedClosePrice2.pkl')
# #             predictedClosePrice.append(testlabel[0])
#             print("**********************************************************")
#             print("Time Taken to Model %s seconds ---" % (time.time() - starttime))
            


# In[21]:

if __name__ == "__main__":
    # Obtain DOW 30  pricing data from Yahoo Finance
    print('************************************************************************')
    print("Step 1:Downloading the DOW 30 data from Yahoo Finance ")
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
    
    joblib.dump(predictedClosePrice2, 'predictedClosePrice2.pkl')
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
    clf = joblib.load('predictedClosePrice2.pkl')
    print("**************************************")
    print(clf)
    



# In[23]:

def connectionString():
    con=psycopg2.connect(dbname= 'userTable', host= configdata['PostgreHost'], 
    port= configdata['PostgrePort'], user= configdata['postgreUser'], password= configdata['PostgrePassword'])
    con.autocommit = True
    return con
def connectionStringclose(con):
    con.close()
def getUsers(con):
    curs = con.cursor()
    print curs
    query =  "select username,email from account"
    curs.execute(query)
    records = curs.fetchall()
    dataf = pd.DataFrame(records)
    con.commit()
    curs.close()
    return dataf


# In[24]:

def sendemail(from_addr, to_addr_list,
              subject, name,
              login, password,
              smtpserver='smtp.gmail.com:587'):
    clf= joblib.load('predictedClosePrice2.pkl')
    msg = email.message.Message()
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr_list
    msg.add_header('Content-Type','text/html')
    
    message= """<p>Hi """+name+"""!<br>
    <h3>Dow30's next 7 days prediction</h3>
    <table border=1>
    <tr>
    <th>Day</th>
    <th>Predicted value</th>
    </tr>
    <tr>
    <td>Day 1</td>
    <td>""" + str(clf[0]) + """</td>
    </tr>  
        <tr>
    <td>Day 2</td>
    <td>""" + str(clf[1]) + """</td>
    </tr>   
        <tr>
    <td>Day 3</td>
    <td>""" + str(clf[2]) + """</td>
    </tr>   
        <tr>
    <td>Day 4</td>
    <td>""" + str(clf[3]) + """</td>
    </tr>   
        <tr>
    <td>Day 5</td>
    <td>""" + str(clf[4]) + """</td>
    </tr>   
        <tr>
    <td>Day 6</td>
    <td>""" + str(clf[5]) + """</td>
    </tr>   
        <tr>
    <td>Day 7</td>
    <td>""" + str(clf[6]) + """</td>
    </tr>   
    </table>
    </p>"""
    msg.set_payload(message)

    s = smtplib.SMTP(smtpserver)
    s.starttls()
    s.login(login,
            password)
    s.sendmail(msg['From'], [msg['To']], msg.as_string())


# In[44]:

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


# In[45]:

con = connectionString()
user= getUsers(con)
AWS_ACCESS_KEY_ID= configdata['AWSAccess'] #raw_input('Amazon Access Key')
AWS_SECRET_ACCESS_KEY= configdata['AWSSecret']#raw_input('Amazon Secret Key')

for index, row in user.iterrows():
    
    sendemail(from_addr    = configdata['notificationEmail'], 
          to_addr_list = str(row[1]),
          subject      = 'Stock Prediction for next 7 days', 
          name      = str(row[0]), 
          login        = configdata['notificationEmail'], 
          password     = configdata['password'],
         )
upload_files_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,'predictedClosePrice2.pkl','results_models' )


# In[42]:


    


# In[ ]:



