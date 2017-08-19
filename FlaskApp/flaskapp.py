
# coding: utf-8

# In[1]:

from flask import Flask
from flask import render_template,request,jsonify
import json
import urllib2
import re
import os
import time
from jinja2.utils import Markup
import psycopg2
import pandas as pd
import pickle
import sys
import requests
import numpy as np

# In[13]:

app=Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():

    page = request.args.get("page")
    if page == "stockanalysis":
        return render_template('stockanalysis.html')
    elif page == "algorithms":
        return render_template('algorithms.html')
    elif page == "register":
        if request.method == 'POST':
            print "post"
            username=request.form['inputName']
            email=request.form['inputEmail']
            phonenumber=request.form['inputPhonenum']
            con=connectionString()
            curs = con.cursor()
            query =  "select * from account where email= '"+ str(email) + "';"
            curs.execute(query)
            records = curs.fetchall()
            df = pd.DataFrame(records)
            print df
            query_two =  "select * from account where username= '"+ str(username) + "';"
            curs.execute(query_two)
            records_two = curs.fetchall()
            dftwo = pd.DataFrame(records_two)
            if df.empty and dftwo.empty:
                query =  "INSERT INTO account (username,email,phonenumber) VALUES (%s, %s, %s);"
                data = (username,email,phonenumber)
                print("trying to print data")
                result =curs.execute(query, data)
                con.commit()
                curs.close()      
                connectionStringclose(con)
                return render_template('registeruser.html' , Status="Registerd for receiving mails")     
            return render_template('registeruser.html' , Warning="User Already exists")
        else:
            return render_template('registeruser.html')
    elif page == "prediction":
        if request.method == 'POST':
            value=request.form['selectList']
            if value=="DOW30":
                with open("/home/ubuntu/flaskapp/predictedClosePrice2.pkl","rb") as f:
                    clf = pickle.load(f)
                    return render_template('prediction.html', Status=clf)
            elif value=="APL":
                with open("/home/ubuntu/flaskapp/AAPLpredictedClosePrice2.pkl","rb") as f:
                    clf = pickle.load(f)
                    return render_template('prediction.html', Status=clf)
            elif value=="ATT":
                with open("/home/ubuntu/flaskapp/TpredictedClosePrice2.pkl","rb") as f:
                    clf = pickle.load(f)
                    return render_template('prediction.html', Status=clf)
            elif value=="BAC":
                with open("/home/ubuntu/flaskapp/BACpredictedClosePrice2.pkl","rb") as f:
                    clf = pickle.load(f)
                    return render_template('prediction.html', Status=clf)
            elif value=="JNJ":
                with open("/home/ubuntu/flaskapp/JNJpredictedClosePrice2.pkl","rb") as f:
                    clf = pickle.load(f)
                    return render_template('prediction.html', Status=clf)  
            elif value=="XOM":
                with open("/home/ubuntu/flaskapp/XOMpredictedClosePrice2.pkl","rb") as f:
                    clf = pickle.load(f)
                    return render_template('prediction.html', Status=clf)  
        return render_template('prediction.html')
    elif page == "twitter":
        if request.method == 'POST':
            value=request.form['selectList']
            if value=="DOW":
                con=connectionString()
                curs = con.cursor()
                query =  "select * from twitter where Stock= 'DOW' ;"
                curs.execute(query)
                records = curs.fetchall()
                df = pd.DataFrame(records)
                con.commit()
                curs.close()      
                connectionStringclose(con)
                return render_template('twitter.html' , Status= df) 
            elif value=="AAPL":
                con=connectionString()
                curs = con.cursor()
                query =  "select * from twitter where Stock= 'AAPL' ;"
                curs.execute(query)
                records = curs.fetchall()
                df = pd.DataFrame(records)
                con.commit()
                curs.close()      
                connectionStringclose(con)
                return render_template('twitter.html' , Status= df) 
            elif value=="XOM":
                con=connectionString()
                curs = con.cursor()
                query =  "select * from twitter where Stock= 'XOM' ;"
                curs.execute(query)
                records = curs.fetchall()
                df = pd.DataFrame(records)
                con.commit()
                curs.close()      
                connectionStringclose(con)
                return render_template('twitter.html' , Status= df) 
            elif value=="JNJ":
                con=connectionString()
                curs = con.cursor()
                query =  "select * from twitter where Stock= 'JNJ' ;"
                curs.execute(query)
                records = curs.fetchall()
                df = pd.DataFrame(records)
                con.commit()
                curs.close()      
                connectionStringclose(con)
                return render_template('twitter.html' , Status= df) 
            elif value=="T":
                con=connectionString()
                curs = con.cursor()
                query =  "select * from twitter where Stock= 'T' ;"
                curs.execute(query)
                records = curs.fetchall()
                df = pd.DataFrame(records)
                con.commit()
                curs.close()      
                connectionStringclose(con)
                return render_template('twitter.html' , Status= df) 
            elif value=="BAC":
                con=connectionString()
                curs = con.cursor()
                query =  "select * from twitter where Stock= 'BAC' ;"
                curs.execute(query)
                records = curs.fetchall()
                df = pd.DataFrame(records)
                con.commit()
                curs.close()      
                connectionStringclose(con)
                return render_template('twitter.html' , Status= df) 
            
        return render_template('twitter.html')
    return render_template('login.html')


# In[14]:

@app.route('/predict', methods=['POST','GET'])
def send():
    if request.method == 'POST':
        print ("POST")
        username=request.form['inputName']
        email=request.form['inputEmail']
        phonenumber=request.form['inputPhonenum']
        return render_template('algorithms.html')
    return render_template('algorithms.html')


# In[15]:

def connectionString():
    con=psycopg2.connect(dbname= 'userTable', host='stockdataanalysis.cuc3vel4graj.us-west-2.rds.amazonaws.com', 
    port= '5432', user= 'Khushbuprkh', password= 'Ads12345')
    con.autocommit = True
    return con
def connectionStringclose(con):
    con.close()


# In[16]:

if __name__=="__main__":
    app.run()


# In[ ]:



