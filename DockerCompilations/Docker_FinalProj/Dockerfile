FROM ubuntu:latest

RUN apt-get update -q && apt-get install -yqq \
    apt-utils \
    git \
    vim \
    nano \
    ssh \
    gcc \
    make \
    build-essential \
    libkrb5-dev \
    sudo 
	
RUN apt-get install -y python python-dev python-distribute python-pip
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install boto3
RUN pip install -U boto
RUN pip install bokeh
RUN pip install jupyter


RUN pip install cython


RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr && \
  make && \
  make install
RUN pip install TA-Lib
RUN pip install keras
RUN pip install pandas_datareader
RUN pip install plotly
RUN pip install tweepy
RUN pip install h5py
RUN pip install seaborn
RUN pip install sklearn
RUN pip install scipy
RUN pip install tensorflow
RUN pip install bs4
RUN pip install quandl
Run pip install matplotlib
Run pip install requests
RUN pip install multiprocessing
RUN pip install psycopg2



ADD DataIngestion.py DataIngestion.py
ADD DowPredictionJob.py DowPredictionJob.py
ADD DowPredictionEmail.py DowPredictionEmail.py
ADD AllStockPredictionModelling.py AllStockPredictionModelling.py
ADD AllStockTestModel.py AllStockTestModel.py
ADD config.json config.json
ADD run.sh run.sh
ADD Makefile Makefile

#ENTRYPOINT ["bash" , "run.sh"]