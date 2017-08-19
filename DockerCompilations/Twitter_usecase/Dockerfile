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


RUN pip install tweepy
RUN pip install psycopg2
RUN pip install textblob
RUN pip install pytrends


ADD twitter.py twitter.py
ADD config.json config.json

#ENTRYPOINT ["bash" , "run.sh"]