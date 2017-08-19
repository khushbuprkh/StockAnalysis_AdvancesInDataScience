# Machine Learning on Stock Prediction 

## Folders to watch : 
### DockerCompilations :
It contains all the required Python Script to run and make our models and recreate the Scenario that we have done. 

### Ipython Notebook :
Contains all the files for EDA , Training all Prediction(Linear Regression,Random Forest,Knn with Neural Network, LSTM, Twitter Sentiment Analysis Models interactively. 

## MLAlgorithm: 
Conatins the raw file generated file with can be used ditrectly , wothout compling the models. All Models are saved in Json and  HDF5 binary data format 


## Live Application Running on : http://ec2-54-200-158-241.us-west-2.compute.amazonaws.com/


## Scope : DOW 30 and TOP 5 Companies around the Industry: 
Apple , AT&T , Exxon Mobil,Bank of America ,Jhonson & Jhonson 

## Prediction day Range : 1 -7 Days 
## Final Model Chosen : 
Feed Forward Neural Netowork 
## MAPE:1.3% 


## Docker Execution
docker pull khushbuprkh/twitter
docker run -ti khushbuprkh/twitter
- in the bash : Vim config.json ## update the config file
- in the bash : python twitter.py

docker pull khushbuprkh/final
docker run -ti khushbuprkh/final
- in the bash : Vim config.json ## update the config file
- in the bash : make

## Video Link
https://youtu.be/LETQ6bjLGhQ

## Thank you 


Note : This is a part of Final Project Submission for ADS INFO 7250 Summer 2017 Course 
