
'''
CISC452 Group Project
Matthew Sherar 10093010

Predicting cost of insurance claim for a company based on 116 categorical attributes
and 14 numerical attributes

Neural Network used is part of Keras library

'''

#Import neccasary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Activation, Dropout
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD

#import the data from csv
#approximately 181 000 entries
train  = pd.read_csv("train.csv")


'''for each categroical column in the training set, the label encoder converts the 
categorical value to a numerical one'''
for i in range(116):
	train['cat%d'%(i+1)] = preprocessing.LabelEncoder().fit_transform(train['cat%d'%(i+1)])

#this was used while testing to reduce execution time 
#train.to_csv("trainNoCats.csv")

#set of 'loss' column to  be the target column we wish to predict
y = train['loss']
#remove the loss column from the rest of the  data
train = train.drop('loss', axis=1)

#split up the data into training and test sets, with a test set size of 30% 
x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.3)


#convert the data from DataFrames to arrays to be used in the neural network
x_train = x_train.as_matrix()
x_test = x_test.as_matrix()
y_train = y_train.as_matrix()
y_test = y_test.as_matrix()



#function to build the actual neural network
def nnModel():
	#sequential layer model
     model = Sequential()
     #first layer has input dimension of 132, 1024 nodes, activation function 'relu' for rectified linear unit
     model.add(Dense(1024, input_dim=132, init='normal', activation='relu'))
     #dropout is a simple way to prevent overfitting
     model.add(Dropout(0.5))
     #add 2 more layers with 1024 nodes
     model.add(Dense(1024, init='normal', activation='relu'))
     model.add(Dropout(0.5))
     model.add(Dense(1024, init='normal', activation='relu'))
     model.add(Dropout(0.5))
     #one final node in the last layer, no activation function since we want to predict a continuous value
     model.add(Dense(1, init='normal'))
     #use mean absolute error as error calculation
     #optimizer Nadam is like Root Mean Square propogation with momentum and small learning rate
     model.compile(loss='mean_absolute_error', optimizer='Nadam')
     return model


#put the nn together
estimators = []
#converts array values to numbers
estimators.append(('standardize', StandardScaler()))
#sets batch size and run time (epochs)
estimators.append(('mlp', KerasRegressor(build_fn=nnModel, nb_epoch=10, batch_size=32, verbose=1)))
pipeline = Pipeline(estimators)
#trains the nn on the trianing data
pipeline.fit(x_train, y_train)






