'''
CISC452 Group Project
Matthew Sherar 10093010

Predicting cost of insurance claim for a company based on 116 categorical attributes
and 14 numerical attributes

Neural Network used is part of Keras library running on Tensorflow

'''

#Import neccasary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Activation, Dropout
from sklearn.pipeline import Pipeline
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.decomposition import PCA
import sklearn.metrics
from sklearn.metrics import mean_absolute_error

train  = pd.read_csv("train.csv")



'''for each categroical column in the training set, the label encoder converts the 
categorical value to a numerical one'''
for i in range(116):
	train['cat%d'%(i+1)] = preprocessing.LabelEncoder().fit_transform(train['cat%d'%(i+1)])

#create new column to specify if the claim is over $10000
def f(row):
     if row['loss'] > 10000:
          val = 1
     else:
          val = 0
     return val

train['large'] = train.apply(f, axis=1) #apply the function

t = train[train.large == 1]  
t1 = train[train.large == 0].sample(5880)
t = t.append(t1) 
t = t.sample(frac = 1)  
train = t 



#set of 'large' column to  be the target column we wish to predict
y = train['large']
#remove the loss column from the rest of the  data
train = train.drop('loss', axis=1)
train = train.drop('large', axis=1)



#split up the data into training and test sets, with a test set size of 30% 
x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.3)

#drop other unnecassary columns
x_train = x_train.drop("Unnamed: 0", axis=1)
x_train = x_train.drop("id", axis =1)
x_train = x_train.as_matrix()
x_test = x_test.drop("Unnamed: 0", axis=1)
x_test = x_test.drop("id", axis =1)
x_test = x_test.as_matrix()
y_train = y_train.as_matrix()
#Tested with PCA
#pca = PCA(n_components = 20)
#x_train = pca.fit_transform(x_train)
#y_train = pca.fit_transform(y_train)
#x_test = pca.fit_transform(x_test)



model = Sequential()
#first layer has input dimension of 130, 400 nodes, activation function 'relu' for rectified linear unit
model.add(Dense(400, input_dim=130, init='normal'))
model.add(Activation('relu'))
#add more layers
model.add(Dense(350, init='normal'))
model.add(Activation('relu'))
model.add(Dense(250, init='normal'))
model.add(Activation('relu'))
model.add(Dense(250, init='normal'))
model.add(Activation('relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))

#use mean absolute error as error calculation
#optimizer adadelta is like Root Mean Square propogation with momentum and small stochastic learning rate
model.compile(loss='binary_crossentropy', optimizer='adadelta')

##train the model
model.fit(x_train, y_train, nb_epoch=10, batch_size=32)


#predict test set data
pred = pd.DataFrame(model.predict(x_test))

yTest = pd.DataFrame(y_test)

#binary high or low claim decision
def cat(row):
	if row[0] > 0.5:
		val  = 1
	else:
		val = 0
	return val

pred['cat'] = pred.apply(cat, axis=1)


#Print binary classifier metrics
print(sklearn.metrics.classification_report(yTest, pred['cat']))
print(sklearn.metrics.confusion_matrix(yTest, pred['cat']))


#NN models trained on low, high, and all claims respectively
def lowModel():
     modelLow = Sequential()
     modelLow.add(Dense(1000, input_dim=130, init='he_normal'))
     modelLow.add(Activation('relu'))
     modelLow.add(Dense(650, init='he_normal'))
     modelLow.add(Activation('relu'))
     modelLow.add(Dense(650,  init='he_normal'))
     modelLow.add(Activation('relu'))
     modelLow.add(Dense(450,  init='he_normal'))
     modelLow.add(Activation('relu'))
     modelLow.add(Dense(1, init='he_normal'))
     modelLow.compile(loss='mae', optimizer='adagrad')
     return modelLow

def highModel():
     modelHigh = Sequential()
     modelHigh.add(Dense(1000, input_dim=130, init='he_normal'))
     modelHigh.add(Activation('relu'))
     modelHigh.add(Dense(650, init='he_normal'))
     modelHigh.add(Activation('relu'))
     modelHigh.add(Dense(450, init='he_normal'))
     modelHigh.add(Activation('relu'))
     modelHigh.add(Dense(250, init='he_normal'))
     modelHigh.add(Activation('relu'))
     modelHigh.add(Dense(1, init='he_normal'))
     modelHigh.compile(loss='mae', optimizer='adagrad')
     return modelHigh


def normalModel():
     modelNorm = Sequential()
     modelNorm.add(Dense(1000, input_dim=130, init='he_normal'))
     modelNorm.add(Activation('relu'))
     modelNorm.add(Dense(650, init='he_normal'))
     modelNorm.add(Activation('relu'))
     modelNorm.add(Dense(450, init='he_normal'))
     modelNorm.add(Activation('relu'))
     modelNorm.add(Dense(250, init='he_normal'))
     modelNorm.add(Activation('relu'))
     modelNorm.add(Dense(1, init='he_normal'))
     modelNorm.compile(loss='mae', optimizer='adagrad')
     return modelNorm

#read in data again - this has already been encoded
train  = pd.read_csv("trainNoCats.csv")

def f(row):
     if row['loss'] > 10000:
          val = 1
     else:
          val = 0
     return val

train['large'] = train.apply(f, axis=1)
y = train['loss']

x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.3)

#Select all data for the normal network
n = x_train.drop('large', axis=1)
n = n.drop('loss', axis=1)
n = n.drop('id', axis=1)
n = n.drop('Unnamed: 0', axis =1)
n = n.as_matrix()
estimatorsNorm = []
estimatorsNorm.append(('standardize', StandardScaler()))
estimatorsNorm.append(('mlp', KerasRegressor(build_fn=normalModel, nb_epoch=10, batch_size=50, verbose=1)))
pipelineNorm = Pipeline(estimatorsNorm)
#train the model
pipelineNorm.fit(n, y_train)



#select all the small claims and a few large claims for the 'low' model
t = x_train[x_train.large == 0]
t1 = x_train[x_train.large == 1].sample(1000)
t = t.append(t1)
t = t.sample(frac=1) #shuffles the data
tY = t['loss']
t = t.drop('loss', axis=1)
t = t.drop('id', axis=1)
t = t.drop('Unnamed: 0', axis =1)
t = t.drop('large', axis=1)
t = t.as_matrix()
tY = tY.as_matrix()
estimatorsLow = []
estimatorsLow.append(('standardize', StandardScaler()))
estimatorsLow.append(('mlp', KerasRegressor(build_fn=lowModel, nb_epoch=5, batch_size=50, verbose=1)))
pipelineLow = Pipeline(estimatorsLow)
#train the low model
pipelineLow.fit(t, tY)

#select all the high claims and some of the low claims to train the model
k = x_train[x_train.large == 1]
k1 = x_train[x_train.large == 0].sample(1000)
k=k.append(k1)
k =k.sample(frac=1) # shuffles the data
kY = k['loss']
k = k.drop('loss', axis=1)
k = k.drop('id', axis=1)
k = k.drop('large', axis=1)
k = k.drop('Unnamed: 0', axis=1)
k = k.as_matrix()
kY = kY.as_matrix()
estimatorsHigh = []
estimatorsHigh.append(('standardize', StandardScaler()))
estimatorsHigh.append(('mlp', KerasRegressor(build_fn=highModel, nb_epoch=30, batch_size=50, verbose=1)))
pipelineHigh = Pipeline(estimatorsHigh)
#train the model
pipelineHigh.fit(k, kY)

#putting all three models together
x_test = x_test.drop('loss', axis=1)
x_test = x_test.drop('large', axis=1)
x_test = x_test.drop('Unnamed: 0', axis=1)
x_test = x_test.drop('id', axis=1)
x_test = x_test.as_matrix()
pred = pd.DataFrame(model.predict(x_test)) #predict binary clasification
lowPred = pipelineLow.predict(x_test) #prediction from low model
highPred = pipelineHigh.predict(x_test) #prediction from high model
normPred = pipelineNorm.predict(x_test) #prediction from normal model

#Put all predicitions into one DataFrame
pred['low'] = lowPred
pred['high'] = highPred
pred['norm'] = normPred

y_test = pd.DataFrame(y_test)
y_test = y_test.set_index(pred.index)

#Add actual loss value to the same DataFrame
pred['act'] = y_test

#Weighted sum of the model outputs
def out(row):
     if row[0] > 0.95:
          return (row['high']*0.75 + row['norm']*0.25)
     else:
          return (row['norm']*0.5 + row['low']*0.5)

pred['out'] = pred.apply(out, axis=1)

##Error stats
print("Normal Error")
print(mean_absolute_error(pred['act'], pred['norm']))

print("Weighted MultiNetwork Error")
print(mean_absolute_error(pred['act'], pred['out']))

#Figure of the error graph
'''
fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_title('Prediction vs Actual')
ax.set_xlabel('Predicted Claim')
ax.set_ylabel('Actual Claim')
ax.scatter(pred['norm'],pred['act'])
plt.show()


'''













