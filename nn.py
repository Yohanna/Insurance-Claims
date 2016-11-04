from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers import Activation, Dropout
from keras.utils.np_utils import to_categorical

import pandas as pd
import numpy as np

#Read dataset
fullDataset = pd.read_csv("train.csv")
sampleDataset = fullDataset.sample(frac=0.15)
sampleTestSet = fullDataset.sample(frac=0.1)
#select certain columns
data = sampleDataset[['num_var4','var38', 'var15', 'imp_op_var40_efect_ult1', 'var3', 'var21']]
data_y = sampleDataset.TARGET
data = data.as_matrix()
data_y = data_y.as_matrix()

#full testdataset
#fullTest = pd.read_csv("test.csv")
testCols = sampleTestSet[['num_var4','var38', 'var15', 'imp_op_var40_efect_ult1', 'var3', 'var21']]
test_y = sampleTestSet.TARGET
test_y = test_y.as_matrix()
testCols = testCols.as_matrix()

model = Sequential()
model.add(Dense(1, input_dim=6, activation='sigmoid'))

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=6, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))

#model.add(Dense(6, input_dim=6, init='normal', activation='relu'))
#model.add(Dense(1, init='normal', activation='sigmoid'))


#model.add(Dense(output_dim=12, input_dim=6))
#model.add(Activation("relu"))
#model.add(Dense(output_dim=2))
#model.add(Activation("softmax"))


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(data, data_y, nb_epoch=5, batch_size=32)

loss_and_metrics = model.evaluate(testCols, test_y, batch_size=32)

