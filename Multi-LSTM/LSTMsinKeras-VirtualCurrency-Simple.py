#!/usr/bin/env python
# coding: utf-8

# refer to https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# 
# to tune parameters
# refer to http://yangguang2009.github.io/2017/01/08/deeplearning/grid-search-hyperparameters-for-deep-learning/

# In[1]:


from __future__ import print_function

import json
import numpy as np
import os
import pandas as pd
import urllib
import math

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


# connect to poloniex's API
url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1546300800&end=9999999999&period=300&resolution=auto'

# parse json returned from the API to Pandas DF
openUrl = urllib.request.urlopen(url)
r = openUrl.read()
openUrl.close()
d = json.loads(r.decode())
df = pd.DataFrame(d)

original_columns=[u'date', u'close',  u'high', u'low', u'open', u'volume']
new_columns = ['Timestamp','Close','High','Low','Open','Volume']
df = df.loc[:,original_columns]
df.columns = new_columns
df.to_csv('bitcoin201901to201905.csv',index=None)


# In[2]:


df = df.set_index('Timestamp')
df.head()


# In[3]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import seaborn as sns
import numpy as np
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# In[4]:


pyplot.plot(df['Close'].values, label='price')
pyplot.legend()
pyplot.show()


# In[5]:


sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0)


# In[6]:



# load dataset
#dataset = read_csv('update_20190301_bitbank_f.csv', header=0, index_col=0)
#values = dataset.values

#dataset.head()

values = df['Close'].values
values = values.reshape(-1, 1)
print(values)


# In[7]:



# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

#test =  series_to_supervised(values, 1, 1)
#print(test.head())
#print(test.shape)


# In[8]:


print(values.shape)
print(reframed.shape)
print('---------')
#print(reframed.columes)

# split into train and test sets
values = reframed.values
print(values.shape)

n_train_rate = 0.7
n_train = values.shape[0] * n_train_rate
n_train = math.floor(n_train)
print(n_train)

train = values[:n_train, :]
test = values[n_train:, :]


# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[9]:


import math

# drop columns we don't want to predict
# 只留下 close 列
#reframed.drop(reframed.columns[[6, 7, 8, 10, 11]], axis=1, inplace=True)
#print(reframed.head())
 
# split into train and test sets
values = reframed.values
print(values.shape)

n_train_rate = 0.7
n_train = values.shape[0] * n_train_rate
n_train = math.floor(n_train)
print(n_train)

train = values[:n_train, :]
test = values[n_train:, :]


# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[10]:


#!pip install tqdm --upgrade
#!pip install hyperopt --upgrade
#!pip install hyperas --upgrade
type(train_X)


# In[16]:


def data():
    global train_X, test_X, train_y, test_y
    return train_X, test_X, train_y, test_y

# design network
def model(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_X, train_y, epochs={{choice([10, 25, 50])}}, batch_size={{choice([8, 16, 32,50])}}, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    score, acc = model.evaluate(test_X, test_y, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials(),
                                          notebook_name='LSTMsinKeras-VirtualCurrency-Simple')
print("Evalutation of best performing model:")
print(best_model.evaluate(test_X, test_y))


# In[ ]:



# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[ ]:


# make a prediction
yhat = model.predict(test_X)
print('yhat.shape', yhat.shape, yhat[0:5, :])
test_X_reshape = test_X.reshape((test_X.shape[0], test_X.shape[2]))
print(test_X_reshape.shape, test_X_reshape[0:5, -7:])
      
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X_reshape[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
print('inv_yhat.shape', inv_yhat.shape, inv_yhat[0:5, :])

inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X_reshape[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
# 因为inv_y 预测是下一时刻的值，所以需要把 inv_yhat 往后 shift 一个时刻
rmse = sqrt(mean_squared_error(inv_y[:-1], inv_yhat[1:]))
print('Test RMSE: %.3f' % rmse)


# In[ ]:


print(test_X.shape)
#print(range(test_X.shape))


#pyplot.plot( inv_y[-100:-1], label='predict')
#pyplot.plot( inv_yhat[-99:], label='actual')
pyplot.plot( inv_y, label='predict')
pyplot.plot( inv_yhat, label='actual')
pyplot.legend()
pyplot.show()

#涨跌的判准率

#获取预测跟实际对应元素值，是否大于0
a = np.diff(inv_y) > 0
b = np.diff(inv_yhat) > 0

#比较相同值的个数
print(sum(a ==b)/a.shape[0])


# In[14]:




x = 6
def func():
    global x
    print(x)
    return x
func()


# In[ ]:




