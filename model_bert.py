# -*- coding: utf-8 -*-
"""Model_bert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-pHFBalsHKnNBvpIfDqfDI8qsLDKN3Ca
"""

import pickle
import os

from google.colab import drive

drive.mount('/content/drive')

root_path = '/content/drive/My Drive/'  #change dir to your project folder

os.chdir(root_path)

def store_data(file,db):
    '''file = name of pickle file'''
    dbfile = open(file, 'wb')
    # source, destination 
    pickle.dump(db, dbfile)
    dbfile.close()
    
    
def load_data(file): 
    '''file = name of pickle file'''    
    dbfile = open(file, 'rb')
    db = pickle.load(dbfile)
    dbfile.close() 
    return db

cased_embds = load_data('uncasedTrainedData.p')

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model = Sequential()

num_words = 100 #number of words in vocab
training_length = 50

# Embedding layer
# model.add(
#     Embedding(input_dim=num_words,
#               input_length = training_length,
#               output_dim=100,
#               weights=[embedding_matrix],
#               trainable=False,
#               mask_zero=True))

# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(LSTM(64, return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_words, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, GlobalMaxPooling1D,BatchNormalization,MaxPooling1D, Input, Concatenate, Activation, Average, Add, Maximum, Dropout
from keras.models import Model
import tensorflow as tf
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

dataset_loc = 'casedTrainedData.p'

vec_size = 768

def one_hot_encod(labels):
    #labels is a 2d array
    # 1. INSTANTIATE
    enc = preprocessing.OneHotEncoder()

    # 2. FIT
    enc.fit(labels)

    # 3. Transform
    onehotlabels = enc.transform(labels).toarray()
    print(onehotlabels.shape)
    return onehotlabels

a= np.array([1,2]).reshape(-1,1)
print(a.shape)
print(one_hot_encod(a))

# def one_hot_encod(labels):
#   arr = np.ndarray((len(labels), 2))
#   print(arr.shape)
#   for l in range(len(labels)):
#     ll = labels[l][0]
#     if ll==0:
#       arr[l,0] = 1
#       arr[l,1] = 0
#     else:
#       arr[l,1] = 1
#       arr[l,0] = 0
#   return arr
    # arr = []
    # for l in labels:
    #     l = l[0]
    #     if(l==0):
    #         arr.append([0,1])
    #     else:
    #         arr.append([1,0])
    # m = len(labels)
    # arr = np.array(arr).reshape(m,2)
    # return arr

sentences = load_data(dataset_loc)

train_out_temp = np.ones((len(sentences),1), dtype='int')

train_vecs_temp = np.zeros((len(sentences), vec_size), dtype='float')

for i in range(len(sentences)):
  temp = sentences[i]['is_hyper']
  if temp == True:
    train_out_temp[i,0] = 1
  else:
    train_out_temp[i,0] = 0
  train_vecs_temp[i,:] = sentences[i]['tensor']
  # print(i.keys())

# print(train_out)

train_vecs_univ, vecs_hold, train_out_univ, out_hold = train_test_split(train_vecs_temp, train_out_temp, test_size=0.2, random_state=42)

# train_out_temp = one_hot_encod(train_out_temp)

train_vecs, test_vecs, train_out, test_out = train_test_split(train_vecs_univ, train_out_univ, test_size=0.2, random_state=42)
# print("After split")
# print('Train Inp Shape: ', train_vecs.shape)
# print('Train Out Shape: ', train_out.shape)
# print("YO")

train_out = one_hot_encod(train_out)
test_out = one_hot_encod(test_out)

print('Train Inp Shape: ', train_vecs.shape)
print('Train Out Shape: ', train_out.shape)
print('Test Inp Shape: ', test_vecs.shape)
print('Test Out Shape: ', test_out.shape)

train_vecs = np.expand_dims(train_vecs, axis=2)

print("tvs",train_vecs.shape)
print('Train Inp Shape: ', train_vecs.shape)
print('Train Out Shape: ', train_out.shape)
print('Test Inp Shape: ', test_vecs.shape)
print('Test Out Shape: ', test_out.shape)

model = Sequential()
model.add(Conv1D(200, kernel_size=2, activation='relu',batch_input_shape=(None,vec_size,1)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print("shape now",train_vecs.shape)
print("shape", train_out.shape)
print(train_out[0])
print(train_out[1])
model.fit(x=train_vecs, y=train_out,batch_size=5,epochs=8)
model.summary()

test_vecs = np.expand_dims(test_vecs, axis=2)

model.evaluate(test_vecs,test_out,batch_size=5)

# # CNN with LSTM

# model = Sequential()
# model.add(LSTM(200, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, batch_input_shape=(None,vec_size,1)))
# # model.add(Conv1D(200, kernel_size=4, activation='relu',batch_input_shape=(None,vec_size,1)))
# # model.add(Flatten())
# # model.add(Dropout(0.5))
# # model.add(Conv1D(100, kernel_size=3, activation='relu',batch_input_shape=(None,200,1)))
# # model.add(Flatten())
# # model.add(Dropout(0.5))
# # model.add(Conv1D(50, kernel_size=2, activation='relu',batch_input_shape=(None,100,1)))
# model.add(Dropout(0.5))
# # model.add(Flatten())
# model.add(Dense(2, activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# print("shape now",train_vecs.shape)
# print("shape", train_out.shape)
# print(train_out[0])
# print(train_out[1])
# model.fit(x=train_vecs, y=train_out,batch_size=5,epochs=8)
# model.summary()
# model.evaluate(test_vecs,test_out,batch_size=5)

model = Sequential()
model.add(Conv1D(200, kernel_size=4, activation='relu',batch_input_shape=(None,vec_size,1)))
# model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Conv1D(100, kernel_size=3, activation='relu',batch_input_shape=(None,200,1)))
model.add(BatchNormalization(momentum=0.9))
model.add(MaxPooling1D())
# model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Conv1D(50, kernel_size=2, activation='relu',batch_input_shape=(None,100,1)))
model.add(BatchNormalization(momentum=0.9))
model.add(MaxPooling1D())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# print("shape now",train_vecs.shape)
# print("shape", train_out.shape)
# print(train_out[0])
# print(train_out[1])
# model.fit(x=train_vecs, y=train_out,batch_size=5,epochs=10)
# model.summary()
# model.evaluate(test_vecs,test_out,batch_size=5)

model = Sequential()
model.add(Conv1D(200, kernel_size=4, activation='relu',batch_input_shape=(None,vec_size,1)))
# model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Conv1D(100, kernel_size=3, activation='relu',batch_input_shape=(None,200,1)))
model.add(BatchNormalization(momentum=0.9))
model.add(MaxPooling1D())
# model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Conv1D(50, kernel_size=2, activation='relu',batch_input_shape=(None,100,1)))
model.add(BatchNormalization(momentum=0.9))
model.add(MaxPooling1D())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print("shape now",train_vecs.shape)
print("shape", train_out.shape)
print(train_out[0])
print(train_out[1])
model.fit(x=train_vecs, y=train_out,batch_size=5,epochs=10)
model.summary()
model.evaluate(test_vecs,test_out,batch_size=5)

# def conv1d_BN(max_len, embed_size):
#     '''
#     CNN with Batch Normalisation.
#     :param max_len: maximum sentence numbers, default=200
#     :param embed_size: ELMo embeddings dimension, default=1024
#     :return: CNN with BN model
#     '''
#     filter_sizes = [2, 3, 4, 5, 6]
#     num_filters = 128
#     inputs = Input(shape=(max_len,embed_size), dtype='float32')
#     conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]))(inputs)
#     act_0 = Activation('relu')(conv_0)
#     bn_0 = BatchNormalization(momentum=0.7)(act_0)

#     conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]))(inputs)
#     act_1 = Activation('relu')(conv_1)
#     bn_1 = BatchNormalization(momentum=0.7)(act_1)

#     conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]))(inputs)
#     act_2 = Activation('relu')(conv_2)
#     bn_2 = BatchNormalization(momentum=0.7)(act_2)

#     conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]))(inputs)
#     act_3 = Activation('relu')(conv_3)
#     bn_3 = BatchNormalization(momentum=0.7)(act_3)

#     conv_4 = Conv1D(num_filters, kernel_size=(filter_sizes[4]))(inputs)
#     act_4 = Activation('relu')(conv_4)
#     bn_4 = BatchNormalization(momentum=0.7)(act_4)

#     maxpool_0 = MaxPooling1D(pool_size=(max_len - filter_sizes[0]))(bn_0)
#     maxpool_1 = MaxPooling1D(pool_size=(max_len - filter_sizes[1]))(bn_1)
#     maxpool_2 = MaxPooling1D(pool_size=(max_len - filter_sizes[2]))(bn_2)
#     maxpool_3 = MaxPooling1D(pool_size=(max_len - filter_sizes[3]))(bn_3)
#     maxpool_4 = MaxPooling1D(pool_size=(max_len - filter_sizes[4]))(bn_4)

#     concatenated_tensor = Concatenate()([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
#     flatten = Flatten()(concatenated_tensor)
#     output = Dense(units=1, activation='sigmoid')(flatten)

#     model = Model(inputs=inputs, outputs=output)
#     #model = multi_gpu_model(model, gpus=gpus)
#     model.summary()
#     model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
#     return model

# max_len=512
# embed_size=768
# model = conv1d_BN(, embed_size)

# model = Sequential()
# model.add(Conv1D(200, kernel_size=2, activation='relu',batch_input_shape=(5,vec_size,1)))
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# print("shape now",train_vecs.shape)
# model.fit(x=train_vecs, y=train_out,batch_size=5,epochs=10)
# model.summary()

"""Main CNN Algorithm"""

import random
from sklearn.model_selection import StratifiedKFold
seed = 7

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

def call_fit(model,X,Y):
    X = np.expand_dims(X, axis=2)
    Y = one_hot_encod(Y)
    model.fit(x=X,y=Y, batch_size=16,epochs=10)
    return model

def call_eval(model,X,Y):
      X = np.expand_dims(X, axis=2)
      Y = one_hot_encod(Y)
      acc = model.evaluate(X,Y,batch_size=5)[1]
      return acc

def getFirstModel():
  model = Sequential()
  model.add(Conv1D(200, kernel_size=4, activation='relu',batch_input_shape=(None,vec_size,1)))
  model.add(BatchNormalization(momentum=0.9))
  model.add(MaxPooling1D())  
  # model.add(Flatten())
  # model.add(Dropout(0.4))
  model.add(Conv1D(100, kernel_size=3, activation='relu',batch_input_shape=(None,200,1)))
  model.add(BatchNormalization(momentum=0.9))
  model.add(MaxPooling1D())
  # model.add(Flatten())
  #model.add(Dropout(0.4))
  model.add(Conv1D(50, kernel_size=2, activation='relu',batch_input_shape=(None,100,1)))
  model.add(BatchNormalization(momentum=0.9))
  model.add(MaxPooling1D())
  model.add(Dropout(0.3))
  model.add(Flatten())
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  return model

def getBertha(max_len, embed_size):
  filter_sizes = [2,3,4,5,6]
  num_filters = 256
  inputs = Input(batch_shape=(None, embed_size, 1), dtype='float32')

  num_parallel = 2

  conv = [None for i in range(num_parallel)]
  act = [None for i in range(num_parallel)]
  bn = [None for i in range(num_parallel)]
  maxpool = [None for i in range(num_parallel)]

  acti = 'relu'

  for i in range(num_parallel):
    temp = Conv1D(200, kernel_size=filter_sizes[i], activation=acti)(inputs)
    temp = BatchNormalization(momentum=0.9)(temp)
    temp = MaxPooling1D()(temp)
    temp = Conv1D(100, kernel_size=filter_sizes[i], activation=acti)(temp)
    temp = BatchNormalization(momentum=0.9)(temp)
    temp = MaxPooling1D()(temp)
    temp = Conv1D(50, kernel_size=filter_sizes[i], activation=acti)(temp)
    temp = BatchNormalization(momentum=0.9)(temp)
    temp = MaxPooling1D()(temp)    
    temp = Dropout(0.4)(temp)
    maxpool[i] = Flatten()(temp)
    # conv[i] = Conv1D(num_filters, kernel_size=(filter_sizes[i]))(inputs)
    # act[i] = Activation('relu')(conv[i])
    # bn[i] = BatchNormalization(momentum=0.9)(act[i])
    # maxpool[i] = MaxPooling1D(pool_size=(max_len-filter_sizes[i]))(bn[i])

  # output = Dense(2, activation='softmax')(maxpool[0])

  conc_tensor = Concatenate()(maxpool)
  # flatten = Flatten()(conc_tensor)
  flatten = conc_tensor
  output = Dense(2, activation='softmax')(flatten)

  model = Model(input=inputs, outputs=output)
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.summary()
  return model

  # model = Sequential()
  # model.add(Conv1D(200, kernel_size=4, activation='relu',batch_input_shape=(None,vec_size,1)))
  # model.add(BatchNormalization(momentum=0.9))
  # model.add(MaxPooling1D())  
  # # model.add(Flatten())
  # # model.add(Dropout(0.5))
  # model.add(Conv1D(100, kernel_size=3, activation='relu',batch_input_shape=(None,200,1)))
  # model.add(BatchNormalization(momentum=0.9))
  # model.add(MaxPooling1D())
  # # model.add(Flatten())
  # #model.add(Dropout(0.5))
  # model.add(Conv1D(50, kernel_size=2, activation='relu',batch_input_shape=(None,100,1)))
  # model.add(BatchNormalization(momentum=0.9))
  # model.add(MaxPooling1D())
  # model.add(Dropout(0.3))
  # model.add(Flatten())
  # model.add(Dense(2, activation='softmax'))
  # model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  # return model

i=0
vals = []
model_arr = []
for train, test in kfold.split(train_vecs_univ, train_out_univ):
    i += 1
    # model = getModel()
    model = getBertha(512, 768)

    train_vecs,test_vecs = train_vecs_temp[train],train_vecs_temp[test]
    train_out,test_out = train_out_temp[train],train_out_temp[test]
    print(np.shape(train_vecs))
    print(np.shape(test_vecs))
    print(np.shape(train_out))
    print(np.shape(test_out))
    print("Fold: %s " % i)
    model = call_fit(model,train_vecs,train_out)
    acc = call_eval(model,test_vecs,test_out)
    print("Fold %d | Accuracy %f " % (i, acc))
    vals.append(acc)
    model_arr.append(model)

vals = np.array(vals)
# print(vals)
print("Mean Accuracy over all Folds:", np.mean(vals))
# print("Final score: %.4f%% (+/- %.4f%%)" % (np.mean(cvscores), np.std(cvscores)))

print(type(out_hold))
final_models = []
final_vals = []

for _ in range(3):
  maxi = 0
  maxe = vals[0]
  for i in range(len(vals)):
    for j in range(len(vals)):
      if vals[j] > maxe:
        maxe = vals[j]
        maxi = j
  final_models.append(model_arr[maxi])
  final_vals.append(vals[maxi])
  vals[maxi] = -1

print(final_vals)
print(final_models)
vecs_hold = np.expand_dims(vecs_hold, axis=2)

"""Ensemble Learning"""

# vecs_hold, out_hold

one_out_hold = one_hot_encod(out_hold)
pred1 = np.argmax(final_models[0].predict(vecs_hold), axis=1)
pred2 = np.argmax(final_models[1].predict(vecs_hold), axis=1)
pred3 = np.argmax(final_models[2].predict(vecs_hold), axis=1)
one = 0
two = 0
three = 0
accc = 0
for i in range(len(out_hold)):
  counter = 0
  if pred1[i] == out_hold[i]:
    counter+=1
  if pred2[i] == out_hold[i]:
    counter+=1
  if pred3[i] == out_hold[i]:
    counter+=1
  if counter > 2:
    accc+=1
print(accc/len(out_hold))
