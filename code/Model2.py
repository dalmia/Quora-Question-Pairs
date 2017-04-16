
# coding: utf-8

# In[13]:

import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, concatenate, LSTM
from keras.layers import Bidirectional
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from  nltk.tokenize import word_tokenize

from embeddings import Embedding


# In[7]:
print('Loading pickles...')
train_question_1 = pickle.load(open('train_question_1.pkl', 'rb'))
train_question_2 = pickle.load(open('train_question_2.pkl', 'rb'))
train_labels = pickle.load(open('train_labels.pkl', 'rb'))
map_index_vec = pickle.load(open('map_index_vec.pkl', 'rb'))
print('Done.')

# In[9]:

maxlen1, maxlen2 = 0, 0
for one, two in zip(train_question_1, train_question_2):
    maxlen1 = max(maxlen1, len(one))
    maxlen2 = max(maxlen2, len(two))


# In[10]:

N = len(train_question_1)


# In[3]:

dim = 50
batch_size = 256
epochs = 100
log_dir = './logs'


# In[15]:

train_labels = np.array(train_labels)


# In[17]:
print('Padding questions...')
train_question_1 = pad_sequences(train_question_1, maxlen=maxlen1, padding='post')
train_question_2 = pad_sequences(train_question_2, maxlen=maxlen2, padding='post')
print('Done...')

# In[19]:

train_question_1 = np.vstack(train_question_1)
train_question_2 = np.vstack(train_question_2)


# In[8]:

n_symbols = len(map_index_vec)
embedding_weights = np.zeros((n_symbols, 50))
for index, vec in map_index_vec.items():
    embedding_weights[index,:] = vec


# In[12]:
print('Creating model...')
embedding = Embedding(output_dim=dim, input_dim=n_symbols, input_length=None,
               weights=[embedding_weights])

in1 = Input(shape=(None,), dtype='int32', name='in1')
x1 = embedding(in1)

in2 = Input(shape=(None,), dtype='int32', name='in2')
x2 = embedding(in2)

l = Bidirectional(LSTM(units=100, return_sequences=False))
y1 = l(x1)
y2 = l(x2)

y = concatenate([y1, y2])
out = Dense(1, activation='sigmoid')(y)
model = Model(inputs=[in1, in2], outputs=[out])
print('Done.')
print(model.summary())


# In[14]:

sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
checkpointer = ModelCheckpoint(filepath="./weights/weights.hdf5", verbose=1, save_best_only=True)


# In[15]:
model.fit([train_question_1, train_question_2], train_labels,
          epochs=epochs,
          validation_split=0.2,
          batch_size=batch_size, callbacks=[checkpointer, tb])


# In[ ]: