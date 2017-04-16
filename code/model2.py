import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, concatenate, LSTM
from keras.layers import Embedding, Bidirectional
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

print('Loading data...')
data = pd.read_csv('../input/train_clean.csv')
data = data.dropna()
print('Done.')

print('Loading glove vectors...')
glove_model = pickle.load(open('reduce_glove_wiki.pkl', 'rb'))
glove_model_index = pickle.load(open('reduce_glove_wiki_index.pkl', 'rb'))
print('Done.')

N = len(data['question1'])

train_data_1 = []
train_data_2 = []
train_labels = []
maxlen1 = 0
maxlen2 = 0

print('Creating training data...')
for i, sentence1, sentence2, label in zip(range(1,N), data['question1'].values, 
                                          data['question2'].values, data['is_duplicate']):
    if i % 50000 == 0:
        print('Iter: %s' % i)
  
    vector1 = [glove_model_index[word] for word in word_tokenize(unicode(sentence1, errors='ignore').encode('ascii','ignore')) if word in glove_model_index]
    vector2 = [glove_model_index[word] for word in word_tokenize(unicode(sentence2, errors='ignore').encode('ascii','ignore')) if word in glove_model_index]
    
    l1 = len(vector1)
    l2 = len(vector2)
    maxlen1 = max(maxlen1, l1)
    maxlen2 = max(maxlen2, l2)
    
    if l1 == 0:
        vector1 = [glove_model_index['unk']]
        
    if l2 == 0:
        vector2 = [glove_model_index['unk']]
        
    train_data_1.append(vector1)
    train_data_2.append(vector2)
    train_labels.append(label)
print('Done.')
 
dim = 50
batch_size = 256
epochs = 100
log_dir = './logs'

train_labels = np.array(train_labels)
print(len(train_data_1))

print('Padding sequences...')
train_data_1 = pad_sequences(train_data_1, maxlen=maxlen1, padding='post')
train_data_2 = pad_sequences(train_data_2, maxlen=maxlen2, padding='post') 
print('Done.')

print(len(train_data_1))
train_data_1 = np.vstack(train_data_1)
train_data_2 = np.vstack(train_data_2)

print('Creating weight matrix...')
n_symbols = len(glove_model_index) + 1          # adding 1 to account for 0th index
embedding_weights = np.zeros((n_symbols + 1, 50))
for word,index in glove_model_index.items():
    embedding_weights[index,:] = glove_model[word]
print('Done.')

print('Building model...')
in1 = Input(shape=(maxlen1,), dtype='int32', name='in1')
x1 = Embedding(output_dim=dim, input_dim=n_symbols + 1, input_length=maxlen1, 
               weights=[embedding_weights], name='x1')(in1)

in2 = Input(shape=(maxlen2,), dtype='int32', name='in2')
x2 = Embedding(output_dim=dim, input_dim=n_symbols + 1, input_length=maxlen2, 
               weights=[embedding_weights], name='x2')(in2)

l = Bidirectional(LSTM(units=100, return_sequences=False))
y1 = l(x1)
y2 = l(x2)

y = concatenate([y1, y2])
out = Dense(1, activation='sigmoid')(y)
model = Model(inputs=[in1, in2], outputs=[out])
print('Done.')
print(model.summary())

sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
model.fit([train_data_1, train_data_2], train_labels,
          epochs=epochs,
          validation_split=0.2,
          batch_size=batch_size, callbacks=[tb])
