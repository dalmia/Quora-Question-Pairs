import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

print('Imports done.')

print('Reading csv...')
data = pd.read_csv('../input/train_cleaned.csv')
data = data.dropna()
print('Done...')

print('Loading glove vectors...')
glove_model = pickle.load(open('model_glove_common_crawl.pkl', 'rb'))
print('Done.')

dim = 300
N = len(data['question1'])

train_data = []
train_labels = []
batch_size = 256
epochs = 20
log_dir = './logs'

print('Creating Training data...');
for i, sentence1, sentence2, label in zip(range(1,N), data['question1'].values,
                                          data['question2'].values, data['is_duplicate']):
    if i % 10000 == 0:
        print(i)
    vector1 = [glove_model[word] for word in sentence1.split(" ") if word in glove_model]
    if len(vector1) == 0:
        mean_vec1 = glove_model['unk']
    else:
        mean_vec1 = np.mean(vector1, axis=0)

    vector2 = [glove_model[word] for word in sentence2.split(" ") if word in glove_model]

    if len(vector2) == 0:
        mean_vec2 = glove_model['unk']
    else:
        mean_vec2 = np.mean(vector2, axis=0)

    train_data.append(np.concatenate((mean_vec1, mean_vec2)))
    train_labels.append(label)
print('Done.')

print('Converting data to array...')
train_data = np.array(train_data)
train_labels = np.array(train_labels)
print('Done.')

print('Building model...')
model = Sequential()
model.add(Dense(1, activation='sigmoid', input_shape=(2 * dim,)))
sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
print('Done...')

print('Training...')
model.fit(train_data, train_labels,
          epochs=epochs,
          validation_split=0.2,
          batch_size=batch_size, callbacks=[tb])
print('Done.')
