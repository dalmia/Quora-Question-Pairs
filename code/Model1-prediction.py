
# coding: utf-8

# In[23]:

from keras.models import load_model
import pickle
import numpy as np
import pandas as pd


# In[3]:

test_question_1 = pickle.load(open('test_question_1.pkl', 'r'))
test_question_2 = pickle.load(open('test_question_2.pkl', 'r'))
map_index_vec = pickle.load(open('map_index_vec.pkl', 'r'))


# In[4]:

N = len(test_question_1)


# In[5]:

test_data = []
print('Creating Testing data...');
for i, sentence1, sentence2 in zip(range(1, N), test_question_1, test_question_2):
    if i % 100000 == 0:
        print(i)

    vector1 = [map_index_vec[index] for index in sentence1]
    vector2 = [map_index_vec[index] for index in sentence2]
    mean_vec1 = np.mean(vector1, axis=0)
    mean_vec2 = np.mean(vector2, axis=0)

    test_data.append(np.concatenate((mean_vec1, mean_vec2)))
print('Done.')


# In[32]:

test_data = np.array(test_data)


# In[6]:

model = load_model('./weights_1/weights.hdf5')


# In[13]:

predictions = model.predict_classes(test_data, verbose=1)


# In[20]:

indices = range(len(predictions))


# In[21]:

predictions = predictions.reshape(-1)


# In[27]:

df = pd.DataFrame({'is_duplicate': predictions})


# In[29]:

df.index.name = 'test_id'


# In[31]:

df.to_csv('test_submission_model_1.csv')


# In[ ]:



