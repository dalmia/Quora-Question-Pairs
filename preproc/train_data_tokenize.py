
# coding: utf-8

# In[1]:

import pickle
from nltk.tokenize import StanfordTokenizer
import pandas as pd
import numpy as np
print('Imports done.')

print('Loading glove vectors...')
glove_model = pickle.load(open('preproc/model_glove_wiki.pkl', 'r'))
print('Done.')

print('Loading data...')
train_data = pd.read_csv('input/train.csv')
print('Done.')

dim = 50

train_question_1 = []
train_question_2 = []
train_labels = []
map_index_vec = dict()
map_word_index = dict()

tokenizer = StanfordTokenizer(options={"ptb3Escaping": True})

words = set()
for col in ['question1', 'question2']:
    sentences = []
    print('Processing column: %s' % col)
    for i, sentence in enumerate(train_data[col]):
        if i % 10000 == 0:
            print('Sentence: %d' % i)
            
        split = tokenizer.tokenize(sentence)
        new_sentence = []
        for word in split:
            word = word.encode('utf-8').strip()
            word = word.lower()
                        
            if word in glove_model:
                if word not in words:
                    words.add(word)
                new_sentence.append(word)
            else:
                if 'unk' not in words:
                    words.add('unk')
                new_sentence.append('unk')
            
        sentences.append(" ".join(new_sentence))
    
    train_data[col] = sentences
print('Done.')

print('Saving cleaned data...')
train_data.to_csv('input_clean/train_clean.csv', )
train_data.fillna('', inplace=True)
print('Done.')

words = list(words)

map_index_vec[0] = np.zeros(dim) 

for i, word in enumerate(words):
    map_index_vec[i + 1] = glove_model[word]
    map_word_index[word] = i + 1

N = len(train_data)


# In[ ]:

print('Creating training data...')
for i, sentence1, sentence2, label in zip(range(1,N+1), train_data['question1'],
                            train_data['question2'], train_data['is_duplicate']):
    if i % 10000 == 0:
        print('Index: %s' % i)
    
    vector1 = [map_word_index[word] for word in sentence1.split(" ")]
    vector2 = [map_word_index[word] for word in sentence2.split(" ")]
    
    train_question_1.append(vector1)
    train_question_2.append(vector2)
    train_labels.append(label)
print('Done.')


# In[ ]:

with open('preproc/map_index_vec.pkl', 'wb') as output:
    pickle.dump(map_index_vec, output, pickle.HIGHEST_PROTOCOL)

# In[ ]:

with open('preproc/map_word_index.pkl', 'wb') as output:
    pickle.dump(map_word_index, output, pickle.HIGHEST_PROTOCOL)
