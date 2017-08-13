
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
print 'Import done'


# In[2]:

df_train = pd.read_csv('input/train.csv')
df_train.head()

df_test = pd.read_csv('input/test.csv')
df_test.head()
print 'Input read'


# In[4]:

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in re.findall(r"[\w']+|[.,!?;]", str(row['question1']).lower()):
        if word not in stops:
            q1words[word] = 1
    for word in re.findall(r"[\w']+|[.,!?;]", str(row['question2']).lower()):
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = np.float32(len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


# In[5]:

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)


# In[6]:

words = (" ".join(train_qs)).lower().split()
tfid = TfidfVectorizer(min_df=1)

tfid.fit(words)
print 'tfid.fit done'


# In[7]:

def tfidf_word_match_share(row, tfid):
    q1words = {}
    q2words = {}
    for word in re.sub(r'[^\w\s]',' ',str(row['question1']).lower()).split():
        if word not in stops and word in tfid.vocabulary_.keys():
            q1words[word] = 1
    for word in re.sub(r'[^\w\s]',' ',str(row['question1']).lower()).split():
        if word not in stops and word in tfid.vocabulary_.keys():
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = 0.0
    total_weights = 0.0
    for w in q1words.keys():
        total_weights += tfid.idf_[tfid.vocabulary_[w]]
        if w in q2words:
            shared_weights += tfid.idf_[tfid.vocabulary_[w]]
    for w in q2words.keys():
        total_weights += tfid.idf_[tfid.vocabulary_[w]]
        if w in q1words:
            shared_weights += tfid.idf_[tfid.vocabulary_[w]]

    R = np.float32(shared_weights) / (total_weights)
    return R


# In[8]:

ques = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
print ques.shape


# In[9]:

q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])
print 'qdict created'


# In[10]:

def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))


# In[11]:

x_train = pd.DataFrame()
x_test = pd.DataFrame()


# In[ ]:

x_train['id'] = df_train['id']
x_train['qid1'] = df_train['qid1']
x_train['qid2'] = df_train['qid2']
x_train['word_match'] = df_train.apply(word_match_share, axis=1, raw=True)
print 'word_match done'
x_train['q1_q2_intersect'] = df_train.apply(q1_q2_intersect, axis=1, raw=True)
print 'q1_q2_intersect done'
x_train['is_duplicate'] = df_train['is_duplicate']


# In[13]:

temp = []
for i in x_train.index:
    if i%10000==0:
        print i
    R = tfidf_word_match_share({'question1':df_train.loc[i,'question1'],'question2':df_train.loc[i,'question2']}, tfid)
    temp.append(R)
x_train['tfidf_word_match'] = temp
# x_train['tfidf_word_match'] = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
print 'tfidf_word_match done'

dprob = pd.read_csv('submission/train_pred.csv')
x_train['kernel_prob'] = dprob['is_duplicate']
del dprob
print 'kernel_prob done'


# In[ ]:

x_train.to_csv('input_clean/magic_train.csv',index=False)


# In[ ]:

del x_train


# In[ ]:

x_test['test_id'] = df_test['test_id']
x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
print 'word_match done'
x_test['q1_q2_intersect'] = df_test.apply(q1_q2_intersect, axis=1, raw=True)
print 'q1_q2_intersect done'


# In[ ]:

temp = []
for i in x_test.index:
    if i%10000==0:
        print i
    R = tfidf_word_match_share({'question1':df_test.loc[i,'question1'],'question2':df_test.loc[i,'question2']})
    temp.append(R)
x_test['tfidf_word_match'] = temp
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
print 'tfidf_word_match done'

dprob = pd.read_csv('submission/test_pred.csv')
x_test['kernel_prob'] = dprob['is_duplicate']
del dprob
print 'kernel_prob done'


# In[ ]:

x_test.to_csv('input_clean/magic_test.csv',index=False)