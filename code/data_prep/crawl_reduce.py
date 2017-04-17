
# coding: utf-8

# In[5]:

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import re
import pickle
from  nltk.tokenize  import  word_tokenize
import nltk
from spellcheck import *
from spacy.en import English


# In[2]:

print('Loading glove vectors...')
model = pickle.load(open('model_glove_wiki.pkl', 'rb'))
print('Done.')


# In[3]:

# def clean_sentence(sentence):
#     "remove chars that are not letters or numbers, downcase, then remove stop words"
#     sentence = sentence.lower()
#     sentence = re.sub('[0-9]+.[0-9]+','9',sentence)
#     return sentence


# In[4]:

# def clean_dataframe(data):
#     "drop nans, then apply 'clean_sentence' function to question1 and 2"
#     data = data.dropna()

#     for col in ['question1', 'question2']:
#         print 'Cleaning col ' + col
#         data[col] = data[col].apply(clean_sentence)

#     return data


# In[3]:

print('Loading train data from csv..')
traindata = pd.read_csv('./input/train.csv')
traindata = traindata.dropna()
print('Done.')


# In[75]:

# print('Cleaning train data...')
# cleaned_train_data = clean_dataframe(traindata)
# print('Data cleaning done...')


# In[7]:

print('Loading test data from csv..')
testdata = pd.read_csv('./input/test.csv')
testdata = testdata.dropna()
print('Done.')


# In[6]:

# print('Cleaning test data...')
# cleaned_test_data = clean_dataframe(testdata)
# print('Data cleaning done...')


# In[6]:

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


# In[9]:

s ='Time is money.Can some one explain, how it works?'
sent_detector.tokenize(s)


# In[70]:

def reduceGloveModel(gloveFile, model, index, last_index, missed, missed_set, data):
    print "Starting Glove Model Reduction..."
    nlp = English()
    clean_sentences = list()
    q = last_index
    for i, question in enumerate(data):
        if i%10000 == 0:
            print 'i: ',i
        
        doc = nlp(unicode(question))
        sentences = [sent.string.strip() for sent in doc.sents]
        word_list = []
        for sentence in sentences:
            sentence = sentence.lower()
            sentence = re.sub('[0-9]+.[0-9]+',' 9 ',sentence)
            sentence = re.sub('[0-9]+',' 9 ',sentence)
            sentence = re.sub(' [a-zA-Z0-9]+.com[ .,?!]+',' website ',sentence)

            new_sentence = []
            for word in word_tokenize(unicode(sentence, errors='ignore').encode('ascii','ignore')):
                if word == '':
                    continue
                if word in gloveFile and word in model:
                    new_sentence.append(word)
                    continue
                if word in gloveFile and word not in model:
                    q = q+1
                    index[word] = q
                    model[word] = gloveFile[word]
                    new_sentence.append(word)
                    continue
                if word not in gloveFile:
                    split_words = re.split("[+_/\\\\=-]+",word)
                    for nword in split_words:
                        if nword in gloveFile and nword in model:
                            new_sentence.append(nword)
                            continue
                        if nword in gloveFile and nword not in model:
                            q = q+1
                            index[nword] = q
                            model[nword] = gloveFile[nword]
                            new_sentence.append(nword)
                            continue
                        if nword not in gloveFile:
                            cword = correction(nword)
                        if cword in gloveFile and cword in model:
                            new_sentence.append(cword)
                            continue
                        if cword in gloveFile and cword not in model:
                            q = q+1
                            index[cword] = q
                            model[cword] = gloveFile[cword]
                            new_sentence.append(cword)
                            continue
                        new_sentence.append('unk')
    #                     print nword
                        missed_set.add(nword)
                        missed = len(missed_set)
                        if missed % 100 == 0:
                            print 'missed: ',missed
            
            word_list.extend(new_sentence)
        # Add spellchecked and verified words to list of wentences
        clean_sentences.append(" ".join(word_list))
      
    print "Done. Model reduced to ",len(model)
    print "Done. Index reduced to ",len(index)
    return model,index,q,missed,clean_sentences,missed_set


# In[18]:

# c1 = list(data['question1'].values)
# c2 = list(data['question2'].values)
# q_concat = c1 + c2
# new_model = {}
# for i, sentence in enumerate(q_concat):
# #     t = [word for word in sentence.split(" ") if word in model]
#     new_model.update( dict([ (word, model[word]) for word in sentence.split(" ") if word in model ]) )


# In[ ]:

new_model = {}
new_index = {}
new_index['unk']=0
new_model['unk']=model['unk']
missed_words = 0
last_index = 0


# In[63]:

cleaned_train_data = traindata


# In[64]:

new_model, new_index, last_index, missed_words, cleaned_train_data['question1'], missed_set = reduceGloveModel(model, new_model, 
                                                           new_index, last_index, missed_words, cleaned_train_data['question1'].values)


# In[71]:

new_model, new_index, last_index, missed_words, cleaned_train_data['question2'], missed_set = reduceGloveModel(model, new_model, 
                                                           new_index, last_index, missed_words, missed_set,cleaned_train_data['question2'].values)


# In[89]:

cleaned_test_data = testdata


# In[90]:

new_model, new_index, last_index, missed_words, cleaned_test_data['question1'], missed_set = reduceGloveModel(model, new_model, 
                                                           new_index, last_index, missed_words, missed_set,cleaned_test_data['question1'].values)


# In[91]:

new_model, new_index, last_index, missed_words, cleaned_test_data['question2'], missed_set = reduceGloveModel(model, new_model, 
                                                           new_index, last_index, missed_words, missed_set,cleaned_test_data['question2'].values)


# In[92]:

print 'Original word bank: ',len(model)
print 'Reduced word bank: ',len(new_model)
print 'Percent reduction: ',float(len(model)-len(new_model))*100/len(model)
print 'Missed words: ',missed_words
print last_index


# In[93]:

print('Saving model...')
with open('reduce_glove_wiki.pkl', 'wb') as output:
    pickle.dump(new_model, output, pickle.HIGHEST_PROTOCOL)
print('Done.')


# In[94]:

print('Saving indices...')
with open('reduce_glove_wiki_index.pkl', 'wb') as output:
    pickle.dump(new_index, output, pickle.HIGHEST_PROTOCOL)
print('Done.')


# In[ ]:

cleaned_train_data.to_csv('./train_cleaned.csv')


# In[95]:

cleaned_test_data.to_csv('./test_cleaned.csv')


# In[ ]:

##################################################################################################################
##################################################################################################################
##################################################################################################################


# In[3]:

print('Loading test data from csv..')
cleaned_train_data = pd.read_csv('./input/train_cleaned.csv')
print('Done.')


# In[6]:

c1 = list(cleaned_train_data['question1'].values)
c2 = list(cleaned_train_data['question2'].values)
q_concat = c1 + c2
c=set()
for i, sentence in enumerate(q_concat):
    for word in word_tokenize(unicode(sentence, errors='ignore').encode('ascii','ignore')):
        if word not in model and word != '':
            c.add(word)
            if i%10000 == 0:
                print i
            
print len(c)
print c
# f = set(e)
# print len(f)
# print float(len(d))/len(f)*100


# In[14]:

print('Reading csv...')
data1 = pd.read_csv('./input/train_cleaned_e.csv')
data1 = data1.dropna()
print('Done...')


# In[86]:

len(new_index)


# In[102]:

model["plagiarisms"]


# In[96]:

missed_set


# In[85]:

cleaned_train_data['question2'][10]

