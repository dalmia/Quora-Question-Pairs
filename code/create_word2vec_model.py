import gensim
import pickle

print('Loading word2vec pretrained weights...')
model = gensim.models.KeyedVectors.load_word2vec_format('../input/GoogleNews-vectors-negative300.bin', binary=True) 
print('Done.')

print('Saving model....')
with open('model_word2vec.pkl', 'wb') as output:
    pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
print('Done.')
