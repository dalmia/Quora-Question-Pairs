# Preprocessing

- `create_glove_wiki_model.py` - Makes a dictionary mapping the words to their corresponding GloVe vector representations.
- `train_data_tokenize.py` - Tokenizes the questions in `train.csv` using nltk's implementation of the Stanford Tokenizer.
- `test_data_tokenize.py` - Tokenizes the questions in `test.csv` using nltk's implementation of the Stanford Tokenizer.
- `train_data_augment.py` - Augments the tokenized training data. Data augmentation explained in arXiv paper (coming soon).
- `convert_train_test_to_indices.py` - Prepares the tokenized train and test data to be passed as input to the RNN model.
