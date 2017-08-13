import pickle
import pandas as pd
import numpy as np
print('Imports done.')

sent_cutoff = 40

print('Loading glove vectors...')
map_word_index = pickle.load(open('input_clean/map_word_index.pkl', 'r'))
print('Done.')

def convert_with_labels(question1, question2, labels):

	v1_list = []
	v2_list = []
	label_list = []

	N = len(question1)

	for i, sentence1, sentence2, label in zip(range(1,N+1), question1, question2, labels):
		if i % 10000 == 0:
			print('Index: %s' % i)

		if type(sentence1)!=float:
			if len(sentence1)>0:
				vector1 = [map_word_index[word] for word in sentence1.split(" ")]
			else:
				print 'Empty sentence1 at %d' % i
				vector1 = [0]
		else:
			print 'NaN sentence1 at %d' % i
			vector1 = [0]
		
		if type(sentence2)!=float:
			if len(sentence2)>0:
				vector2 = [map_word_index[word] for word in sentence2.split(" ")]
			else:
				print 'Empty sentence2 at %d' % i
				vector2 = [0]
		else:
			print 'NaN sentence2 at %d' % i
			vector2 = [0]

		vector1 = vector1[:sent_cutoff]
		vector2 = vector2[:sent_cutoff]

		v1 = np.pad(np.asarray(vector1), (sent_cutoff-len(vector1), 0), 'constant').astype(np.int32)
		v2 = np.pad(np.asarray(vector2), (sent_cutoff-len(vector2), 0), 'constant').astype(np.int32)

		v1_list.append([v1])
		v2_list.append([v2])
		# v1_list = np.append(v1_list, [v1], axis=0)
		# v2_list = np.append(v2_list, [v2], axis=0)	
		label_list.append(label.astype(np.float32))

		return v1_list, v2_list, label_list

def convert_without_labels(question1, question2):

	v1_list = []
	v2_list = []

	N = len(question1)

	for i, sentence1, sentence2 in zip(range(1,N+1), question1, question2):
		if i % 10000 == 0:
			print('Index: %s' % i)

		if type(sentence1)!=float:
			if len(sentence1)>0:
				vector1 = [map_word_index[word] for word in sentence1.split(" ")]
			else:
				print 'Empty sentence1 at %d' % i
				vector1 = [0]
		else:
			print 'NaN sentence1 at %d' % i
			vector1 = [0]
		
		if type(sentence2)!=float:
			if len(sentence2)>0:
				vector2 = [map_word_index[word] for word in sentence2.split(" ")]
			else:
				print 'Empty sentence2 at %d' % i
				vector2 = [0]
		else:
			print 'NaN sentence2 at %d' % i
			vector2 = [0]

		vector1 = vector1[:sent_cutoff]
		vector2 = vector2[:sent_cutoff]

		v1 = np.pad(np.asarray(vector1), (sent_cutoff-len(vector1), 0), 'constant').astype(np.int32)
		v2 = np.pad(np.asarray(vector2), (sent_cutoff-len(vector2), 0), 'constant').astype(np.int32)

		v1_list.append([v1])
		v2_list.append([v2])

		return v1_list, v2_list

def convert_csv(in_filepath, out_filepath, is_train=None):

	print('Loading data...')
	data = pd.read_csv(in_filepath)
	print('Done.')

	N = len(data)
	print 'len(data) : %d' % N

	if is_train:
		print('Converting training data to indices...')
		v1_list, v2_list, label_list = convert_with_labels(data['question1'], data['question2'], data['is_duplicate'])

	else:
		v1_list, v2_list = convert_with_labels(data['question1'], data['question2'])

	v1_list1 = np.concatenate(v1_list, axis = 0)
	v2_list2 = np.concatenate(v2_list, axis = 0)
	v_large = np.concatenate((v1_list1, v2_list2), axis=1)

	conv = pd.DataFrame(v_large)
	if is_train:
		conv['is_duplicate'] = label_list
		del label_list

	del v1_list
	del v2_list
	del v_large
	
	conv.to_csv(out_filepath,index=False)

convert_csv('input_clean/train_clean.csv', 'input_clean/train_nonaug_conv.csv', True)
convert_csv('input_clean/train_aug.csv', 'input_clean/train_conv.csv', True)
convert_csv('input_clean/test_clean.csv', 'input_clean/test_conv.csv')

'''
train, valid = conv_shuffle[:int(len(conv_shuffle)*0.8)],conv_shuffle[int(len(conv_shuffle)*0.8):]
del conv_shuffle

print 'Saving train data...'
train.to_csv('input_clean/train_conv.csv',index=False)
print 'Saving validation data...'
valid.to_csv('input_clean/valid_conv.csv',index=False)
'''
print 'Clean EXIT'