import pickle
import os
import numpy as np
import pandas as pd

MAX_STEP = 40

def expand_vec(data, col_name, max_len=0):
	val = np.empty((0,max_len),np.int32)
	for sentence_vec in data[col_name].values:
		temp_vec = np.pad(sentence_vec, (MAX_STEP-len(sentence_vec), 0), 'constant').astype(int)
		val = np.vstack((val,temp_vec))
		
	return val

class SimpleDataIterator():
	def __init__(self, inputs, shuffle = True, test = False):
		print 'util.py : __init__'
		# train_question_1 = pickle.load(open(inputs['q1'], 'rb'))
		# train_question_2 = pickle.load(open(inputs['q2'], 'rb'))
		# train_labels = pickle.load(open(inputs['label'], 'rb'))
		# print 'util2.py : Pickles loaded'
		# self.df = pd.DataFrame(data={'vec1':train_question_1,'vec2':train_question_2,'is_duplicate':train_labels})
		self.df = inputs
		self.size = len(self.df)
		print 'Original data size : ',self.size
		self.epochs = 0
		if shuffle:
			self.shuffle()
		else:
			self.cursor = 0
		self.test = test
		
		print 'util2.py : Proper init'

	def shuffle(self):
		self.df = self.df.sample(frac=1).reset_index(drop=True)
		self.cursor = 0

	def next_batch(self, n):
		if self.cursor+n-1 > self.size:
			self.epochs += 1
			self.shuffle()
		res = self.df.ix[self.cursor:self.cursor+n-1]
		self.cursor += n
		return res['vec1'], res['vec2'], res['is_duplicate']

class PaddedDataIterator(SimpleDataIterator):
	def next_batch(self, n):
		epoch_complete = 0
		if self.test:
			if self.cursor+n < self.size:
				res = self.df.ix[self.cursor:self.cursor+n-1]
				self.cursor += n
			else:
				res = self.df.ix[self.cursor:self.size-1]
				self.cursor = self.size
				epoch_complete += 1

			batch = {}

			# uncomment the line below if you want the CSV to include 'tid', but would need additional code changes
			# batch['tid'] = res['test_id'].values.astype(np.int64)
			batch['vec1'] = res.as_matrix([str(i) for i in range(MAX_STEP)])
			batch['vec2'] = res.as_matrix([str(i) for i in range(MAX_STEP,2*MAX_STEP)])

			ol = []
			for j in range(len(res)):
				o1 = 0
				o2 = 0
				for x in batch['vec1'][j].tolist():
					if x==0:
						continue
					for y in batch['vec2'][j].tolist():
						if y==0:
							continue
						if x==y:
							o1 += 1
							o2 += 1
				try:
					o1 = float(o1)/sum(batch['vec1'][j] != 0)
				except ZeroDivisionError:
					o1 = 0.0
				try:
					o2 = float(o2)/sum(batch['vec2'][j] != 0)
				except ZeroDivisionError:
					o2 = 0.0
				ol.append([o1,o2])

			batch['overlap'] = np.asarray(ol).astype(np.float32)

		else:
			if self.cursor+n > self.size:
				self.epochs += 1
				self.shuffle()
				epoch_complete = 1
			res = self.df.ix[self.cursor:self.cursor+n-1]
			self.cursor += n

			batch = {}
			batch['is_duplicate'] = res['is_duplicate'].values.astype(np.float64)
			batch['vec1'] = res.as_matrix([str(i) for i in range(MAX_STEP)])
			batch['vec2'] = res.as_matrix([str(i) for i in range(MAX_STEP,2*MAX_STEP)])
			batch['overlap'] = res.as_matrix(['overlap1','overlap2'])
		
		return batch, epoch_complete

	def validation_batch(self, batch_size, seed = 10587):
		valid = self.df.sample(n=batch_size, random_state=seed)
		self.df.drop(valid.index, inplace = True)
		self.df = self.df.reset_index(drop=True)
		self.size = len(self.df)
		print 'Train data size: ',len(self.df)
		batch = {}
		batch['is_duplicate'] = valid['is_duplicate'].values.astype(np.float64)
		batch['vec1'] = valid.as_matrix([str(i) for i in range(MAX_STEP)])
		batch['vec2'] = valid.as_matrix([str(i) for i in range(MAX_STEP,2*MAX_STEP)])
		batch['overlap'] = valid.as_matrix(['overlap1','overlap2'])
		return batch