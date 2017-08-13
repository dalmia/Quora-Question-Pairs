import multiprocessing
import pickle
import pandas as pd
import numpy as np
import os
import sys
from nltk.tokenize import StanfordTokenizer
print('Imports Done.')

class Consumer(multiprocessing.Process):
	def __init__(self, task_queue, result_queue):
		
		multiprocessing.Process.__init__(self)
		self.task_queue = task_queue
		self.result_queue = result_queue

		self.tokenizer = StanfordTokenizer(options={"ptb3Escaping": True})
		print '%s: Loading pickles...' % self.name
		self.map_word_index = pickle.load(open('preproc/map_word_index.pkl', 'r'))
		print '%s: Done.' % self.name

	def run(self):
		proc_name = self.name
		sys.stdout = open( 'preproc/log5/proc.' + str(os.getpid()) + ".out", "w+")
		while True:
			next_task = self.task_queue.get()
			if next_task is None:
				# Poison pill means shutdown
				print '%s: Exiting' % proc_name
				self.task_queue.task_done()
				break
			print '%s: %s' % (proc_name, next_task)
			answer = next_task.run(self.tokenizer,self.map_word_index)
			self.task_queue.task_done()
			self.result_queue.put(answer)
			print '%s: Done %s' % (proc_name, next_task)
			sys.stdout.flush()
		return


class Task(object):
	def __init__(self, data, loc):
		self.data = data
		self.loc = loc

	def __str__(self):
		return '%s' % (self.loc)

	def run(self,tokenizer,map_word_index):
		for col in ['question1', 'question2']:
			sentence = self.data.loc[self.loc,col]
			# if pd.isnull(sentence):
			if len(str(sentence))==0:
				print('Skipping sentence %d on empty'%self.loc)
				self.data.loc[self.loc,col] = ''
				continue
			try:
				split = tokenizer.tokenize(sentence)
			except TypeError:
				# Avoid error on NaN
				print('Skipping sentence %d on NaN'%self.loc)
				self.data.loc[self.loc,col] = ''
				continue
			new_sentence = []
			for word in split:
				word = word.encode('utf-8').strip()
				word = word.lower()
				
				if word in map_word_index:
					new_sentence.append(word)
				else:
					new_sentence.append('unk')
			self.data.loc[self.loc,col] = " ".join(new_sentence)
		return self.data

def wrapper( j,CHUNK ):
	# Establish communication queues
	tasks = multiprocessing.JoinableQueue()
	results = multiprocessing.Queue()
	
	# Start consumers
	num_consumers = multiprocessing.cpu_count()
	print 'Creating %d consumers' % num_consumers
	consumers = [ Consumer(tasks, results)
				  for i in xrange(num_consumers) ]
	
	for w in consumers:
		w.start()
	
	# Create jobs
	print('Loading data...')
	test_data = pd.read_csv('input/test.csv')
	#test_data = test_data.loc[:28]
	print('Done.')

	for i in range(j*CHUNK,min(len(test_data),(j+1)*CHUNK )):
		if i%(CHUNK/5)==0:
			print 'Queuing task %s' % i
			sys.stdout.flush()
		tasks.put(Task(test_data.loc[i:i],i))		# a little trick for maintaining shape
		# test_data.drop(i,axis=0,inplace=True)

	# Add a poison pill for each consumer
	for i in xrange(num_consumers):
		tasks.put(None)

	# Wait for all of the tasks to finish
	tasks.join()

	test_clean = pd.DataFrame(columns=test_data.columns)
	# Start printing results
	print 'Gathering results...'
	for i in range(j*CHUNK,min(len(test_data),(j+1)*CHUNK )):		
		if results.empty():
			print 'Possible error'
			break
		if ((i%(CHUNK/5) == 0) or (i>((j+1)*CHUNK - num_consumers))):
			print 'Stacking result %s' % i
			sys.stdout.flush()
		result = results.get()
		# print 'Result:', result
		test_clean = pd.concat([test_clean, result])
		# i = i + 1

	print 'Sanity check: %d' % results.empty()
	print 'Length test_clean: %d' % len(test_clean)
	print 'CHUNK: %d' % CHUNK

	print 'Saving to csv...'
	test_clean.sort_index(inplace=True)
	test_clean.to_csv('input_clean/v_test_cleaned_'+str(j)+'.csv',index=False)	

if __name__ == '__main__':
	#CHUNK = 10
	CHUNK = 10000
	print('Loading data...')
	test_data = pd.read_csv('input/test.csv')
	#test_data = test_data.loc[:28]
	print('Done.')
	for j in range(np.int64( np.ceil(np.float(len(test_data))/CHUNK) ) ):
		wrapper(j, CHUNK)

	test_tokenized_df = pd.DataFrame(columns=test_data.columns)
	for j in range(np.int64( np.ceil(np.float(len(test_data))/CHUNK) ) ):
		chunk_df = pd.read_csv('input_clean/v_test_cleaned_'+str(j)+'.csv')
		test_tokenized_df = pd.concat([test_tokenized_df, chunk_df],ignore_index=True)

	test_tokenized_df.to_csv('input_clean/test_clean.csv', index=False)