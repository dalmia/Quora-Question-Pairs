import pandas as pd
import numpy as np
import random as rd
rd.seed(149263)

train = pd.read_csv('input_clean/train_clean.csv')

print('Augmenting Data...')
#	Flipped order
aug = train.cpoy()
aug.rename(columns={'qid1':'qid2','qid2':'qid1','question1':'question2','question2':'question1'},inplace=True)

aug = pd.concat([train,aug],ignore_index=True)

#	Duplicate with self
questions = dict()
for i,q in enumerate(train['qid1']):
	questions[q] = train.loc[i,'question1']
for i,q in enumerate(train['qid2']):
	questions[q] = train.loc[i,'question2']

print 'Unique (approx) questions : %d' % len(questions)

same = pd.DataFrame({'id':np.arange(len(questions))
	'qid1' : questions.keys()
	'qid2' : questions.keys()
	'question1' : questions.values()
	'question2' : questions.values()
	'is_duplicate' : np.ones(len(questions), dtype = np.int64)
	})

aug = pd.concat([aug,same],ignore_index=True)

#	Random non-matches
duplicates = dict()
pairs = set()
for i,d in enumerate(train['is_duplicate']):
	if d==1:
		try:
			duplicates[train.loc[i,'qid1']].add(train.loc[i,'qid2'])
		except KeyError:
			duplicates[train.loc[i,'qid1']] = set([train.loc[i,'qid2']])

		try:
			duplicates[train.loc[i,'qid2']].add(train.loc[i,'qid1'])
		except KeyError:
			duplicates[train.loc[i,'qid2']] = set([train.loc[i,'qid1']])
	pairs.add( (train.loc[i,'qid1'],train.loc[i,'qid2']) )

NON_DUP = 326400/2
MAX_QID = max(questions.keys())
diff = pd.DataFrame(columns = train.columns)
for i in range(NON_DUP):
	GEN = 0
	d=[]
	while not GEN:
		q1 = -1
		while q1 not in questions.keys():
			q1 = rd.randint(len(MAX_QID))
		q2 = -1
		while q2 not in questions.keys():
			q2 = rd.randint(len(MAX_QID))
		if((q1 != q2)&&(q1 not in duplicates(q2))&&(q2 not in duplicates(q1))&&(duplicates(q2).isdisjoint(duplicates(q1)))):
			if(((q1,q2) not in pairs) and ((q2,q1) not in pairs)):
				GEN = 1
				d = [{'id':i, 'qid1':q1, 'qid2':q2, 'question1':questions[q1], 'question2':questions[q2],
				'is_duplicate':0},{'id':NON_DUP+i, 'qid1':q2, 'qid2':q1, 'question1':questions[q2], 'question2':questions[q1],
				'is_duplicate':0}]
	diff = pd.concat([diff,pd.DataFrame(d)],ignore_index=True)

aug = pd.concat([aug,diff],ignore_index=True)

#	Save augmented training set
aug.to_csv('input_clean/train_aug.csv',index=False)
print('Data Augmentation done.')