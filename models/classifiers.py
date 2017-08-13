import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier


print 'Reading data'
df_train = pd.read_csv('../input_clean/magic_train.csv').fillna(0.0)

X = df_train.as_matrix(['word_match','q1_q2_intersect','tfidf_word_match','kernel_prob'])
Y = np.ravel(df_train.as_matrix(['is_duplicate']))

print 'Splitting...'
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, random_state=4327)
print 'Done'

clf_svm = svm.SVC( kernel = 'rbf', probability = True, verbose = True, random_state = 4273)
clf_rf = RandomForestClassifier(n_estimators=300,min_samples_split=3,oob_score=True,n_jobs=-1, random_state = 4273, max_depth=5, bootstrap=True)
clf_adaboost = AdaBoostClassifier()

for clf, name in zip([clf_svm, clf_rf, clf_adaboost], ['SVM', 'Random_Forest', 'Adaboost']):

	print('Fitting %s classifier' % name) 
	clf.fit(x_train,y_train)
	print 'Classes:'
	print clf.classes_

	print('%s Prediction on validation set' % name)
	pred = clf.predict_proba(x_valid)
	print "LOSS: ",log_loss(y_valid, pred, eps = 1e-15)

	print 'Prediction on test'
	df_test = pd.read_csv('../input_clean/magic_test.csv').fillna(0.0)

	X_test = df_test.as_matrix(['word_match','q1_q2_intersect','tfidf_word_match','kernel_prob'])
	pred = clf.predict_proba(X_test)
	test_pred = pd.DataFrame({'test_id':df_test['test_id'].values.astype(np.int32),'is_duplicate':pred[:,1]})
	test_pred = test_pred[['test_id','is_duplicate']]
	out_filename = name.lower() + '_pred.csv'
	test_pred.to_csv(out_filename,index=False)