"""
tf_1.10.0
RNN over concatenated sentence embeddings
"""
print 'Importing'
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
from datetime import datetime
from util import *
print 'Done'

INPUT_PATH = 'input_clean/'
EMBED_PATH = 'input_clean/'
CHECKPOINT_PATH = 'models/log5/check_points/'
LOGDIR = 'models/log5/'
LEARNING_RATE = 0.001
BATCH_SIZE = 256
MAX_STEP = 40
EMBED_SIZE = 50
EPOCHS = 100
N_HIDDEN_1 = 250
N_HIDDEN_2 = 500
N_HIDDEN_3 = 250

def train(inputs, epochs, batch_size = BATCH_SIZE):
	global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

	print 'Loading embeddings...'
	map_index_vec = pickle.load(open(EMBED_PATH + inputs['embed'], 'rb'))
	print('Done.')

	n_symbols = len(map_index_vec)
	ew = np.zeros((n_symbols, 50),dtype=np.float32)
	for index, vec in map_index_vec.items():
		ew[index,:] = vec

	with tf.name_scope("data"):
		in1 = tf.placeholder(tf.int32, shape=[None, MAX_STEP], name='in1')
		in2 = tf.placeholder(tf.int32, shape=[None, MAX_STEP], name='in2')
		# overlap = tf.placeholder(tf.float32, shape=[None, 2], name='overlap')
		target = tf.placeholder(tf.float32, shape=[None], name='target')
		tf.add_to_collection("in1",in1)
		tf.add_to_collection("in2",in2)
		# tf.add_to_collection("overlap",overlap)
		# print 'target : ',target

	with tf.name_scope("embedding"):
		embedding_weights = tf.Variable(initial_value = ew, name = 'embedding_weights')
		q1 = tf.nn.embedding_lookup(embedding_weights, in1, name='embed_q1')
		q2 = tf.nn.embedding_lookup(embedding_weights, in2, name='embed_q2')
		# q1 = tf.transpose(eq1, [0,2,1], name='q1')
		# q2 = tf.transpose(eq2, [0,2,1], name='q2')
		print 'q2 : ',q2

	with tf.variable_scope('gru1') as scope:
		x1 = tf.unstack( q1, MAX_STEP, 1 )
		x2 = tf.unstack( q2, MAX_STEP, 1 )		
		gru_cell = tf.contrib.rnn.GRUCell(N_HIDDEN_1)
		y11, _ = tf.contrib.rnn.static_rnn(gru_cell, x1, dtype=tf.float32)
		scope.reuse_variables()
		y12, _ = tf.contrib.rnn.static_rnn(gru_cell, x2, dtype=tf.float32)

	with tf.variable_scope('gru2') as scope:
		gru_cell = tf.contrib.rnn.GRUCell(N_HIDDEN_2)
		y21, _ = tf.contrib.rnn.static_rnn(gru_cell, y11, dtype=tf.float32)
		scope.reuse_variables()
		y22, _ = tf.contrib.rnn.static_rnn(gru_cell, y12, dtype=tf.float32)

	with tf.variable_scope('gru3') as scope:
		gru_cell = tf.contrib.rnn.GRUCell(N_HIDDEN_3)
		_, y1 = tf.contrib.rnn.static_rnn(gru_cell, y21, dtype=tf.float32)
		scope.reuse_variables()
		_, y2 = tf.contrib.rnn.static_rnn(gru_cell, y22, dtype=tf.float32)

	with tf.variable_scope('process_state'):
		y_d = tf.squared_difference(y1,y2,name='h_sub_sq')
		y_cos = tf.reduce_prod(tf.stack(values=[y1, y2],axis=2,name='h_concat'),reduction_indices=2,name='h_dot')
		y = tf.concat(values=[y1, y2, y_d, y_cos], axis=1)
		print 'y : ',y

	with tf.variable_scope('dense') as scope:
		w = tf.Variable(tf.truncated_normal([4*N_HIDDEN_3,1], stddev=0.1, dtype=tf.float32), name='weights')
		b = tf.Variable(tf.zeros([1], dtype=tf.float32), name="bias")

		logits = tf.matmul(y, w) + b
		logits = tf.reshape(logits, [-1], name='logits')
		print 'logits : ',logits

	with tf.name_scope('lr') as scope:
		cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=target))#,keep_dims=True)
		print 'cross_entropy : ',cross_entropy

	optimizer = tf.train.AdamOptimizer(1e-4)
	train_step = optimizer.minimize(cross_entropy,global_step=global_step)

	prediction = tf.sigmoid(logits, name='prediction')
	tf.add_to_collection("prediction", prediction)
	correct_prediction = tf.equal(tf.round(prediction), target)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.name_scope("summaries"):
		tf.summary.scalar("loss", cross_entropy)
		tf.summary.scalar("accuracy", accuracy)
		tf.summary.histogram("histogram_loss", cross_entropy)
		summary_op = tf.summary.merge_all()

	with tf.Session() as sess:
		print 'Starting session'
		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver()

		# it = PaddedDataIterator( pd.read_csv('../input_clean/train_conv.csv') )

		if not os.path.exists(CHECKPOINT_PATH):
			os.makedirs(CHECKPOINT_PATH)

		ckpt = tf.train.get_checkpoint_state(os.path.dirname(CHECKPOINT_PATH))
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print global_step.eval()

		print '1'
		writer = tf.summary.FileWriter(LOGDIR, sess.graph)
		print '2'

		it = PaddedDataIterator( pd.read_csv(inputs['train_file']) )
		print '3'
		int_step = 0
		while epochs>0:
			# print 'Iteration %d' % int_step

			batch,epoch_complete = it.next_batch(batch_size)
			
			if int_step % 100 == 0:
				# print batch['vec1'].shape
				train_accuracy = sess.run(accuracy, feed_dict={in1: batch['vec1'],
					target: batch['is_duplicate'], in2: batch['vec2']})#, overlap: batch['overlap']})
				print('Step %d: Training accuracy %g' % (int_step, train_accuracy))
				print("{} Saving checkpoint of model...".format(datetime.now()))

				#save checkpoint of the model
				checkpoint_name = os.path.join(CHECKPOINT_PATH, 'model_step')
				save_path = saver.save(sess, checkpoint_name, global_step=global_step)
				#saver.export_meta_graph(save_path+'.meta')

				print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
			
			_,summary = sess.run([train_step,summary_op], feed_dict={in1: batch['vec1'], in2: batch['vec2'],
				target: batch['is_duplicate']})#, overlap: batch['overlap']})
			writer.add_summary(summary, global_step = global_step.eval())

			int_step = global_step.eval()
			epochs = epochs - epoch_complete
			if epoch_complete:
				print 'Epochs left = ',epochs
			sys.stdout.flush()

		checkpoint_name = os.path.join(CHECKPOINT_PATH, 'model_train.ckpt')
		save_path = saver.save(sess, checkpoint_name)
		#saver.export_meta_graph(save_path+'.meta')
		del it
		writer.close()

def predict(test_file, output_file, num_test):
	with tf.Session() as sess:
		print 'Starting session'
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(CHECKPOINT_PATH))
		print 'Restoring meta graph'
		new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
		print 'Restoring model'
		new_saver.restore(sess, ckpt.model_checkpoint_path)

		print 'Loading ops'
		in1 = tf.get_collection('in1')[0]
		in2 = tf.get_collection('in2')[0]
		pred = tf.get_collection('prediction')[0]

		writer = tf.summary.FileWriter('.', sess.graph)

		for j in range(num_test):
			print 'Loading test file %d'%j
			if num_test>1:
				it = PaddedDataIterator( pd.read_csv(test_file+str(j)+'.csv'), shuffle = False, test = True)
			else:
				it = PaddedDataIterator( pd.read_csv(test_file), shuffle = False, test = True)

			epoch_complete = 0
			tid = np.empty(0, dtype=np.int64)
			preds = np.empty(0,dtype = np.float32)
			
			while not epoch_complete:
				
				batch,epoch_complete = it.next_batch(500)
				# if epoch_complete:
				# 	print 'tid',batch['tid'][0]
				# 	break
				tid = np.concatenate((tid,batch['tid']),axis=0)

				p = sess.run(pred, feed_dict={in1:batch['vec1'],in2:batch['vec2']})#,overlap:batch['overlap']})
				
				# print p
				preds = np.concatenate([preds,np.asarray(p)],axis = 0)
				
			out = pd.DataFrame({'test_id':tid, 'is_duplicate':preds})
			out = out[['test_id','is_duplicate']]
			if num_test>1:
				out.to_csv(output_file+str(j)+'.csv', index=False)
			else:
				out.to_csv(output_file, index=False)
			del it

		writer.close()

if __name__ == "__main__":
	inputs = {'embed':'map_index_vec.pkl',
	'train_file':INPUT_PATH + 'train_conv.csv'}

	train(inputs, EPOCHS)

	predict('input_clean/test_conv.csv','submission/test_pred.csv',1)
	predict('input_clean/train_nonaug_conv.csv','submission/train_pred.csv',1)
