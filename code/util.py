import pickle
import os
import tensorflow as tf
from datetime import datetime
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

checkpoint_path = '/tmp/quora_question_pairs/'

def load_pickle(filename):
    pkl_file = open(filename, 'rb')
    data = pickle.load(pkl_file)
    return data


class SimpleDataIterator():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

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


class SimpleArrayIterator():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = len(self.x)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = range(self.cursor, self.cursor+n)
        self.cursor += n
        return self.x[res], self.y[res]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def train(trainX, trainY, validX, validY, batch_size, n_epoch=10):

    # x = tf.placeholder(tf.float32, [batch_size, 200])
    # y = tf.placeholder(tf.float32, [batch_size])
    #
    # W = weight_variable([200, 1])
    # b = bias_variable([1])
    #
    # logits = tf.matmul(x, W) + b
    # logits = tf.reshape(logits, [-1])
    #
    # cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.round(tf.sigmoid(logits)), y)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    #
    # saver = tf.train.Saver()
    # it = SimpleArrayIterator(inputs, labels)
    #
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)
    #
    # for i in range(iterations):
    #     batch = it.next_batch(batch_size)
    #
    #     if i % 10 == 0:
    #         train_accuracy = sess.run(accuracy, feed_dict={
    #         x: batch[0], y: batch[1]})
    #         print('Step %d: Training accuracy %g' % (i, train_accuracy))
    #         print("{} Saving checkpoint of model...".format(datetime.now()))
    #
    #         #save checkpoint of the model
    #         checkpoint_name = os.path.join(checkpoint_path, 'model_step'+str(i)+'.ckpt')
    #         save_path = saver.save(sess, checkpoint_name)
    #
    #         print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
    #     sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
    #
    # checkpoint_name = os.path.join(checkpoint_path, 'model_train.ckpt')
    # save_path = saver.save(sess, checkpoint_name)
    validY = to_categorical(validY, nb_classes=2)
    trainY = to_categorical(trainY, nb_classes=2)

    net = tflearn.input_data([None, 100])
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001,
                             loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(validX, validY), show_metric=True,
              batch_size=batch_size, snapshot_epoch=True, n_epoch=n_epoch)
