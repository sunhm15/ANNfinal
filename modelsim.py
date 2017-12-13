import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn, static_rnn
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops.nn import rnn_cell


PAD_ID = 0
UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']


class RNN(object):
    def __init__(self,
                 num_symbols,
                 num_embed_units,
                 num_units,
                 num_labels,
                 batch_size,
                 embed,
                 learning_rate=0.001,
                 max_gradient_norm=5.0
                 ):
        
        self.texts1 = tf.placeholder(tf.string, [batch_size, None], name='texts1')
        self.texts2 = tf.placeholder(tf.string, [batch_size, None], name='texts2')  # shape: batch*len
        self.texts_length = tf.placeholder(tf.int32, [None], name='texts_length')  # shape: batch
        learning_rate_decay_factor=0.9
        self.len = tf.constant(1.0, shape=[batch_size])
        self.labels = tf.placeholder(
            tf.int64, [None], name='labels')  # shape: batch
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.embed_units = num_embed_units
        self.num_units = num_units
        self.batch_size = batch_size
        self._initializer = tf.truncated_normal_initializer(stddev=0.1)
        self.symbol2index = MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=UNK_ID,
            shared_name="in_table",
            name="in_table",
            checkpoint=True)
        # build the vocab table (string to index)
        # initialize the training process
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.epoch,
            1, 0.95)

        self.index_input1 = self.symbol2index.lookup(self.texts1)   # batch*len
        self.index_input2 = self.symbol2index.lookup(self.texts2)

        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable(
                'embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable(
                'embed', dtype=tf.float32, initializer=embed)

        self.embed_input1 = tf.nn.embedding_lookup(
            self.embed, self.index_input1)  # batch*len*embed_unit
        self.embed_input2 = tf.nn.embedding_lookup(
            self.embed, self.index_input2)

        with tf.variable_scope('lstm_s'):
            self.lstm_s = rnn_cell.BasicLSTMCell(num_units=num_units, forget_bias=0)
        
        with tf.variable_scope('lstm_r'):
            self.lstm_r = rnn_cell.BasicLSTMCell(num_units=num_units, forget_bias=0)

        out_s1, state_s1 = dynamic_rnn(self.lstm_s, self.embed_input1, self.texts_length, dtype=tf.float32, scope='rnn')
        out_s2, state_s2 = dynamic_rnn(self.lstm_s, self.embed_input2, self.texts_length, dtype=tf.float32, scope='rnn')
        
        self.h_s1 = out_s1
        self.h_s2 = out_s2
        reshaped_s1 = tf.reshape(self.h_s1, [-1, self.num_units])
        reshaped_s2 = tf.reshape(self.h_s2, [-1, self.num_units])
        with tf.variable_scope('Attn_'):
            W_s = tf.get_variable(shape=[self.num_units, self.num_units],
                              initializer=self._initializer, name='W_s')
        self.s_1 = tf.matmul(reshaped_s1, W_s)
        self.s_1 = tf.transpose(tf.reshape(self.s_1, [self.batch_size, -1, self.num_units]), [1,2,0])
        i = tf.constant(0)

        self.length = tf.reduce_max(self.texts_length)
        print(self.length )
        state_r = self.lstm_r.zero_state(batch_size=batch_size, dtype=tf.float32)
        
        def c(t, sr): return tf.less(t, self.length)
        def b(t, sr): return self.attention(t, sr)
        i, state_r = tf.while_loop(cond=c, body=b, loop_vars=(i, state_r))
        
        with tf.variable_scope('fully_connect'):
            w_fc = tf.get_variable(shape=[self.num_units, num_labels],
                                   initializer=self._initializer, name='w_fc')
            b_fc = tf.get_variable(shape=[num_labels],
                                   initializer=self._initializer, name='b_fc')
        logits = tf.matmul(state_r.h, w_fc)+ b_fc

        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=logits), name='loss')
        mean_loss = self.loss / \
            tf.cast(tf.shape(self.labels)[0], dtype=tf.float32)
        predict_labels = tf.argmax(logits, 1, 'predict_labels')
        self.accuracy = tf.reduce_sum(
            tf.cast(tf.equal(self.labels, predict_labels), tf.int64), name='accuracy')

        self.params = tf.trainable_variables()
        # calculate the gradient of parameters
        for item in tf.global_variables():
            print (item)   
        opt = tf.train.GradientDescentOptimizer (self.learning_rate)
        gradients = tf.gradients(mean_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(
            gradients, max_gradient_norm)
        self.update = opt.apply_gradients(
            zip(clipped_gradients, self.params), global_step=self.global_step)
        
        #self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(mean_loss, global_step=self.global_step,
                                                                            #var_list=self.params)
        tf.summary.scalar('loss/step', self.loss)
        for each in tf.trainable_variables():
            tf.summary.histogram(each.name, each)

        self.merged_summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


    def attention(self, k, hr):
        with tf.variable_scope('Attn_'):
            W_o = tf.get_variable(shape=[self.num_units, self.num_units],
                              initializer=self._initializer, name='W_o')
            W_e = tf.get_variable(shape=[self.num_units, 1],
                              initializer=self._initializer, name='W_e')
            W_a = tf.get_variable(shape=[self.num_units, self.num_units],
                              initializer=self._initializer, name='W_a')
        e_kj = tf.tanh( self.s_1 + tf.transpose( tf.matmul(self.h_s2[:,k,:], W_o)+tf.matmul(hr.h, W_a) ) )
        print( e_kj.get_shape())

        e_kj = tf.matmul(tf.reshape(tf.transpose(e_kj,[2,0,1]),[-1, self.num_units]), W_e)

        e_kj = tf.reshape(e_kj, [-1, self.batch_size])
        print (e_kj.get_shape())
        alpha_kj = tf.nn.softmax(e_kj, dim=0)
        print (alpha_kj.get_shape())

        akj = alpha_kj*tf.transpose(self.h_s1,[2,1,0])
        print (akj.get_shape())
        ak = tf.reduce_sum(tf.transpose(akj,[2,1,0]), 1)
        print(ak.get_shape())

        concat_layer = tf.concat([ak,self.h_s2[:,k,:]], 1)
        print(concat_layer.get_shape())
        with tf.variable_scope('lstm_r'):
            out_r, hr = self.lstm_r(inputs=concat_layer, state=hr)

        k = tf.add(k,1)        
        return k, hr
        

    def _length(self, sequence):
        mask = tf.sign(tf.abs(sequence))
        length = tf.reduce_sum(mask, axis=-1)
        return length

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def train_step(self, session, data, summary=False):
        input_feed = {self.texts1: data['texts1'],
                      self.texts2: data['texts2'],
                      self.texts_length: data['texts_length'],
                      self.labels: data['labels'],
                      self.keep_prob: data['keep_prob']}
        output_feed = [self.loss, self.accuracy, #self.train_op]
                       self.gradient_norm, self.update]
        '''
                       ,self.assign_op1,
                       self.assign_op2, self.assign_op3, self.assign_op4,
                       self.assign_op5, self.ini_op1,
                       self.ini_op2, self.ini_op3, self.ini_op4, self.ini_op5]
        '''
        #print self.symbol2index.lookup(data['texts1'])
        if summary:
            output_feed.append(self.merged_summary_op)
        #print session.run([self.texts1[0,:10],self.index_input1[0,:10]], input_feed)
        return session.run(output_feed, input_feed)
