import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
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
        # todo: implement placeholders
        self.texts1 = tf.placeholder(tf.string, [batch_size, None], name='texts1')
        self.texts2 = tf.placeholder(tf.string, [batch_size, None], name='texts2')  # shape: batch*len
        self.texts_length = tf.placeholder(tf.int32, [None], name='texts_length')  # shape: batch
        self.len = tf.constant(1.0, shape=[batch_size])
        self.labels = tf.placeholder(
            tf.int64, [None], name='labels')  # shape: batch
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.embed_units = num_embed_units
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

        self.index_input1 = self.symbol2index.lookup(self.texts1)   # batch*len
        self.index_input2 = self.symbol2index.lookup(self.texts2)
        '''
        self.h_s1 = tf.Variable(tf.constant(0.0,shape=[num_units+1, batch_size, num_embed_units]), trainable=False)
        self.h_s2 = tf.Variable(tf.constant(0.0,shape=[num_units+1, batch_size, num_embed_units]), trainable=False)
        self.h_r = tf.Variable(tf.constant(0.0,shape=[num_units+1, batch_size, num_embed_units]), trainable=False)
        self.a1 = tf.Variable(tf.constant(0.0,shape=[num_units+1, batch_size, num_embed_units]), trainable=False)
        self.a2 = tf.Variable(tf.constant(0.0,shape=[num_units+1, batch_size, num_embed_units]), trainable=False)
        '''
        self.h_s1 = []
        self.h_s2 = []
        self.h_r = []
        self.a1 = []
        self.a2 = []
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
            self.lstm_s = rnn_cell.BasicLSTMCell(num_units=num_embed_units, forget_bias=0)
        '''
        out_s1, state_s1 = tf.nn.dynamic_rnn(self.lstm_s, self.embed_input1, self.texts_length, dtype=tf.float32)
        out_s2, state_s2 = tf.nn.dynamic_rnn(self.lstm_s, self.embed_input2, self.texts_length, dtype=tf.float32)
        self.h_s1 = state_s1
        self.h_s2 = state_s2
        '''
        with tf.variable_scope('lstm_r'):
            self.lstm_r = rnn_cell.BasicLSTMCell(num_units=num_embed_units, forget_bias=0)
        '''
        self.ini_op1 = tf.assign(self.h_s1[0], self.lstm_s.zero_state(batch_size=batch_size, dtype=tf.float32))
        self.ini_op2 = tf.assign(self.h_s2[0], self.lstm_s.zero_state(batch_size=batch_size, dtype=tf.float32))
        self.ini_op3 = tf.assign(self.h_r[0], self.lstm_r.zero_state(batch_size=batch_size, dtype=tf.float32))
        self.ini_op4 = tf.assign(self.a1[0], self.lstm_r.zero_state(batch_size=batch_size, dtype=tf.float32))
        self.ini_op5 = tf.assign(self.a2[0], self.lstm_r.zero_state(batch_size=batch_size, dtype=tf.float32))
        '''
        
        self.h_s1.append(self.lstm_s.zero_state(batch_size=batch_size, dtype=tf.float32))
        self.h_s2.append(self.lstm_s.zero_state(batch_size=batch_size, dtype=tf.float32))
        self.h_r.append(self.lstm_s.zero_state(batch_size=batch_size, dtype=tf.float32))
        self.a1.append(self.lstm_r.zero_state(batch_size=batch_size, dtype=tf.float32))
        self.a2.append(self.lstm_r.zero_state(batch_size=batch_size, dtype=tf.float32)) 
        
        W = tf.Variable(self._initializer(shape=[num_embed_units, num_labels],dtype=tf.float32))
        bias = tf.Variable(tf.constant(0.0, shape=[num_labels]), dtype=tf.float32)

        i = tf.constant(1, dtype=tf.int64)
        # print self.index_input1[1].get_shape()
        length = self._length(self.index_input1[1])
        self.ind = 1
        state_s1 = self.lstm_s.zero_state(batch_size=batch_size, dtype=tf.float32)
        state_s2 = self.lstm_s.zero_state(batch_size=batch_size, dtype=tf.float32)
        state_r = self.lstm_r.zero_state(batch_size=batch_size, dtype=tf.float32)
        def c(t, s1, s2, sr): return tf.less(t, length+1)

        def b(t, s1, s2, sr): return self.attention(t, s1, s2, sr)
        i, state_s1, state_s2, state_r = tf.while_loop(cond=c, body=b, loop_vars=(i, state_s1, state_s2, state_r))

        
        logits = tf.matmul(state_r.h, W) + bias

        #logits = tf.layers.dense(outputs, num_labels)

        # todo: implement unfinished networks

        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=logits), name='loss')
        mean_loss = self.loss / \
            tf.cast(tf.shape(self.labels)[0], dtype=tf.float32)
        predict_labels = tf.argmax(logits, 1, 'predict_labels')
        self.accuracy = tf.reduce_sum(
            tf.cast(tf.equal(self.labels, predict_labels), tf.int32), name='accuracy')

        self.params = tf.trainable_variables()
        # calculate the gradient of parameters
        '''
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(mean_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(
            gradients, max_gradient_norm)
        self.update = opt.apply_gradients(
            zip(clipped_gradients, self.params), global_step=self.global_step)
        '''
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(mean_loss, global_step=self.global_step,
                                                                            var_list=self.params)
        tf.summary.scalar('loss/step', self.loss)
        for each in tf.trainable_variables():
            tf.summary.histogram(each.name, each)

        self.merged_summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
    
    def attention(self, t, s1, s2, sr):
        '''
        h_s1_j = tf.reshape(x1[t], [1, -1])
        h_s2_j = tf.reshape(x2[t], [1, -1])
        h_s1_p = tf.slice(s1, begin=[0, 0], size=[t, self.embed_units])
        h_s2_p = tf.slice(s2, begin=[0, 0], size=[t, self.embed_units])
        '''
        
        s1_t = tf.concat([self.embed_input1[:,t-1], sr.h],1)
        s2_t = tf.concat([self.embed_input2[:,t-1], sr.h],1)
        r_t = tf.concat([self.a1[self.ind-1].h, self.a2[self.ind-1].h],1)
        

        with tf.variable_scope('lstm_s'):
            out_s1, state_s1 = self.lstm_s(inputs=s1_t, state=s1)
            out_s2, state_s2 = self.lstm_s(inputs=s2_t, state=s2)
        with tf.variable_scope('lstm_r'):
            out_r, state_r = self.lstm_r(inputs=r_t, state=sr)
        '''
        self.assign_op1 = tf.assign(self.h_s1[t], state_s1)
        self.assign_op2 = tf.assign(self.h_s2[t], state_s2)
        self.assign_op3 = tf.assign(self.h_r[t], state_r)
        '''
        self.h_s1.append(state_s1)
        self.h_s2.append(state_s2)
        self.h_r.append(state_r)

        a1t = tf.constant(0.0, shape = [self.batch_size, self.embed_units], dtype=tf.float32)
        a2t = tf.constant(0.0, shape = [self.batch_size, self.embed_units], dtype=tf.float32)
        
        def c1(j, t, a1tj, a2tj): return tf.less(j, t)
        def b1(j, t, a1tj, a2tj): return self.match(j,t, a1tj, a2tj)
        k = tf.constant(1, dtype=tf.int64)
        self.j = 1
        k, q, a1t, a2t = tf.while_loop(cond=c1, body=b1, loop_vars=[k ,t, a1t, a2t], shape_invariants=None)
        '''
        self.assign_op4 = tf.assign(self.a1[t], a1t)
        self.assign_op5 = tf.assign(self.a2[t], a2t)
        '''
        self.a1.append(a1t)
        self.a2.append(a2t)
        
        t=tf.add(t,1)
        self.ind+=1
        return t, state_s1, state_s2, state_r


    def match(self, j, t, a1tj, a2tj):
        with tf.variable_scope('Attn_'):
            W_s = tf.get_variable(shape=[self.embed_units, self.embed_units],
                              initializer=self._initializer, name='W_s')
            W_o = tf.get_variable(shape=[self.embed_units, self.embed_units],
                              initializer=self._initializer, name='W_o')
            W_e = tf.get_variable(shape=[self.embed_units, 1],
                              initializer=self._initializer, name='W_e')
            W_a = tf.get_variable(shape=[self.embed_units, self.embed_units],
                              initializer=self._initializer, name='W_a')
        
        e1_tj = tf.matmul(tf.tanh(tf.matmul(self.h_s1[self.j].h, W_s) +
                                       tf.matmul(W_o, self.h_s2[self.ind].h, transpose_b=True) + 
                                       tf.matmul(W_a, self.h_r[self.ind-1].h, transpose_b=True)), W_e)
        e2_tj = tf.matmul(tf.tanh(tf.matmul(W_s, self.h_s2[self.j].h, transpose_b=True) +
                                       tf.matmul(W_o, self.h_s1[self.ind].h, transpose_b=True) + 
                                       tf.matmul(W_a, self.h_r[self.ind-1].h, transpose_b=True)), W_e)
            
        alpha1_tj = tf.reshape(tf.nn.softmax(e1_tj, dim=1),[-1])
        alpha2_tj = tf.reshape(tf.nn.softmax(e2_tj, dim=1),[-1])
        '''
        with tf.variable_scope('atten'):
            a1tj = tf.get_variable(shape = [self.embed_units, batch_size], initializer=tf.constant_initializer(), name='a1tj')
            a2tj = tf.get_variable(shape = [self.embed_units, batch_size], initializer=tf.constant_initializer(), name='a2tj')
        self.add_op1 = tf.assign_add(a1tj, tf.transpose(self.h_s1[j])*alpha1_tj)
        self.add_op2 = tf.assign_add(a2tj, tf.transpose(self.h_s2[j])*alpha2_tj)
        '''

        a1tj = tf.add(a1tj, tf.transpose(self.h_s1[self.j].h)*alpha1_tj)
        a2tj = tf.add(a2tj, tf.transpose(self.h_s2[self.j].h)*alpha2_tj)
        j = tf.add(j,1)        
        self.j+=1
        return j, t, a1tj, a2tj
        

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
        output_feed = [self.loss, self.accuracy, self.train_op]
                       #self.gradient_norm, self.update]
        '''
                       ,self.assign_op1,
                       self.assign_op2, self.assign_op3, self.assign_op4,
                       self.assign_op5, self.ini_op1,
                       self.ini_op2, self.ini_op3, self.ini_op4, self.ini_op5]
        '''
        if summary:
            output_feed.append(self.merged_summary_op)
        return session.run(output_feed, input_feed)
