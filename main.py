import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from read_file import deserialize
import sys
import json
from modelsim import RNN, _START_VOCAB
import os
import time
import random
random.seed(1229)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_boolean(
    "read_graph", False, "Set to False to build graph.")
tf.app.flags.DEFINE_integer("symbols", 400002, "vocabulary size.")
tf.app.flags.DEFINE_integer("labels", 3, "Number of labels.")
tf.app.flags.DEFINE_integer("epoch", 20, "Number of epoch.")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./mlstmtrain", "Training directory.")
tf.app.flags.DEFINE_boolean("log_parameters", True,
                            "Set to True to show the parameters")

FLAGS = tf.app.flags.FLAGS


def load_data(path, fname):
    print('Creating %s dataset...' % fname)
    train_data = deserialize(os.path.join(
        FLAGS.data_dir, 'snli_1.0_train.bin'))
    valid_data = deserialize(os.path.join(FLAGS.data_dir, 'snli_1.0_dev.bin'))
    test_data = deserialize(os.path.join(FLAGS.data_dir, 'snli_1.0_test.bin'))
    return train_data, valid_data, test_data


def load_dict():
    word2index_or = deserialize(os.path.join(FLAGS.data_dir, 'word2index_glove.bin'))
    word2index = sorted(word2index_or.items(), key=(lambda x: x[1]))
    word2index = [a[0] for a in word2index]
    print(word2index[0],len(word2index))
    word_embeddings = deserialize(os.path.join(FLAGS.data_dir, 'word_embeddings_glove.bin'))
    print(len(word_embeddings))
    return word2index, word_embeddings


def gen_batch_data(data):
    def padding(sent, l):
        return sent + ['_PAD'] * (l - len(sent))

    max_len = max([max(len(item[0]), len(item[1])) for item in data])
    texts1, texts2, texts_length, labels = [], [], [], []

    for item in data:
        texts1.append(padding(item[0], max_len))
        texts2.append(padding(item[1], max_len))
        texts_length.append(len(item[0]))
        labels.append(np.array(item[2]))

    batched_data = {'texts1': np.array(texts1), 'texts2': np.array(texts2), 'texts_length': texts_length, 'labels': labels, 'keep_prob': 0.5}

    return batched_data


def train(model, sess, dataset):
    st, ed, loss, accuracy = 0, 0, .0, .0
    gen_summary = True
    while ed + FLAGS.batch_size < len(dataset):
        st, ed = ed, ed + FLAGS.batch_size if ed + \
            FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        # print(batch_data['texts2'][0])
        outputs = model.train_step(sess, batch_data, summary=gen_summary)
        if gen_summary:
            summary = outputs[-1]
            gen_summary = False
        loss += outputs[0]
        accuracy += outputs[1]
    sess.run(model.epoch_add_op)

    return loss / len(dataset), accuracy / len(dataset), summary


def evaluate(model, sess, dataset):
    st, ed, loss, accuracy = 0, 0, .0, .0
    while ed + FLAGS.batch_size < len(dataset):
        st, ed = ed, ed + FLAGS.batch_size if ed + FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        #print batch_data['texts2']
        outputs = sess.run(['loss:0', 'accuracy:0'], {
                           'texts1:0': batch_data['texts1'], 'texts2:0': batch_data['texts2'], 'texts_length:0': batch_data['texts_length'], 'labels:0': batch_data['labels'], 'keep_prob:0': 1.0})
        loss += outputs[0]
        accuracy += outputs[1]
    return loss / len(dataset), accuracy / len(dataset)


def inference(model, sess, dataset):
    st, ed, loss, accuracy = 0, 0, .0, .0
    result = []
    while ed < len(dataset):
        st, ed = ed, ed + FLAGS.batch_size if ed + \
            FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        outputs = sess.run(['predict_labels:0'], {
                           'texts:0': batch_data['texts'], 'texts_length:0': batch_data['texts_length']})
        result += outputs[0].tolist()

    with open('result.txt', 'w') as f:
        for label in result:
            f.write('%d\n' % label)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        print(FLAGS.__flags)
        
        data_train = deserialize(os.path.join(FLAGS.data_dir, 'snli_1.0_train.bin'))
        data_dev = deserialize(os.path.join(FLAGS.data_dir, 'snli_1.0_dev.bin'))
        data_test = deserialize(os.path.join(FLAGS.data_dir, 'snli_1.0_test.bin'))
        # vocab = build_vocab(FLAGS.data_dir, 'vocab_list.txt')
        # print vocab[0]
        # embed = np.loadtxt('data/vec.txt').astype(np.float32)
        word2index, word_embeddings = load_dict()

        
        model = RNN(
            FLAGS.symbols,
            FLAGS.embed_units,
            FLAGS.units,
            FLAGS.labels,
            FLAGS.batch_size,
            word_embeddings,
            learning_rate=0.005)
        if FLAGS.log_parameters:
            model.print_parameters()

        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(
                sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            op_in = model.symbol2index.insert(constant_op.constant(word2index),
               constant_op.constant(list(range(FLAGS.symbols)), dtype=tf.int64))
            sess.run(op_in)

        summary_writer = tf.summary.FileWriter(
            '%s/log' % FLAGS.train_dir, sess.graph)
        while model.epoch.eval() < FLAGS.epoch:
            epoch = model.epoch.eval()
            random.shuffle(data_train)
            start_time = time.time()
            loss, accuracy, summary = train(model, sess, data_train)
            summary_writer.add_summary(summary, epoch)
            summary = tf.Summary()
            summary.value.add(tag='loss/train', simple_value=loss)
            summary.value.add(tag='accuracy/train', simple_value=accuracy)

            model.saver.save(sess, '%s/checkpoint' %
                             FLAGS.train_dir, global_step=model.global_step)
            print("epoch %d learning rate %.4f epoch-time %.4f loss %.8f accuracy [%.8f]" % (
                epoch, model.learning_rate.eval(), time.time() - start_time, loss, accuracy))
            # todo: implement the tensorboard code recording the statistics of development and test set
            loss, accuracy = evaluate(model, sess, data_dev)
            summary.value.add(tag='loss/dev', simple_value=loss)
            summary.value.add(tag='accuracy/dev', simple_value=accuracy)
            print("        dev_set, loss %.8f, accuracy [%.8f]" % (
                loss, accuracy))

            loss, accuracy = evaluate(model, sess, data_test)
            summary.value.add(tag='loss/test', simple_value=loss)
            summary.value.add(tag='accuracy/test', simple_value=accuracy)
            summary_writer.add_summary(summary, epoch)
            print("        test_set, loss %.8f, accuracy [%.8f]" % (
                loss, accuracy))
