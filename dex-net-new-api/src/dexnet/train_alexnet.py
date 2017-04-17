###############################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

import IPython
import json
import logging
import os
import numpy as np
import sys
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import random
import threading
import signal
import urllib

import tensorflow as tf

from caffe_classes import class_names
import dnn

# global vars - to config file in future
BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
IM_HEIGHT = 227
IM_WIDTH = 227
NUM_CHANNELS = 3
NUM_CATEGORIES = 320
DECAY_STEP = 100000
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
LOG_FREQUENCY = 1  # Number of steps between evaluations.
FILES_PER_EVAL = 25
NUM_EPOCHS = 100

NUM_TRAIN_FILES = 1000
NUM_VAL_FILES = 278
TOTAL_FILES = 1278
DATAPOINTS_PER_FILE = 100

SEED = 66478  # Set to None for random seed.

class AlexNetWeights(object):
  def __init__(self):
    pass

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def error_rate_in_batches(data_filenames, label_filenames, sess, eval_prediction, num_files, data_mean):
  """Get all predictions for a dataset by running it in small batches."""
  error_rates = []
  data_and_labels_filenames = zip(data_filenames, label_filenames)
  random.shuffle(data_and_labels_filenames)
  for data_filename, label_filename in data_and_labels_filenames[:num_files]:
    # load next file
    data = np.tile(np.load(data_filename)['arr_0'], [1,1,1,3]) - data_mean
    labels = np.load(label_filename)['arr_0']

    # setup buffers
    size = data.shape[0]
    if size < VAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, NUM_CATEGORIES), dtype=np.float32)

    # make predictions
    for begin in xrange(0, size, VAL_BATCH_SIZE):
      end = begin + VAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
          eval_prediction,
          feed_dict={val_data_node: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
          eval_prediction,
          feed_dict={val_data_node: data[-VAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]

    # get error rate
    error_rates.append(error_rate(predictions, labels))
  
    # clean up
    del data
    del labels

  # return average error rate over all files (assuming same size)
  return np.mean(error_rates)

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    """
    Convolution layer helper function
    From https://github.com/ethereon/caffe-tensorflow
    """
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
    
def build_alexnet_weights(out_size, net_data, reinit_fc6=False, reinit_fc7=False):
  """ Build a set of convnet weights for AlexNet """
  #conv1
  #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
  k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
  conv1W = tf.Variable(net_data["conv1"][0])
  conv1b = tf.Variable(net_data["conv1"][1])
  
  #conv2
  #conv(5, 5, 256, 1, 1, group=2, name='conv2')
  k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
  conv2W = tf.Variable(net_data["conv2"][0])
  conv2b = tf.Variable(net_data["conv2"][1])
  
  #conv3
  #conv(3, 3, 384, 1, 1, name='conv3')
  k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
  conv3W = tf.Variable(net_data["conv3"][0])
  conv3b = tf.Variable(net_data["conv3"][1])
  
  #conv4
  #conv(3, 3, 384, 1, 1, group=2, name='conv4')
  k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
  conv4W = tf.Variable(net_data["conv4"][0])
  conv4b = tf.Variable(net_data["conv4"][1])
  
  #conv5
  #conv(3, 3, 256, 1, 1, group=2, name='conv5')
  k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
  conv5W = tf.Variable(net_data["conv5"][0])
  conv5b = tf.Variable(net_data["conv5"][1])
    
  #fc6
  #fc(4096, name='fc6')
  fc6_in_size = net_data["fc6"][0].shape[0]
  fc6_out_size = net_data["fc6"][0].shape[1]
  if reinit_fc6:
    fc6W = tf.Variable(tf.truncated_normal([fc6_in_size, fc6_out_size],
                                           stddev=0.005,
                                           seed=SEED))
    fc6b = tf.Variable(tf.constant(0.1, shape=[fc6_out_size]))
  else:
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])

  #fc7
  #fc(4096, name='fc7') 
  fc7_in_size = fc6_out_size
  fc7_out_size = net_data["fc7"][0].shape[1]
  if reinit_fc7:
    fc7W = tf.Variable(tf.truncated_normal([fc7_in_size, fc7_out_size],
                                           stddev=0.005,
                                           seed=SEED))
    fc7b = tf.Variable(tf.constant(0.1, shape=[fc7_out_size]))
  else:
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    
  #fc8
  #fc(num_cats, relu=False, name='fc8')                       
  fc8_in_size = fc7_out_size
  fc8_out_size = out_size
  fc8W = tf.Variable(tf.truncated_normal([fc8_in_size, fc8_out_size],
                                         stddev=0.01,
                                         seed=SEED))
  fc8b = tf.Variable(tf.constant(0.0, shape=[fc8_out_size]))
  
  # make return object
  weights = AlexNetWeights()
  weights.conv1W = conv1W
  weights.conv1b = conv1b
  weights.conv2W = conv2W
  weights.conv2b = conv2b
  weights.conv3W = conv3W
  weights.conv3b = conv3b 
  weights.conv4W = conv4W
  weights.conv4b = conv4b 
  weights.conv5W = conv5W
  weights.conv5b = conv5b 
  weights.fc6W = fc6W
  weights.fc6b = fc6b
  weights.fc7W = fc7W
  weights.fc7b = fc7b
  weights.fc8W = fc8W
  weights.fc8b = fc8b
  return weights

def build_alexnet(data_node, weights, drop_fc6=False, drop_fc7=False):
  """ Connects graph of alexnet from weights """
  #conv1
  #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
  k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
  conv1_in = conv(data_node, weights.conv1W, weights.conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
  conv1 = tf.nn.relu(conv1_in)
    
  #lrn1
  #lrn(2, 2e-05, 0.75, name='norm1')
  radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
  lrn1 = tf.nn.local_response_normalization(conv1,
                                            depth_radius=radius,
                                            alpha=alpha,
                                            beta=beta,
                                            bias=bias)
  
  #maxpool1
  #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
  k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
  maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
  
  #conv2
  #conv(5, 5, 256, 1, 1, group=2, name='conv2')
  k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
  conv2_in = conv(maxpool1, weights.conv2W, weights.conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
  conv2 = tf.nn.relu(conv2_in)


  #lrn2
  #lrn(2, 2e-05, 0.75, name='norm2')
  radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
  lrn2 = tf.nn.local_response_normalization(conv2,
                                            depth_radius=radius,
                                            alpha=alpha,
                                            beta=beta,
                                            bias=bias)
  
  #maxpool2
  #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
  k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
  maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
  
  #conv3
  #conv(3, 3, 384, 1, 1, name='conv3')
  k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
  conv3_in = conv(maxpool2, weights.conv3W, weights.conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
  conv3 = tf.nn.relu(conv3_in)
    
  #conv4
  #conv(3, 3, 384, 1, 1, group=2, name='conv4')
  k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
  conv4_in = conv(conv3, weights.conv4W, weights.conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
  conv4 = tf.nn.relu(conv4_in)

  #conv5
  #conv(3, 3, 256, 1, 1, group=2, name='conv5')
  k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
  conv5_in = conv(conv4, weights.conv5W, weights.conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
  conv5 = tf.nn.relu(conv5_in)
    
  #maxpool5
  #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
  k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
  maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

  #fc6
  #fc(4096, name='fc6')
  fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), weights.fc6W, weights.fc6b)
  if drop_fc6:
    fc6 = tf.nn.dropout(fc6, 0.5)
    
  #fc7
  #fc(4096, name='fc7') 
  fc7 = tf.nn.relu_layer(fc6, weights.fc7W, weights.fc7b)
  if drop_fc7:
    fc7 = tf.nn.dropout(fc7, 0.5)
    
  #fc8
  #fc(num_cats, relu=False, name='fc8')
  fc8 = tf.nn.xw_plus_b(fc7, weights.fc8W, weights.fc8b)
  return fc8

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(SEED)

    # read params
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    reinit_fc6 = True
    reinit_fc7 = True
    drop_fc6 = True
    drop_fc7 = True
    
    # read in categories
    f = open(os.path.join(data_dir, 'categories.json'), 'r')
    categories = json.load(f)
    category_map = {}
    for key, val in categories.iteritems():
        category_map[val] = key
    for i in range(NUM_CATEGORIES):
        if i in category_map.keys():
            cur_category = category_map[i]
    category_map[i] = cur_category

    # create indices
    indices = np.arange(TOTAL_FILES-1) # minus 1 to remove last file of non-uniform size
    np.random.shuffle(indices)
    indices = indices.tolist()

    # load training data
    train_size = NUM_TRAIN_FILES * DATAPOINTS_PER_FILE
    train_indices = indices[:NUM_TRAIN_FILES]
    train_data_filenames = []
    train_labels_filenames = []
    for i, j in enumerate(train_indices):
      train_data_filename = os.path.join(data_dir, 'train_depths_%04d.npz' %(j))
      train_label_filename = os.path.join(data_dir, 'train_labels_%04d.npz' %(j))
      train_data_filenames.append(train_data_filename)
      train_labels_filenames.append(train_label_filename)

    val_indices = indices[NUM_TRAIN_FILES:NUM_TRAIN_FILES+NUM_VAL_FILES]
    val_data_filenames = []
    val_labels_filenames = []
    for i, j in enumerate(train_indices):
      val_data_filename = os.path.join(data_dir, 'train_depths_%04d.npz' %(j))
      val_label_filename = os.path.join(data_dir, 'train_labels_%04d.npz' %(j))
      val_data_filenames.append(val_data_filename)
      val_labels_filenames.append(val_label_filename)

    # setup nodes
    train_data_batch = tf.placeholder(tf.float32, (BATCH_SIZE, IM_HEIGHT, IM_WIDTH, NUM_CHANNELS))
    train_labels_batch = tf.placeholder(tf.int64, (BATCH_SIZE,))
    val_data_node = tf.placeholder(tf.float32, (VAL_BATCH_SIZE, IM_HEIGHT, IM_WIDTH, NUM_CHANNELS))

    # create queue
    queue_capacity = 100
    q = tf.FIFOQueue(queue_capacity, [tf.float32, tf.int64], shapes=[(BATCH_SIZE, IM_HEIGHT, IM_WIDTH, NUM_CHANNELS),
                                                                     (BATCH_SIZE,)])
    enqueue_op = q.enqueue([train_data_batch, train_labels_batch])
    train_data_node, train_labels_node = q.dequeue()

    # get conv weights
    net_data = np.load('data/cnn/bvlc_alexnet.npy').item()
    data_mean = np.load('data/cnn/mean.npy')
    weights = build_alexnet_weights(NUM_CATEGORIES, net_data,
                                    reinit_fc6=reinit_fc6,
                                    reinit_fc7=reinit_fc7)

    # start tf session
    start_time = time.time()
    saver = tf.train.Saver()
    sess = tf.Session()
    
    # function for loading and enqueuing training data
    files_per_batch = int(BATCH_SIZE / DATAPOINTS_PER_FILE) + 1
    points_per_file = BATCH_SIZE / files_per_batch

    term_event = threading.Event()
    term_event.clear()
    def load_and_enqueue():
      train_data = np.zeros([BATCH_SIZE, IM_HEIGHT, IM_WIDTH, NUM_CHANNELS])
      label_data = np.zeros(BATCH_SIZE)
      all_ind = np.arange(DATAPOINTS_PER_FILE)

      while not term_event.is_set():
        # loop through data
        for j in range(files_per_batch):
          # gen file index uniformly at random
          i = int(NUM_TRAIN_FILES * np.random.rand() + 0.5)

          # get batch indices uniformly at random
          np.random.shuffle(all_ind)
          ind = all_ind[:points_per_file]

          # get indices
          start_i = j * points_per_file
          end_i = start_i + points_per_file

          # enqueue training data batch
          train_data_arr = np.load(train_data_filenames[i])['arr_0']
          train_label_arr = np.load(train_labels_filenames[i])['arr_0']
          train_data[start_i:end_i, ...] = np.tile(train_data_arr[ind,...], [1,1,1,3]) - data_mean
          label_data[start_i:end_i] = train_label_arr[ind]
          del train_data_arr
          del train_label_arr
          
        # send data to queue
        sess.run(enqueue_op, feed_dict={train_data_batch: train_data,
                                        train_labels_batch: label_data})
          
    #prob
    #softmax(name='prob'))
    train_net_output = build_alexnet(train_data_node, weights, drop_fc6=drop_fc6, drop_fc7=drop_fc7)
    val_net_output = build_alexnet(val_data_node, weights, drop_fc6=False, drop_fc7=False)

    train_predictions = tf.nn.softmax(train_net_output)
    val_predictions = tf.nn.softmax(val_net_output)

    # form loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(train_net_output, train_labels_node))
    layer_weights = weights.__dict__.values()
    regularizers = tf.nn.l2_loss(layer_weights[0])
    for w in layer_weights:
      regularizers = regularizers + tf.nn.l2_loss(w)
    loss += 5e-4 * regularizers

    # setup learning rate
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.9).minimize(loss,
                                                         global_step=batch)

    # setup data thread
    def handler(signum, frame):
      term_event.set()
    signal.signal(signal.SIGINT, handler)
    try:
      t = threading.Thread(target=load_and_enqueue)
      t.start()

      # init and run tf sessions
      init = tf.initialize_all_variables()
      sess.run(init)
      print('Initialized!')

      # Loop through training steps.
      train_eval_iters = []
      train_losses = []
      train_errors = []
      val_eval_iters = []
      val_errors = []
      for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
        # run optimization
        _, l, lr, predictions, batch_labels = sess.run(
          [optimizer, loss, learning_rate, train_predictions, train_labels_node])
        if step % LOG_FREQUENCY == 0:
          elapsed_time = time.time() - start_time
          start_time = time.time()
          print('Step %d (epoch %.2f), %.1f ms' %
                (step, float(step) * BATCH_SIZE / train_size,
                 1000 * elapsed_time / EVAL_FREQUENCY))
          print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
          train_error = error_rate(predictions, batch_labels)
          print('Minibatch error: %.3f%%' %train_error)
          sys.stdout.flush()

          train_eval_iters.append(step)
          train_errors.append(train_error)
          train_losses.append(l)

        if step % EVAL_FREQUENCY == 0:
          val_error = error_rate_in_batches(
            val_data_filenames, val_labels_filenames, sess, val_predictions, FILES_PER_EVAL, data_mean)
          print('Validation error: %.1f%%' %val_error)
          sys.stdout.flush()

          val_eval_iters.append(step)
          val_errors.append(val_error)

          # save everything!
          np.save(os.path.join(output_dir, 'train_eval_iters.npy'), train_eval_iters)
          np.save(os.path.join(output_dir, 'val_eval_iters.npy'), val_eval_iters)
          np.save(os.path.join(output_dir, 'train_losses.npy'), train_losses)
          np.save(os.path.join(output_dir, 'train_errors.npy'), train_errors)
          np.save(os.path.join(output_dir, 'val_errors.npy'), val_errors)

          saver.save(sess, os.path.join(output_dir, 'model.ckpt'))

    except Exception as e:
      handler(0,0)
      raise e
      
