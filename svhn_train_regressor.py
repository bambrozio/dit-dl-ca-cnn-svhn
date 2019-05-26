from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import tensorflow as tf

from svhn_preprocessing import load_svhn_data
from svhn_model import regression_head
from datetime import datetime
from svhn_train_classifier import CLASSIFIER_CKPT

# Avoid Warning logs
tf.logging.set_verbosity(tf.logging.ERROR)

# Avoid suggestion on log console like:
# ...Your CPU supports... AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Run Options
BATCH_SIZE = 32
NUM_EPOCHS = 128 # 128 complete iteration of the entire data set 
TENSORBOARD_SUMMARIES_DIR = 'logs/svhn_regression_logs'
TENSOR_BOARD_TRAIN_WRITER = TENSORBOARD_SUMMARIES_DIR+'/train'
TENSOR_BOARD_VALID_WRITER = TENSORBOARD_SUMMARIES_DIR+'/validation'
REGRESSION_CKPT_DIR = TENSORBOARD_SUMMARIES_DIR+'/ckpt'
REGRESSION_CKPT = REGRESSION_CKPT_DIR+'/regression.ckpt'


# Image Settings
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CHANNELS = 3

# Label Settings
NUM_LABELS = 11
LABELS_LEN = 6

# LEARING RATE HYPER PARAMS
LEARN_RATE = 0.075
DECAY_RATE = 0.95
STAIRCASE = True


def prepare_log_dir():
    '''Clears the log files then creates new directories to place
        the tensorbard log file.'''
    if not tf.gfile.Exists(TENSORBOARD_SUMMARIES_DIR):
        tf.gfile.MakeDirs(TENSORBOARD_SUMMARIES_DIR)


def fill_feed_dict(data, labels, x, y_, step):
    set_size = labels.shape[0]
    
    # Compute the offset of the current minibatch in the data.
    offset = (step * BATCH_SIZE) % (set_size - BATCH_SIZE)
    batch_data = data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = labels[offset:(offset + BATCH_SIZE)]
    return {x: batch_data, y_: batch_labels}

# With batch size we update the weights after passing the data samples each batch
# means the gradients are calculated after passing each batch.    
def train_regressor(train_data, train_labels, valid_data, valid_labels,
                    test_data, test_labels, train_size, saved_weights_path):

    global_step = tf.Variable(0, trainable=False)
    # This is where training samples and labels are fed to the graph.
    with tf.name_scope('input'):
        images_placeholder = tf.placeholder(tf.float32,
                                            shape=(BATCH_SIZE, IMG_HEIGHT,
                                                   IMG_WIDTH, NUM_CHANNELS))

    with tf.name_scope('image'):
        tf.summary.image('input', images_placeholder, 10)

    labels_placeholder = tf.placeholder(tf.int32,
                                        shape=(BATCH_SIZE, LABELS_LEN))

    [logits_1, logits_2, logits_3, logits_4, logits_5] = regression_head(images_placeholder, True)

    # Calc the mean of elements across dimensions of each softmax function.
    # Computes sparse softmax cross entropy between logits and labels.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=labels_placeholder[:, 1])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_2, labels=labels_placeholder[:, 2])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3, labels=labels_placeholder[:, 3])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_4, labels=labels_placeholder[:, 4])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_5, labels=labels_placeholder[:, 5]))

    learning_rate = tf.train.exponential_decay(LEARN_RATE, global_step*BATCH_SIZE, train_size, DECAY_RATE)
    tf.summary.scalar('learning_rate', learning_rate)

    # Optimizer: set up a variable that's incremented once per batch
    with tf.name_scope('train'):
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # predicts using softmax activation function the most likely result 
    # for each of the possible 5 digits
    prediction = tf.stack([tf.nn.softmax(regression_head(images_placeholder)[0]),
                                tf.nn.softmax(regression_head(images_placeholder)[1]),
                                tf.nn.softmax(regression_head(images_placeholder)[2]),
                                tf.nn.softmax(regression_head(images_placeholder)[3]),
                                tf.nn.softmax(regression_head(images_placeholder)[4])])

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    start_time = time.time()
    # Create a local session to run the training.
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        init_op = tf.initialize_all_variables()
        # Restore variables from disk.
        if(saved_weights_path):
            saver.restore(sess, saved_weights_path)
        print("Model restored.")

        reader = tf.train.NewCheckpointReader(CLASSIFIER_CKPT)
        reader.get_variable_to_shape_map()

        # Run all the initializers to prepare the trainable parameters.
        sess.run(init_op)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                best = tf.transpose(prediction, [1, 2, 0])  # permute n_steps and batch_size
                lb = tf.cast(labels_placeholder[:, 1:6], tf.int64)
                correct_prediction = tf.equal(tf.argmax(best, 1), lb)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) / prediction.get_shape().as_list()[1] / prediction.get_shape().as_list()[0]
            tf.summary.scalar('accuracy', accuracy)

        # Prepare vairables for the tensorboard
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(TENSOR_BOARD_TRAIN_WRITER)  # create writer
        train_writer.add_graph(sess.graph)

        valid_writer = tf.summary.FileWriter(TENSOR_BOARD_VALID_WRITER)  # create writer
        valid_writer.add_graph(sess.graph)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        saver = tf.train.Saver()
        saver.save(sess, save_path=TENSOR_BOARD_TRAIN_WRITER, global_step=global_step)
        train_writer = tf.summary.FileWriter(TENSOR_BOARD_TRAIN_WRITER)  # create writer
        train_writer.add_graph(sess.graph)

        saver.save(sess, save_path=TENSOR_BOARD_VALID_WRITER, global_step=global_step)
        valid_writer = tf.summary.FileWriter(TENSOR_BOARD_VALID_WRITER)  # create writer
        valid_writer.add_graph(sess.graph)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()        

        # Loop through training steps.
        for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
        #for step in xrange(5):
            duration = time.time() - start_time
            examples_per_sec = BATCH_SIZE / duration

            # Run the graph and fetch some of the nodes.
            # This dictionary maps the batch data (as a numpy array)
            train_feed_dict = fill_feed_dict(train_data, train_labels, images_placeholder, labels_placeholder, step)
            _, l, lr, acc, predictions = sess.run([optimizer, loss, learning_rate,
                                                  accuracy, prediction],
                                                  feed_dict=train_feed_dict)

            train_batched_labels = train_feed_dict.values()[1]

            # every 1000 steps print the accuracy of the training data set
            if step % 1000 == 0:
                valid_feed_dict = fill_feed_dict(valid_data, valid_labels, images_placeholder, labels_placeholder, step)
                valid_batch_labels = valid_feed_dict.values()[1]

                valid_summary, _, l, lr, valid_acc = sess.run([merged, optimizer, loss, learning_rate, accuracy],
                feed_dict=valid_feed_dict, options=run_options, run_metadata=run_metadata)
                print('Validation Accuracy: %.2f' % valid_acc)
                valid_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                valid_writer.add_summary(valid_summary, step)

                train_summary, _, l, lr, train_acc = sess.run([merged, optimizer, loss, learning_rate, accuracy],
                    feed_dict=train_feed_dict)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                train_writer.add_summary(train_summary, step)
                print('Training Set Accuracy: %.2f' % train_acc)
                print('Adding run metadata for', step)

            # every 100 steps print out the accuracy of the mini batches     
            elif step % 100 == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()

                format_str = ('%s: step %d, loss = %.2f  learning rate = %.2f  (%.1f examples/sec; %.3f ''sec/batch)')
                print (format_str % (datetime.now(), step, l, lr, examples_per_sec, duration))

                print('Minibatch accuracy2: %.2f' % acc)
                sys.stdout.flush()

        test_feed_dict = fill_feed_dict(test_data, test_labels, images_placeholder, labels_placeholder, step)
        _, l, lr, test_acc = sess.run([optimizer, loss, learning_rate, accuracy], feed_dict=test_feed_dict, options=run_options, run_metadata=run_metadata)

        ret = 'Test Accuracy: %.5f%%' % test_acc
        print(ret)

        # Save the variables to regression.ckpt in disk.
        save_path = saver.save(sess, REGRESSION_CKPT)
        print("Model saved in file: %s" % save_path)

        train_writer.close()
        valid_writer.close()
        sess.close();

    return ret


def main(saved_weights_path):
    prepare_log_dir()
    train_data, train_labels = load_svhn_data("train", "full")
    valid_data, valid_labels = load_svhn_data("valid", "full")
    test_data, test_labels = load_svhn_data("test", "full")

    print("TrainData", train_data.shape)
    print("Valid Data", valid_data.shape)
    print("Test Data", test_data.shape)

    train_size = len(train_labels)
    return train_regressor(train_data, train_labels, valid_data, valid_labels,
                    test_data, test_labels, train_size, saved_weights_path)


def run():
    print("Regressor training - Start")

    if not os.path.exists(CLASSIFIER_CKPT+'.index'):
        raise EnvironmentError("File [{}] not found. Please, be sure to run svhn_train_classifier.py first".format(CLASSIFIER_CKPT))

    print("Loading Saved Checkpoints From: ", REGRESSION_CKPT)

    saved_weights_path = None
    if os.path.isdir(REGRESSION_CKPT_DIR):
        for file in os.listdir(os.path.dirname(REGRESSION_CKPT)):
            if os.path.isfile(os.path.join(REGRESSION_CKPT_DIR, file)) and '.ckpt' in file:
                saved_weights_path = REGRESSION_CKPT
                break

    if saved_weights_path is None:
        print("No weights file found. Starting from scratch...")

    ret_class = main(saved_weights_path)
    print("Regressor training - Done")

    return ret_class


if __name__ == '__main__':
    run()
