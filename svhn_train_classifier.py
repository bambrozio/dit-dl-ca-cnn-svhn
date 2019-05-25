from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import tensorflow as tf

from svhn_preprocessing import load_svhn_data
from svhn_model import classification_head
from datetime import datetime

TENSORBOARD_SUMMARIES_DIR = 'logs/svhn_classifier_logs'
TENSOR_BOARD_TRAIN_WRITER = TENSORBOARD_SUMMARIES_DIR+'/train'
TENSOR_BOARD_VALID_WRITER = TENSORBOARD_SUMMARIES_DIR+'/validation'
CLASSIFIER_CKPT = TENSORBOARD_SUMMARIES_DIR+"/ckpt/classifier.ckpt"

#0-9
NUM_LABELS = 10
IMG_ROWS = 32
IMG_COLS = 32
NUM_CHANNELS = 3 # (32,32,3)

BATCH_SIZE = 256 # every 256 images the model and its WEIGHTS are updated
NUM_EPOCHS = 128 # total iteration 

# LEARNING RATE HYPER PARAMS
#LR controls the adjust in the weights with respect to loss gradient. 
LEARN_RATE = 0.075 
DECAY_RATE = 0.95 #This prevents the weights from growing too large  
STAIRCASE = True

def prepare_log_dir():
    '''Clears the log files then creates new directories to place
        the tensorbard log file.'''
    if tf.gfile.Exists(TENSORBOARD_SUMMARIES_DIR):
        tf.gfile.DeleteRecursively(TENSORBOARD_SUMMARIES_DIR)
    tf.gfile.MakeDirs(TENSORBOARD_SUMMARIES_DIR)


def fill_feed_dict(data, labels, x, y_, step):
    size = labels.shape[0]
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    offset = (step * BATCH_SIZE) % (size - BATCH_SIZE)
    batch_data = data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = labels[offset:(offset + BATCH_SIZE)]
    return {x: batch_data, y_: batch_labels}

# main helper method for training the classifer 
def train_classification(train_data, train_labels,
                         valid_data, valid_labels,
                         test_data, test_labels,
                         train_size, saved_weights_path):
    
    #This creates a global tf variable named "global_step" 
    global_step = tf.Variable(0, trainable=False) 

    # This is where training samples and labels are fed to the graph.
    with tf.name_scope('input'):
        images_placeholder = tf.placeholder(tf.float32,
                                            shape=[BATCH_SIZE, IMG_ROWS,
                                                   IMG_COLS, NUM_CHANNELS], name="Images_Input")
        labels_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS], name="Labels_Input")

    with tf.name_scope('image'):
        tf.summary.image('train_input', images_placeholder, 10)


    # Training computation: logits + cross-entropy loss.
    # Computes softmax cross entropy between logits and labels
    logits = classification_head(images_placeholder, train=True)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                          logits=logits, labels=labels_placeholder))
        tf.summary.scalar('loss', loss)
        
    #applies an exponential decay function to a provided initial learning rate. 
    #It requires a global_step value to compute the decayed learning rate. 
    #Decay prevents the weights from growing too large  
    learning_rate = tf.train.exponential_decay(LEARN_RATE,
                                               global_step*BATCH_SIZE,
                                               train_size,
                                               DECAY_RATE,
                                               staircase=STAIRCASE)

    #Outputs a Summary protocol buffer containing a single scalar value.
    tf.summary.scalar('learning_rate', learning_rate)
    
    '''Optimizer: set up a variable that's incremented
      once per batch and controls the learning rate decay.'''
    with tf.name_scope('train'):
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(classification_head(images_placeholder, train=False))

    init_op = tf.initialize_all_variables()

    # Accuracy ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

        # Restore variables and WEIGHTS checkpoint from disk. 
        if(saved_weights_path):
            saver.restore(sess, saved_weights_path)
            print("Model restored.")

        sess.run(init_op)
        # Run all the initializers to prepare the trainable parameters.

        # Add histograms for trainable variables.
        # histogram summary to visualize data's distribution in TensorBoard. 
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add accuracy to tensorboard
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(train_prediction, 1),
                                              tf.argmax(labels_placeholder, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        # Prepare vairables for the tensorboard
        merged = tf.summary.merge_all()

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
            # Run the graph and fetch some of the nodes.
            # This dictionary maps the batch data (as a numpy array) to the
            feed_dict = fill_feed_dict(train_data, train_labels,
                                       images_placeholder, labels_placeholder,
                                       step)
            _, l, lr, acc = sess.run([optimizer, loss, learning_rate, accuracy], feed_dict=feed_dict)
            duration = time.time() - start_time

            if step % 1000 == 0:
                valid_feed_dict = fill_feed_dict(valid_data, valid_labels,
                                                  images_placeholder,
                                                  labels_placeholder, step)
                valid_summary, _, l, lr, valid_acc = sess.run([merged, optimizer, loss, learning_rate, accuracy], feed_dict=valid_feed_dict, options=run_options, run_metadata=run_metadata)
                valid_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                print('Validation Accuracy: %.2f%%' % valid_acc)
                valid_writer.add_summary(valid_summary, step)

                train_summary, _, l, lr, train_acc = sess.run([merged, optimizer, loss, learning_rate, accuracy],
                    feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                print('Training Accuracy: %.2f%%' % train_acc)
                train_writer.add_summary(train_summary, step)

                print('Adding run metadata for', step)

            if step % 100 == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                examples_per_sec = BATCH_SIZE / duration
                format_str = ('%s: step %d, loss = %.2f  learning rate = %.6f  (%.1f examples/sec; %.2f ''sec/batch)')
                print (format_str % (datetime.now(), step, l, lr, examples_per_sec, duration))

                print('Mini-Batch Accuracy: %.2f%%' % acc)

                sys.stdout.flush()

        # Save the variables to CLASSIFIER_CKPT.
        save_path = saver.save(sess, CLASSIFIER_CKPT)
        print("Model saved in file: %s" % save_path)

        test_feed_dict = fill_feed_dict(test_data, test_labels, images_placeholder, labels_placeholder, step)
        summary, acc = sess.run([merged, accuracy], feed_dict=test_feed_dict)
        print('Test Accuracy: %.5f%%' % acc)

        train_writer.close()
        valid_writer.close()


def main(saved_weights_path):
    prepare_log_dir()
    
    # Load the data arrays from its correct file location and assign variables
    train_data, train_labels = load_svhn_data("train", "cropped")
    valid_data, valid_labels = load_svhn_data("valid", "cropped")
    test_data, test_labels = load_svhn_data("test", "cropped")

    print("Training", train_data.shape)
    print("Valid", valid_data.shape)
    print("Test", test_data.shape)

    train_size = train_labels.shape[0]
    
    # call main training method to kick off classification training
    train_classification(train_data, train_labels,
                         valid_data, valid_labels,
                         test_data, test_labels, train_size,
                         saved_weights_path)

# passing the weights chkpt file location when calling for prediction
# if chkpt file exists use it, if not kick off the training    
def run(saved_weights_path = None):
    if saved_weights_path is not None:
        print("Loading Saved Checkpoints From: ", saved_weights_path)
        if os.path.isfile(saved_weights_path):
            saved_weights_path = saved_weights_path
        else:
            raise EnvironmentError("The weights file [%] cannot be opened." % saved_weights_path)
    else:
        print("No weights file informed. Starting from scratch...")
    main(saved_weights_path)

if __name__ == '__main__':
    saved_weights_path = sys.argv[1] if len(sys.argv) > 1 else None
    run(saved_weights_path)

