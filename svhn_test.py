import sys
import os
import numpy as np
import tensorflow as tf
import PIL.Image as Image

# Uncomment only when running in a GUI IDE
#import matplotlib.pyplot as plt
import time

from svhn_model import regression_head
from svhn_train_regressor import REGRESSION_CKPT

# Avoid Warning logs
tf.logging.set_verbosity(tf.logging.ERROR)

# Avoid suggestion on log console like: ...Your CPU supports... AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def prediction_to_string(pred_array):
    pred_str = ""
    for i in range(len(pred_array)):
        if pred_array[i] != 10:
            pred_str += str(pred_array[i])
        else:
            return pred_str
    return pred_str


def resize(sample_img, to_size):
    print("Resizing image to [{} x {}]...".format(to_size, to_size))
    return sample_img.resize((to_size, to_size), Image.ANTIALIAS)


def crop(sample_img, to_size):
    width, height = sample_img.size
    print("Image with width [{}] and height [{}].".format(width, height))
    print('Cropping to the centre on [{} x {}]...').format(to_size, to_size)

    left = (width - to_size) / 2
    top = (height - to_size) / 2
    right = (width + to_size) / 2
    bottom = (height + to_size) / 2
    return sample_img.crop((left, top, right, bottom))


# Detect is called by the main program to find the image path
# Do preprocessing of the input image with resize, reshape, decode_png
def detect(img_path):
    sample_img = Image.open(img_path)

    # preprocessing of input image
    width, height = sample_img.size

    if width == height:
        if width != 64:
            sample_img = resize(sample_img, 64)
    else:
        sample_img = crop(sample_img, 64)
        #sample_img = crop(sample_img, 32*1.2) # 32 + 20%
        #sample_img = resize(sample_img, 64)

        
    #sample_img = sample_img.convert('L')

    # Uncomment only when running in a GUI IDE
    # plt.imshow(sample_img)
    # plt.show()

    pix = np.array(sample_img)
    norm_pix = (255-pix)*1.0/255.0
    exp = np.expand_dims(norm_pix, axis=0)

    X = tf.placeholder(tf.float32, shape=(1, 64, 64, 3))
    
    # call regression head method that takes the image and passes it through the network
    # returns an array of 5 logits   
    [logits_1, logits_2, logits_3, logits_4, logits_5] = regression_head(X)

    # each logit is passed to a softmax tf function that produces the most likely prediction
    predict = tf.stack([tf.nn.softmax(logits_1),
                      tf.nn.softmax(logits_2),
                      tf.nn.softmax(logits_3),
                      tf.nn.softmax(logits_4),
                      tf.nn.softmax(logits_5)])

    # transpose prediction
    # argmax returns the index with the largest value across axes of a tensor
    best_prediction = tf.transpose(tf.argmax(predict, 2))

    saver = tf.train.Saver()
    with tf.Session() as session:
        print "Loading model..."
        saver.restore(session, REGRESSION_CKPT)

        feed_dict = {X: exp}
        start_time = time.time()
        predictions = session.run(best_prediction, feed_dict=feed_dict)
        pred = prediction_to_string(predictions[0])
        end_time = time.time()
        print("Best Prediction: [{}]. Time spent in seconds: [{}]").format(pred, end_time - start_time)
        return pred


def classify(img_path = None):
    if img_path is None:
        raise EnvironmentError("You must pass an image file to process")

    print("Reading Image file: [{}]").format(img_path)
    if not os.path.isfile(img_path):
        raise EnvironmentError("Image file cannot be opened.")

    detect(img_path)

if __name__ == "__main__":
    classify(sys.argv[1])
