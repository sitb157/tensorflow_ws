import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import datetime

### Load mnist Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

frozen_output_path = 'training_1/'
frozen_model = 'saved_model'

if __name__ == '__main__':
    ### Load Model 
    model = tf.saved_model.load('training_1/')

    trt_model = tf.saved_model.load('trt_training_1/', tags=['serve'])
    trt_infer = trt_model.signatures['serving_default']
