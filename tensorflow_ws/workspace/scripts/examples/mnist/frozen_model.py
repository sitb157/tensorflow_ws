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
    model = tf.keras.models.load_model('training_1/model.keras')
    model.summary()

    ### Convert to Concrete Function Format
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    ### Print frozen model layers
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 60)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 60)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    ### Save Frozen Model
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=frozen_output_path,
                      name=f'{frozen_model}.pb',
                      as_text=False)

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=frozen_output_path,
                      name=f'{frozen_model}.pbtxt',
                      as_text=True)
 
    ### Defined Model with softmax 
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
        ])
    
    print(probability_model(x_test[:5]))
