import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

### Load mnist Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

### Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ]) 

### Choose the optimizer, loss_function, performance_metric
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
              optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy']
             )

### Train and Test 
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

### Defined Model with softmax 
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
    ])

print(probability_model(x_test[:5]))
