import tensorflow as tf
import os
import datetime

### Load mnist Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

### Function to make model
def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name='input'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax', name='output')
    ])

### Choose the optimizer, loss_function, performance_metric
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "/home/sitb157/datas/training_1/model.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)

if __name__ == '__main__':
    ### Generate Model 
    model = create_model()
    model.compile(
              optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy']
             )

    model.summary()
    ### Train Model
    model.fit(x_train, 
              y_train, 
              epochs=5,
              validation_data=(x_test, y_test),
              callbacks=[cp_callback])
    
    ### Defined Model with softmax 
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
        ])
    
    print(probability_model(x_test[:5]))
    model.save('/home/sitb157/datas/saved_model')
