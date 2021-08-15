import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = X_train.reshape(X_train.shape[0], np.prod(X_train[0,:,:].shape))
X_test = X_test.reshape(X_test.shape[0], np.prod(X_test[0,:,:].shape))

# one-hot encoding:
nb_classes = 10
Y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
Y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
X = X_train # (60000, 784)
Y = Y_train # (60000, 10)

intermediate_size = 256
batch_size = 128
learning_rate = 0.01

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(intermediate_size,
                           #kernel_initializer='glorot_uniform',
                           kernel_initializer=tf.keras.initializers.RandomNormal(),
                           activation='relu'),  # 'sigmoid'),
     tf.keras.layers.Dense(10,
                           #kernel_initializer='glorot_uniform',
                           kernel_initializer=tf.keras.initializers.RandomNormal(),
                           activation='softmax')
     ]
  )


def train_step(inputs, labels, model):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    with tf.GradientTape() as tape:
        prediction = model(inputs)
        loss = cce(labels, prediction)

    grads = tape.gradient(loss, model.trainable_variables)
    return loss, grads


opt = tf.keras.optimizers.SGD(learning_rate)

total_steps = int(X.shape[0] / batch_size)

for step in range(total_steps):
    x = X[step*batch_size:(step+1)*batch_size]
    y = Y[step*batch_size:(step+1)*batch_size]

    loss, grads = train_step(x, y, model)
    print(f'step: {step} loss: {loss:.3f}')
    #print(tf.math.reduce_max(grads[0]))
    opt.apply_gradients(zip(grads, model.trainable_variables))
