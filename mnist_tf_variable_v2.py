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


class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        stddev = 0.05
        # to match default tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        self.w1 = tf.Variable(tf.random.normal((intermediate_size, X.shape[1]), stddev=stddev), name='w1')
        self.b1 = tf.Variable(tf.zeros(intermediate_size, dtype=tf.float32), name='b1')
        self.w2 = tf.Variable(tf.random.normal((Y.shape[1], intermediate_size), stddev=stddev), name='w2')
        self.b2 = tf.Variable(tf.zeros(Y.shape[1], dtype=tf.float32), name='b2')

    def __call__(self, x):
        y1 = tf.nn.relu(tf.matmul(x, tf.transpose(self.w1)) + self.b1)
        y2 = y1 @ tf.transpose(self.w2) + self.b2
        return tf.nn.softmax(logits=y2)


def train_step(model, inputs, labels):
    with tf.GradientTape() as tape:
        prediction = model(inputs)
        loss = tf.reduce_mean(-tf.reduce_sum(
            labels * tf.math.log(tf.math.maximum(prediction, 1e-15)), axis=-1))
    # print(model.trainable_variables)  # KGF: determnistic ordering? alphabetical?
    dl_dw1, dl_dw2, dl_db1, dl_db2 = tape.gradient(
        loss, [model.w1, model.w2, model.b1, model.b2])  # model.trainable_variables)
    return loss, dl_dw1, dl_dw2, dl_db1, dl_db2


def apply_grads(model, dl_dw1, dl_dw2, dl_db1, dl_db2, learning_rate):
    model.w1.assign_sub(learning_rate * dl_dw1)
    model.w2.assign_sub(learning_rate * dl_dw2)
    model.b1.assign_sub(learning_rate * dl_db1)
    model.b2.assign_sub(learning_rate * dl_db2)


total_steps = int(X.shape[0] / batch_size)

mnist_model = MyModel()

for step in range(total_steps):
    x = X[step*batch_size:(step+1)*batch_size]
    y = Y[step*batch_size:(step+1)*batch_size]

    loss, dl_dw1, dl_dw2, dl_db1, dl_db2 = train_step(mnist_model, x, y)
    print(f'step: {step} loss: {loss:.3f}')
    apply_grads(mnist_model, dl_dw1, dl_dw2, dl_db1, dl_db2, learning_rate)
