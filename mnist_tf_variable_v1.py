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

stddev = 0.05
# to match default tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
w1 = tf.Variable(tf.random.normal((intermediate_size, X.shape[1]), stddev=stddev), name='w1')
b1 = tf.Variable(tf.zeros(intermediate_size, dtype=tf.float32), name='b1')
w2 = tf.Variable(tf.random.normal((Y.shape[1], intermediate_size), stddev=stddev), name='w2')
b2 = tf.Variable(tf.zeros(Y.shape[1], dtype=tf.float32), name='b2')


def train_step(inputs, labels):
    #with tf.GradientTape(persistent=True) as tape:  (only if calling tape after tape.gradient())
    with tf.GradientTape() as tape:
        y1 = tf.nn.relu(tf.matmul(inputs, tf.transpose(w1)) + b1)
        y2 = y1 @ tf.transpose(w2) + b2
        prediction = tf.nn.softmax(logits=y2)

        # cross-entropy error:
        # https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks
        # https://stackoverflow.com/questions/42799818/tensorflow-softmax-cross-entropy-with-logits-versus-tf-reduce-mean-tf-reduce-su
        # max(y_hat, eps) might not be necessary if ReLU is replaced with Sigmoid
        loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(tf.math.maximum(prediction, 1e-15)), axis=-1))

    dl_dw1, dl_dw2, dl_db1, dl_db2 = tape.gradient(loss, [w1, w2, b1, b2])
    # print(type(dl_dw1))
    # print(type(w1))
    # print(type(y1))  # EagerTensor
    #print(dl_dw1, dl_dw2, dl_db1, dl_db2)
    return loss, dl_dw1, dl_dw2, dl_db1, dl_db2


def apply_grads_globals(dl_dw1, dl_dw2, dl_db1, dl_db2, learning_rate):
    global w1, w2, b1, b2
    # print(type(dl_dw1))
    # print(type(w1))
    w1.assign_sub(learning_rate * dl_dw1)
    #print(type(w1))
    w2.assign_sub(learning_rate * dl_dw2)
    b1.assign_sub(learning_rate * dl_db1)
    b2.assign_sub(learning_rate * dl_db2)


def apply_grads(dl_dw1, dl_dw2, dl_db1, dl_db2, w1, w2, b1, b2, learning_rate):
    # w1 = w1 - learning_rate * dl_dw1
    # EagerTensor <---- tf.ResourceVariable - tf.EagerTensor * tf.EagerTensor
    # use tf.Variable.assign*() class methods to perform any changes to a variable (do not
    # make it the LHS of a statement)
    # https://stackoverflow.com/questions/40919706/updating-variable-values-in-tensorflow
    # https://medium.com/analytics-vidhya/3-different-ways-to-perform-gradient-descent-in-tensorflow-2-0-and-ms-excel-ffc3791a160a
    # See bottom snippet from TF documentation

    # basic tf.Tensors cannot be updated once declared

    #print(type(w1))
    w1.assign_sub(learning_rate * dl_dw1)
    #print(type(w1))
    w2.assign_sub(learning_rate * dl_dw2)
    b1.assign_sub(learning_rate * dl_db1)
    b2.assign_sub(learning_rate * dl_db2)
    return w1, w2, b1, b2


total_steps = int(X.shape[0] / batch_size)

for step in range(total_steps):
    x = X[step*batch_size:(step+1)*batch_size]
    y = Y[step*batch_size:(step+1)*batch_size]

    loss, dl_dw1, dl_dw2, dl_db1, dl_db2 = train_step(x, y)
    print(f'step: {step} loss: {loss:.3f}')
    #print(tf.math.reduce_max(dl_dw1))
    #tf.print(dl_dw1, summarize=-1)
    #print(w1.trainable)
    apply_grads_globals(dl_dw1, dl_dw2, dl_db1, dl_db2, learning_rate)
    # w1, w2, b1, b2 = apply_grads(dl_dw1, dl_dw2, dl_db1, dl_db2, w1, w2, b1, b2, learning_rate)

# TensorFlow reminder:
# print(a + b, "\n") # element-wise addition
# print(a * b, "\n") # element-wise multiplication
# print(a @ b, "\n") # matrix multiplication


# https://github.com/tensorflow/tensorflow/issues/33131
# TensorFlow tensor object is not assignable, so you cannot use it on the left-hand side
# of an assignment.


# https://www.tensorflow.org/guide/basic_training_loops
# Given a callable model, inputs, outputs, and a learning rate...
# def train(model, x, y, learning_rate):

#   with tf.GradientTape() as t:
#     # Trainable variables are automatically tracked by GradientTape
#     current_loss = loss(y, model(x))

#   # Use GradientTape to calculate the gradients with respect to W and b
#   dw, db = t.gradient(current_loss, [model.w, model.b])

#   # Subtract the gradient scaled by the learning rate
#   model.w.assign_sub(learning_rate * dw)
#   model.b.assign_sub(learning_rate * db)


# https://www.tensorflow.org/api_docs/python/tf/Variable?version=nightly
# The Variable() constructor requires an initial value for the variable, which can be a
# Tensor of any type and shape. This initial value defines the type and shape of the
# variable. After construction, the type and shape of the variable are fixed. The value
# can be changed using one of the assign methods.
