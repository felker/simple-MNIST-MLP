# simple-MNIST-MLP

ATPESC 2021

2 layer MLP for classifying MNIST using different levels of manual implementation and TensorFlow, Keras API. 
- [`mnist_tf_variable_v1.py`](./mnist_tf_variable_v1.py): network and layer definitions bassed on `tf.Variable`, autodiff via `tf.GradientTape`, manual implementation of loss function and SGD, no Keras used beyond loading MNIST dataset.
- [`mnist_tf_variable_v2.py`](./mnist_tf_variable_v2.py): same as v1, but variables collected in a `tf.Module` subclass. Should update `tape.gradient()` call to use `model.trainable_variables` and output a single dictionary of wgrads.
- [`mnist_tf_variable_v3.py`](./mnist_tf_variable_v3.py): same as v1, but using Keras `tf.keras.optimizers` and loss function.
- [`mnist_keras.py`](./mnist_keras.py): using Keras for everything, including `tf.keras.Sequential` for the model. No use of `tf.Variable` or `tf.Module`
