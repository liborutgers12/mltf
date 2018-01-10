# This python3 file implements the linear regression using TensorFlow.
# The example uses california housing data from scikit-learn
#
# The source code partially refers to book titled "Hands-on ML with Scikit-learn & TensorFlow"

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
print('The dataset is of shape: ')
print(m,n)
trainSize = int(np.floor(m*0.7))
testSize = m - trainSize

data_train = housing.data[:trainSize, :]
target_train = housing.target[:trainSize]

data_test = housing.data[trainSize:, :]
target_test = housing.target[trainSize:]

housing_data_with_bias = np.c_[np.ones((trainSize,1)), data_train]

X = tf.constant(housing_data_with_bias, dtype=tf.float32, name="X")
y = tf.constant(target_train.reshape(-1, 1), dtype=tf.float32, name="y")

# Compute theta using least squares
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
# with tf.Session() as sess:
#    theta_value = theta.eval()

n_epochs = 1000
learning_rate = 0.01

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_predict = tf.matmul(X, theta, name="predictions")
error = y_predict -y
mse = tf.reduce_mean(tf.square(error), name="mse")

# Manully computing the gradients for linear regression
# gradients = 2/trainSize * tf.matmul(tf.transpose(X), error)
# training_op = tf.assign(theta, theta - learning_rate * gradients)

# Using an Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)
    theta_value = theta.eval()

print("The best theta value is ", theta_value.reshape(-1))

def eval_accuracy(data, target, theta):
    data_num, n = data.shape
    housing_data_with_bias = np.c_[np.ones((data_num, 1)) ,data]
    y_predict = np.matmul(housing_data_with_bias, theta_value)
    print('The first 5 predicted housing prices: ', y_predict[:5].reshape(-1))
    print('The first 5 true housing prices: ', target[:5])
    print('The predict price average absolute error over the average price: ', \
    np.mean(np.abs(y_predict-target.reshape(-1, 1)))/np.mean(target))

# Accuracy on the training dataset
print('ON TRAINING DATASET')
eval_accuracy(data_train, target_train, theta_value)

# Accuracy on the test dataset
print('ON TEST DATASET')
eval_accuracy(data_test, target_test, theta_value)
