# This python3 file implements the linear regression using TensorFlow.
# The example uses california housing data from scikit-learn
#
# The source code partially refers to book titled "Hands-on ML with Scikit-learn & TensorFlow"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()    # not a very good example for linear regression looking at the relation
data = housing.data
target = housing.target
""" Generated dataset"""
m = 1000;
x = 20 * np.random.rand(m,1) + 10;
data = x.reshape(-1,1)
target = 0.5*data + 2 + 0.1* np.random.randn(m,1)

m, n = data.shape
print('The dataset is of shape: ', data.shape)
# plt.plot(data[:,0], target,"b.")

trainSize = int(np.floor(m*0.7))
testSize = m - trainSize

data_train = data[:trainSize, :]
target_train = target[:trainSize]

data_test = data[trainSize:, :]
target_test = target[trainSize:]
#plt.plot(data_train,target_train,"r*")
#plt.show()

data_with_bias = np.c_[np.ones((trainSize,1)), data_train]

n_epochs = 1000
learning_rate = 0.00001   # too large learning_rate will make the gradient descent unbounded

""" Gradient descent using numpy
X = data_with_bias
y = target_train.reshape(-1,1)

print(X.shape,y.shape)
theta = np.random.randn(n+1,1)

mse = np.zeros(n_epochs)
for epoch in range(n_epochs):
    err = X.dot(theta) - y
    #print(error.shape)
    gradients = 2.0/trainSize * X.T.dot(err)
    if epoch > 100:
        learning_rate = learning_rate/(epoch/10.0)
    theta = theta - learning_rate * gradients
    mse[epoch] = err.T.dot(err)/trainSize
    if epoch %100 == 0:
        print("Epoch: ", epoch, "MSE: ", mse[epoch])

print(theta)
plt.figure(2)
plt.plot(mse)
plt.show()
"""

X = tf.constant(data_with_bias, dtype=tf.float32, name="X")
y = tf.constant(target_train.reshape(-1, 1), dtype=tf.float32, name="y")

# Compute theta using least squares
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
# with tf.Session() as sess:
#    theta_value = theta.eval()

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), dtype=tf.float32, name="theta")
y_predict = tf.matmul(X, theta, name="predictions")
error = y_predict - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# Manully computing the gradients for linear regression
gradients = 2/trainSize * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * (gradients))

# Using an Optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
#                                        momentum=0.9)
#training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_history = np.zeros(n_epochs)
theta_history = np.zeros((n+1, n_epochs))
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        sess.run(training_op)
        if epoch % 20 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        mse_history[epoch] = mse.eval()
        theta_history[:,epoch] = theta.eval().reshape(-1)
    theta_value = theta.eval()

print("The best theta value is ", theta_value.reshape(-1))
plt.figure(2)
plt.plot(mse_history)
plt.title("The MSE curve vs training epochs")


def eval_accuracy(data, target, theta):
    data_num, n = data.shape
    data_with_bias = np.c_[np.ones((data_num, 1)) ,data]
    y_predict = np.matmul(data_with_bias, theta_value)
    #print('The first 5 predicted housing prices: ', y_predict[:5].reshape(-1))
    #print('The first 5 true housing prices: ', target[:5])
    mse = np.mean((y_predict-target.reshape(-1, 1))**2)
    #print('The predict price average absolute error over the average price: ', mse)
    return mse

val_err = np.zeros(n_epochs)
for epoch in range(n_epochs):
    theta_value = theta_history[:, epoch]
    val_err[epoch] = eval_accuracy(data_test, target_test, theta_value.reshape(-1,1))

plt.plot(val_err,"r")
plt.show()

# Accuracy on the training dataset
print('ON TRAINING DATASET')
mse = eval_accuracy(data_train, target_train, theta_value)
print('The predict price average absolute error over the average price: ', mse)

# Accuracy on the test dataset
print('ON TEST DATASET')
mse = eval_accuracy(data_test, target_test, theta_value)
print('The predict price average absolute error over the average price: ', mse)
