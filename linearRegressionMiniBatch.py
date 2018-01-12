# This python3 file implements the linear regression using TensorFlow.
# The example uses california housing data from scikit-learn --> changed to simulated dataset
# This code is a slight twist of linearRegression.py using mini-batch training
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

trainSize = int(np.floor(m*0.7))
testSize = m - trainSize

data_train = data[:trainSize, :]
target_train = target[:trainSize]

data_test = data[trainSize:, :]
target_test = target[trainSize:]

data_with_bias = np.c_[np.ones((trainSize,1)), data_train]

n_epochs = 1000
learning_rate = 0.00001   # too large learning_rate will make the gradient descent unbounded

X = tf.placeholder(dtype=tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")

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
saver = tf.train.Saver()

# For mini-batch training
batchSize = 10
n_batches = int(np.ceil(trainSize/batchSize))
def fetchBatch(epoch, batch_index, batchSize):
    X_batch = data_with_bias[batch_index*batchSize:(batch_index+1)*batchSize,:]
    y_batch = target_train[batch_index*batchSize:(batch_index+1)*batchSize].reshape(-1,1)
    return X_batch, y_batch
# For tensorboard visualizing
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "./tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

mse_history = np.zeros(n_epochs)
theta_history = np.zeros((n+1, n_epochs))
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetchBatch(epoch, batch_index, batchSize)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        if epoch % 50 == 0:
            save_path = saver.save(sess, "./tmp/my_model.ckpt")
            print("Epoch", epoch, "MSE = ", mse.eval(feed_dict={X: X_batch, y: y_batch}))
        mse_history[epoch] = mse.eval(feed_dict={X: X_batch, y: y_batch})
        theta_history[:,epoch] = theta.eval().reshape(-1)
    theta_value = theta.eval()
    save_path = saver.save(sess, "./tmp/my_model_final.ckpt")
file_writer.close()

print("The best theta value is ", theta_value.reshape(-1))
plt.figure(2)
plt.plot(mse_history)
plt.title("The MSE curve vs training epochs")
plt.show()
