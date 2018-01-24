import numpy as np
import tensorflow as tf

n_inputs = 28*28        #mnist
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.001
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

logdir = "./tf_logs/dnn"
loss_summary = tf.summary.scalar('Loss-cross entropy', loss)
acc_train_summary = tf.summary.scalar('Train accuracy',accuracy)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_epochs = 20
batch_size = 50
n_batches = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            if iteration % 10 == 0:
                summary_str = loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
                acc_train_summary_str = acc_train_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + iteration
                file_writer.add_summary(summary_str, step)
                file_writer.add_summary(acc_train_summary_str, step)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        acc_val = accuracy.eval(feed_dict={X: mnist.validation.images,
                                           y: mnist.validation.labels})
        print("Epoch#", epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")

# Use the trained DNN
from sklearn.metrics import accuracy_score

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    X_test = mnist.test.images[1,:]
    Z = logits.eval(feed_dict={X: X_test})
    print(mnist.test.labels[1])
    print(np.argmax(Z, axis=1))
    X_test, y_test = mnist.test.images, mnist.test.labels
    y_predict = logits.eval(feed_dict={X: X_test})
    print(accuracy_score(y_test, y_predict['classes']))
