import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)]                # petal length and petal width
y = (iris.target == 0).astype(np.int)   # Iris Setosa?

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

y_predict = per_clf.predict([[2, 0.5]])
print(y_predict)


"""
Training an Multi-Layer Perceptron (MLP) using TensorFlow's high-level APT
"""
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data")
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"].astype(np.int32)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
print(X_train.shape)
print(y_train[:10])

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                        feature_columns=feature_cols)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)    # if tensorflow>=1.1
dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)

from sklearn.metrics import accuracy_score
y_predict = dnn_clf.predict(X_test)
print(accuracy_score(y_test, y_predict['classes']))
