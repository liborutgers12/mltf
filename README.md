# Machine Learning with Tensorflow
This repository contains several examples when I read/learn/practice ML.

Most of the codes follow guidelines from the book *Hands-on Machine Learning with Scikit-Learn & TensorFlow* by Aurelien Geron

## **Linear Regression** using TensorFlow 

  * [linearRegression.py](./linearRegression.py): This python3 file implements the linear regression using TensorFlow. The example uses california housing data from scikit-learn.

  * [linearRegressionMiniBatch.py](./linearRegressionMiniBatch.py): This code is a slight twist of linearRegression.py using mini-batch training.

## **A toy DNN** for MNIST using plain TensorFlow

  * [dnn.py](./dnn.py): This code implements a toy DNN for MNIST digits classification composed of two hidden dense layers with ReLU and one output dense layer. Cross entropy is used as the loss.  
  In the training phase, the epoch number and batch size are 20 and 50 respectively.   
  Accuracy score of the trained DNN is printed in the test phase.  
  The DNN graph is saved and can be checked from the Tensorboard via logdir ./tf_logs/dnn


