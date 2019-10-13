# -*- coding: utf-8 -*-
"""
Author:-aam35
Analyzing Forgetting in neural networks
"""

import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import time

# from MLP import MLP2, MLP3, MLP4
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.enable_eager_execution()
tf.executing_eagerly()

size_input = 784
size_hidden = 256
size_output = 10
number_of_train_examples = 60000
number_of_test_examples = 10000
num_tasks_to_run = 10
num_epochs_per_task = 20
minibatch_size = 32
learning_rate = 0.001

class MLP2(object):
    def __init__(self, size_input, size_hidden, size_output, device=None, lr=1e-2):
        """
        size_input: int, size of input layer
        size_hidden: int, size of hidden layer
        size_output: int, size of output layer
        device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
        """
        self.lr = lr
        self.size_input, self.size_hidden, self.size_output, self.device = \
            size_input, size_hidden, size_output, device

        # Initialize weights between input layer and hidden layer
        self.W1 = tf.Variable(tf.random.normal([self.size_input, self.size_hidden]), dtype=tf.float32)
        # Initialize biases for hidden layer
        self.b1 = tf.Variable(tf.random.normal([1, self.size_hidden]), dtype=tf.float32)
        # Initialize weights between hidden layer and output layer
        self.W2 = tf.Variable(tf.random.normal([self.size_hidden, self.size_output]), dtype=tf.float32)
        # Initialize biases for output layer
        self.b2 = tf.Variable(tf.random.normal([1, self.size_output]), dtype=tf.float32)

        # Define variables to be updated during backpropagation
        self.variables = [self.W1, self.W2, self.b1, self.b2]

    def forward(self, X):
        """
        forward pass
        X: Tensor, inputs
        """
        if self.device is not None:
            with tf.device('gpu:0' if self.device == 'gpu' else 'cpu'):
                self.y = self.compute_output(X)
        else:
            self.y = self.compute_output(X)

        return self.y

    def loss(self, y_pred, y_true):
        '''
        y_pred - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        '''
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true_tf, logits=y_pred_tf))
        # return tf.compat.v1.losses.mean_squared_error(y_true_tf, y_pred_tf)

    def backward(self, X_train, y_train):
        """
        backward pass
        """
        # optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.lr)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr)
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train)
        grads = tape.gradient(current_loss, self.variables)
        optimizer.apply_gradients(zip(grads, self.variables),
                                  global_step=tf.compat.v1.train.get_or_create_global_step())

    def compute_output(self, X):
        """
        Custom method to obtain output tensor during forward pass
        """
        # Cast X to float32
        X_tf = tf.cast(X, dtype=tf.float32)
        # Remember to normalize your dataset before moving forward
        # Compute values in hidden layer
        what = tf.matmul(X_tf, self.W1) + self.b1
        hhat = tf.nn.relu(what)
        # Compute output
        output = tf.matmul(hhat, self.W2) + self.b2

        # Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this
        # Second add tf.Softmax(output) and then return this variable
        # return tf.compat.v2.nn.softmax(output)
        return output

def one_hot(feature, label, seed):
    feature = tf.cast(feature, dtype=tf.float32)
    feature /= 255.0
    feature = tf.reshape(feature, (784,))
    return tf.random.shuffle(feature, seed=seed), tf.one_hot(label, depth=10)


mnist_train, train_info = tfds.load(name="mnist", with_info=True, as_supervised=True, split=tfds.Split.TRAIN)
# convert your labels in one-hot
seed = 10
mnist_train = mnist_train.map(lambda feature, label: one_hot(feature, label, seed))
# you can batch your data here
mnist_train = mnist_train.batch(minibatch_size)


X_test, _ = tfds.load(name="mnist", with_info=True, as_supervised=True, split=tfds.Split.TEST)
X_test = X_test.map(lambda feature, label: one_hot(feature, label, seed)).batch(number_of_test_examples)


## Permuted MNIST
train_iterator = tf.compat.v1.data.make_one_shot_iterator(mnist_train)

# Initialize model using CPU
mlp_on_cpu = MLP2(size_input, size_hidden, size_output, device='gpu', lr=1e-2)

time_start = time.time()
for epoch in range(num_epochs_per_task):
    loss_total = tf.Variable(0, dtype=tf.float32)

    for inputs, outputs in train_iterator:
        preds = mlp_on_cpu.forward(inputs)
        loss_total = loss_total + mlp_on_cpu.loss(preds, outputs)
        mlp_on_cpu.backward(inputs, outputs)
    print('Number of Epoch = {} - Average MSE:= {:.4f}'.format(epoch + 1, loss_total.numpy() / (
    number_of_train_examples / minibatch_size)))
    train_iterator = tf.compat.v1.data.make_one_shot_iterator(mnist_train)
test_iterator = tf.compat.v1.data.make_one_shot_iterator(X_test)
test_input, test_labels = test_iterator.next()
model_output = mlp_on_cpu.forward(test_input)

correct_prediction = tf.equal(tf.argmax(model_output, 1), tf.argmax(test_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy = {:.4f}'.format(accuracy))

time_taken = time.time() - time_start

print('\nTotal time taken (in seconds): {:.2f}'.format(time_taken))
#For per epoch_time = Total_Time / Number_of_epochs

#Based on tutorial provided create your MLP model for above problem
#For TF2.0 users Keras can be used for loading trainable variables and dataset.
#You might need google collab to run large scale experiments