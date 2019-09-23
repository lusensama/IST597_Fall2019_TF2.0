"""
author:-aam35
"""
import time
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

# Create data
NUM_EXAMPLES = 500

#define inputs and outputs with some noise 
X = tf.random.normal([NUM_EXAMPLES])  #inputs
noise = tf.random.normal([NUM_EXAMPLES]) #noise
y = X * 10 + 22 + noise  #true output

# Create variables.
W = tf.Variable(0.)
b = tf.Variable(0.)


train_steps = 3000
learning_rate = 0.1

# Define the linear predictor.
def prediction(x):

  return None

# Define loss functions of the form: L(y, y_predicted)
def squared_loss(y, y_predicted):
    return tf.reduce_mean(tf.square(y_predicted-y))

def huber_loss(y, y_predicted, m=0.1):
    """Huber loss."""
    error = tf.abs(y-y_predicted)
    above = m * (error - 0.5*m)
    below = 0.5*tf.square(error)
    return tf.reduce_mean(tf.where(error<m, below, above))

prevloss = -1.0
# with tf.device('/cpu:0'):
for i in range(train_steps):
    # t0 = time.time()
    with tf.GradientTape() as tape:
        # tape.watch(W)
        yhat = X * W + b
        # loss = squared_loss(y, yhat)
        loss = huber_loss(y, yhat)
        dW, db = tape.gradient(loss, [W,b])

        if tf.equal(loss, prevloss):
            learning_rate /= 2
        W.assign_sub(dW * learning_rate)
        b.assign_sub(db * learning_rate)
        prevloss = loss


    if i % 100 == 0:
        print(("Loss at step {:03d}: {:.3f}".format(i, loss)))
        # print('time cost is '+ str(t1-t0))
    ###TO DO ## Calculate gradients
print("W={},b={}".format(W, b))
plt.plot(X, y, 'bo',label='org')
plt.plot(X, y * W.numpy() + b.numpy(), 'ro',
         label="huber loss")
plt.legend()
plt.show()
print(" ")
