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
y = X * 3 + 2 + noise  #true output

# Create variables.
W = tf.Variable(0.)
b = tf.Variable(0.)


train_steps = 1000
learning_rate = 0.001

# Define the linear predictor.
def prediction(x):

  return None

# Define loss functions of the form: L(y, y_predicted)
def squared_loss(y, y_predicted):
    return tf.reduce_mean(tf.square(y_predicted-y))

def huber_loss(y, y_predicted, m=1.0):
    """Huber loss."""
    if np.absolute(y_predicted-y) <= m:
        return .5*((y_predicted - y)**2)
    else:
      return  m*np.absolute(y_predicted-y)-.5*(m**2)


for i in range(train_steps):
    with tf.GradientTape() as tape:
        tape.watch(X)
        yhat = X * W + b
        loss = [squared_loss(y[i], yhat[i]) for i in range(len(X))]
        dW, db = tape.gradient(loss, [W, b])
        W.assign_sub(dW * learning_rate)
        b.assign_sub(db * learning_rate)
    if i % 100 == 0:
        print(("Loss at step {:03d}: {:.3f}".format(i, loss)))
    ###TO DO ## Calculate gradients
plt.plot(X, y, 'bo',label='org')
plt.plot(X, y * W.numpy() + b.numpy(), 'r',
         label="huber regression")
plt.legend()
plt.show
print(" ")
