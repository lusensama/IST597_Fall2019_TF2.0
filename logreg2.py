import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define paramaters for the model
learning_rate = 0.001
batch_size = 32
n_epochs = 2
n_train = None
n_test = None
n_classes = 10
n_channel = 1
pixel_depth = 255
image_size = 28
seed = None
val_size = 0.2

# Step 1: Read in data
fmnist_folder = 'None'


# Create dataset load function [Refer fashion mnist github page for util function]
# Create train,validation,test split
# train, val, test = utils.read_fmnist(fmnist_folder, flatten=True)
def load_mnist(path, kind='train', val_size=0.0):
    import os
    import gzip
    import numpy as np
    from sklearn.model_selection import train_test_split

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    # if kind=='train':
    #     return train_test_split(images, labels, test_size=val_size, random_state=13)
    # else:
    # img = images.reshape([-1, 28, 28, 1])
    return images, labels


# Step 2: Create datasets and iterator
# create training Dataset and batch it
fmnist_folder = './data/'
train_dataset, train_labelset = load_mnist('./data/')
test_dataset, test_labelset = load_mnist('./data/', kind='t10k')
np.random.seed(seed)
train_index = np.random.choice(len(train_dataset), round(len(train_dataset) * (1.0 - val_size)), replace=False)
val_index = np.array(list(set(range(len(train_dataset))) - set(train_index)))
train_data = train_dataset[train_index]
train_label = train_labelset[train_index]
val_data = train_dataset[val_index]
val_label = train_labelset[val_index]

# train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).batch(batch_size)
# # val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label)).batch(batch_size)


# create one iterator and initialize it with different datasets
# iterator = tf.data.Iterator.from_structure(train_data.output_types,
#                                            train_data.output_shapes)
# img, label = iterator.get_next()

# train_init = iterator.make_initializer(train_data)	# initializer for train_data
# test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.Variable(tf.random.normal([batch_size, 28 * 28]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([batch_size, 1], dtype=tf.float32))

batch_index = np.random.choice(len(train_data), size=batch_size)
batch_train_X = train_data[batch_index]
batch_train_y = np.matrix(train_label[batch_index]).T


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# iterator = train_dataset.__iter__()
# train_img, train_lbl = iterator.next()
# itert = test_dataset.__iter__()
# test_img, test_lbl = iter.next()

def logi(data, W, b):
    return data * W + b


logits = logi(batch_train_X, w, b)


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
def loss_func(logit, y):
    return tf.reduce_mean(tf.compat.v2.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))


loss = loss_func(logits, batch_train_y)

# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, -1), tf.argmax(val_label, -1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

print(accuracy.numpy())

for epoch in range(10):
    # Generate random batch index
    batch_index = np.random.choice(len(train_data), size=batch_size)
    batch_train_X = train_data[batch_index]
    batch_train_y = np.matrix(train_label[batch_index]).T
    with tf.GradientTape() as tape:
        logits = logi(batch_train_X, w, b)
        loss = loss_func(logits, batch_train_y)
        grads = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(grads, [w, b]))

    # dW, db = tape.gradient(loss, [w, b])
    #
    # # if tf.equal(loss, prevloss):
    # #     learning_rate /= 2
    # w.assign_sub(dW * learning_rate)
    # b.assign_sub(db * learning_rate)

    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, -1), tf.argmax(val_label, -1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    print(accuracy.numpy())