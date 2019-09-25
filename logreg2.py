import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
# import tensorflow.contrib.eager as tfe
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow_datasets as tfds
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10)


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

train_data = tfds.load(name='fashion_mnist', split=tfds.Split.TRAIN.subsplit(tfds.percent[:80])).shuffle(buffer_size=1000).batch(batch_size=batch_size)


val_data = tfds.load(name='fashion_mnist', split=tfds.Split.TRAIN.subsplit(tfds.percent[80:])).shuffle(buffer_size=1000).batch(batch_size=batch_size)
test_data = tfds.load(name='fashion_mnist', split=tfds.Split.TEST)


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

# fmnist_folder = './data/'
# train_dataset, train_labelset = load_mnist('./data/')
# test_dataset, test_labelset = load_mnist('./data/', kind='t10k')
# np.random.seed(seed)
# train_index = np.random.choice(len(train_dataset), round(len(train_dataset) * (1.0 - val_size)), replace=False)
# val_index = np.array(list(set(range(len(train_dataset))) - set(train_index)))
# train_data = train_dataset[train_index]
# train_label = train_labelset[train_index]
# val_data = train_dataset[val_index]
# val_label = train_labelset[val_index]

# print(train_index)
# print(val_data)
# print(val_label)
# t_data = tf.data.Dataset.from_tensor_slices((train_data, train_label)).batch(batch_size)
# rfc.fit(train_data, train_label)
# print(rfc.score(val_data, val_label))
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
W = tf.Variable(tf.random.normal(shape=(784, 10)), dtype=tf.float32)
b = tf.Variable(tf.random.normal(shape=(10,)), dtype=tf.float32)

# batch_index = np.random.choice(len(train_label), size=batch_size)
# batch_train_X = train_data[batch_index]
# batch_train_y = np.matrix(train_label[batch_index]).T


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# iterator = train_dataset.__iter__()
# train_img, train_lbl = iterator.next()
# itert = test_dataset.__iter__()
# test_img, test_lbl = iter.next()

def model(img):
    image_batch = tf.cast(tf.reshape(img, (batch_size, 784)), tf.float32)
    image_batch /= 255.0
    model_output = tf.matmul(image_batch, W) + b
    return model_output

def softmax_model(image_batch):
    image_batch = tf.reshape(image_batch, (batch_size, 784))

    model_output = tf.nn.softmax(tf.matmul(tf.cast(image_batch, tf.float32), W) + b)
    return model_output
    # return tf.map_fn(model, image_batch)



def cross_entropy(model_output, label_batch):
    label_batch = tf.one_hot(label_batch, depth=10)
    loss = tf.compat.v2.nn.softmax_cross_entropy_with_logits(
        label_batch,
        model_output,
        axis=-1,
        name=None
    )
    # loss = tf.reduce_mean(
    #     -tf.reduce_sum(label_batch * tf.math.log(model_output),
    #     keepdims=True))
    return loss

# @tf.implicit_value_and_gradients
def cal_gradient(image_batch, label_batch):
    return cross_entropy(model(image_batch), label_batch)


# iterator = tf.compat.v1.data.make_one_shot_iterator(train_data)
# iterator.next()
# batch_train_X, batch_train_y = iterator.next()
# logits = softmax_model(batch_train_X)


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
# def loss_func(logit, y):
#     return tf.reduce_mean(tf.compat.v2.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))
# def squared_loss(y, y_predicted):
#     return tf.reduce_mean(tf.square(y_predicted-y))



# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

# train_img, train_label = train_data.get_next()
# train_init = train_data.make_initializer(train_data)	# initializer for train_data
# val_init = val_data.make_initializer(test_data)	# initializer for train_data
# Step 7: calculate accuracy with test set
# preds = tf.nn.softmax(logits)
# loss = cross_entropy(preds, batch_train_y)
# correct_preds = tf.equal(tf.argmax(preds, -1), tf.argmax(batch_train_y, -1))
# accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
#
# print(accuracy.numpy())

for epoch in range(10):
    step = 0
    for data in train_data:
        image_batch, label_batch = data['image'], data['label']
        with tf.GradientTape() as tape:
            loss = cal_gradient(image_batch, label_batch)
            grads = tape.gradient(loss, [W, b])
            optimizer.apply_gradients(zip(grads[0], W), zip(grads[1], b))
        if (step % 100 == 0):
            print("step: {}  loss: {}".format(step, loss.numpy()))
        model_test_output = softmax_model(train_data.test.images)
        model_test_label = train_data.test.labels

        correct_prediction = tf.equal(tf.argmax(model_test_output, 1), tf.argmax(model_test_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("test accuracy = {}".format(accuracy.numpy()))
    # # Generate random batch index
    # batch_index = np.random.choice(len(train_label), size=batch_size)
    # batch_train_X = train_data[batch_index]
    # batch_train_y = np.matrix(train_label[batch_index]).T
    # with tf.GradientTape() as tape:
    #     logits = softmax_model(batch_train_X)
    #     loss = cal_gradient(logits, batch_train_y)
    #     grads = tape.gradient(loss, [W, b])
    #     optimizer.apply_gradients(zip(grads, [W, b]))
    #
    # # dW, db = tape.gradient(loss, [W, b])
    # # w.assign_sub(dW * learning_rate)
    # # b.assign_sub(db * learning_rate)
    # #print(w.numpy())
    # batch_v_index = np.random.choice(len(val_label), size=batch_size)
    # batch_val_X = val_data[batch_v_index]
    # batch_val_y = np.matrix(val_label[batch_v_index]).T
    # logits = softmax_model(batch_val_X)
    # preds = tf.nn.softmax(logits)
    # print("val label")
    # print(tf.argmax(batch_val_y, -1))
    # print("pred")
    # print(tf.argmax(preds, -1))
    # correct_preds = tf.equal(tf.argmax(preds, axis=-1), tf.argmax(batch_val_y, axis=-1))
    # accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    # print(accuracy.numpy())