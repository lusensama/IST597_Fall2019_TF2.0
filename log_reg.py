""" 
author:-aam35
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

import time
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
import tensorflow as tf
import tensorflow_datasets as tfds





def load_mnist(path, kind='train', val_size = 0.0):
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
    return images, labels


# Define paramaters for the model
learning_rate = 0.01
batch_size = 64
n_epochs = 10
n_train = None
n_test = None
n_classes = 10
n_channel = 1
pixel_depth = 255
image_size = 28
seed = None

# Step 1: Read in data
fmnist_folder = './data/'
train_data, train_label= load_mnist('./data/')
test_data, test_label = load_mnist('./data/', kind='t10k')
#Create train,validation,test split
#train, val, test = utils.read_fmnist(fmnist_folder, flatten=True)

# train_data = tfds.load('fashion_mnist', tfds.Split.TRAIN)
# eval_data = tfds.load('fashion_mnist', tfds.Split.VALIDATION)
# test_data = tfds.load('fashion_mnist', tfds.Split.TEST)
# Step 2: Create datasets and iterator
# create training Dataset and batch it
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).batch(32)
# val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))

# create testing Dataset and batch it


train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# create one iterator and initialize it with different datasets

# iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
#                                            train_dataset.output_shapes)
# img, label = iterator.get_next()
#
# train_init = iterator.make_initializer(train_data)	# initializer for train_data
# test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
# weight_initializer = tf.random_normal(mean=0.0, stddev=0.01)
# step 2: create the weight variable with proper initialization
# w = tf.get_variable(name="Weight", dtype=tf.float32, shape=(train_data.shape[1],train_data.shape[0]), initializer=weight_initializer)

# initialize biases as zero
# step 1: create the initializer for biases
# b =tf.constant(0., shape=train_label.shape, dtype=tf.float32)
#############################
########## TO DO ############
#############################

conv1_weights = tf.Variable(
    tf.random.normal([5, 5, n_channel, 32],  # 5x5 filter, depth 32.
                        stddev=0.1,
                        seed=seed, dtype=tf.float32))
conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
conv2_weights = tf.Variable(tf.random.normal(
    [5, 5, 32, 64], stddev=0.1,
    seed=seed, dtype=tf.float32))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
fc1_weights = tf.Variable(  # fully connected, depth 512.
    tf.random.normal([image_size // 4 * image_size // 4 * 64, 512],
                        stddev=0.1,
                        seed=seed,
                        dtype=tf.float32))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
fc2_weights = tf.Variable(tf.random.normal([512, n_classes],
                                              stddev=0.1,
                                              seed=seed,
                                              dtype=tf.float32))
fc2_biases = tf.Variable(tf.constant(
    0.1, shape=[n_classes], dtype=tf.float32))




# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=seed)
    return tf.matmul(hidden, fc2_weights) + fc2_biases


#############################
########## TO DO ############
#############################


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function

def loss_func(logit, y):
    return tf.losses.softmax_cross_entropy_with_logits(labels=y, logits=logit)


def train(inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_func(inputs, targets)
    dc1w, dc1b ,dc2w, dc2b, df1w, df1b, df2w, df2b= tape.gradient(loss_value, [conv1_weights, conv1_biases,
                                                                                conv2_weights, conv2_biases,
                                                                                fc1_weights, fc1_biases,
                                                                                fc2_weights, fc2_biases,])
    conv1_weights.assign_sub(learning_rate * dc1w)
    conv1_biases.assign_sub(learning_rate * dc1b)
    conv2_weights.assign_sub(learning_rate * dc2w)
    conv2_biases.assign_sub(learning_rate * dc2b)
    fc1_weights.assign_sub(learning_rate * df1w)
    fc1_biases.assign_sub(learning_rate * df1b)
    fc2_weights.assign_sub(learning_rate * df2w)
    fc2_biases.assign_sub(learning_rate * df2b)

for i in range(n_epochs):
    data_iter = train_dataset.__iter__()
    for j in range(train_data.shape[0]//batch_size):
        batch_train, batch_label = data_iter.next()
        logits = model(batch_train, train=True)
        train(logits, batch_label)




# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.optimizer.Adam(learning_rate)
optimizer.apply_gradients(zip(gradient, logits.trainable_variables))
#############################
########## TO DO ############
#############################


# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

#Step 8: train the model for n_epochs times
for i in range(n_epochs):
	total_loss = 0
	n_batches = 0
	#Optimize the loss function

	print("Train and Validation accuracy")
	################################
	###TO DO#####
	############
	
#Step 9: Get the Final test accuracy

#Step 10: Helper function to plot images in 3*3 grid
#You can change the function based on your input pipeline

def plot_images(images, y, yhat=None):
    assert len(images) == len(y) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if yhat is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], yhat[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#Get image from test set 
images = test_data[0:9]

# Get the true classes for those images.
y = test_label[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, y=y)


#Second plot weights 

def plot_weights(w=None):
    # Get the values for the weights from the TensorFlow variable.
    #TO DO ####
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = None
    #TO DO## obtains these value from W
    w_max = None

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

