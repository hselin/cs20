""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 100
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.shuffle(10000) # if you want to shuffle your data
test_data = test_data.batch(batch_size)


# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3: create weights and bias
layer1_size = 200
layer2_size = 100
layer3_size = 60
layer4_size = 30

w1 = tf.get_variable(name='weights_1', shape=[784, layer1_size], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
b1 = tf.get_variable(name='bias_1', shape=[1, layer1_size], initializer=tf.zeros_initializer())

w2 = tf.get_variable(name='weights_2', shape=[layer1_size, layer2_size], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
b2 = tf.get_variable(name='bias_2', shape=[1, layer2_size], initializer=tf.zeros_initializer())

w3 = tf.get_variable(name='weights_3', shape=[layer2_size, layer3_size], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
b3 = tf.get_variable(name='bias_3', shape=[1, layer3_size], initializer=tf.zeros_initializer())

w4 = tf.get_variable(name='weights_4', shape=[layer3_size, layer4_size], initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
b4 = tf.get_variable(name='bias_4', shape=[1, layer4_size], initializer=tf.zeros_initializer())

w5 = tf.get_variable(name='weights_5', shape=[layer4_size, 10], initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
b5 = tf.get_variable(name='bias_5', shape=[1, 10], initializer=tf.zeros_initializer())

# Step 4: build model
'''
out1 = tf.nn.sigmoid(tf.matmul(img, w1) + b1)
out2 = tf.nn.sigmoid(tf.matmul(out1, w2) + b2)
out3 = tf.nn.sigmoid(tf.matmul(out2, w3) + b3)
out4 = tf.nn.sigmoid(tf.matmul(out3, w4) + b4)
logits = tf.matmul(out4, w5) + b5
'''

'''
out1 = tf.matmul(img, w1) + b1
out2 = tf.matmul(out1, w2) + b2
out3 = tf.matmul(out2, w3) + b3
out4 = tf.matmul(out3, w4) + b4
logits = tf.matmul(out4, w5) + b5
'''

out1 = tf.nn.relu(tf.matmul(img, w1) + b1)
out2 = tf.nn.relu(tf.matmul(out1, w2) + b2)
out3 = tf.nn.relu(tf.matmul(out2, w3) + b3)
out4 = tf.nn.relu(tf.matmul(out3, w4) + b4)
logits = tf.matmul(out4, w5) + b5


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
print(label)
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
loss = tf.reduce_mean(entropy)


# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
lr_start = 0.001
lr = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

#writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
   
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        lr = lr_start
        try:
            while True:
                #lr = lr / 2
                #lr = max(lr, 0.001)
                #print("lr: " + str(lr))
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1

        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))
#writer.close()