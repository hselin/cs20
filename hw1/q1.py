"""
Simple exercises to get used to TensorFlow API
You should thoroughly test your code.
TensorFlow's official documentation should be your best friend here
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

sess = tf.InteractiveSession()
###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.greater(x, y), lambda: x + y, lambda: x - y)
print(sess.run(out))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

x = tf.random_uniform([], minval=-1, maxval=1)  # Empty array as shape creates a scalar.
y = tf.random_uniform([], minval=-1, maxval=1)
out = tf.case( [(tf.less(x, y), lambda: x + y), (tf.greater(x, y), lambda: x - y)], default=lambda:tf.constant(0, dtype=tf.float32))
print(sess.run(out))


###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

x = tf.constant([[0, -2, -1], [0, 1, 2]])
y = tf.zeros_like(x)
out = tf.equal(x, y)
print(sess.run(out))

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

x = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
 				 30.97266006,  26.67541885,  38.08450317,  20.74983215,
 				 34.94445419,  34.45999146,  29.06485367,  36.01657104,
 				 27.88236427,  20.56035233,  30.20379066,  29.51215172,
 				 33.71149445,  28.59134293,  36.05556488,  28.66994858])

#x = tf.constant([[100, 1], [31, 51]])

y = tf.where(x > 30)
out = tf.gather(x, y) #can use gather_nd instead for multi-dimension tensor
print(sess.run(out))

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

diagonal = tf.range(1, 7, delta=1)
out = tf.diag(diagonal)
print(sess.run(out))

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

x = tf.random_normal([10, 10])
out = tf.matrix_determinant(x)
print(sess.run(out))

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

x = [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9]
out, idx = tf.unique(x)
print(sess.run(out))

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.cond() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

#x = tf.random_normal([300])
#y = tf.random_normal([300])

x = tf.constant([1, 2, 3])
y = tf.constant([2, 2, 6])

diff = x - y
#z = tf.reduce_sum(tf.square(diff)) / tf.shape(x)[0]
#zz = tf.reduce_sum(tf.abs(diff))

out = tf.cond(tf.less(tf.reduce_sum(diff), 0), lambda: tf.reduce_sum(tf.square(diff)) / tf.shape(x)[0], lambda: tf.cast(tf.reduce_sum(tf.abs(diff)), tf.float64))

print(sess.run([out]))