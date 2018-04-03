import tensorflow as tf

'''
A vectorized Tensorflow Python function that takes 
the hidden activations from the previous layer then 
return the weighted sum of the inputs(i.e. the z) for 
the current hidden layer. You will also initialize the
weight matrix and the biases in the same function.
You should use Xavier initialization for the weight
matrix. Your function should be able to compute the
weighted sum for all the data points in your mini-batch
at once using matrix multiplication. It should not
contain loops over the training examples in the
mini-batch. The function should accept two arguments,
the input tensor and the number of the hidden units.
Include the snippets of the Python code. [3 pt.]
'''

# vectorized layer
def get_layer(input_tensor, num_hidden_units):
	weight_initializer = tf.contrib.layers.xavier_initializer()
	num_input_units = input_tensor.shape[0]
	w = tf.get_variable(name="weights", shape=(num_input_units, num_hidden_units), dtype=tf.float32, initializer=weight_initializer)
	b = tf.Variable(tf.zeros(shape=(num_hidden_units)), name="bias")
	z = tf.add(tf.matmul(input_tensor), b)
	return z

