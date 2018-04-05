import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# data
def get_data():
	with np.load("notMNIST.npz") as data:
		Data, Target = data ["images"], data["labels"]
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data = Data[randIndx]/255.
		Target = Target[randIndx]
		trainData, trainTarget = Data[:15000], Target[:15000]
		validData, validTarget = Data[15000:16000], Target[15000:16000]
		testData, testTarget = Data[16000:], Target[16000:]

		# number of classifications
		num_targets = 10

		# reshape data
		trainData = trainData.reshape(trainData.shape[0], 784)
		validData = validData.reshape(validData.shape[0], 784)
		testData = testData.reshape(testData.shape[0], 784)

		# reshape targets
		trainTarget = tf.Session().run(tf.one_hot(trainTarget, num_targets))
		validTarget = tf.Session().run(tf.one_hot(validTarget, num_targets))
		testTarget = tf.Session().run(tf.one_hot(testTarget, num_targets))

		return trainData, trainTarget, validData, validTarget, testData, testTarget

# vectorized layer
def get_layer(input_tensor, num_input, num_output):
	weight_initializer = tf.contrib.layers.xavier_initializer()
	W = tf.get_variable(
			name="weights",
			shape=(num_input, num_output),
			dtype=tf.float32,
			initializer=weight_initializer,
			#regularizer=tf.contrib.layers.l2_regularizer(0.1)
	)
	b = tf.Variable(tf.zeros(shape=(num_output)), name="bias")
	z = tf.add(tf.matmul(input_tensor, W), b)
	return z, W


def gary_round(num, num2):
	return int(num * num2) / float(num2)

trainData, trainTarget, validData, validTarget, testData, testTarget = get_data()

num_samples = trainData.shape[0]

''' hyper parameters ''' 
learning_rate = 0.005 # 0.01
image_dim = 28 * 28 # 784
num_classifications = 10
weight_decay = 3e-4
training_steps = 500
batch_size = 500
keep_prob_value = 0.5

num_hidden_unit = 1000
keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

num_batches = int(num_samples / batch_size)
num_epochs = int(training_steps / num_batches)
print("num_epochs: {}".format(num_epochs))


''' start defining variables '''
X = tf.placeholder(tf.float32, shape=(None, image_dim), name="input")
Y = tf.placeholder(tf.float32, shape=(None, num_classifications), name="output")


''' build the model '''
with tf.variable_scope("weights1" + str(num_hidden_unit)):
	
	z1, W1 = get_layer(X, image_dim, num_hidden_unit)
	#print ("z1 shape: {}".format(z1.shape))
	h1 = tf.nn.relu(z1)
	h1_drop = tf.nn.dropout(h1, keep_prob)

with tf.variable_scope("weights2" + str(num_hidden_unit)):
	z_out, W2 = get_layer(h1_drop, num_hidden_unit, num_classifications)
	#print ("z_out shape: {}".format(z_out.shape))

''' output '''
softmax = tf.nn.softmax(z_out)
prediction = tf.cast(tf.argmax(softmax, 1), tf.float64)

correct = tf.reduce_sum(tf.cast(tf.equal(prediction, tf.cast(tf.argmax(tf.cast(Y, tf.float64), 1), tf.float64)), tf.float64))
accuracy = tf.cast(correct, tf.float64) / tf.cast(tf.shape(prediction)[0], tf.float64)
classification_error = 1.0 - accuracy

''' cost definition '''
lD = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(Y, tf.int32), logits=z_out))
lW = (tf.reduce_sum(W1 * W1) + tf.reduce_sum(W2 * W2)) * weight_decay / 2
cost = lD + lW
report_cost = lD

''' checkpoint '''
saver = tf.train.Saver()

''' plot array definition '''
train_accs = list()
valid_accs = list()
test_accs = list()

train_errors = list()
valid_errors = list()
test_errors = list()

train_losses = list()
valid_losses = list()
test_losses = list()

''' optimizer '''
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

''' tensorflow session '''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

epoch = 0
checkpoint1 = int(training_steps / 4.0)
num_neurons = 100
weight_dim = (28, 28)

for step in range(training_steps):
	batch_num = (step % num_batches) * batch_size
	dataBatchi = trainData[batch_num: batch_num + batch_size]
	targetBatchi = trainTarget[batch_num: batch_num + batch_size]

	''' run training '''
	keep_prob_value = 1.0
	sess.run(train, feed_dict = {X: dataBatchi, Y: targetBatchi, keep_prob: keep_prob_value})

	if batch_num == 0:
		keep_prob_value = 1.0
		train_acc, train_loss, train_error = sess.run([accuracy, cost, classification_error], feed_dict = {X: trainData, Y: trainTarget, keep_prob: keep_prob_value})
		valid_acc, valid_loss, valid_error = sess.run([accuracy, cost, classification_error], feed_dict = {X: validData, Y: validTarget, keep_prob: keep_prob_value})
		test_acc, test_loss, test_error = sess.run([accuracy, cost, classification_error], feed_dict = {X: testData, Y: testTarget, keep_prob: keep_prob_value})

		train_accs.append(train_acc)
		valid_accs.append(valid_acc)
		test_accs.append(test_acc)

		train_errors.append(train_error)
		valid_errors.append(valid_error)
		test_errors.append(test_error)

		train_losses.append(train_loss)
		valid_losses.append(valid_loss)
		test_losses.append(test_loss)

		print("Epoch: {}, Train loss: {}, acc: {}".format(epoch, gary_round(train_loss,1000), gary_round(train_acc,1000)))
		epoch += 1

	keep_prob_value = 1.0
	if step == checkpoint1:
		print ("25 % REACHED")
		fig, axs = plt.subplots(10,10, figsize=(28, 28), facecolor='w', edgecolor='k')
		fig.subplots_adjust(hspace = .5, wspace=.001)
		axs = axs.ravel()
		for layer_num in range(num_neurons):
			print("rendering neuron: {}".format(layer_num))
			w_vis = sess.run(tf.reshape(W1[:, layer_num], weight_dim))
			axs[layer_num].imshow(w_vis, cmap="gray")

		plt.tight_layout()
		plt.savefig("3_2_25_" + ("drop" if keep_prob_value == 0.5 else "nodrop") + ".png")

	if step == training_steps - 1:
		print ("100 % REACHED")
		fig, axs = plt.subplots(10,10, figsize=(28, 28), facecolor='w', edgecolor='k')
		fig.subplots_adjust(hspace = .5, wspace=.001)
		axs = axs.ravel()
		for layer_num in range(num_neurons):
			print("rendering neuron: {}".format(layer_num))
			w_vis = sess.run(tf.reshape(W1[:, layer_num], weight_dim))
			axs[layer_num].imshow(w_vis, cmap="gray")

		plt.tight_layout()
		plt.savefig("3_2_100_" + ("drop" if keep_prob_value == 0.5 else "nodrop") + ".png")


# redefine accuracy for the no dropout case, 
# keep_prob_value = 1.0

# print ("Num hidden units: {}".format(num_hidden_unit))
# valid_acc, valid_loss, valid_error = sess.run([accuracy, report_cost, classification_error], feed_dict = {X: validData, Y: validTarget, keep_prob: keep_prob_value})
# print ("Valid loss: {}, acc: {}, error: {}".format(valid_loss, valid_acc, valid_error))

# test_acc, test_loss, test_error = sess.run([accuracy, report_cost, classification_error], feed_dict = {X: testData, Y: testTarget, keep_prob: keep_prob_value})
# print ("Test loss: {}, acc: {}, error: {}".format(test_loss, test_acc, test_error))


