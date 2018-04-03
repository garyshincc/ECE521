import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

'''
Early stopping is the simplest procedure to avoid overfitting.
Determine and highlight the early stopping point on the
classification error plot from question 1.1.2, and report
the training, validation and test classification error at
the early stopping point. Are the early stopping points
the same on the two plots? Why or why not? Which plot should
be used for early stopping, and why?
'''
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
	W = tf.get_variable(name="weights", shape=(num_input, num_output), dtype=tf.float32, initializer=weight_initializer)
	b = tf.Variable(tf.zeros(shape=(num_output)), name="bias")
	z = tf.add(tf.matmul(input_tensor, W), b)
	return z


trainData, trainTarget, validData, validTarget, testData, testTarget = get_data()

num_samples = trainData.shape[0]

''' hyper parameters ''' 
learning_rate = 0.01 # 0.01
num_hidden_units = 1000 # 1000
image_dim = 28 * 28 # 784
bias_init = 0
num_classifications = 10
weight_decay = 3e-4
training_steps = 100
batch_size = 1500


num_batches = int(num_samples / batch_size)
num_epochs = int(training_steps / num_batches)


''' start defining variables '''
X = tf.placeholder(tf.float32, shape=(None, image_dim), name="input")
Y = tf.placeholder(tf.float32, shape=(None, num_classifications), name="output")

''' build the model '''
with tf.variable_scope("weights1"):
	z1 = get_layer(X, image_dim, num_hidden_units)
	#print ("z1 shape: {}".format(z1.shape))
	h1 = tf.nn.relu(z1)

with tf.variable_scope("weights2"):
	z_out = get_layer(h1, num_hidden_units, num_classifications)
	#print ("z_out shape: {}".format(z_out.shape))

''' output '''
softmax = tf.nn.softmax(z_out)
prediction = tf.cast(tf.argmax(softmax, 1), tf.float64)

correct = tf.reduce_sum(tf.cast(tf.equal(prediction, tf.cast(tf.argmax(tf.cast(Y, tf.float64), 1), tf.float64)), tf.float64))
accuracy = tf.cast(correct, tf.float64) / tf.cast(tf.shape(prediction)[0], tf.float64)
classification_error = 1.0 - accuracy
''' cost definition '''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(Y, tf.int32), logits=z_out))

''' checkpoint '''
weight_saver = tf.train.Saver()

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

print("learning rate: {}".format(learning_rate))
''' optimizer '''
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

''' tensorflow session '''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

epoch = 0
for step in range(training_steps):
	batch_num = (step % num_batches) * batch_size
	dataBatchi = trainData[batch_num: batch_num + batch_size]
	targetBatchi = trainTarget[batch_num: batch_num + batch_size]


	''' run training '''
	sess.run(train, feed_dict = {X: dataBatchi, Y: targetBatchi})

	#if batch_num == 0:
	train_acc, train_loss, train_error = sess.run([accuracy, cost, classification_error], feed_dict = {X: trainData, Y: trainTarget})
	valid_acc, valid_loss, valid_error = sess.run([accuracy, cost, classification_error], feed_dict = {X: validData, Y: validTarget})
	test_acc, test_loss, test_error = sess.run([accuracy, cost, classification_error], feed_dict = {X: testData, Y: testTarget})

	train_accs.append(train_acc)
	valid_accs.append(valid_acc)
	test_accs.append(test_acc)

	train_errors.append(train_error)
	valid_errors.append(valid_error)
	test_errors.append(test_error)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)
	test_losses.append(test_loss)

	print("Epoch: {}".format(epoch))
	print("Training loss: {0:.2f}, accuracy: {0:.2f}".format(train_loss, train_acc))
	epoch += 1

train_acc, train_loss = sess.run([accuracy, cost], feed_dict = {X: trainData, Y: trainTarget})
print ("Learning rate: {}, train loss: {0:.2f}, acc: {0:.2f}".format(learning_rate, train_loss, train_acc))



''' plot '''


steps = np.linspace(0, training_steps, num=training_steps)
fig = plt.figure()
fig.patch.set_facecolor('white')
plt.plot(steps, train_accs, "r-")
plt.plot(steps, valid_accs, "c-")
plt.plot(steps, test_accs, "b-")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
red_patch = mpatches.Patch(color='red', label='Training Set')
cyan_patch = mpatches.Patch(color='cyan', label='Validation Set')
blue_patch = mpatches.Patch(color='blue', label='Test Set')
plt.legend(handles=[red_patch, cyan_patch, blue_patch], loc=0)
plt.savefig("1_3_acc.png")

fig = plt.figure()
fig.patch.set_facecolor('white')
plt.plot(steps, train_losses, "r-")
plt.plot(steps, valid_losses, "c-")
plt.plot(steps, test_losses, "b-")

plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
red_patch = mpatches.Patch(color='red', label='Training Set')
cyan_patch = mpatches.Patch(color='cyan', label='Validation Set')
blue_patch = mpatches.Patch(color='blue', label='Test Set')
plt.legend(handles=[red_patch, cyan_patch, blue_patch], loc=0)
plt.savefig("1_3_loss.png")

fig = plt.figure()
fig.patch.set_facecolor('white')
plt.plot(steps, train_errors, "r-")
plt.plot(steps, valid_errors, "c-")
plt.plot(steps, test_errors, "b-")

plt.xlabel("Epochs")
plt.ylabel("Classification Error")
red_patch = mpatches.Patch(color='red', label='Training Set')
cyan_patch = mpatches.Patch(color='cyan', label='Validation Set')
blue_patch = mpatches.Patch(color='blue', label='Test Set')
plt.legend(handles=[red_patch, cyan_patch, blue_patch], loc=0)
plt.savefig("1_3_error.png")

plt.show()































