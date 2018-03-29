'''
Author: Shin Won Young
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# params
weight_decay = 0.01 # 0.01
training_epochs = 1500 # 5000
display_step = 100 # 100
batch_size = 300 # 300
image_dim = 32 * 32 # 32 * 32
learning_rate = 0.005 # 0.005
num_targets = 6 # 6 

def data_segmentation(data_path, target_path, task):

	data = np.load(data_path) / 255.0
	data = np.reshape(data, [-1, 32*32])

	target = np.load(target_path)
	np.random.seed(45689)
	rnd_idx = np.arange(np.shape(data)[0])
	np.random.shuffle(rnd_idx)
	trBatch = int(0.8*len(rnd_idx))
	validBatch = int(0.1*len(rnd_idx))
	trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
			data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
			data[rnd_idx[trBatch + validBatch+1:-1],:]
	trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
			target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
			target[rnd_idx[trBatch + validBatch + 1:-1], task]

	trainData = trainData.reshape(trainData.shape[0], image_dim)
	validData = validData.reshape(validData.shape[0], image_dim)
	testData = testData.reshape(testData.shape[0], image_dim)

	trainTarget = tf.Session().run(tf.one_hot(trainTarget, num_targets))
	validTarget = tf.Session().run(tf.one_hot(validTarget, num_targets))
	testTarget = tf.Session().run(tf.one_hot(testTarget, num_targets))


	return trainData, validData, testData, trainTarget, validTarget, testTarget

#trainData, trainTarget, validData, validTarget, testData, testTarget = get_data()
trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy", "target.npy", 0)


print (trainData.shape)
print (trainTarget.shape)
num_samples = trainData.shape[0]

X = tf.placeholder(tf.float32, shape=(None, image_dim))
Y = tf.placeholder(tf.float32, shape=(None, num_targets))
W = tf.Variable(tf.ones((image_dim, num_targets)), name="weight1")
b = tf.Variable(tf.ones(1), name="bias")

z = tf.add(tf.matmul(X, W), b)

softmax = tf.nn.softmax(z)
prediction = tf.argmax(softmax, 1)

correct = tf.reduce_sum(tf.cast(tf.equal(tf.cast(prediction, tf.float64), tf.cast(tf.argmax(Y, 1), tf.float64)), tf.float64))
accuracy = tf.cast(correct, tf.float64) / tf.cast(tf.shape(prediction)[0], tf.float64)

training_loss_for_plot = list()
validation_loss_for_plot = list()

training_accuracy_for_plot = list()
validation_accuracy_for_plot = list()

#weight_decays = [0.1, 0.01, 0.001]
#learning_rates = [0.005, 0.001, 0.0001]

# cost definition
lD = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=z))
lW = weight_decay * tf.reduce_sum(tf.matmul(W, W, transpose_a=True)) / 2

with tf.Session() as sess:
	# set cost with weight decay params
	cost = lD + lW

	# optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss=cost)
	num_batches = int(trainData.shape[0] / batch_size)

	init = tf.global_variables_initializer()
	sess.run(init)	

	for epoch in range(training_epochs):
		for i in range(num_batches):
			trainBatchi = trainData[i*batch_size: (i+1) * batch_size]
			trainTargeti = trainTarget[i*batch_size: (i+1) * batch_size]
			sess.run(optimizer, feed_dict={X: trainBatchi, Y: trainTargeti})

		# debugging
		# p = sess.run(z, feed_dict={X: validData, Y: validTarget})
		# print (p)

		# loss calculation
		train_loss = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
		validation_loss = sess.run(cost, feed_dict={X: validData, Y: validTarget})

		training_loss_for_plot.append(train_loss)
		validation_loss_for_plot.append(validation_loss)

		# accuracy calculation
		train_acc = sess.run(accuracy, feed_dict={X: trainData, Y: trainTarget})
		validation_acc = sess.run(accuracy, feed_dict={X: validData, Y: validTarget})

		training_accuracy_for_plot.append(train_acc)
		validation_accuracy_for_plot.append(validation_acc)

		# print information
		if epoch % display_step == 0:
			print ("epoch: " + str(epoch) + ", loss: " + str(train_loss) + ", acc: " + str(train_acc))

	train_acc = sess.run(accuracy, feed_dict = {X: trainData, Y: trainTarget})
	print ("Train accuracy: " + str(train_acc))
	validation_acc = sess.run(accuracy, feed_dict = {X: validData, Y: validTarget})
	print ("Validation accuracy: " + str(validation_acc))
	test_acc = sess.run(accuracy, feed_dict = {X: testData, Y: testTarget})
	print ("Test accuracy: " + str(test_acc))

# Plot loss vs number of training steps
steps = np.linspace(0, training_epochs, num=training_epochs)
fig = plt.figure()
axes = plt.gca()
fig.patch.set_facecolor('white')
plt.plot(steps, training_loss_for_plot, "r-")
plt.plot(steps, validation_loss_for_plot, "c-")

plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
red_patch = mpatches.Patch(color='red', label='Training Set')
cyan_patch = mpatches.Patch(color='cyan', label='Validation Set')
plt.legend(handles=[red_patch, cyan_patch])
plt.savefig("cross_entrpy_loss.png")

plt.figure()
axes = plt.gca()
fig.patch.set_facecolor('white')
plt.plot(steps, training_accuracy_for_plot, "r-")
plt.plot(steps, validation_accuracy_for_plot, "c-")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
red_patch = mpatches.Patch(color='red', label='Training Set')
cyan_patch = mpatches.Patch(color='cyan', label='Validation Set')
plt.legend(handles=[red_patch, cyan_patch])
plt.savefig("accuracy.png")

plt.show()


