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
display_step = 50 # 100
batch_size = 500 # 1500
image_dim = 28 * 28 # 28 * 28
learning_rate = 0.001

def get_data():
	with np.load("notMNIST.npz") as data:
		Data, Target = data["images"], data["labels"]
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data = Data[randIndx]/255.
		Target = Target[randIndx]
		trainData, trainTarget = Data[:15000], Target[:15000]
		validData, validTarget = Data[15000:16000], Target[15000:16000]
		testData, testTarget = Data[16000:], Target[16000:]

		print (trainTarget.shape)

		trainData = trainData.reshape(trainData.shape[0], image_dim)
		validData = validData.reshape(validData.shape[0], image_dim)
		testData = testData.reshape(testData.shape[0], image_dim)
		trainTarget = tf.Session().run(tf.one_hot(trainTarget, 10))
		validTarget = tf.Session().run(tf.one_hot(validTarget, 10))
		testTarget = tf.Session().run(tf.one_hot(testTarget, 10))
		print (trainTarget.shape)
		return trainData, trainTarget, validData, validTarget, testData, testTarget


trainData, trainTarget, validData, validTarget, testData, testTarget = get_data()
num_samples = trainData.shape[0]

X = tf.placeholder(tf.float32, shape=(None, image_dim))
Y = tf.placeholder(tf.float32, shape=(None, 10))
W = tf.Variable(tf.random_normal((image_dim, 10), stddev=0.01), name="weight1")
b = tf.Variable(tf.zeros(10), name="bias")

logit = tf.add(tf.matmul(X, W), b)
softmax = tf.nn.softmax(logit)
prediction = tf.argmax(softmax, 1)

correct = tf.reduce_sum(tf.cast(tf.equal(tf.cast(prediction, tf.float64), tf.cast(tf.argmax(Y, 1), tf.float64)), tf.float64))
accuracy = tf.cast(correct, tf.float64) / tf.cast(tf.shape(prediction)[0], tf.float64)


#lD = tf.reduce_sum(-1 * p * tf.log(q) - (1 - p) * tf.log(1 - q))
lD = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logit))
lW = weight_decay * tf.reduce_sum(tf.matmul(W, W, transpose_a=True)) / 2


training_loss_for_plot = list()
validation_loss_for_plot = list()

training_accuracy_for_plot = list()
validation_accuracy_for_plot = list()



print ("learning rate: " + str(learning_rate))


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

	test_acc = sess.run(accuracy, feed_dict = {X: testData, Y: testTarget})
	print ("Test accuracy: " + str(test_acc))				


# Plot loss vs number of training steps
steps = np.linspace(0, training_epochs, num=training_epochs)
plt.plot(steps, training_loss_for_plot, "r-")
plt.plot(steps, validation_loss_for_plot, "c-")

plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
red_patch = mpatches.Patch(color='red', label='Training Set')
cyan_patch = mpatches.Patch(color='cyan', label='Validation Set')
plt.legend(handles=[red_patch, cyan_patch])
plt.savefig("a.png")


plt.clf()
plt.plot(steps, training_accuracy_for_plot, "r-")
plt.plot(steps, validation_accuracy_for_plot, "c-")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
red_patch = mpatches.Patch(color='red', label='Training Set')
cyan_patch = mpatches.Patch(color='cyan', label='Validation Set')
plt.legend(handles=[red_patch, cyan_patch])
plt.savefig("b.png")



