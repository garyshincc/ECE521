'''
Author: Shin Won Young
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

# params
weight_decay = 0.01
training_epochs = int(5000/7)
display_step = 50
batch_size = 500
learning_rate = 0.001

def get_data():
	with np.load("notMNIST.npz") as data :
		Data, Target = data ["images"], data["labels"]
		posClass = 2
		negClass = 9
		dataIndx = (Target==posClass) + (Target==negClass)
		Data = Data[dataIndx]/255.
		Target = Target[dataIndx].reshape(-1, 1)
		Target[Target==posClass] = 1
		Target[Target==negClass] = 0
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data, Target = Data[randIndx], Target[randIndx]
		trainData, trainTarget = Data[:3500], Target[:3500]
		validData, validTarget = Data[3500:3600], Target[3500:3600]
		testData, testTarget = Data[3600:], Target[3600:]

		trainData = trainData.reshape(trainData.shape[0], 784)
		validData = validData.reshape(validData.shape[0], 784)
		testData = testData.reshape(testData.shape[0], 784)
		return trainData, trainTarget, validData, validTarget, testData, testTarget


trainData, trainTarget, validData, validTarget, testData, testTarget = get_data()
num_samples = trainData.shape[0]

X = tf.placeholder(tf.float32, shape=(None, 784))
Y = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable(tf.zeros((784, 1)), name="weight")
b = tf.Variable(tf.ones(1), name="bias")

z = tf.add(tf.matmul(X, W), b)

prediction = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))
classification = tf.cast(tf.greater(prediction, 0.5), tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(classification, tf.cast(Y, tf.float64)), tf.float64))


#lD = tf.reduce_sum(-1 * p * tf.log(q) - (1 - p) * tf.log(1 - q))
# define cost values
lD = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=z))
lW = weight_decay * tf.reduce_sum(tf.pow(W, 2)) / 2

training_loss_sgd = list()
validation_loss_sgd = list()

training_loss_adam = list()
validation_loss_adam = list()

adam = 1
for i in range(2):
	with tf.Session() as sess:
		# set cost with weight decay params
		cost = lD + lW

		# optimizer
		if adam == 1:
			optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss=cost)
		else:
			optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=cost)
		num_batches = int(trainData.shape[0] / batch_size)

		init = tf.global_variables_initializer()
		sess.run(init)	

		for epoch in range(training_epochs):
			for i in range(num_batches):
				trainBatchi = trainData[i*batch_size: (i+1) * batch_size]
				trainTargeti = trainTarget[i*batch_size: (i+1) * batch_size]
				sess.run(optimizer, feed_dict={X: trainBatchi, Y: trainTargeti})

			if epoch % display_step == 0:
				c = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
				print("Epoch: " + str(epoch) + ", cost: " + str(c))

			# loss calculation
			train_loss = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
			validation_loss = sess.run(cost, feed_dict={X: validData, Y: validTarget})

			if adam == 1:
				training_loss_adam.append(train_loss)
				validation_loss_adam.append(validation_loss)
			else:
				training_loss_sgd.append(train_loss)
				validation_loss_sgd.append(validation_loss)

		# redefine cost and calculate
		cost = lD

		train_loss = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
		train_acc = sess.run(accuracy, feed_dict = {X: trainData, Y: trainTarget})
		test_acc = sess.run(accuracy, feed_dict = {X: testData, Y: testTarget})

		print("Train cost: " + str(train_loss))
		print ("Train accuracy: " + str(train_acc))
		print ("Test accuracy: " + str(test_acc))
	adam = 0



# Plot loss vs number of training steps
steps = np.linspace(0, training_epochs, num=training_epochs)
fig = plt.figure()
axes = plt.gca()
fig.patch.set_facecolor('white')
plt.plot(steps, training_loss_adam, "r-")
plt.plot(steps, validation_loss_adam, "c-")
plt.plot(steps, training_loss_sgd, "b-")
plt.plot(steps, validation_loss_sgd, "g-")

plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
red_patch = mpatches.Patch(color='red', label='Adam Train')
cyan_patch = mpatches.Patch(color='cyan', label='Adam Validation')
blue_patch = mpatches.Patch(color='blue', label='SGD Train')
green_patch = mpatches.Patch(color='green', label='SGD Validation')
plt.legend(handles=[red_patch, cyan_patch, blue_patch, green_patch])



plt.show()


