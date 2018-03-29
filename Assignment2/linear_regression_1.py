'''
Author: Shin Won Young
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Parameters
training_epochs = 20000
batch_size = 500
display_step = 1000
weight_decay_param = 0.0


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
n_samples = trainData.shape[0]


X = tf.placeholder(tf.float32, shape=(None, 784))
Y = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable(tf.ones((784, 1)), name="weight")
b = tf.Variable(tf.ones(1), name="bias")

pred = tf.add(tf.matmul(X, W), b)

# defind the cost
lD = tf.reduce_mean((Y-pred)**2) / 2
lW = weight_decay_param * tf.reduce_sum(tf.pow(W, 2)) / 2

# tune the learning rate
learning_rates = [0.005, 0.001, 0.0001]
losses = list()
training_loss = list()

for lr in learning_rates:
	with tf.Session() as sess:
		# initialize tf sessions
		init = tf.global_variables_initializer()
		sess.run(init)
		num_batches = int(n_samples / batch_size)
		training_loss_on_this_specific_learning_rate = list()

		# set cost with weight decay params
		cost = lD + lW

		# optimizer
		optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss=cost)

		for epoch in range(training_epochs):
			for i in range(num_batches):
				trainBatchi = trainData[i*batch_size: (i+1) * batch_size]
				trainTargeti = trainTarget[i*batch_size: (i+1) * batch_size]
				sess.run(optimizer, feed_dict={X: trainBatchi, Y: trainTargeti})

			c = sess.run(cost, feed_dict={X: trainData, Y:trainTarget})
			training_loss_on_this_specific_learning_rate.append(c)
			if epoch % display_step == 0:
				print ("epoch: " + str(epoch) + ", loss: " + str(c))

		# redefine and calculate loss
		cost = lD
		train_cost = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})

		print("Train cost: " + str(train_cost))
		losses.append(train_cost)
		training_loss.append(training_loss_on_this_specific_learning_rate)

print (losses)


# Plot loss vs number of training steps
steps = np.linspace(0, training_epochs, num=training_epochs)
fig = plt.figure()
axes = plt.gca()
axes.set_ylim([0,10])
fig.patch.set_facecolor('white')
plt.plot(steps, training_loss[0], "r-")
plt.plot(steps, training_loss[1], "c-")
plt.plot(steps, training_loss[2], "b-")

plt.xlabel("Epochs")
plt.ylabel("Training loss")
red_patch = mpatches.Patch(color='red', label='Learning Rate: 0.005')
cyan_patch = mpatches.Patch(color='cyan', label='Learning Rate: 0.001')
blue_patch = mpatches.Patch(color='blue', label='Learning Rate: 0.0001')
plt.legend(handles=[red_patch, cyan_patch, blue_patch])    

plt.show()

