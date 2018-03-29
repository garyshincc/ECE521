'''
Author: Shin Won Young
'''
import tensorflow as tf
import numpy as np
import time

# Parameters
training_epochs = 20000
learning_rate = 0.005
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

# define cost
lD = tf.losses.mean_squared_error(labels=Y, predictions=pred) / 2
lW = weight_decay_param * tf.reduce_sum(tf.pow(W, 2)) / 2

# tuning batch
batch_sizes = [500, 1500, 3500]
times = list()
losses = list()
for bs in batch_sizes:
	start = time.time()

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)

		# set cost with weight decay
		cost = lD + lW

		# optimizer
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=cost)
		num_batches = int(n_samples / bs)

		for epoch in range(training_epochs):
			for i in range(num_batches):
				trainBatchi = trainData[i*bs: (i+1) * bs]
				trainTargeti = trainTarget[i*bs: (i+1) * bs]
				sess.run(optimizer, feed_dict={X: trainBatchi, Y: trainTargeti})

			if epoch % display_step == 0:
				c = sess.run(cost, feed_dict={X: trainData, Y:trainTarget})
				print("Epoch: " + str(epoch) + ", cost: " + str(c))


		# redefine cost and calculate
		cost = lD
		train_cost = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
		
		print("Train cost: " + str(train_cost))
		losses.append(train_cost)

	end = time.time()
	times.append(end-start)



print (losses)
print (times)

