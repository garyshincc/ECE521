'''
Author: Shin Won Young
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

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


'''
LINEAR REGRESSION NORMAL EQUATION
For zero weight decay, compare the train, 
validation and test classification accuracy
of the linear regression using "normal equation"
'''
trainData, trainTarget, validData, validTarget, testData, testTarget = get_data()
n_samples = trainData.shape[0]

X = tf.placeholder(tf.float32, shape=(None, 785))
Y = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.placeholder(tf.float32, shape=(None, 785))
bias_factor_train = np.ones((trainData.shape[0], 1))
trainData = tf.Session().run(tf.concat([bias_factor_train, trainData], 1))

bias_factor_valid = np.ones((validData.shape[0], 1))
validData = tf.Session().run(tf.concat([bias_factor_valid, validData], 1))

bias_factor_test = np.ones((testData.shape[0], 1))
testData = tf.Session().run(tf.concat([bias_factor_test, testData], 1))

W_opt = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(X), X)), tf.transpose(X)), Y)

pred = tf.matmul(X, W_opt)
classification = tf.cast(tf.greater(pred, 0.5), tf.float64)
correct = tf.reduce_sum(tf.cast(tf.equal(classification, tf.cast(Y, tf.float64)), tf.float64))
accuracy = tf.cast(correct, tf.float64) / tf.cast(tf.shape(classification)[0], tf.float64)


start = time.time()
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	cost = tf.reduce_sum(tf.norm(pred - Y)) / (2 * trainData.shape[0])
	W_train_opt = sess.run(W_opt, feed_dict={X: trainData, Y: trainTarget})

	train_loss = sess.run(cost, feed_dict={X: trainData, Y: trainTarget, W_opt: W_train_opt})
	train_acc = sess.run(accuracy, feed_dict={X: trainData, Y: trainTarget, W_opt: W_train_opt})

	cost = tf.reduce_sum(tf.norm(pred - Y)) / (2 * validData.shape[0])
	valid_loss = sess.run(cost, feed_dict={X: validData, Y: validTarget, W_opt: W_train_opt})
	valid_acc = sess.run(accuracy, feed_dict={X: validData, Y: validTarget, W_opt: W_train_opt})

	cost = tf.reduce_sum(tf.norm(pred - Y)) / (2 * testData.shape[0])
	test_loss = sess.run(cost, feed_dict={X: testData, Y: testTarget, W_opt: W_train_opt})
	test_acc = sess.run(accuracy, feed_dict={X: testData, Y: testTarget, W_opt: W_train_opt})

	print ("Train loss: " + str(train_loss))
	print ("Train acc: " + str(train_acc))
	print ("Valid loss: " + str(valid_loss))
	print ("Valid acc: " + str(valid_acc))
	print ("Test loss: " + str(test_loss))
	print ("Test acc: " + str(test_acc))


end = time.time()
print ("Time: " + str(end-start))


'''
LOGISTIC REGRESSION adam MODEL
'''
# params
weight_decay = 0.0
training_epochs = int(5000/7)
display_step = 50
batch_size = 500
learning_rate = 0.001

trainData, trainTarget, validData, validTarget, testData, testTarget = get_data()
num_samples = trainData.shape[0]

X = tf.placeholder(tf.float32, shape=(None, 784))
Y = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable(tf.ones((784, 1)), name="weight")
b = tf.Variable(tf.ones(1), name="bias")

z = tf.add(tf.matmul(X, W), b)

prediction = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))
classification = tf.cast(tf.greater(prediction, 0.5), tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(classification, tf.cast(Y, tf.float64)), tf.float64))

#lD = tf.reduce_sum(-1 * p * tf.log(q) - (1 - p) * tf.log(1 - q))
# define cost
lD = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=z))
lW = weight_decay * tf.reduce_sum(tf.pow(W, 2)) / 2

logistic_regression_train_ce = list()

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

		if epoch % display_step == 0:
			c = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
			print("Epoch: " + str(epoch) + ", cost: " + str(c))

		lrtm = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
		logistic_regression_train_ce.append(lrtm)





'''
LINEAR REGRESSION ADAM
'''


trainData, trainTarget, validData, validTarget, testData, testTarget = get_data()
n_samples = trainData.shape[0]

X = tf.placeholder(tf.float32, shape=(None, 784))
Y = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable(tf.ones((784, 1)), name="weight")
b = tf.Variable(tf.ones(1), name="bias")

pred = tf.add(tf.matmul(X, W), b)
classification = tf.cast(tf.greater(pred, 0.5), tf.float64)
correct = tf.reduce_sum(tf.cast(tf.equal(classification, tf.cast(Y, tf.float64)), tf.float64))
accuracy = tf.cast(correct, tf.float64) / tf.cast(tf.shape(classification)[0], tf.float64)

# define cost
lD = tf.losses.mean_squared_error(labels=Y, predictions=pred) / 2
cost = lD
linear_regression_train_mse = list()

with tf.Session() as sess:
	# optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss=cost)
	num_batches = int(trainData.shape[0] / batch_size)

	init = tf.global_variables_initializer()
	sess.run(init)

	for epoch in range(training_epochs):
		c = None
		for i in range(num_batches):
			trainBatchi = trainData[i*batch_size: (i+1) * batch_size]
			trainTargeti = trainTarget[i*batch_size: (i+1) * batch_size]
			sess.run(optimizer, feed_dict={X: trainBatchi, Y: trainTargeti})
			if epoch % display_step == 0:
				c = sess.run(cost, feed_dict={X: trainBatchi, Y:trainTargeti})

		if epoch % display_step == 0:	
			print("Epoch: " + str(epoch) + ", cost: " + str(c))

		# for train set
		train_mse = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
		linear_regression_train_mse.append(train_mse)


# Plot loss vs number of training steps
steps = np.linspace(0, training_epochs, num=training_epochs)
fig = plt.figure()
axes = plt.gca()
fig.patch.set_facecolor('white')
plt.plot(steps, logistic_regression_train_ce, "r-")
plt.plot(steps, linear_regression_train_mse, "c-")

plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
red_patch = mpatches.Patch(color='red', label='Logistic Regression Cross Entropy')
cyan_patch = mpatches.Patch(color='cyan', label='Linear Regression Mean Squared Error')
plt.legend(handles=[red_patch, cyan_patch])



plt.show()





