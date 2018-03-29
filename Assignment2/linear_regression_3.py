'''
Author: Shin Won Young
'''
import tensorflow as tf
import numpy as np

# Parameters
training_epochs = 20000 # 20000
learning_rate = 0.005 # 0.005
display_step = 1000 # 1000
batch_size = 500


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
classification = tf.cast(tf.greater(pred, 0.5), tf.float64)
correct = tf.reduce_sum(tf.cast(tf.equal(classification, tf.cast(Y, tf.float64)), tf.float64))
accuracy = tf.cast(correct, tf.float64) / tf.cast(tf.shape(classification)[0], tf.float64)

# define cost
lD = tf.losses.mean_squared_error(labels=Y, predictions=pred) / 2
#lW = weight_decay_param * tf.reduce_sum(tf.pow(W, 2)) / 2

# tuning weight decay lambda parameters
weight_decays = [0.0, 0.001, 0.1, 1.0]

train_accs = list()
valid_accs = list()
test_accs = list()

for wd in weight_decays:
	print ("Weight decay: " + str(wd))
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		
		# define cost, redefine weight decay param
		lW = wd * tf.reduce_sum(tf.pow(W, 2)) / 2
		cost = lD + lW

		# optimizer
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=cost)
		num_batches = int(trainData.shape[0] / batch_size)

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
		train_acc = sess.run(accuracy, feed_dict={X: trainData, Y: trainTarget})
		train_accs.append(train_acc)
		print ("train accuracy: " + str(train_acc))

		# for validation set
		validation_acc = sess.run(accuracy, feed_dict={X: validData, Y: validTarget})
		valid_accs.append(validation_acc)
		print ("validation accuracy: " + str(validation_acc))

		# for test set
		test_acc = sess.run(accuracy, feed_dict={X: testData, Y: testTarget})
		test_accs.append(test_acc)
		print ("test accuracy: " + str(test_acc))

print ("Train Accuracy: " + str(train_accs))
print ("Validation Accuracy: " + str(valid_accs))
print ("Test Accuracy: " + str(test_accs))

