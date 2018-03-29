'''
Author: Shin Won Young
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# params
weight_decay = 0.01
training_epochs = int(5000/7)
display_step = 100
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
num_samples = trainData.shape[0]

X = tf.placeholder(tf.float32, shape=(None, 784))
Y = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable(tf.zeros((784, 1)), name="weight")
b = tf.Variable(tf.ones(1), name="bias")

z = tf.add(tf.matmul(X, W), b)

prediction = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))
classification = tf.cast(tf.greater(prediction, 0.5), tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(classification, tf.cast(Y, tf.float64)), tf.float64))
#accuracy = tf.cast(correct, tf.float64) / tf.cast(tf.shape(classification)[0], tf.float64)

#lD = tf.reduce_sum(-1 * p * tf.log(q) - (1 - p) * tf.log(1 - q))
# define cost values
lD = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=z))
lW = weight_decay * tf.reduce_sum(tf.matmul(W, W, transpose_a = True)) / 2

# tune the learning rate!
learning_rates = [0.005, 0.001, 0.0001]
#learning_rates = [0.005] # best found <--
losses = list()
train_accs = list()
test_accs = list()

training_loss_for_plot = list()
validation_loss_for_plot = list()

training_accuracy_for_plot = list()
validation_accuracy_for_plot = list()

for learning_rate in learning_rates:

	print ("learning rate: " + str(learning_rate))

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		
		# set cost with weight decay params
		cost = lD + lW

		# optimizer
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=cost)
		num_batches = int(trainData.shape[0] / batch_size)

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

			training_loss_for_plot.append(train_loss)
			validation_loss_for_plot.append(validation_loss)

			train_acc = sess.run(accuracy, feed_dict={X: trainData, Y: trainTarget})
			validation_acc = sess.run(accuracy, feed_dict={X: validData, Y: validTarget})

			training_accuracy_for_plot.append(train_acc)
			validation_accuracy_for_plot.append(validation_acc)

		# redefine cost and calculate
		cost = lD

		train_loss = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
		train_acc = sess.run(accuracy, feed_dict = {X: trainData, Y: trainTarget})
		test_acc = sess.run(accuracy, feed_dict = {X: testData, Y: testTarget})

		losses.append(train_loss)
		train_accs.append(train_acc)
		test_accs.append(test_acc)

		print ("Train cost: " + str(train_loss))
		print ("Train accuracy: " + str(train_acc))
		print ("Test accuracy: " + str(test_acc))


print (losses)
print (train_accs)
print (test_accs)


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



plt.show()


