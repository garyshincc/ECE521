import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

assert (tf.__version__ == "1.1.0")

batch_size = 500
learning_rate = 0.0001
hidden_units = 1000
num_epoch = 101
image_dim = 28 * 28
num_targets = 10
weight_decay = 3e-4

with np.load("notMNIST.npz") as data:
    Data, Target = data["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx] / 255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]

trainData = trainData.reshape(trainData.shape[0], image_dim)
validData = validData.reshape(validData.shape[0], image_dim)
testData = testData.reshape(testData.shape[0], image_dim)

trainTarget = tf.Session().run(tf.one_hot(trainTarget, num_targets))
validTarget = tf.Session().run(tf.one_hot(validTarget, num_targets))
testTarget = tf.Session().run(tf.one_hot(testTarget, num_targets))


def layer(input_tensor, output_size):
    input_size = input_tensor.shape[1]

    # initialize bias
    bias = tf.get_variable('bias', shape=(1, output_size), dtype=tf.float64)

    # initialize weight using Xavier initializer
    weight = tf.get_variable('weight',
                             shape=[input_size, output_size],
                             regularizer=tf.contrib.layers.l2_regularizer(scale=3e-3),
                             initializer=tf.contrib.layers.xavier_initializer(),
                             dtype=tf.float64)

    # output as the weighted sum of input data plus bias
    output = tf.add(tf.matmul(input_tensor, weight), bias)

    return output


data = trainData
target = trainTarget

# define variables
X = tf.placeholder(tf.float64, shape=[None, image_dim])
y = tf.placeholder(tf.float64, shape=[None, num_targets])

# forward propagation
with tf.variable_scope('hidden'):
    h = tf.nn.relu(layer(X, hidden_units))
with tf.variable_scope('output'):
    y_ = layer(h, num_targets)

prediction = tf.argmax(y_, axis=1)

# back propagation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
update = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# run optimizer
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

display_step = 5
display_train_loss = []
display_valid_loss = []
display_test_loss = []

display_train_acc = []
display_valid_acc = []
display_test_acc = []

for epoch in range(num_epoch):
    for i in range(int(len(data) / batch_size)):
        data_batch = data[i * batch_size: (i + 1) * batch_size]
        target_batch = target[i * batch_size: (i + 1) * batch_size]
        sess.run(update, feed_dict={X: data_batch, y: target_batch})

    train_loss = sess.run(cost, feed_dict={X: trainData, y: trainTarget})
    validation_loss = sess.run(cost, feed_dict={X: validData, y: validTarget})
    test_loss = sess.run(cost, feed_dict={X: testData, y: testTarget})

    accuracy = np.mean(np.argmax(target, axis=1) == sess.run(prediction, feed_dict={X: data, y: target}))
    validation_acc = np.mean(np.argmax(validTarget, axis=1) == sess.run(prediction, feed_dict={X: validData, y: validTarget}))
    test_acc = np.mean(np.argmax(testTarget, axis=1) == sess.run(prediction, feed_dict={X: testData, y: testTarget}))

    display_train_acc.append(accuracy)
    display_train_loss.append(train_loss)

    display_valid_acc.append(validation_acc)
    display_valid_loss.append(validation_loss)

    display_test_acc.append(test_acc)
    display_test_loss.append(test_loss)


    if epoch % display_step == 0:
        print("Epoch = %d, Train accuracy = %.2f%%, cost = %.2f" % (
        epoch + 1, 100. * accuracy, 100. * train_loss))
        print("Epoch = %d, Valid accuracy = %.2f%%, cost = %.2f" % (
        epoch + 1, 100. * validation_acc, 100. * validation_loss))
        print("Epoch = %d, Test accuracy = %.2f%%, cost = %.2f" % (
        epoch + 1, 100. * test_acc, 100. * test_loss))


# Plot
steps = np.linspace(0, num_epoch, num=num_epoch)

fig = plt.figure()
axes = plt.gca()
fig.patch.set_facecolor('white')
plt.plot(steps, display_train_loss, "b-")
plt.plot(steps, display_valid_loss, "c-")
plt.plot(steps, display_test_loss, "r-")

plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
blue_patch = mpatches.Patch(color='blue', label='Training Set')
cyan_patch = mpatches.Patch(color='cyan', label='Validation Set')
red_patch = mpatches.Patch(color='red', label='Test Set')
plt.legend(handles=[blue_patch, cyan_patch, red_patch])
plt.savefig("cross_entrpy_loss.png")

plt.figure()
axes = plt.gca()
fig.patch.set_facecolor('white')
plt.plot(steps, display_train_acc, "b-")
plt.plot(steps, display_valid_acc, "c-")
plt.plot(steps, display_test_acc, "r-")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
blue_patch = mpatches.Patch(color='blue', label='Training Set')
cyan_patch = mpatches.Patch(color='cyan', label='Validation Set')
red_patch = mpatches.Patch(color='red', label='Test Set')
plt.legend(handles=[blue_patch, cyan_patch, red_patch])
plt.savefig("accuracy.png")

plt.show()