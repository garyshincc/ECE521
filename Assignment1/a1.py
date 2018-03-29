'''
University of Toronto
ECE521 Assignment 1
Due date: Feb 2nd, 2018
Name, student number:
	Won Young Shin, 1002077637
	Zinan Lin, 1001370287

'''
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

sess = tf.Session()

def euclidean_distance(X, Z):
	"""
	:param x: a n-by-d dimensional array
	:param z: an m-by-d dimensional array
	:return: a n-by-m dimensional array which is the pairwise
	        euclidean distance of input x and z.
	"""
	global sess
	x_sq = tf.reshape(tf.reduce_sum(X*X, 1), [-1, 1])
	z_sq = tf.reshape(tf.reduce_sum(Z*Z, 1), [1, -1])
	D = x_sq - 2*tf.matmul(X, tf.transpose(Z)) + z_sq
	matrix = sess.run(tf.sqrt(D))
	return matrix

def nearest_neighbors(distance_matrix, x_star, index, k):
	global sess
	inv_e_dist = 1.0 / distance_matrix[index]
	values, indices = tf.nn.top_k(inv_e_dist, k)
	resp = np.zeros(np.shape(inv_e_dist))
	run_ind = sess.run(indices)
	np.put(resp, run_ind, 1.0 / k)
	return resp

def nearest_neighbors_part3(distance_matrix, x_star, index, k):
	# x = training data = list
	# k = number of neighbors
	global sess
	inv_e_dist = 1 / distance_matrix[index]
	values, indices = tf.nn.top_k(inv_e_dist, k)
	return (sess.run(indices))



def mean_squared_error(prediction, target):
	return ((prediction - target) ** 2).mean() / 2

def anton_error(prediction, target):
	num_error = 0
	for i in range(len(prediction)):
		if prediction[i] != target[i]:
			num_error += 1
	# accuracy
	return (1.0 - (num_error / float(len(target))))

def get_data():
	Data = np.linspace(1.0, 10.0, num = 100) [:, np.newaxis]
	Target = np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
	return Data, Target

def partition_data(Data, Target):
	np.random.seed(521)
	randIdx = np.arange(100)
	np.random.shuffle(randIdx)
	trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
	validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
	testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]
	return trainData, trainTarget, validData, validTarget, testData, testTarget

def test_k_values(data_points, trainData, trainTarget, data_target):
	k_list = [1, 3, 5, 50]
	prediction_list = list()
	data_mse = list()
	e_dist = euclidean_distance(data_points, trainData)
	for k in k_list:
		full_rank_resp = list()
		for index in range(len(data_points)):
			resp = nearest_neighbors(e_dist, data_points[index], index, k)
			full_rank_resp.append(resp)

		prediction = np.matmul(full_rank_resp, trainTarget)
		mse = mean_squared_error(prediction, data_target)
		prediction_list.append(prediction)
		data_mse.append(mse)

	return data_mse, prediction_list

def predict_part3(data_points, trainData, trainTarget, data_target):
	"""
	:param data_points: the matrix of data points that will be predicted
	:param trainData: the training data that will be used for the model
	:param trainTarget: the training target that the training data will use
	:param data_target: the prediction values of data_points to calculate error
	:return: list of error percentage by k, list of prediction by k.
	"""
	k_list = [10]
	prediction_list = list()
	error_list = list()
	e_dist = euclidean_distance(data_points, trainData)
	for k in k_list:
		print ("starting k: " + str(k))
		full_rank_resp = list()
		for index in range(len(data_points)):
			indices = nearest_neighbors_part3(e_dist, data_points[index], index, k)
			values = list()
			# collect values into a matrix
			for i in indices:
				values.append(trainTarget[i])
			y, idx, count = tf.unique_with_counts(values)
			y, idx, count = sess.run(y), sess.run(idx), list(sess.run(count))
			# collect into a full rank prediction / responsibility matrix
			prediction = y[count.index(max(count))]
			full_rank_resp.append(prediction)

			if prediction != data_target[index]:
				print ("What we thought was: " + str(prediction))
				print ("The person is supposed to be: " + str(data_target[index]))
				plt.imshow(np.reshape(data_points[index], (32,32)), cmap="Greys_r")
				plt.show()
				for i in indices:
					neighbor = trainTarget[i]
					print ("Error detected!!!!!")
					print ("neighbor value: " + str(neighbor))

					plt.imshow(np.reshape(trainData[i],(32,32)), cmap='Greys_r')
					plt.show()

		prediction_list.append(full_rank_resp)
		# use a classification error instead of a regression error
		error = anton_error(data_target, full_rank_resp)
		error_list.append(error)
	return error_list, prediction_list


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
	return trainData, validData, testData, trainTarget, validTarget, testTarget


def main():
	sess = tf.Session()

	np.random.seed(521)
	Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
	Target = np.sin( Data ) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(100 , 1)
	randIdx = np.arange(100)
	np.random.shuffle(randIdx)
	trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
	validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
	testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]
	###########
	# plt.scatter(trainData, trainTarget, c="g", alpha=0.5)
	# plt.scatter(validData, validTarget, c="r", alpha=0.5)
	# plt.scatter(testData, testTarget, c="b", alpha=0.5)
	# plt.xlabel("X")
	# plt.ylabel("Y")
	# plt.legend(loc=2)

	###########

	'''
	Part 2. 2 Prediction
	For the dataset data1D, compute the above k-NN prediction function with 
	k = {1, 3, 5, 50}. For each of these values of k, compute and report the 
	training MSE loss, validation MSE loss and test MSE loss. Choose the best
	k using the validation error.
	'''

	'''
	#Train MSE
	print ("Train MSE")
	train_mse, prediction_list = test_k_values(trainData, trainData, trainTarget, trainTarget)
	print (train_mse)
	#print (prediction_list)

	#Validation MSE
	print ("Validation MSE")
	valid_mse, prediction_list = test_k_values(validData, trainData, trainTarget, validTarget)
	print (valid_mse)
	#print (prediction_list)

	#Test MSE
	print ("Test MSE")
	test_mse, prediction_list = test_k_values(testData, trainData, trainTarget, testTarget)
	print (test_mse)
	#print (prediction_list)

	#random points MSE
	print ("X MSE")
	X = np.linspace(0.0, 11.0, num = 1000)[:, np.newaxis]
	x_mse, prediction_list = test_k_values(X, trainData, trainTarget, trainTarget)
	print (x_mse)
	print (prediction_list)

	quit()
	'''
	'''
	# for prediction in prediction_list:
	# 	plt.figure()
	# 	plt.scatter(Data, Target, c = "b", alpha = 0.5)
	# 	plt.plot(X, np.transpose(prediction), c = "g")
	# plt.show()


	Part 3. 1 Predicting class label
	Modify the prediction function for regression in section 1 and use 
	majority voting over k nearest neighbors to predict the final class. 
	You may find tf.unique with counts helpful for this task. Include 
	the relevant snippet of code for this task.
	'''
	# task 0
	trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy", "target.npy", 0)
	error_list, prediction_list = predict_part3(testData, trainData, trainTarget, testTarget)
	
	for i in range(len(error_list)):
		prediction = prediction_list[i]
		error = error_list[i]
		y, idx, count = tf.unique_with_counts(tf.transpose(prediction))
		print ("error: " + str(error))
		print (sess.run(y))
		print (sess.run(idx))
		print (sess.run(count))





if __name__ == "__main__":
	main()


























