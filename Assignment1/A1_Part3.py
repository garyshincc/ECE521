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

def nearest_neighbors_part3(distance_matrix, x_star, index, k):
	# x = training data = list
	# k = number of neighbors
	global sess
	inv_e_dist = 1 / distance_matrix[index]
	values, indices = tf.nn.top_k(inv_e_dist, k)
	return (sess.run(indices))


def anton_error(prediction, target):
	num_error = 0
	for i in range(len(prediction)):
		if prediction[i] != target[i]:
			num_error += 1
	# accuracy
	return (1.0 - (num_error / float(len(target))))


def predict_part3(data_points, trainData, trainTarget, data_target):
	"""
	:param data_points: the matrix of data points that will be predicted
	:param trainData: the training data that will be used for the model
	:param trainTarget: the training target that the training data will use
	:param data_target: the prediction values of data_points to calculate error
	:return: list of error percentage by k, list of prediction by k.
	"""
	k_list = [1, 5, 10, 25, 50, 100, 200]
	prediction_list = list()
	error_list = list()
	e_dist = euclidean_distance(data_points, trainData)
	for k in k_list:
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

	'''
	Part 3. 1 Predicting class label
	Modify the prediction function for regression in section 1 and use 
	majority voting over k nearest neighbors to predict the final class. 
	You may find tf.unique with counts helpful for this task. Include 
	the relevant snippet of code for this task.
	'''
	# task 0
	trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy", "target.npy", 0)
	
	error_list, prediction_list = predict_part3(trainData, trainData, trainTarget, trainTarget)
	error_list, prediction_list = predict_part3(validData, trainData, trainTarget, validTarget)
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


























