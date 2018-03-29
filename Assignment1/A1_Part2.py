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


def mean_squared_error(prediction, target):
	return ((prediction - target) ** 2).mean() / 2

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

	'''
	Part 2. 2 Prediction
	For the dataset data1D, compute the above k-NN prediction function with 
	k = {1, 3, 5, 50}. For each of these values of k, compute and report the 
	training MSE loss, validation MSE loss and test MSE loss. Choose the best
	k using the validation error.
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


if __name__ == "__main__":
	main()


























