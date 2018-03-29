'''
University of Toronto
ECE521 Assignment 1
Part 1
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

	train_distance_matrix = euclidean_distance(trainData, trainData)
	valid_distance_matrix = euclidean_distance(validData, trainData)
	test_distance_matrix = euclidean_distance(testData, trainData)








if __name__ == "__main__":
	main()


























