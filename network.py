#!/usr/bin/env python3

# imports
import tensorflow as tf
import numpy as np
from libs import data

'''
input > weight > hidden layer 1 (activation function) > 
weights > hidden l2 (activation function) > weights > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer ... SGD, AdaGrad)


Backpropagation

feed forward + backprop = epoch
'''


# code

# 10 classes, 0-9

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 5 # how many outputs (5 companies that were in)
batch_size = 100 # adjust this for batchsize

# height x width
x  = tf.placeholder('float', [None, 5]) # input data
y = tf.placeholder('float')

def neural_network_model(data):

	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([5, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}

	# (input_data * weights) + biases

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output


def train_neural_network(x, hm_epochs=10):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

	#								learning rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())


		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range():
				# epoch_x, epoch_y = 				train data here
				print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)

		# for epoch in range(hm_epochs):
		# 	epoch_loss = 0
		# 	for _ in range(int(mnist.train.num_examples/batch_size)):
		# 		epoch_x, epoch_y = mnist.train.next_batch(batch_size)
		# 		_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
		# 		epoch_loss += c
		# 	print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)

		# correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		# accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		# print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))




def create_test_sets(arr, test_size=100):
	features = []

	for _ in arr:
		if _ >= 0:
			features += [[_, [1,0]]]
		else:
			features += [[_, [0,1]]]


	testing_size = int(test_size*len(features))

	features = np.array(features)
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y

#run code

AAPL = data.get_file_data('AAPL')
MSFT = data.get_file_data('MSFT')
GOOGL = data.get_file_data('GOOGL')
AMZN = data.get_file_data('AMZN')
ADBE = data.get_file_data('ADBE')

AAPL, MSFT, GOOGL, AMZN, ADBE = data.size_test_results(AAPL, MSFT, GOOGL, AMZN, ADBE)

print('AAPL: ', len(AAPL), ' data points')
print('MSFT: ', len(MSFT), ' data points')
print('GOOGL: ', len(GOOGL), ' data points')
print('AMZN: ', len(AMZN), ' data points')
print('ADBE: ', len(ADBE), ' data points')

print(AAPL[len(AAPL)-1])
AAPL = data.flip_array(AAPL)
print(AAPL[0])

val1, val2, val3, val4 = data.create_test_sets(AAPL)
print(val[0], test_size=1000)



# train_neural_network(x)