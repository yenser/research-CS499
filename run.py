#!/usr/bin/env python3

# imports
import tensorflow as tf
import numpy as np

filenameMeta='./models/4in_model.ckpt.meta'
filename='./models/4in_model.ckpt'

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2 # how many outputs
batch_size = 3392 # adjust this for batchsize

input_size = 5
x = tf.placeholder('float') # input data
y = tf.placeholder('float')

current_epoch = tf.Variable(1)

hidden_1_layer = {'f_fum':n_nodes_hl1,'weights': tf.Variable(tf.random_normal([input_size, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes])),}

def neural_network_model(data):

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


saver = tf.train.import_meta_graph(filenameMeta)


def use_neural_network(input_data, comp_name):
    prediction = neural_network_model(x)
            
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,filename)

        features = np.array(list(input_data))
        print("features:", features)

        # up: [1,0] , argmax: 0
        # down: [0,1] , argmax: 1
        result = sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1))
        print(prediction.eval(feed_dict={x:[features]}))
        if result[0] == 0:
            print(comp_name,' will go down: ',input_data)
        elif result[0] == 1:
            print(comp_name,' will go up: ',input_data)



use_neural_network([1,1,1,1,1], 'ORACLE')