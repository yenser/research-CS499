#!/usr/bin/env python3

# imports
import tensorflow as tf
import numpy as np
from create_test_data import get_data_and_create_run_set
import libs.bcolors as c




company = 'ORCL'
run_x, run_y, batch_size = get_data_and_create_run_set('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'ADBE', 'ORCL')


work_path = './models/'+company+'/'
filename=work_path+company+'_model.ckpt'
filenameMeta=work_path+company+'_model.ckpt.meta'

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

correct = 0

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


def use_neural_network(input_data, hm_epchs=34):

    prediction = neural_network_model(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,filename)

        features = np.array(list(input_data))
        print(c.HEADER,"\nInput:", features)

        # up: [1,0] , argmax: 0
        # down: [0,1] , argmax: 1
        result = sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1))
        print("prediction: ",prediction.eval(feed_dict={x:[features]}))
        print("output: ", result)
        if result[0] == 1:
            print(company,' will go down')
        elif result[0] == 0:
            print(company,' will go up')

        return result[0]


print(len(run_x))

runTotal = 34

testAccuracy = open("dataNew/dataForGraphing/testAccuracy" + company + ".txt", "w")
testPrediction = open("dataNew/dataForGraphing/testPrediction" + company + ".txt", "w")
testActual = open("dataNew/dataForGraphing/ testActual" + company + ".txt", "w")

for i in range(runTotal):
    res = use_neural_network(run_x[i], company)
    testActual.write(str(res) + '\n')
    if (res == 1 and run_y[i] == [1,0]) or (res == 0 and run_y[i] == [0,1]):
        print(c.OKGREEN, 'CORRECT')
        correct += 1

        testPrediction.write(str(res) + '\n')

    else:
        print(c.FAIL, 'FAIL')

        if(res == 1):
            testPrediction.write(str(0) + '\n')
        else:
            testPrediction.write(str(1) + '\n')

    print(c.WARNING,'TEST [',i+1,'|',runTotal,']')

    testAccuracy.write(str(correct/runTotal) + "\n")

testAccuracy.close()
print(c.OKGREEN, correct, ' are Correct\n', c.FAIL, runTotal-correct, ' are Wrong\n', c.HEADER, (correct/runTotal)*100, '% Correctness')
# use_neural_network([89,2,-100,30,-300], 'ORACLE')
# use_neural_network([0,0,0,0,0], 'ORACLE')
# use_neural_network([1,1,0,1,0], 'ORACLE')
# use_neural_network([0,1,1,1,1], 'ORACLE')
# use_neural_network([1,0,0,0,1], 'ORACLE')
