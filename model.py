import tensorflow as tf


def model():
    #Declare the system specs
    #How fast the system learns
    #Too low and the system may not have enough data to train
    #Too high and the system might adjust too quickly and miss the target
    learning_rate = 0.5

    #How many times the model is trained on the test data.
    epochs = 10

    #How much data is trained on at a time?
    batch_size =

    #Declare the input layer 1 input node per stock
    inputLayer = tf.placeholder(tf.float32, [None, 5])

    #Declare the output layer 1 output node per stock
    outputLayer = tf.placeholder(tf.float32, [None, 5])

    #Weights for the connections between the 5 node input layer and the first 100 node hidden layer
    W1 = tf.Variable(tf.random_normal([5, 100], stddev = 0.03), name = 'W1')
    b1 = tf.Variable(tf.random_normal([100]), name = 'b1')

    #Weights for the connections between the first 100 node hidden layer and the second 100 node hidden layer
    W2 = tf.Variable(tf.random_normal([100, 100], stddev = 0.03), name = 'W2')
    b2 = tf.Variable(tf.random_normal([100]), name = 'b2')

    #Weights for the connection between the second 100 node hidden layer and the 5 nod output layer
    W3 = tf.Variable(tf.random_normal([100, 5] stddev = 0.03), name = 'W3')
    b3 = tf.variable(tf.random_normal([5]), name = 'b3')

    #Begin Calculations
    first_out = tf.add(tf.matmul(inputLayer, W1), b1)
    first_out = tf.nn.relu(first_out)

    second_out = tf.add(tf.matmul(first_out, W2), b2)
    second_out = tf.nn.relu(second_out)

    output = tf.add(tf.matmul(second_out, W3), b3)
    output = tf.nn.softmax(output)

    output_clipped = tf.clip_by_value(output, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(outputLayer * tf.log(output_clipped) + (1 - outputLayer) * tf.log(1 - output_clipped), axis = 1))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)

    init_op = tf.global_variables_initializer()

    numCorrect = 0.0;
    expected = 0;
    actual = 0;

    for x range (0,4)
        if output[x] > .5
            expected = 1.0;
        else
            expected = 0.0;

        if outputLayer[x] > 0
            actual = 1.0;
        else
            expected = 0.0
        if expected == actual
        numCorrect += 1.0

        accuracy = numCorrect / 5.0
