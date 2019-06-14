import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 

x = tf.placeholder('float')
y = tf.placeholder('float')

#input_data * weights + biases 
#why do we have a bias? if all input data is 0 times weights no neuron would ever fire, not ideal
#so the bias adds a value to that so that atleast some neurons can fire

def neural_network_model(data):

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					  'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output 

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	#cycles of feed forward + backprop
	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0 
			for _ in range(int( mnist.train.num_examples / batch_size )):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c 
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)

#relu is your threshold function 1 or 0

#10 classes, 0-9
# one_hot 0 = [1,0,0,0,0,0,0,0,0,0,] etc 1 moves one along for each number increase



''' NN  Feed Forward data straight to output
input > weight > hidden layer 1 (activation function) > weights > hidden Layer 2 
(activation function) > weights > output layer 

At end we compare out to intended output (how close it is we measure with a) > cost function (e.g. cross entropy)
Optimization function (optimizer) > attempts to minimize that cost (e.g. AdamnOptizer ... stochastic gradient descent etc.)

Then it goes backwards and manipulates the weights this is called
backpropagation

feed forward + backprop = epoch (one cycle of feed forward backprop)
each cycle you lower the cost function cost is really high and drops down and lowers 

'''
