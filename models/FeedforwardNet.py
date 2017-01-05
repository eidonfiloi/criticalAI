import tensorflow as tf


class FeedforwardNet(object):
    """
    Class representing simple multilayer feedforward net
    """

    def __init__(self, params, start_weights=None):
        self.params = params
        self.network_shape = self.params['network_shape']
        self.input_dim = self.network_shape[0]
        self.output_dim = self.network_shape[-1]
        self.start_weights = start_weights
        self.weights = self.initialize_weights()

        self.activation_function = self.params['activation_function']
        self.optimizer_ = self.params['optimizer']
        # model
        self.input = tf.placeholder(tf.float32, [None, self.input_dim])
        self.output = tf.placeholder(tf.float32, [None, self.output_dim])

        self.activation = self.input
        for i in range(len(self.weights)):
            self.activation = self.activation_function(tf.matmul(self.activation, self.weights['W_{0}'.format(i)]))

        # cost
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.activation, self.output))
        self.optimizer = self.optimizer_.minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def initialize_weights(self):
        all_weights = dict()
        if self.start_weights is not None:
            for i in range(1, len(self.network_shape)):
                all_weights["W_{0}".format(i)] = tf.Variable(self.start_weights[i].astype('float32'), dtype=tf.float32)
        else:
            for i in range(0, len(self.network_shape) - 1):
                all_weights["W_{0}".format(i)] = tf.Variable(tf.random_normal([self.network_shape[i], self.network_shape[i+1]], stddev=0.01))
        return all_weights

    def partial_fit(self, X, Y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.input: X, self.output: Y})
        return cost

    def calc_total_cost(self, X, Y):
        return self.sess.run(self.cost, feed_dict={self.input: X, self.output: Y})

    def get_weight(self, weight_name=None):
        if weight_name is not None:
            return [self.sess.run(self.weights[weight_name])]
        else:
            return [self.sess.run(self.weights['W_{0}'.format(i)]) for i in range(0, len(self.weights))]

