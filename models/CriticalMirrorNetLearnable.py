import tensorflow as tf
from utils.Utils import *


class CriticalMirrorNetLearnable(object):
    """
    Class representing a critical echo net
    """

    def __init__(self, params, hidden_weights=None):
        self.params = params
        self.network_shape = self.params['network_shape']
        self.input_dim = self.network_shape[0]
        self.output_dim = self.network_shape[-1]
        self.batch_size = self.params['batch_size']
        self.hidden_weights = hidden_weights
        self.tensorboard_dir = self.params['tensorboard_dir']

        self.activation_function = self.params['activation_function']
        self.optimizer_ = self.params['optimizer']
        self.hs_optimizer_ = self.params['optimizer']
        # model
        self.input = tf.placeholder(tf.float32, [None, self.input_dim], name="input")
        self.output = tf.placeholder(tf.float32, [None, self.output_dim], name="output")
        self.activation_patterns = {}
        self.hidden_state_activation_patterns = {}
        self.activation = self.input
        self.hidden_states = {}
        self.hidden_states_update_ops = {}
        for i in range(1, len(self.network_shape) - 1):
            with tf.name_scope("layer{0}".format(i)):
                h = tf.Variable(tf.truncated_normal([self.network_shape[i]]), name="hidden_state", trainable=False)
                # h = tf.truncated_normal([self.batch_size, self.network_shape[i]])
                self.hidden_states["hs_{0}".format(i)] = h
                Utils.variable_summaries(self.hidden_states["hs_{0}".format(i)], "hs_{0}".format(i))
        if self.hidden_weights is not None:
            H_tune = tf.Variable(1.0, trainable=False, name="H_tune")
            # Utils.variable_summaries(H_tune, "H_tune")
        else:
            H_tune = tf.Variable(1, trainable=False, name="H_tune")

        self.H_grad_modifier_list = []
        for i in range(len(self.network_shape) - 1):
            with tf.name_scope("layer{0}".format(i + 1)):
                if i < len(self.network_shape) - 2:
                    with tf.name_scope("hidden"):

                        # input weight and bias
                        W = tf.Variable(
                            tf.random_normal([self.network_shape[i], self.network_shape[i + 1]], stddev=0.05),
                            name="W")
                        bW = tf.Variable(tf.random_normal([self.network_shape[i + 1]], stddev=0.05), name="bW")
                        Utils.variable_summaries(W, "W")
                        Utils.variable_summaries(bW, "bW")
                        H_name = "H_{0}".format(i + 1)
                        if self.hidden_weights is not None and H_name in self.hidden_weights.keys():
                            H = tf.Variable(self.hidden_weights[H_name].astype('float32'), dtype=tf.float32,
                                            trainable=True, name="H")
                            Utils.variable_summaries(H, "H")
                            H_grad_modifier = tf.Variable(tf.random_normal([self.network_shape[i + 1], self.network_shape[i + 1]], stddev=0.05),
                                trainable=True, name="h_grad_modifier")
                            Utils.variable_summaries(H_grad_modifier, "h_grad_modifier")
                            self.H_grad_modifier_list.append(H_grad_modifier)
                        else:
                            H = tf.Variable(
                                tf.random_normal([self.network_shape[i + 1], self.network_shape[i + 1]], stddev=0.05),
                                trainable=False, name="H")

                        input_for_hidden = tf.matmul(self.activation, W) + bW
                        tiled_h = tf.reshape(tf.tile(self.hidden_states["hs_{0}".format(i + 1)], [self.batch_size]), [self.batch_size, -1])
                        hidden_update = tf.nn.tanh(tf.add(input_for_hidden,
                                                             tf.matmul(tiled_h, tf.scalar_mul(H_tune, H))))

                    with tf.name_scope("mirror"):
                        # mirror input and bias
                        M = tf.Variable(tf.random_normal([self.network_shape[i], self.network_shape[i + 1]], stddev=0.05), name="M")
                        bM = tf.Variable(tf.random_normal([self.network_shape[i + 1]], stddev=0.05), name="bM")
                        Utils.variable_summaries(M, "M")
                        Utils.variable_summaries(bM, "bM")
                        input_for_mirror = tf.nn.tanh(tf.matmul(self.activation, M) + bM)

                    with tf.name_scope("readout"):
                        # readout weights and biases
                        R = tf.Variable(tf.random_normal([self.network_shape[i + 1], self.network_shape[i + 1]], stddev=0.05), name="R")
                        bR = tf.Variable(tf.random_normal([self.network_shape[i + 1]], stddev=0.05), name="bR")
                        Utils.variable_summaries(R, "R")
                        Utils.variable_summaries(bR, "bR")
                        readout = tf.nn.tanh(tf.matmul(hidden_update, R) + bR)

                    with tf.name_scope("activation"):
                        self.activation = self.activation_function(tf.multiply(readout, input_for_mirror))
                    # self.hidden_state_activation_patterns['hidden_state_layer_{0}'.format(i + 1)] = self.hidden_states[self.hidden_states["hs_{0}".format(i+1)]]
                    self.hidden_states_update_ops["hs_{0}".format(i+1)] = self.hidden_states["hs_{0}".format(i+1)].assign(hidden_update[0])
                    # self.hidden_states["hs_{0}".format(i+1)] = hidden_update
                else:
                    W = tf.Variable(
                        tf.random_normal([self.network_shape[i], self.network_shape[i + 1]], stddev=0.05),
                        name="W")
                    bW = tf.Variable(tf.random_normal([self.network_shape[i + 1]], stddev=0.05), name="bW")
                    Utils.variable_summaries(W, "W")
                    Utils.variable_summaries(bW, "bW")
                    with tf.name_scope("activation"):
                        self.activation = self.activation_function(tf.matmul(self.activation, W) + bW)
                act = self.activation
                if i > 0:
                    self.activation_patterns['layer_{0}'.format(i)] = act

        # cost
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.activation, self.output))
            tf.summary.scalar('cost', self.cost)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.activation, 1), tf.argmax(self.output, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        gv = self.optimizer_.compute_gradients(self.cost)
        self.hidden_modifier_cost = self.optimize_hidden_modifier(gv)
        tf.summary.scalar('hidden_modifier_cost', self.hidden_modifier_cost)

        fm_gv = self.filter_modify_gradients(gv)

        gv_mod = self.hs_optimizer_.compute_gradients(self.hidden_modifier_cost)

        self.hs_optimizer = self.hs_optimizer_.apply_gradients([t for t in gv_mod if "hidden/h_grad_modifier" in t[1].name])

        self.optimizer = self.optimizer_.apply_gradients(fm_gv)

        self.sess = tf.Session()

        self.merged = tf.summary.merge_all()
        self.summ_writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        init = tf.global_variables_initializer()

        self.sess.run(init)

    def partial_fit(self, X, Y):
        fetches = [self.cost, self.optimizer, self.merged, self.hidden_modifier_cost, self.hs_optimizer]
        for _, v in self.hidden_states_update_ops.items():
            fetches.append(v)
        cost, opt, merged, hs_cost, hs_opt, _ = self.sess.run(fetches, feed_dict={self.input: X, self.output: Y})

        return cost, merged

    def set_batch_size(self, bs):
        self.batch_size = bs

    def inference(self, X, Y):
        bs = tf.shape(X)[0]
        if bs != self.batch_size:
            self.set_batch_size(bs)
        return self.sess.run(self.cost, feed_dict={self.input: X, self.output: Y})

    def optimize_hidden_modifier(self, gv):
        grad = []
        for tup in gv:
            if("hidden/H" in tup[1].name):
                grad.append(tup[0])
            else:
                continue

        costs = []
        for (tupM, tup) in zip(self.H_grad_modifier_list, grad):
            costs.append(tf.reduce_sum(tf.abs(tf.reduce_sum(tf.multiply(tupM, tup), axis=1))))

        return tf.add_n(costs)

    def filter_modify_gradients(self, gv):
        modifiers = []
        to_modify = []
        rest = []
        for tup in gv:
            if("h_grad_modifier" in tup[1].name):
                modifiers.append(tup)
            elif("hidden/H" in tup[1].name):
                to_modify.append(tup)
            else:
                rest.append(tup)

        for (tupM, tup) in zip(modifiers, to_modify):
            rest.append((tf.multiply(tupM[1], tup[0]), tup[1]))

        return rest

    def get_accuracy(self, X, Y):
        bs = tf.shape(X)[0]
        if bs != self.batch_size:
            self.set_batch_size(bs)
        return self.sess.run(self.accuracy, feed_dict={self.input: X, self.output: Y})

    def calc_total_cost(self, X, Y):
        bs = tf.shape(X)[0]
        if bs != self.batch_size:
            self.set_batch_size(bs)
        return self.sess.run(self.cost, feed_dict={self.input: X, self.output: Y})

    def get_activations(self, X):
        return {k: self.sess.run(v, feed_dict={self.input: X}) for k, v in self.activation_patterns.items()}

    def get_activation_pattern(self, input_, activation_name=None):
        if activation_name is not None:
            return [self.sess.run(self.activation_patterns[activation_name], feed_dict={self.input: input_})]
        else:
            return [self.sess.run(self.activation_patterns['activation_layer_{0}'.format(i + 1)], feed_dict={self.input: input_})
                    for i in range(len(self.network_shape) - 1)]

    def get_hidden_state_pattern(self, input_, activation_name=None):
        if activation_name is not None:
            return [self.sess.run(self.activation_patterns[activation_name], feed_dict={self.input: input_})]
        else:
            return [self.sess.run(self.hidden_state_activation_patterns['hidden_state_layer_{0}'.format(i + 1)], feed_dict={self.input: input_})
                    for i in range(len(self.network_shape) - 1)]

    def get_weight(self, weight_name=None):
        if weight_name is not None:
            return [self.sess.run(self.weights[weight_name])]
        else:
            return [self.sess.run(self.weights['W_{0}'.format(i)]) for i in range(len(self.weights))]

