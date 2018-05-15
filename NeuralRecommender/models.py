import numpy as np
from scipy.special import expit
import tensorflow as tf

def sigmoid(X, W, b):
    '''
    Calculates the activations of a layer using the
    sigmoid function, in [0,1].
    '''
    xw = np.dot(X, W)
    #return 1.0 / (1 + np.exp(- xw - b))
    return expit(xw+b)


def tanh(X, W, b):
    '''
    Calculates the activations of a layer using the
    tanh function, in [-1,1].
    '''
    z = np.dot(X,W) + b
    return (np.exp(z) - np.exp(-z)) // (np.exp(z) + np.exp(-z))


class RBM:
    author = 'Jorge'

    def __init__(self, n_visible, n_hidden, alpha=1.0, verbose=False):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """
        self.verbose = verbose
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.vb = tf.placeholder("float", [self.n_visible])  # Number of unique movies

        self.hb = tf.placeholder("float", [self.n_hidden])  # Number of features we're going to learn
        self.W = tf.placeholder("float", [self.n_visible, self.n_hidden])

        # Phase 1: Input Processing
        self.v0 = tf.placeholder("float", [None, self.n_visible])
        self._h0 = tf.nn.sigmoid(tf.matmul(self.v0, self.W) + self.hb)
        self.h0 = tf.nn.relu(tf.sign(self._h0 - tf.random_uniform(tf.shape(self._h0))))
        # Phase 2: Reconstruction
        self._v1 = tf.nn.sigmoid(tf.matmul(self.h0, tf.transpose(self.W)) + self.vb)
        self.v1 = tf.nn.relu(tf.sign(self._v1 - tf.random_uniform(tf.shape(self._v1))))
        self.h1 = tf.nn.sigmoid(tf.matmul(self.v1, self.W) + self.hb)

        # Learning rate
        self.alpha = alpha
        # Create the gradients
        self.w_pos_grad = tf.matmul(tf.transpose(self.v0), self.h0)
        self.w_neg_grad = tf.matmul(tf.transpose(self.v1), self.h1)
        # Calculate the Contrastive Divergence to maximize
        self.CD = (self.w_pos_grad - self.w_neg_grad) / tf.to_float(tf.shape(self.v0)[0])
        # Create methods to update the weights and biases
        self.update_w = self.W + self.alpha * self.CD
        self.update_vb = self.vb + self.alpha * tf.reduce_mean(self.v0 - self.v1, 0)
        self.update_hb = self.hb + self.alpha * tf.reduce_mean(self.h0 - self.h1, 0)

        self.err = self.v0 - self.v1
        self.err_sum = tf.reduce_mean(self.err * self.err)

        # Current weight
        self.cur_w = np.zeros([self.n_visible, self.n_hidden], np.float32)
        # Current visible unit biases
        self.cur_vb = np.zeros([self.n_visible], np.float32)
        # Current hidden unit biases
        self.cur_hb = np.zeros([self.n_hidden], np.float32)
        # Previous weight
        self.prv_w = np.zeros([self.n_visible, self.n_hidden], np.float32)
        # Previous visible unit biases
        self.prv_vb = np.zeros([n_visible], np.float32)
        # Previous hidden unit biases
        self.prv_hb = np.zeros([self.n_hidden], np.float32)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.errors = None

    def fit(self, X, learning_rate=0.001, decay=0, momentum=False, batchsize=1, n_epochs=10):
        self.errors = []
        for epoch in range(n_epochs):
            for start, end in zip(range(0, len(X), batchsize), range(batchsize, len(X), batchsize)):
                batch = X[start:end]
                d = {self.v0: batch, self.W: self.prv_w, self.vb: self.prv_vb, self.hb: self.prv_hb}
                cur_w = self.sess.run(self.update_w, feed_dict=d)
                cur_vb = self.sess.run(self.update_vb, feed_dict=d)
                cur_nb = self.sess.run(self.update_hb, feed_dict=d)
                self.prv_w = cur_w
                self.prv_vb = cur_vb
                self.prv_hb = cur_nb
            self.errors.append(self.sess.run(self.err_sum, feed_dict=
                                        {self.v0: X, self.W: self.cur_w, self.vb: self.cur_vb, self.hb: cur_nb}))
            if self.verbose:
                print('Epoch %d, with an error of %.8f' % (epoch, self.errors[-1]))

    def evaluate(self, samples):
        # Feeding in the user and reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(self.v0, self.W) + self.hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(self.W)) + self.vb)
        feed = self.sess.run(hh0, feed_dict={self.v0: samples, self.W: self.prv_w, self.hb: self.prv_hb})
        rec = self.sess.run(vv1, feed_dict={hh0: feed, self.W: self.prv_w, self.vb: self.prv_vb})
        return rec
