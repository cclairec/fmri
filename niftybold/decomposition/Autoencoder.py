import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pdb
import math
import scipy.optimize


class Autoencoder(object):

    def __init__(self, n_components, weight_decay_param=3e-4, beta=3, sparsity_param=0.1, max_iter = 300, second_nonlinear=False, bias=True):
        self._number_of_individual_autoencoder_components = n_components
        self._beta = beta
        self._weight_decay_param = weight_decay_param
        self._sparsity_param = sparsity_param
        self._maxiter = max_iter
        self._second_nonlinear = second_nonlinear
        self._bias = bias

    def model(self, theta, x_mat):
        # The input theta is a vector (because minFunc expects the parameters to be a vector).
        # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
        # follows the notation convention of the lecture notes.

        W1 = theta[0:self._number_of_individual_autoencoder_components * self._visible_size].reshape(self._number_of_individual_autoencoder_components, self._visible_size)
        W2 = theta[self._number_of_individual_autoencoder_components * self._visible_size:2 * self._number_of_individual_autoencoder_components * self._visible_size].reshape(self._visible_size, self._number_of_individual_autoencoder_components)
        if self._bias:
            b1 = theta[2 * self._number_of_individual_autoencoder_components * self._visible_size:2 * self._number_of_individual_autoencoder_components * self._visible_size + self._number_of_individual_autoencoder_components]
            b2 = theta[2 * self._number_of_individual_autoencoder_components * self._visible_size + self._number_of_individual_autoencoder_components:]
        else:
            b1 = np.zeros(self._number_of_individual_autoencoder_components)
            b2 = np.zeros(self._visible_size)


        # Number of training examples
        m = x_mat.shape[1]
        # Forward propagation
        z2 = W1.dot(x_mat) + np.tile(b1, (m, 1)).transpose()
        h = sigmoid(z2)
        z3 = W2.dot(h) + np.tile(b2, (m, 1)).transpose()
        if self._second_nonlinear:
            x_mat_tilde = sigmoid(z3)
        else:
            x_mat_tilde = z3

        # Sparsity
        rho_hat = np.sum(h, axis=1) / m
        rho = np.tile(self._sparsity_param, self._number_of_individual_autoencoder_components)

        # Cost function
        cost = np.sum((x_mat_tilde - x_mat) ** 2) / (2 * m) + \
               self._weight_decay_param * (l2_loss(W1-self._params_cost_bias['W1']) + l2_loss(W2-self._params_cost_bias['W2']) + l2_loss(b1-self._params_cost_bias['b1']) + l2_loss(b2-self._params_cost_bias['b2']) ) + \
               self._beta * np.sum(KL_divergence(rho, rho_hat))

        # Backprop
        sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()

        if self._second_nonlinear:
            delta3 = -(x_mat - x_mat_tilde) * sigmoid_prime(z3)
        else:
            delta3 = -(x_mat - x_mat_tilde)

        W2grad = delta3.dot(h.transpose()) / m + self._weight_decay_param * (W2-self._params_cost_bias['W2'])
        delta2 = (W2.transpose().dot(delta3) + self._beta * sparsity_delta) * sigmoid_prime(z2)
        W1grad = delta2.dot(x_mat.transpose()) / m + self._weight_decay_param * (W1-self._params_cost_bias['W1'])

        # concate gradient
        if self._bias:
            b1grad = np.sum(delta2, axis=1) / m
            b2grad = np.sum(delta3, axis=1) / m
            grad = np.concatenate((W1grad.reshape(self._number_of_individual_autoencoder_components * self._visible_size),
                               W2grad.reshape(self._number_of_individual_autoencoder_components * self._visible_size),
                               b1grad.reshape(self._number_of_individual_autoencoder_components),
                               b2grad.reshape(self._visible_size)))
        else:
            grad = np.concatenate((W1grad.reshape(self._number_of_individual_autoencoder_components * self._visible_size),
                               W2grad.reshape(self._number_of_individual_autoencoder_components * self._visible_size)))

        return cost, grad

    def fit(self, x_mat, params=None, params_cost_bias=None, optimization=True):

        self._visible_size = x_mat.shape[0]
        r = np.sqrt(6) / np.sqrt(self._number_of_individual_autoencoder_components + self._visible_size + 1)
        W1 = np.random.random((self._number_of_individual_autoencoder_components, self._visible_size)) * 2 * r - r
        W2 = np.random.random((self._visible_size, self._number_of_individual_autoencoder_components)) * 2 * r - r
        b1 = np.zeros(self._number_of_individual_autoencoder_components, dtype=np.float64)
        b2 = np.zeros(self._visible_size, dtype=np.float64)
        if params:
            W1 = params['W1']
            W2 = params['W2']
            b1 = params['b1']
            b2 = params['b2']

        self._params_cost_bias = {}
        if params_cost_bias is None:
            self._params_cost_bias['W1'] = np.zeros((W1.shape))
            self._params_cost_bias['W2'] = np.zeros((W2.shape))
            self._params_cost_bias['b1'] = np.zeros((b1.shape))
            self._params_cost_bias['b2'] = np.zeros((b2.shape))

        else:
            self._params_cost_bias['W1'] = params_cost_bias['W1']
            self._params_cost_bias['W2'] = params_cost_bias['W2']
            self._params_cost_bias['b1'] = params_cost_bias['b1']
            self._params_cost_bias['b2'] = params_cost_bias['b2']

        if optimization:
            if self._bias:
                theta_init = np.concatenate((W1.reshape(self._number_of_individual_autoencoder_components * self._visible_size),
                                    W2.reshape(self._number_of_individual_autoencoder_components * self._visible_size),
                                    b1.reshape(self._number_of_individual_autoencoder_components),
                                    b2.reshape(self._visible_size)))
            else:
                theta_init = np.concatenate((W1.reshape(self._number_of_individual_autoencoder_components * self._visible_size),
                                    W2.reshape(self._number_of_individual_autoencoder_components * self._visible_size)))

            J = lambda x: self.model(x, x_mat)
            options_ = {'maxiter': self._maxiter, 'disp': True}
            result = scipy.optimize.minimize(J, theta_init, method='L-BFGS-B', jac=True, options=options_)
            theta_opt = result.x
            W1 = theta_opt[0:self._number_of_individual_autoencoder_components * self._visible_size].reshape(self._number_of_individual_autoencoder_components, self._visible_size)
            W2 = theta_opt[self._number_of_individual_autoencoder_components * self._visible_size:2 * self._number_of_individual_autoencoder_components * self._visible_size].reshape( self._visible_size, self._number_of_individual_autoencoder_components)
            if self._bias:
                b1 = theta_opt[2 * self._number_of_individual_autoencoder_components * self._visible_size:2 * self._number_of_individual_autoencoder_components * self._visible_size + self._number_of_individual_autoencoder_components]
                b2 = theta_opt[2 * self._number_of_individual_autoencoder_components * self._visible_size + self._number_of_individual_autoencoder_components:2 * self._number_of_individual_autoencoder_components * self._visible_size + self._number_of_individual_autoencoder_components+self._visible_size]
            else:
                b1 = None
                b2 = None

        if self._bias:
            m = x_mat.shape[1]
            h = W1.dot(x_mat) + np.tile(b1, (m, 1)).transpose()
        else:
            h = W1.dot(x_mat)


        self._h = sigmoid(h)
        self._W1 = W1
        self._W2 = W2
        self._b1 = b1
        self._b2 = b2
        return self

    @property
    def components_(self):
        return self._h

    @property
    def x_mat_tilde(self):
	return self._x_mat_tilde

    @property
    def h(self):
        return self._h

    @property
    def W(self):
        return self._W1

    @property
    def W1(self):
        return self._W1

    @property
    def W2(self):
        return self._W2

    @property
    def b1(self):
        return self._b1

    @property
    def b2(self):
        return self._b2


def l2_loss(x):
    return np.sum(x ** 2) / 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def KL_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))
