import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pdb
import math
import scipy


class RICA(object):

    def __init__(self, n_components, penalty=0.1, eps=0.01, max_iter=200, weight_decay_param=3e-4):
        self._n_components = n_components
        self._penalty = penalty
        self._eps = eps
        self._maxiter = max_iter
        self._weight_decay_param = weight_decay_param

    def l2rowscaled(self, x, alpha):
        normeps = 1e-5
        epssumsq = np.sum(x**2,axis=1) + normeps
        l2rows=np.sqrt(epssumsq)*alpha
        y=x/l2rows[:,None]
        return y

    def l2rowscaledg(self, x, y, outderv, alpha):
        normeps = 1e-5
        epssumsq = np.sum(x**2,axis=1) + normeps
        l2rows = np.sqrt(epssumsq)*alpha

        if y.size==0:
            y = x/l2rows[:,None]

        dummy = np.sum(outderv*x, axis=1)/epssumsq
        grad = (outderv/l2rows[:,None]) - (y*dummy[:,None])
        return grad

    def model(self, W, X):
        # compute hidden layer

        W = np.reshape(W, (self.feature_dim, self.sample_dim))
        Wold = W
        W = self.l2rowscaled(W, 1)

        h = W.dot(X)
        # compute output layer
        out = W.T.dot(h) - X
        # compute cost
        K = np.sqrt(self._eps + h**2)
        cost = l2_loss(out) + self._penalty * np.sum(K.ravel())
        if self._params_cost_bias:
            cost += self._weight_decay_param * l2_loss(W-self._params_cost_bias['W'])

        W2grad = out.dot(h.T)
        W1grad = (W.dot(out) + self._eps * (h * (1/K))).dot(X.T)
        Wgrad = W1grad + W2grad.T
        if self._params_cost_bias:
            Wgrad += self._weight_decay_param * (W-self._params_cost_bias['W'])

        grad = self.l2rowscaledg(Wold, W, Wgrad, 1).ravel()
        #grad = Wgrad.ravel()

        return cost, grad

    def fit(self, x_mat, params=None, params_cost_bias=None, optimization=True):

        dims_mat = x_mat.shape
        W0 = np.random.randn(self._n_components , dims_mat[0]) * 0.01
        W0 = W0/(np.sqrt(np.sum(W0**2,axis=1))[:,None])
        if params:
            W0 = params['W']

        self.feature_dim = W0.shape[0]
        self.sample_dim = W0.shape[1]
        self._params_cost_bias = params_cost_bias

        W = W0
        if optimization:
            J = lambda x: self.model(x, x_mat)
            options_ = {'maxiter': self._maxiter, 'disp': True}
            result = scipy.optimize.minimize(J, W0.ravel(), method='L-BFGS-B', jac=True, options=options_)
            W = np.reshape(result.x,(self._n_components, dims_mat[0]))

        self._W = W
        self._h = W.dot(x_mat)
        return self

    @property
    def components_(self):
        return self._h

    @property
    def h(self):
        return self._h

    @property
    def W(self):
        return self._W


def l2_loss(x):
    return np.sum(x.ravel() ** 2) / 2
