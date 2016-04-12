import numpy as np
import scipy
import pdb
import math


class InfoMax(object):

    def __init__(self, n_components, penalty=0.1, eps=0.01, maxiter=200):
        self._n_components = n_components
        self._penalty = penalty
        self._eps = eps
        self._maxiter = maxiter

    def cost(self, W, X):
        # compute hidden layer
        #import pdb; pdb.set_trace()
        W = np.reshape(W, (self.feature_dim, self.sample_dim))
        [N,M] = X.shape
        #
        # Y = W.dot(X)
        # #f =-( N*log(abs(det(W))) - sum(sum(log( cosh(S) ))) - N*M*log(pi) );
        # cost = self.feature_dim*np.log(np.abs(np.linalg.det(W))) + (np.log(np.cosh(Y))).sum()
        # #grad = (self.feature_dim*np.identity(self.feature_dim) - np.tanh(Y).dot(X.T).dot(W.T)).dot(W).ravel()
        # grad = (self.feature_dim*np.linalg.pinv(W.T) - np.tanh(Y).dot(X.T)).ravel()
        #import pdb; pdb.set_trace()
        # print cost

        # estimated source signals
        y = W.dot(X)
        # estimated maximum entropy signals
        Y = np.tanh(y)
        detW = np.abs(np.linalg.det(W))
        cost = -np.log(Y).sum() - np.log(detW) #(1.0/N)*   0.5*
        # Find matrix of gradients
        grad = -np.linalg.inv(W.T) + Y.dot(X.T)    #(2.0/N)*
        #grad = (np.identity(self.feature_dim) - np.tanh(Y).dot(X.T).dot(W.T)).dot(W).ravel()

        print cost
        return cost, grad.ravel()

    def fit(self,x_mat):

        dims_mat = x_mat.shape
        W0 = np.random.randn(self._n_components , dims_mat[0]) * 0.01
        W0 = W0/(np.sqrt(np.sum(W0**2,axis=1))[:,None])
        #W0 = np.identity(self._n_components)
        self.feature_dim = W0.shape[0]
        self.sample_dim = W0.shape[1]

        J = lambda x: self.cost(x, x_mat)
        options_ = {'maxiter': self._maxiter, 'disp': True}
        result = scipy.optimize.minimize(J, W0, method='L-BFGS-B', jac=True, options=options_)

        W_opt = np.reshape(result.x,(self._n_components, dims_mat[0]))
        self._W = W_opt
        self._h = W_opt.dot(x_mat)
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
