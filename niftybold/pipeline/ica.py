from niftybold.decomposition.Autoencoder import Autoencoder
from niftybold.decomposition.RICA import RICA
from niftybold.decomposition.InfoMax import InfoMax
from niftybold.img.Image import Image

import argparse
import numpy as np
import scipy.io as sio
from sklearn.decomposition import FastICA,PCA
from numpy.linalg import pinv

parser = argparse.ArgumentParser(description='matrix decomposition')
parser.add_argument('-mat', metavar='mat', type=str, nargs='+', required=True)
parser.add_argument('-ican', metavar='ican', type=int, default=30)
parser.add_argument('-maxiter', metavar='maxiter', type=int, default=400)
parser.add_argument('-autoencoder', metavar='autoencoder', type=bool, default=False)
parser.add_argument('-prefix', metavar='prefix', type=str, default=None)
args = parser.parse_args()

for i,f in enumerate(args.mat):
    print f
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = f[0:f.find('.mat')]
        
    f = sio.loadmat(f)
    if i == 0:
        if 'mat' in f.keys():
            mat = f['mat']

        if 'h' in f.keys():
            mat = f['h']

    else:
        if 'mat' in f.keys():
            mat = np.concatenate((mat, f['mat']))

        if 'h' in f.keys():
            mat = np.concatenate((mat, f['h']))


####################################################################################
################################## PCA #############################################
####################################################################################
# def inverse_transform(self, X):
#         """Transform data back to its original space, i.e.,
#         return an input X_original whose transform would be X
#         Parameters
#         ----------
#         X : array-like, shape (n_samples, n_components)
#             New data, where n_samples is the number of samples
#             and n_components is the number of components.
#         Returns
#         -------
#         X_original array-like, shape (n_samples, n_features)
#         """
#         check_is_fitted(self, 'mean_')
#
#         if self.whiten:
#             return fast_dot(
#                 X,
#                 np.sqrt(self.explained_variance_[:, np.newaxis]) *
#                 self.components_) + self.mean_
#         else:
#             return fast_dot(X, self.components_) + self.mean_


pca = PCA(whiten=True, n_components='mle')
#pca = PCA(whiten=True, n_components=args.ican)
h = pca.fit_transform(mat.T)
sio.savemat("{prefix}.pca.mat".format(prefix=prefix), {'h':h.T, 'W':pca.components_, 'mean': pca.mean_, 'variance_explained': pca.explained_variance_}) # although it's the other way around

####################################################################################
################################## Spatial Autoencoder #############################
####################################################################################
if args.autoencoder:
    autoencoder = Autoencoder(args.ican, max_iter=args.maxiter, second_nonlinear=True, sparsity_param=0.3).fit(mat)
    sio.savemat("{prefix}.autoencoder.mat".format(prefix=prefix), {'h':autoencoder.h,'W1':autoencoder.W1,'W2':autoencoder.W2,'b1':autoencoder.b1, 'b2':autoencoder.b2})
#import pdb; pdb.set_trace()
###################################################################################
################################## Reconstruction ICA #############################
###################################################################################
rica = RICA(args.ican, max_iter=args.maxiter, penalty=0.05).fit(h.T)
sio.savemat("{prefix}.rica.mat".format(prefix=prefix), {'h':rica.h,'W':rica.W})
###################################################################################
#################################### FastICA ######################################
###################################################################################

ica = FastICA(n_components=args.ican, whiten=False)
S_ = ica.fit_transform(h)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
A_inv = pinv(A_)
sio.savemat("{prefix}.ica.mat".format(prefix=prefix), {'h':S_.T,'W':A_,'W_inv':A_inv})

# S_norm = S_
# for i in xrange(S_.shape[1]):
#     S_norm[:,i] = S_norm[:,i] - S_norm[:,i].mean()
#     S_norm[:,i] = S_norm[:,i] / S_norm[:,i].std()
