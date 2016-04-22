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
parser.add_argument('-mat', metavar='mat', type=str, required=True)
parser.add_argument('-ican', metavar='ican', type=int, default=30)
parser.add_argument('-maxiter', metavar='maxiter', type=int, default=400)
args = parser.parse_args()
prefix = args.mat[0:args.mat.find('.')]
f = sio.loadmat(args.mat)
mat = f['mat']

####################################################################################
################################## PCA #############################################
####################################################################################
pca = PCA(args.ican, whiten=True)
S_ = pca.fit_transform(mat.T)
import pdb; pdb.set_trace()
sio.savemat("{prefix}_pca.mat".format(prefix=prefix), {'h':S_.T})
####################################################################################
################################## Spatial Autoencoder #############################
####################################################################################
autoencoder = Autoencoder(args.ican, max_iter=args.maxiter).fit(mat)
sio.savemat("{prefix}_autoencoder.mat".format(prefix=prefix), {'h':autoencoder.h,'W1':autoencoder.W1,'W2':autoencoder.W2,'b1':autoencoder.b1, 'b2':autoencoder.b2})
###################################################################################
################################## Reconstruction ICA #############################
###################################################################################
rica = RICA(args.ican, max_iter=args.maxiter).fit(mat)
sio.savemat("{prefix}_rica.mat".format(prefix=prefix), {'h':rica.h,'W':rica.W})
###################################################################################
#################################### FastICA ######################################
###################################################################################
ica = FastICA(n_components=args.ican, whiten=True)
S_ = ica.fit_transform(mat.T)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
A_inv = pinv(A_)
sio.savemat("{prefix}_ica.mat".format(prefix=prefix), {'h':S_.T,'W':A_,'W_inv':A_inv})

# S_norm = S_
# for i in xrange(S_.shape[1]):
#     S_norm[:,i] = S_norm[:,i] - S_norm[:,i].mean()
#     S_norm[:,i] = S_norm[:,i] / S_norm[:,i].std()
