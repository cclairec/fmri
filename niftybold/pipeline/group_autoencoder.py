import argparse
import numpy as np
from Image import Image
import scipy
import scipy.io as sio
from Autoencoder import Autoencoder
from RICA import RICA
from sklearn.decomposition import FastICA,PCA
from sklearn import linear_model
import os.path
from numpy.linalg import pinv
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from numpy.linalg import norm
from sklearn.metrics import explained_variance_score as evs

parser = argparse.ArgumentParser(description='fMRI analysis pipeline for matrix decomposition')
parser.add_argument('-grouppca', metavar='grouppca', type=str, nargs='+', required=True)
parser.add_argument('-img', metavar='img', type=str, nargs='+', required=False)
parser.add_argument('-mask', metavar='mask', type=str, nargs='+', required=False)
parser.add_argument('-groupmask', metavar='groupmask', type=str, required=True)
parser.add_argument('-fwhm', metavar='fwhm', type=float, default=5.0)
parser.add_argument('-pcan', metavar='pcan', type=int, default=30)
parser.add_argument('-ican', metavar='ican', type=int, default=30)
parser.add_argument('-maxiter', metavar='maxiter', type=int, required=True)
args = parser.parse_args()

if not os.path.isfile("_ica.mat"):
    all_pca_mat = None
    for index, scan in enumerate(args.grouppca):
        print scan
        pca_img = Image(img_filename=scan, mask_filename=args.groupmask,fwhm=args.fwhm)
        pca_mat = pca_img.image_mat_in_mask_normalised.T
        if index==0:
            all_pca_mat = pca_mat
        else:
            all_pca_mat = np.concatenate((all_pca_mat, pca_mat))


    # PCA reduction
    all_pca_mat_reduced = PCA(args.pcan, whiten=True).fit_transform(all_pca_mat.T).T
    pca_img.write_decomposition(maps=all_pca_mat_reduced, filename="_pca.nii.gz", normalise=False)
    ####################################################################################
    ################################## Spatial Autoencoder #############################
    ####################################################################################
    autoencoder = Autoencoder(args.ican, max_iter=args.maxiter).fit(all_pca_mat)
    pca_img.write_decomposition(maps=autoencoder.h, filename="_autoencoder.nii.gz", normalise=False)
    sio.savemat("_autoencoder.mat", {'h':autoencoder.h,'W1':autoencoder.W1,'W2':autoencoder.W2,'b1':autoencoder.b1, 'b2':autoencoder.b2, 'all_pca_mat_reduced':all_pca_mat_reduced})
    ###################################################################################
    ################################## Reconstruction ICA #############################
    ###################################################################################
    rica = RICA(args.ican).fit(all_pca_mat)
    pca_img.write_decomposition(maps=rica.h, filename="_rica.nii.gz", normalise=False)
    sio.savemat("_rica.mat", {'h':rica.h,'W':rica.W,'all_pca_mat_reduced':all_pca_mat})
    ###################################################################################
    #################################### FastICA ######################################
    ###################################################################################
    ica = FastICA(n_components=args.ican)
    S_ = ica.fit_transform(all_pca_mat.T)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix
    A_inv = pinv(A_)
    S_norm = S_
    for i in xrange(S_.shape[1]):
        S_norm[:,i] = S_norm[:,i] - S_norm[:,i].mean()
        S_norm[:,i] = S_norm[:,i] / S_norm[:,i].std()
    pca_img.write_decomposition(maps=S_norm.T,filename="_fastica.nii.gz",normalise=False)
    sio.savemat("_ica.mat", {'h':S_,'W':A_,'all_pca_mat_reduced':all_pca_mat})
else:
    mat = sio.loadmat("_ica.mat")
    S_ = mat['h']

# evs_mat = np.zeros((len(args.grouppca)))
# for index, scan in enumerate(args.grouppca):
#     pca_img = Image(img_filename=scan, mask_filename=args.groupmask, fwhm=args.fwhm)
#     raw_img = Image(img_filename=args.img[index], mask_filename=args.mask[index])
#     pca_mat = pca_img.image_mat_in_mask_normalised.T
#     raw_mat = raw_img.image_mat_in_mask_normalised.T
#     W = np.zeros((S_.shape[1],pca_mat.shape[0]))
#     regr = linear_model.LinearRegression()
#     for t in xrange(pca_mat.shape[0]):
#         regr.fit(S_, pca_mat[t,:].ravel())
#         W[:,t] = regr.coef_
#     mat_filename = (scan[0:scan.find('.nii')]+'.fastica.W.mat')
#     ica_group_filename = (scan[0:scan.find('.nii')]+'.fastica.nii.gz')
#     ica_filename = (scan[0:args.img[index].find('.nii')]+'.fastica.nii.gz')
#     sio.savemat(mat_filename, {'W':W})
#     regr.fit(W.T, pca_mat)
#     pca_img.write_decomposition(maps=regr.coef_.T,filename=ica_group_filename,normalise=True)
#     regr.fit(W.T, raw_mat)
#     raw_mat_tilde = regr.predict(W.T)
#     print "error: "+str(evs(raw_mat.ravel(),raw_mat_tilde.ravel()))
#     #with open((scan[0:args.img[index].find('.nii')]+'.fastica.evs.txt'), 'w') as f:
#     #f.write('%d' % evs(raw_mat.ravel(), raw_mat_tilde.ravel()))
#     evs_mat[index]=evs(raw_mat.ravel(), raw_mat_tilde.ravel())
#     raw_img.write_decomposition(maps=regr.coef_.T,filename=ica_filename,normalise=True)
#     print ica_filename
#
# sio.savemat("./stats/ica_evs.mat",{'evs':evs_mat})
