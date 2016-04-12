import argparse
import sys
import numpy as np
from Image import Image
from ImageDecompositionWrapper import ImageDecompositionWrapper
from Autoencoder import Autoencoder
from RICA import RICA
from sklearn.decomposition import PCA, FastICA
import scipy.io as sio
import nibabel as nib
import os.path
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import explained_variance_score as evs

parser = argparse.ArgumentParser(description='fMRI analysis pipeline for matrix decomposition')
parser.add_argument('-pca', metavar='image', type=str, nargs='+',required=True)
parser.add_argument('-mask', metavar='mask', type=str, nargs='+',required=True)
parser.add_argument('-groupautoencoder', metavar='groupautoencoder', type=str, required=True)
parser.add_argument('-grouprica', metavar='grouprica', type=str, required=True)
parser.add_argument('-atlas', metavar='atlas', type=str, nargs='+')
parser.add_argument('-seg', metavar='seg', type=str, nargs='+')
parser.add_argument('-maxiter', metavar='maxiter', type=int, default=200)
parser.add_argument('-fwhm', metavar='fwhm', type=int, default=0)
parser.add_argument('-weight_decay', metavar='weight_decay', type=float, default=0.0011)
args = parser.parse_args()

if len(args.pca)!=len(args.mask):
    print "ERROR: length of image and mask list is not equal ... exit"
    sys.exit()

atlas_exits = False
if args.atlas is not None:
    if len(args.pca)!=len(args.atlas):
        print "ERROR: length of image and atlas list is not equal ... exit"
        sys.exit()
    else:
        atlas_exits=True

mat_dict = sio.loadmat(args.groupautoencoder)
mat_dict_rica = sio.loadmat(args.grouprica)
W_all = mat_dict_rica['W']
W1_all = mat_dict['W1']
W2_all = mat_dict['W2']
h_all = mat_dict['h']
b1_all = mat_dict['b1']
b2_all = mat_dict['b2']
i=0
j=0
W_all_new = mat_dict_rica['W']
W1_all_new = mat_dict['W1']
W2_all_new = mat_dict['W2']
h_all_new = mat_dict['h']
b1_all_new = mat_dict['b1']
b2_all_new = mat_dict['b2']
stats = np.zeros((len(args.pca),5))
for index, scan in enumerate(args.pca):
    print "Process scan: "+ scan + " with atlas:" + args.mask[index]
    # load subject's pca in its image space
    img = Image(img_filename=scan, mask_filename=args.mask[index], fwhm=args.fwhm)
    # init parameters
    # increase i,j indices
    i=j
    j+=img.image_mat_in_mask_normalised.shape[1]
    params_init_scan = {}
    params_init_scan_rica = {}
    params_init_scan_rica['W'] = W_all[:, i:j]
    params_init_scan['W1'] = W1_all[:, i:j]
    params_init_scan['W2'] = W2_all[i:j, :]
    params_init_scan['b1'] = b1_all
    params_init_scan['b2'] = b2_all[:,i:j]
    # bias on parameter estimation
    params_bias_scan = {}
    params_bias_scan_rica = {}
    params_bias_scan_rica['W'] = W_all[:, i:j]
    params_bias_scan['W1'] = W1_all[:, i:j]
    params_bias_scan['W2'] = W2_all[i:j, :]
    params_bias_scan['b1'] = b1_all
    params_bias_scan['b2'] = b2_all[:,i:j]


    # # do Autoencoder with group bias influence on parameters
    # d = {}; d[0.5]='05'; d[0.1]='01'; d[0.05]='005'; d[0.01]='001'; d[0.001]='0001';
    # for weight_idx,weight in enumerate([0.5,0.1,0.05,0.01,0.001]):
    # 	filename = (scan[0:scan.find('.nii')]+'.fwhm'+str(args.fwhm)+'mm.'+'_backpropagation_groupbias.'+d[weight]+'.nii.gz')
    # 	print filename
    # 	ae_voxels_restricted=Autoencoder(img.image_mat_in_mask_normalised.shape[1], max_iter=args.maxiter, weight_decay_param=weight).fit(img.image_mat_in_mask_normalised.T, params_init_scan, params_bias_scan)
    # 	img.write_decomposition(maps=ae_voxels_restricted.components_,filename=filename,normalise=False)
    # 	#with open((filename[0:filename.find('.nii')]+'.evs.txt'), 'w') as f:
	# #    	f.write('%f' % evs(img.image_mat_in_mask_normalised.T.ravel(), ae_voxels_restricted.x_mat_tilde.ravel()))
    #
	# stats[index,weight_idx]=evs(img.image_mat_in_mask_normalised.T.ravel(), ae_voxels_restricted.x_mat_tilde.ravel())
    # 	print "error: "+str(evs(img.image_mat_in_mask_normalised.T.ravel(), ae_voxels_restricted.x_mat_tilde.ravel()))

    # ... and without

    filename = (scan[0:scan.find('.nii')]+'.fwhm'+str(args.fwhm)+'mm.'+'_backpropagation_groupbias_rica.reference.nii.gz')
    rica=RICA(img.image_mat_in_mask_normalised.shape[1], max_iter=args.maxiter, weight_decay_param=args.weight_decay).fit(img.image_mat_in_mask_normalised.T, params_init_scan_rica, optimization=False)
    img.write_decomposition(maps=rica.components_, filename=filename, normalise=True)

    filename = (scan[0:scan.find('.nii')]+'.fwhm'+str(args.fwhm)+'mm.'+'_backpropagation_groupbias_rica.'+str(args.weight_decay)+'.nii.gz')
    rica=RICA(img.image_mat_in_mask_normalised.shape[1], max_iter=args.maxiter, weight_decay_param=args.weight_decay).fit(img.image_mat_in_mask_normalised.T, params_init_scan_rica, params_bias_scan_rica, optimization=True)
    img.write_decomposition(maps=rica.components_, filename=filename, normalise=True)
    W_all_new[:, i:j] = rica.W 

    filename = (scan[0:scan.find('.nii')]+'.fwhm'+str(args.fwhm)+'mm.'+'_backpropagation_groupbias.reference.nii.gz')
    ae_voxels=Autoencoder(img.image_mat_in_mask_normalised.shape[1], max_iter=args.maxiter, weight_decay_param=args.weight_decay, second_nonlinear=True).fit(img.image_mat_in_mask_normalised.T, params_init_scan, optimization=False)
    img.write_decomposition(maps=ae_voxels.components_,filename=filename, normalise=False)

    filename = (scan[0:scan.find('.nii')]+'.fwhm'+str(args.fwhm)+'mm.'+'_backpropagation_groupbias.'+str(args.weight_decay)+'.nii.gz')
    ae_voxels=Autoencoder(img.image_mat_in_mask_normalised.shape[1], max_iter=args.maxiter, weight_decay_param=args.weight_decay, second_nonlinear=True).fit(img.image_mat_in_mask_normalised.T, params_init_scan, params_bias_scan, optimization=True)
    img.write_decomposition(maps=ae_voxels.components_,filename=filename, normalise=False)

    break
    #S_ = FastICA(n_components=img.image_mat_in_mask_normalised.shape[1]).fit_transform(img.image_mat_in_mask_normalised)
    #img.write_decomposition(maps=S_.T,filename=(scan[0:scan.find('.nii')]+'.fwhm'+str(args.fwhm)+'mm.'+'_ica.nii.gz'), normalise=True)

#sio.savemat('./stats/evs.mat', {'evs':stats})
