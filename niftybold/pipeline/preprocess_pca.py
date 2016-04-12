import argparse
import sys
import numpy as np
from Image import Image
from sklearn.decomposition import PCA
import scipy.io as sio
import nibabel as nib
import os.path

parser = argparse.ArgumentParser(description='fMRI analysis pipeline for matrix decomposition')
parser.add_argument('-img', metavar='image', type=str, nargs='+', required=True)
parser.add_argument('-mask', metavar='mask', type=str, nargs='+', required=True)
parser.add_argument('-n', metavar='n_components', type=int, required=True)
parser.add_argument('-fwhm', metavar='fwhm', type=int, required=True)
args = parser.parse_args()

if len(args.img)!=len(args.mask):
    print "ERROR: length of image and mask list is not equal ... exit"
    sys.exit()

for index, scan in enumerate(args.img):
    print scan+"           "+args.mask[index]
    img = Image(img_filename=scan, mask_filename=args.mask[index], fwhm=args.fwhm)
    mat = img.image_mat_in_mask_normalised
    pca = PCA(n_components=args.n, whiten=True)
    mat_reduced = pca.fit_transform(mat)
    #print "Variance explained "+str(pca.explained_variance_ratio_)
    f = open((scan[0:scan.find('.nii')]+'.pca.txt'),'w')
    f.write(str(pca.explained_variance_ratio_))
    f.close()
    img.write_decomposition(maps=mat_reduced.T,filename=(scan[0:scan.find('.nii')]+'.pca_reduced.nii.gz'),normalise=False)
    #import pdb; pdb.set_trace()
