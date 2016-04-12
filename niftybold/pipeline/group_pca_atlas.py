import argparse
import sys
import numpy as np
from Image import Image
from ImageDecompositionWrapper import ImageDecompositionWrapper
from Autoencoder import Autoencoder
from sklearn.decomposition import PCA
import scipy.io as sio
import nibabel as nib
import os.path

parser = argparse.ArgumentParser(description='fMRI analysis pipeline for matrix decomposition')
parser.add_argument('-img', metavar='image', type=str, nargs='+',required=True,help='# image[s]')
parser.add_argument('-mask', metavar='mask', type=str, nargs='+',required=True,help='# mask comprising all brain voxels')
parser.add_argument('-atlas', metavar='atlas', type=str, nargs='+',help='# parcellation in image space')
parser.add_argument('-seg', metavar='seg', type=str, nargs='+',help='# segmentation in image space')
parser.add_argument('-n', metavar='n_components', type=int,help='# components in matrix decomposition', required=True)
parser.add_argument('-atlasref', metavar='atlasref', type=str, help='one atlas reference scan', required=False)
parser.add_argument('-maxiter', metavar='maxiter', type=int, help='max iteration for optimisation', required=False)
args = parser.parse_args()


def graph_mat_2_atlas_nii(graph_mat,atlas_ref_filename,output_filename):
    atlas = nib.load(atlas_ref_filename).get_data()
    dims_atlas = atlas.shape
    dims_components = graph_mat.shape[0]
    components = np.zeros((dims_atlas[0],dims_atlas[1],dims_atlas[2],dims_components),dtype=np.float32)
    unique_atlas_values = np.unique(atlas)
    unique_atlas_values = unique_atlas_values[unique_atlas_values>100]
    for component in xrange(dims_components):
        atlas_copy = np.zeros((atlas.shape),dtype=np.float32)
        for index,unique_region_value in enumerate(unique_atlas_values):
            atlas_copy[atlas==unique_region_value]=graph_mat[component,index]

        components[:,:,:,component]=atlas_copy


    output_img = nib.Nifti1Image(components, np.eye(4))
    nib.save(output_img, output_filename)


if len(args.img)!=len(args.mask):
    print "ERROR: length of image and mask list is not equal ... exit"
    sys.exit()

atlas_exits = False
if args.atlas is not None:
    if len(args.img)!=len(args.atlas):
        print "ERROR: length of image and atlas list is not equal ... exit"
        sys.exit()
    else:
        atlas_exits=True


all_atlas_mat = None
all_atlas_time_dim = {}
for index, scan in enumerate(args.img):
    print scan
    img = Image(img_filename=scan, mask_filename=args.mask[index], atlas_filename=args.atlas[index], fwhm=5)
    dim = img.image_mat_in_mask_atlas.shape[1]
    #mat = PCA(whiten=True).fit_transform(img.image_mat_in_mask_atlas).T
    mat = img.image_mat_in_mask_atlas.T
    if all_atlas_mat is None:
        all_atlas_mat = mat
        all_atlas_time_dim = dim
    else:
        all_atlas_mat = np.concatenate((all_atlas_mat, mat))
        all_atlas_time_dim = np.append(all_atlas_time_dim, dim)

all_atlas_mat = all_atlas_mat.T
all_atlas_mat=all_atlas_mat-all_atlas_mat.mean(axis=1)[:,None]
all_atlas_mat=all_atlas_mat-all_atlas_mat.std(axis=1)[:,None]
all_atlas_mat=all_atlas_mat.T

pca_group = PCA(whiten=True).fit_transform(all_atlas_mat.T)
graph_mat_2_atlas_nii(pca_group.T,args.atlasref,"group_atlas_pca.nii.gz")
ae_atlas = Autoencoder(args.n, max_iter=args.maxiter).fit(pca_group.T)
sio.savemat("group_atlas.mat", {'h':ae_atlas.h,'W1':ae_atlas.W1,'W2':ae_atlas.W2})
graph_mat_2_atlas_nii(ae_atlas.h,args.atlasref,"group_atlas.nii.gz")
import pdb; pdb.set_trace()
for index, scan in enumerate(args.img):
    print scan
    img = Image(img_filename=scan, mask_filename=args.mask[index], atlas_filename=args.atlas[index])
    pca = PCA(n_components=50,whiten=True).fit_transform(img.image_mat_in_mask_normalised_atlas)
    graph_mat_2_atlas_nii(pca.T,args.atlasref,(scan[0:scan.find('.nii')]+'.pca.nii.gz'))
    Wpca = np.concatenate((pca.T,pca_group.T));
    autoencoder = Autoencoder(args.n, max_iter=args.maxiter).fit(Wpca)
    graph_mat_2_atlas_nii(autoencoder.h,args.atlasref,(scan[0:scan.find('.nii')]+'.autoencoder.nii.gz'))
