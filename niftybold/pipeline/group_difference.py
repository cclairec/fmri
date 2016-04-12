import argparse
import sys
import numpy as np
import nibabel as nib
import pdb
import csv
import scipy.io as sio
from Autoencoder import Autoencoder
from Image import Image
import tools
from scipy.stats import ttest_ind


parser = argparse.ArgumentParser(description='fMRI group graph analysis hidden layer ')
parser.add_argument('-autoencoder', metavar='autoencoder', type=str, nargs='+',required=True)
parser.add_argument('-atlas', metavar='atlas', type=str, nargs='+',
                   help='# parcellation in image space')
parser.add_argument('-atlas_thr', metavar='atlas_thr', type=float,
                   help='# atlas threshold')
parser.add_argument('-meta', metavar='meta', type=str, help='csv file with scan ids and corresponding class information')
parser.add_argument('-atlasref', metavar='atlasref', type=str,
                    help='one atlas reference scan', default='brain_atlas.nii.gz')

args = parser.parse_args()

if args.meta:
    d, dc, dcn = tools.get_d_dc_dcn(args)

if len(args.atlas)==1:
    group_sum = np.zeros((53795, len(args.autoencoder), 30))
    all_labels = np.zeros((len(args.autoencoder)))
    for index, scan in enumerate(args.autoencoder):
        print scan
        first_point_index = scan.find('.')
        last_slash_index = scan.rfind('/')
        autoencoder_id = scan[last_slash_index+1:first_point_index]
        all_labels[index] = dcn[d[autoencoder_id]]
        decomp = Image(img_filename=scan, mask_filename=args.atlas[0], atlas_filename=args.atlas[0])
        group_sum[:,index,:] = decomp.image_mat_in_mask

    control_vs_ad_t = np.zeros((53795,30))
    control_vs_ad_p = np.zeros((53795,30))
    for i in xrange(53795):
        for c in xrange(30):
            t, p = ttest_ind(group_sum[i,all_labels==1,c], group_sum[i,all_labels==3,c], equal_var=False)
            control_vs_ad_t[i,c] = t
            control_vs_ad_p[i,c] = p

    # alpha = 0.05
    # for c in xrange(30):
    #     t_bool, pcorr = tools.fdr(control_vs_ad_p[:,c].ravel(),alpha=alpha)
    #     control_vs_ad_t[t_bool==False,c]=0
    #     control_vs_ad_p[:,c] = pcorr

    decomp.write_decomposition(maps=control_vs_ad_t.T, filename='_ica_control_vs_ad_ttest.nii.gz', normalise=False)
    decomp.write_decomposition(maps=control_vs_ad_p.T, filename='_ica_control_vs_ad_ttest_p.nii.gz', normalise=False)


else:
    ########################################################################################################################
    ########################################################################################################################
    ######################################## COMPUTE INDIVIDUAL AND GROUP DECOMPOSITION ####################################
    ########################################################################################################################
    ########################################################################################################################
    all_labels = np.zeros((len(args.autoencoder)))
    all_decompositions = np.zeros((len(args.autoencoder),98,30))
    for index, scan in enumerate(args.autoencoder):
        print "Using scan: "+scan+" with atlas: "+args.atlas[index]
        first_point_index = scan.find('.')
        last_slash_index = scan.rfind('/')
        autoencoder_id = scan[last_slash_index+1:first_point_index]
        label = dcn[d[autoencoder_id]]
        decomposition = Image(img_filename=scan, mask_filename=args.atlas[index], atlas_filename=args.atlas[index])
        all_labels[index] = label
        all_decompositions[index,:,:] = decomposition.image_mat_in_mask_atlas

    # Control vs AD
    control_vs_ad_t = np.zeros((98,30))
    control_vs_ad_p = np.zeros((98,30))
    for i in xrange(98):
        for c in xrange(30):
            t, p = ttest_ind(all_decompositions[all_labels==1,i,c], all_decompositions[all_labels==3,i,c], equal_var=False)
            control_vs_ad_t[i,c] = t
            control_vs_ad_p[i,c] = p


    alpha = 0.1
    for c in xrange(30):
        t_bool, pcorr = tools.fdr(control_vs_ad_p[:,c].ravel(),alpha=alpha)
        #control_vs_ad_t[t_bool==False,c]=0
        control_vs_ad_p[:,c] = pcorr

    tools.graph_mat_2_atlas_nii(control_vs_ad_t.T,args.atlasref, '_autoencoder_control_vs_ad_ttest.nii.gz')
    tools.graph_mat_2_atlas_nii(control_vs_ad_p.T,args.atlasref, '_autoencoder_control_vs_ad_ttest_p.nii.gz')

    # Control vs PCA
    control_vs_pca_t = np.zeros((98,30))
    control_vs_pca_p = np.zeros((98,30))
    for i in xrange(98):
        for c in xrange(30):
            t, p = ttest_ind(all_decompositions[all_labels==1,i,c], all_decompositions[all_labels==4,i,c], equal_var=False)
            control_vs_pca_t[i,c] = t
            control_vs_pca_p[i,c] = p


    for c in xrange(30):
        t_bool, pcorr = tools.fdr(control_vs_pca_p[:,c].ravel(),alpha=alpha)
        #control_vs_pca_t[t_bool==False,c]=0
        control_vs_pca_p[:,c] = pcorr

    tools.graph_mat_2_atlas_nii(control_vs_pca_t.T,args.atlasref, '_autoencoder_control_vs_pca_ttest.nii.gz')
    tools.graph_mat_2_atlas_nii(control_vs_pca_p.T,args.atlasref, '_autoencoder_control_vs_pca_ttest_p.nii.gz')
