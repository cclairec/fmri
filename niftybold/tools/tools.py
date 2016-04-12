import argparse
import sys
import numpy as np
import nibabel as nib
import pdb
import csv
import scipy.io as sio
from Autoencoder import Autoencoder
from Image import Image

def get_d_dc_dcn(args):
    d={}
    with open(args.meta, 'rb') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
        for index,row in enumerate(reader):
            #pdb.set_trace()
            if index!=0:
                    d[row[0]]=row[1]

    dc=count_pdf(d.values())
    dcn={}
    for i,e in enumerate(dc.keys()):
        dcn[e]=i+1

    return d, dc, dcn

def count_pdf(classes_list):
    d={}
    for c in set(classes_list):
        d[c]=0
        for item in classes_list:
            if item==c:
                d[c]=d[c]+1

    return d

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

def fdr(pvals, alpha=0.05, method='indep'):
    '''pvalue correction for false discovery rate

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests. Both are
    available in the function multipletests, as method=`fdr_bh`, resp. `fdr_by`.

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : {'indep', 'negcorr')

    Returns
    -------
    rejected : array, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----

    If there is prior information on the fraction of true hypothesis, then alpha
    should be set to alpha * m/m_0 where m is the number of tests,
    given by the p-values, and m_0 is an estimate of the true hypothesis.
    (see Benjamini, Krieger and Yekuteli)

    The two-step method of Benjamini, Krieger and Yekutiel that estimates the number
    of false hypotheses will be available (soon).

    Method names can be abbreviated to first letter, 'i' or 'p' for fdr_bh and 'n' for
    fdr_by.



    '''
    pvals = np.asarray(pvals)
    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()
    ecdffactor = np.arange(1,len(pvals_sorted)+1)/float(len(pvals_sorted))
    reject = pvals_sorted < ecdffactor*alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected>1] = 1
    return reject[sortrevind], pvals_corrected[sortrevind]
    #return reject[pvals_sortind.argsort()]
