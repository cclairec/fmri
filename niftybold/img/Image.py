import os
import sys
import numpy as np
# to load fmri nifti file
import nibabel as nib
# from nibabel import processing
import pdb

class Image(object):
    """ Load image, mask it and hold the remaining voxels in the mask for future processing"""
    def __init__(self,img_filename,mask_filename,noise_mask_filename=None,atlas_filename=None,segmentation_filename=None,atlas_thr=100,atlas_exclsion=None,fwhm=0):
        #load imaging data
        print atlas_exclsion
        self._img_filename = img_filename
        self._image_hd = nib.load(img_filename)
  #      if fwhm>0:
  #          self._image_hd = processing.smooth_image(self._image_hd,fwhm)

        self._mask_hd = nib.load(mask_filename)
        self._mask_volume = self._mask_hd.get_data()
        self._image_volume = self._image_hd.get_data()
        self._dims = self._image_volume.shape
        self._mask_v = self._mask_volume.reshape(self._dims[0]*self._dims[1]*self._dims[2])
        self._image_mat = self._image_volume.reshape(self._dims[0]*self._dims[1]*self._dims[2],self._dims[3])


        # if seg is specified, use grey matter mask instead
        if segmentation_filename is not None:
            self._segmentation_filename = segmentation_filename
            self._segmentation_hd = nib.load(segmentation_filename)
            self._segmentation_volume = self._segmentation_hd.get_data()
            self._mask_v = self._segmentation_volume[:,:,:,2].reshape(self._dims[0]*self._dims[1]*self._dims[2])

        self._image_mat_in_mask = self._image_mat[self._mask_v[:]>0,:].astype(float)
        self._dims_mat = self._image_mat_in_mask.shape
        # demean and normalise
        self._image_mat_in_mask = self._image_mat_in_mask-self._image_mat_in_mask.mean(axis=1)[:,None]
        self._image_mat_in_mask_normalised=self._image_mat_in_mask/self._image_mat_in_mask.std(axis=1)[:,None]
        where_are_NaNs = np.isnan(self._image_mat_in_mask_normalised)
        self._image_mat_in_mask_normalised[where_are_NaNs] = 0

        if atlas_filename is not None:
            self._atlas_filename = atlas_filename
            self._atlas_hd = nib.load(atlas_filename)
            self._atlas_volume = self._atlas_hd.get_data()
            self._atlas_v = self._atlas_volume.reshape(self._dims[0]*self._dims[1]*self._dims[2])
            self._atlas_values_unique = np.unique(self._atlas_v)
            self._atlas_thr = atlas_thr
            self._atlas_values_unique = self._atlas_values_unique[self._atlas_values_unique >= self._atlas_thr]
            self._atlas_indices_dict = {}
            self._atlas_v_reduced = self._mask_v[self._mask_v>0]
            self._atlas_values_unique = [item for item in self._atlas_values_unique if item not in atlas_exclsion]
            self._image_mat_in_mask_normalised_atlas = np.zeros((len(self._atlas_values_unique), self._dims[3]))
            self._image_mat_in_mask_atlas = np.zeros((len(self._atlas_values_unique), self._dims[3]))
            # compute atlas-wise scan
            for index, mask_val in enumerate(self._atlas_values_unique):
                print mask_val, index
                self._atlas_indices_dict[mask_val] = indices(self._atlas_v_reduced, lambda x: x == mask_val)
                self._image_mat_in_mask_atlas[index,:] = self._image_mat_in_mask[self._atlas_indices_dict[mask_val],:].mean(axis=0)
            self._image_mat_in_mask_normalised_atlas=self._image_mat_in_mask_atlas-self._image_mat_in_mask_atlas.mean(axis=1)[:,None]
            self._image_mat_in_mask_normalised_atlas=self._image_mat_in_mask_normalised_atlas/self._image_mat_in_mask_normalised_atlas.std(axis=1)[:,None]
            where_are_NaNs = np.isnan(self._image_mat_in_mask_normalised_atlas)
            self._image_mat_in_mask_normalised_atlas[where_are_NaNs] = 0
            self._corr = np.corrcoef(self._image_mat_in_mask_normalised_atlas)

        if noise_mask_filename is not None: # create volume, do temporal PCA and regress
            self._noise_mask_filename = noise_mask_filename
            self._noise_mask_hd = nib.load(noise_mask_filename)
            self._noise_mask_volume = self._noise_mask_hd.get_data()
            self._noise_mask_v = self._noise_mask_volume.reshape(self._dims[0]*self._dims[1]*self._dims[2])
            # todo noise removal from tPCA

    @property
    def atlas_corr(self):
        return self._corr
    @property
    def image_mat_in_mask(self):
        return self._image_mat_in_mask

    @property
    def image_mat_in_mask_atlas(self):
        return self._image_mat_in_mask_atlas

    @property
    def image_mat_in_mask_normalised(self):
        return self._image_mat_in_mask_normalised

    @property
    def image_mat_in_mask_normalised_atlas(self):
        return self._image_mat_in_mask_normalised_atlas

    @property
    def img_filename(self):
        return self._img_filename

    @property
    def atlas_v(self):
        return self._atlas_v

    @property
    def atlas_indices_dict(self):
        return self._atlas_indices_dict

    def write_decomposition(self,maps,timecourses=None,filename="maps.nii.gz",normalise=True):
        if normalise:
            maps=maps-maps.mean(axis=1)[:,None]
            maps=maps/maps.std(axis=1)[:,None]

        dim_features=maps.shape[0]
        maps_mat = np.zeros((self._dims[0]*self._dims[1]*self._dims[2], dim_features))
        maps_mat[self._mask_v[:]>0,:]=maps.T
        maps_volumes = np.reshape(maps_mat,(self._dims[0], self._dims[1], self._dims[2], dim_features))
        img = nib.Nifti1Image(maps_volumes, self._image_hd.get_affine()) # save map in same orientation as original image
        nib.save(img, filename)
        #print "save "+filename

        if hasattr(self, '_atlas_v'):
            graph_mat = np.zeros((self._atlas_values_unique.shape[0],dim_features))
            for index,mask_val in enumerate(self._atlas_values_unique):
                graph_mat[index,:] = maps_mat[self._atlas_v==mask_val,:].mean(axis=0)

            np.savetxt((filename[0:filename.find('.nii')]+'.'+'graph_mat.txt'), graph_mat)
            #print "save "+(filename[0:filename.find('.nii')]+'.'+'graph_mat.txt')


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]
