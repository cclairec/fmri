from niftybold.img.Image import Image

import scipy.io as sio
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='build mat file for matrix decomposition')
parser.add_argument('-groupimg', metavar='groupimg', type=str, nargs='+', required=True)
parser.add_argument('-groupmask', metavar='groupmask', type=str, required=True)
parser.add_argument('-fwhm', metavar='fwhm', type=float, default=5.0)
parser.add_argument('-mat', metavar='mat', type=str, default='ica_ready.mat')
args = parser.parse_args()

mat = None
for index, scan in enumerate(args.groupimg):
    print scan
    img = Image(img_filename=scan, mask_filename=args.groupmask, fwhm=args.fwhm)
    _mat = img.image_mat_in_mask_normalised.T
    sio.savemat("{prefix}.mat".format(prefix=scan[0:scan.find('.nii')]), {'mat': _mat})
    if index==0:
        mat = _mat
    else:
        mat = np.concatenate((mat, _mat))

sio.savemat(args.mat, {'mat': mat})
