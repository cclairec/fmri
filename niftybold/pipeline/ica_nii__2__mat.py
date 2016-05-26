from niftybold.img.Image import Image

import scipy.io as sio
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='build mat file from nifti file')
parser.add_argument('-img', metavar='img', type=str, required=True)
parser.add_argument('-mask', metavar='mask', type=str, required=True)
parser.add_argument('-mat', metavar='mat', type=str, required=True)
parser.add_argument('-normalise', metavar='normalises', type=bool, default=True)
parser.add_argument('-asl', metavar='asl', type=bool, default=False)
parser.add_argument('-fwhm', metavar='fwhm', type=float, default=0.0)
args = parser.parse_args()

img = Image(img_filename=args.img, mask_filename=args.mask, fwhm=args.fwhm, asl=args.asl)
if args.normalise:
    _mat = img.image_mat_in_mask_normalised.T

else:
    _mat = img.image_mat_in_mask.T

sio.savemat(args.mat, {'mat': _mat})
