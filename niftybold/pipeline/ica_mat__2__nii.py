from niftybold.img.Image import Image

import scipy.io as sio
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='build nifti file from mat file')
parser.add_argument('-img', metavar='img', type=str, required=True)
parser.add_argument('-mask', metavar='mask', type=str, required=True)
parser.add_argument('-mat', metavar='mat', type=str, required=True)
parser.add_argument('-normalise', metavar='normalises', type=bool, default=False)
args = parser.parse_args()
f = sio.loadmat(args.mat)
hiddenlayer = f['h']
img = Image(img_filename=args.img, mask_filename=args.mask)
img.write_decomposition(maps=hiddenlayer, filename="{prefix}.nii.gz".format(prefix=args.mat[0:args.mat.find('.mat')]), normalise=args.normalise)
