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
from nilearn import plotting
from PIL import Image as imagemaker

parser = argparse.ArgumentParser(description='ATLAS VOLUME ANALYSIS')
parser.add_argument('-anat', metavar='anat', type=str, nargs='+')
parser.add_argument('-ica', metavar='ica', type=str, nargs='+')
parser.add_argument('-thr', metavar='thr', type=float, default=0.5)
parser.add_argument('-max', metavar='max', type=float, default=1.0)
parser.add_argument('-meta', metavar='meta', type=str, help='csv file with scan ids and corresponding class information')
parser.add_argument('-network', metavar='network', type=int, default=0)
parser.add_argument('-caption', metavar='caption', type=str, default=None)
args = parser.parse_args()

if args.meta:
    d, dc, dcn = tools.get_d_dc_dcn(args)

for index, scan in enumerate(args.anat):
    print index
    first_point_index = scan.find('.')
    last_slash_index = scan.rfind('/')
    autoencoder_id = scan[last_slash_index+1:first_point_index]
    if args.meta:
        label = dcn[d[autoencoder_id]]
    anat_img = nib.load(scan)
    anat = anat_img.get_data()
    fmri_img = nib.load(args.ica[index])
    fmri = fmri_img.get_data()
    # Plotting: vary the 'dim' of the background
    anat = nib.Nifti1Image(anat, np.eye(4))
    dims = anat.shape
    x = dims[0]/2
    y = dims[1]/2
    z = 80#dims[2]/2
    fmri[fmri<=0]=0
    #fmri[:,:,:,args.network] = (fmri[:,:,:,args.network] - fmri[:,:,:,args.network].mean())/fmri[:,:,:,args.network].std()
    tmap = nib.Nifti1Image(fmri[:,:,:,args.network], np.eye(4))

    img_name = args.ica[index][0:args.ica[index].find('.nii')]+'.network'+str(args.network)+'.png'

    if not args.caption:
        caption = autoencoder_id+' '+d[autoencoder_id]
    else:
        caption = args.caption

    plotting.plot_stat_map(tmap,
                           bg_img=anat,
                           cut_coords=(x, y, z),
                           threshold=args.thr, title=caption,
                           dim=0, output_file=img_name,vmax=args.max,draw_cross=False)


# img_arrary = ["autoencoder1.network25.png",
#              '/Volumes/VAULT/YOAD/all/01-001.fmri.rest.2.volreg.band.pca_reduced.fwhm0mm._backpropagation_groupbias.network25.png',
#              '/Volumes/VAULT/YOAD/all/01-004.fmri.rest.2.volreg.band.pca_reduced.fwhm0mm._backpropagation_groupbias.network25.png',
#              '/Volumes/VAULT/YOAD/all/01-005.fmri.rest.2.volreg.band.pca_reduced.fwhm0mm._backpropagation_groupbias.network25.png',
#              "autoencoder1.network21.png",
#              '/Volumes/VAULT/YOAD/all/01-001.fmri.rest.2.volreg.band.pca_reduced.fwhm0mm._backpropagation_groupbias.network21.png',
#              '/Volumes/VAULT/YOAD/all/01-004.fmri.rest.2.volreg.band.pca_reduced.fwhm0mm._backpropagation_groupbias.network21.png',
#              '/Volumes/VAULT/YOAD/all/01-005.fmri.rest.2.volreg.band.pca_reduced.fwhm0mm._backpropagation_groupbias.network21.png',
#              "autoencoder1.network9.png",
#              '/Volumes/VAULT/YOAD/all/01-001.fmri.rest.2.volreg.band.pca_reduced.fwhm0mm._backpropagation_groupbias.network9.png',
#              '/Volumes/VAULT/YOAD/all/01-004.fmri.rest.2.volreg.band.pca_reduced.fwhm0mm._backpropagation_groupbias.network9.png',
#              '/Volumes/VAULT/YOAD/all/01-005.fmri.rest.2.volreg.band.pca_reduced.fwhm0mm._backpropagation_groupbias.network9.png',]
#
# #creates a new empty image, RGB mode, and size 400 by 400.
# new_im = imagemaker.new('RGB', (250*3,100*4))
# #Here I resize my opened image, so it is no bigger than 100,100
# #Iterate through a 4 by 4 grid with 100 spacing, to place my image
# count = 0
# for i in xrange(0,250*3,250):
#     for j in xrange(0,4*100,100):
#         im = imagemaker.open(img_arrary[count])
#         im.thumbnail((250,100))
#         #paste the image at location i,j:
#         new_im.paste(im, (i,j))
#         count += 1
#
# #new_im.show()
# new_im.save('out.bmp')
