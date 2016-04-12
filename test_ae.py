import scipy.io as sio
from niftybold.decomposition.Autoencoder import Autoencoder
from niftybold.decomposition.RICA import RICA
from niftybold.decomposition.InfoMax import InfoMax
from niftybold.img.Image import Image

def test_ae(filename='/Volumes/VAULT/YOAD/all/01-001.fmri.rest.1.volreg.band.nii.gz', filename_mask='/Volumes/VAULT/YOAD/all/registrations/01-001.fmri.rest.1.volreg.cortical_mask.nii.gz'):

    img = Image(img_filename=filename, mask_filename=filename_mask)
    xmat = img.image_mat_in_mask_normalised.T
    ae = Autoencoder(30, max_iter=200, second_nonlinear=True, bias=True).fit(xmat)
    img.write_decomposition(maps=ae.components_, filename='test_ae.nii.gz', normalise=False)

def test_infomax(filename='./data/01-001.fmri.rest.1.volreg.band.pca_reduced.nii.gz',filename_mask='./data/01-001.fmri.rest.1.volreg.cortical_mask.nii.gz'):
    from Image import Image
    from sklearn.decomposition import FastICA
    import scipy.io as sio
    img = Image(img_filename=filename, mask_filename=filename_mask, fwhm=6.0)
    xmat = img.image_mat_in_mask_normalised.T
    from InfoMax import InfoMax
    infomax = InfoMax(30).fit(xmat)
    img.write_decomposition(maps=infomax.components_, filename='test_ica_infomax.nii.gz', normalise=True)

def test_rica(filename='./data/01-001.fmri.rest.1.volreg.nii.gz', filename_mask='./data/01-001.fmri.rest.1.volreg.cortical_mask.nii.gz'):
    from Image import Image
    import scipy.io as sio
    img = Image(img_filename=filename, mask_filename=filename_mask)
    xmat = img.image_mat_in_mask_normalised.T
    from sklearn.decomposition import PCA, FastICA
    pca = PCA(30)
    xmat_red = pca.fit_transform(xmat.T)
    import matplotlib.pyplot as plt
    plt.plot(pca.components_.T)
    plt.show()
    import pdb; pdb.set_trace()
    rica = RICA(30).fit(xmat)
    img.write_decomposition(maps=rica.components_, filename='test_rica.nii.gz', normalise=False)

if __name__ == '__main__':test_ae()
