# evs_mat = np.zeros((len(args.grouppca)))
# for index, scan in enumerate(args.grouppca):
#     pca_img = Image(img_filename=scan, mask_filename=args.groupmask, fwhm=args.fwhm)
#     raw_img = Image(img_filename=args.img[index], mask_filename=args.mask[index])
#     pca_mat = pca_img.image_mat_in_mask_normalised.T
#     raw_mat = raw_img.image_mat_in_mask_normalised.T
#     W = np.zeros((S_.shape[1],pca_mat.shape[0]))
#     regr = linear_model.LinearRegression()
#     for t in xrange(pca_mat.shape[0]):
#         regr.fit(S_, pca_mat[t,:].ravel())
#         W[:,t] = regr.coef_
#     mat_filename = (scan[0:scan.find('.nii')]+'.fastica.W.mat')
#     ica_group_filename = (scan[0:scan.find('.nii')]+'.fastica.nii.gz')
#     ica_filename = (scan[0:args.img[index].find('.nii')]+'.fastica.nii.gz')
#     sio.savemat(mat_filename, {'W':W})
#     regr.fit(W.T, pca_mat)
#     pca_img.write_decomposition(maps=regr.coef_.T,filename=ica_group_filename,normalise=True)
#     regr.fit(W.T, raw_mat)
#     raw_mat_tilde = regr.predict(W.T)
#     print "error: "+str(evs(raw_mat.ravel(),raw_mat_tilde.ravel()))
#     #with open((scan[0:args.img[index].find('.nii')]+'.fastica.evs.txt'), 'w') as f:
#     #f.write('%d' % evs(raw_mat.ravel(), raw_mat_tilde.ravel()))
#     evs_mat[index]=evs(raw_mat.ravel(), raw_mat_tilde.ravel())
#     raw_img.write_decomposition(maps=regr.coef_.T,filename=ica_filename,normalise=True)
#     print ica_filename
#
# sio.savemat("./stats/ica_evs.mat",{'evs':evs_mat})
