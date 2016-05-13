from niftybold.decomposition.Autoencoder import Autoencoder
from niftybold.decomposition.RICA import RICA
from niftybold.decomposition.InfoMax import InfoMax
from niftybold.img.Image import Image
from niftybold.tools.atlas import atlas
import argparse
import scipy.io as sio
from subprocess import call,check_output
from os.path import isfile
import numpy as np
import xlwt

################################################## NOTES ##################################
# - use 3drefit -TR to set TR in header if not correctly set
###########################################################################################
def fmri(fmri_collection, t1_collection=None, seg_collections=None, atlas_collections=None):
    directory_reg = ''.join([fmri_collection[0][0:fmri_collection[0].rfind('/') + 1],'/reg/'])
    call("mkdir -p {directory_reg}".format(directory_reg=directory_reg), shell=True)
    for index, fmri_scan in enumerate(fmri_collection):
        id_start = fmri_scan.rfind('/') + 1
        id_end = fmri_scan.find('.')
        directory = fmri_scan[0:id_start]
        identifier = fmri_scan[id_start:id_end]
        # distortion correction if still required - deprecated
        VOLUMES = check_output('3dinfo -nv {scan}'.format(scan=fmri_scan), shell=True).rstrip()
        TR = check_output('3dinfo -tr {scan}'.format(scan=fmri_scan), shell=True).rstrip()
        print "scan: {scan} TR={TR} VOLUMES={VOLUMES}".format(scan=identifier, TR=TR, VOLUMES=VOLUMES)
        ##########################################################################################################
        ################################# FILE SETUP #############################################################
        ##########################################################################################################
        ### 3ddespike
        in_3ddespike = fmri_scan
        out_3ddespike = "{directory}{fmri_id}.fmri.despike.nii.gz".format(fmri_id=identifier, directory=directory)
        ### 3dvolreg
        in_3dvolreg = out_3ddespike
        out_3dvolreg = "{prefix}.volreg.nii.gz".format(prefix=in_3dvolreg[:-7])
        out_3dvolreg_txt = "{directory}{fmri_id}.motion.1D".format(fmri_id=identifier, directory=directory)
        out_motion = "{directory}{fmri_id}.motion.demean.1D".format(fmri_id=identifier, directory=directory)
        out_motion_derv = "{directory}{fmri_id}.motion.demean.derivative.1D".format(fmri_id=identifier, directory=directory)
        ### 3dvol to extract 4th volume
        in_3dvol = out_3dvolreg
        out_3dvol = "{prefix}.vol4.nii.gz".format(prefix=in_3dvol[:-7])
        # 3dBandpass
        in_3dBandpass = out_3dvolreg
        out_3dBandpass = "{prefix}.band.nii.gz".format(prefix=in_3dBandpass[:-7])
        out_3dBandpass_regressors = "{directory}{fmri_id}.bandpass.1D".format(fmri_id=identifier, directory=directory)
        # 3dDeconvolve
        in_3dDeconvolve = out_3dvolreg
        out_3dDeconvolve_err = "{prefix}.deconvolve.err.nii.gz".format(prefix=in_3dDeconvolve[:-7])
        out_3dDeconvolve_fit = "{prefix}.deconvolve.fit.nii.gz".format(prefix=in_3dDeconvolve[:-7])
        out_3dDeconvolve_stats = "{prefix}.deconvolve.stats.nii.gz".format(prefix=in_3dDeconvolve[:-7])
        # final output
        fmri_preprocessed = out_3dDeconvolve_err

        t1 = args.t1[index]
        seg = args.seg[index]
        parcellation = args.atlas[index]
        fvol = out_3dvol
        aff_t1_2_fmri = "{directory_reg}{fmri_id}.t1__2__{fmri_id}.fmri.despike.volreg.vol4.txt".format(
            fmri_id=identifier, directory_reg=directory_reg)
        aff_fmri_2_t1 = "{directory_reg}{fmri_id}.fmri.despike.volreg.vol4__2__{fmri_id}.t1.txt".format(
            fmri_id=identifier, directory_reg=directory_reg)
        aff_t1_in_fmri ="{directory_reg}{fmri_id}.t1__in__{fmri_id}.fmri.despike.volreg.vol4.nii.gz".format(
            fmri_id=identifier, directory_reg=directory_reg)
        aff_fmri_in_t1 = "{directory_reg}{fmri_id}.fmri.despike.volreg.vol4__in__{fmri_id}.t1.nii.gz".format(
            fmri_id=identifier, directory_reg=directory_reg)

        fmri_seg = "{directory}{fmri_id}.fmri.despike.volreg.vol4.seg.nii.gz".format(
            directory=directory, fmri_id=identifier)
        fmri_atlas = "{directory}{fmri_id}.fmri.despike.volreg.vol4.atlas.nii.gz".format(
            directory=directory, fmri_id=identifier)
        ##########################################################################################################
        ################################# FMRI PREPROCESSING #####################################################
        ##########################################################################################################
        # despiking
        if not isfile(out_3ddespike):
            cmd = "3dDespike -NEW -nomask -prefix {out_} {in_}".format(in_=in_3ddespike, out_=out_3ddespike)
            print cmd
            call(cmd, shell=True)

        # motion realignment
        if not isfile(out_3dvolreg):
            cmd = "3dvolreg -overwrite -zpad 1 -cubic -prefix {out_} -1Dfile {out_txt} {in_}".format(out_txt=out_3dvolreg_txt, in_=in_3dvolreg, out_=out_3dvolreg)
            print cmd
            call(cmd, shell=True)
            cmd = "1d_tool.py -infile {in_} -set_nruns 1 -demean -write {out_}".format(in_=out_3dvolreg_txt, out_=out_motion)
            print cmd
            call(cmd, shell=True)
            cmd = "1d_tool.py -infile {in_} -set_nruns 1 -derivative -demean -write {out_}".format(in_=out_3dvolreg_txt, out_=out_motion_derv)
            print cmd
            call(cmd, shell=True)

        # get one volume
        if not isfile(out_3dvol):
            cmd = "3dcalc -overwrite -a '{in_}[4]' -prefix {out_} -expr a".format(in_=in_3dvol, out_=out_3dvol)
            call(cmd, shell=True)

        # band passs filter signals
        if not isfile(out_3dBandpass_regressors):
            #cmd = "3dBandpass -overwrite -band 0.01 0.1 -prefix {out_} {in_}".format(in_=in_3dBandpass, out_=out_3dBandpass)
            cmd = "1dBport -nodata {VOLUMES} {TR} -band 0.01 0.1 -invert -nozero > {out_}".format(VOLUMES=VOLUMES, TR=TR, out_=out_3dBandpass_regressors)
            print cmd
            call(cmd, shell=True)

        if not isfile(out_3dDeconvolve_err):
            cmd = """3dDeconvolve -input {in_}                             \
            -ortvec {bandpass} bandpass                                        \
            -polort 3                                                                \
            -num_stimts 12                                                           \
            -stim_file 1 {motion}'[0]' -stim_base 1 -stim_label 1 roll_01    \
            -stim_file 2 {motion}'[1]' -stim_base 2 -stim_label 2 pitch_01   \
            -stim_file 3 {motion}'[2]' -stim_base 3 -stim_label 3 yaw_01     \
            -stim_file 4 {motion}'[3]' -stim_base 4 -stim_label 4 dS_01      \
            -stim_file 5 {motion}'[4]' -stim_base 5 -stim_label 5 dL_01      \
            -stim_file 6 {motion}'[5]' -stim_base 6 -stim_label 6 dP_01      \
            -stim_file 7 {motion_derv}'[0]' -stim_base 7 -stim_label 7 roll_02     \
            -stim_file 8 {motion_derv}'[1]' -stim_base 8 -stim_label 8 pitch_02    \
            -stim_file 9 {motion_derv}'[2]' -stim_base 9 -stim_label 9 yaw_02      \
            -stim_file 10 {motion_derv}'[3]' -stim_base 10 -stim_label 10 dS_02    \
            -stim_file 11 {motion_derv}'[4]' -stim_base 11 -stim_label 11 dL_02    \
            -stim_file 12 {motion_derv}'[5]' -stim_base 12 -stim_label 12 dP_02    \
            -fitts {out_fit}                                                     \
            -errts {out_err}                                                     \
            -bucket {out_stats} """.format(in_=in_3dDeconvolve, bandpass=out_3dBandpass_regressors, motion=out_motion, motion_derv=out_motion_derv, out_fit=out_3dDeconvolve_fit, out_err=out_3dDeconvolve_err, out_stats=out_3dDeconvolve_stats)
            call(cmd, shell=True)

        #########################################################################################################################################
        ################################################### AFFINE REGISTRATIONS ################################################################
        #########################################################################################################################################
        #(i) fmri to t1
        if t1_collection and not isfile(aff_fmri_in_t1):
            cmd = "reg_aladin -ref {ref} -flo {flo} -noSym -res {res} -aff {aff}".format(ref=t1, flo=fvol, res=aff_fmri_in_t1, aff=aff_fmri_2_t1)
            print cmd
            call(cmd, shell=True)

        #(ii) t1 to fmri
        if t1_collection and not isfile(aff_t1_in_fmri):
            cmd = "reg_aladin -ref {ref} -flo {flo} -noSym -res {res} -aff {aff}".format(ref=fvol, flo=t1, res=aff_t1_in_fmri, aff=aff_t1_2_fmri)
            print cmd
            call(cmd, shell=True)

        # resample seg (5 tissue classes) in fmri space
        if seg_collections and not isfile(fmri_seg):
            cmd = "reg_resample -ref {ref} -flo {flo} -psf -res {res} -trans {trans}".format(ref=fvol, flo=seg, res=fmri_seg, trans=aff_t1_2_fmri)
            print cmd
            call(cmd, shell=True)

        # resample atlas (parcellations) in fmri space
        if atlas_collections and not isfile(fmri_atlas):
            cmd = "reg_resample -ref {ref} -flo {flo} -inter 0 -res {res} -trans {trans}".format(ref=fvol, flo=parcellation, res=fmri_atlas, trans=aff_t1_2_fmri)
            print cmd
            call(cmd, shell=True)
            # build atlas based correlation matrices
            ventricles_csf = [1, 5, 12, 16, 50, 51]
            exterior_lesions = [43, 44, 54, 55, 64, 65, 70]
            cerebellum = [72, 73, 74] # exclude for Genfi DF1
            atlas_exclusion_list = ventricles_csf+exterior_lesions+cerebellum
            img = Image(img_filename=fmri_preprocessed,
                        mask_filename=fmri_atlas,
                        atlas_filename=fmri_atlas,
                        atlas_thr=1,
                        atlas_exclsion=atlas_exclusion_list)
            sio.savemat("{prefix}.corr.mat".format(prefix=fmri_preprocessed[:-7]),{'CorrMatrix':img.atlas_corr})
            sio.savemat("{prefix}.rois.mat".format(prefix=fmri_preprocessed[:-7]),{'ROISignals':img.image_mat_in_mask_normalised_atlas})
            # adding xls file for brain areas kept, for graphVar use.
            book = xlwt.Workbook()
            sh = book.add_sheet("sheet1")
            n = 0
            for label, name in atlas.items():
                if label in atlas_exclusion_list:
                    sh.write(n, 0, 0)
                else:
                    sh.write(n, 0, 1)

                sh.write(n, 1, label)
                sh.write(n, 2, name)
                n = n+1

            book.save(fmri_preprocessed[:-7]+"BrainRegions.xls")

            labels = np.unique(img.atlas_v)
            labels = labels[labels>0]
            sio.savemat("{prefix}.rois_label.mat".format(prefix=fmri_preprocessed[:-7]), {'ROILabels':labels})


def relate_scans(fmri_collection, t1_collection, t1_template):
    directory_reg = ''.join([fmri_collection[0][0:fmri_collection[0].rfind('/') + 1], 'reg'])
    directory = ''.join([fmri_collection[0][0:fmri_collection[0].rfind('/') + 1]])
    number_of_nrr = 10
    number_of_aff = 10
    #(iii) create parameter file for group registration and start group registration
    if not isfile(''.join([directory_reg,'groupwise_niftyreg_params.sh'])):
        f = open(''.join([directory_reg,'groupwise_niftyreg_params.sh']),'w')
        f.write('#!/bin/sh\n')
        f.write(''.join(['export IMG_INPUT=(`ls ', directory, '*.t1.nii*`)', '\n']))
        f.write(''.join(['export TEMPLATE=', t1_template,'\n']))
        f.write(''.join(['export RES_FOLDER=', directory_reg, '\n']))
        f.write('export AFFINE_args=""\n')
        f.write('export NRR_args=""\n')
        f.write('export AFF_IT_NUM={number_of_aff}\n'.format(number_of_aff=number_of_aff))
        f.write('export NRR_IT_NUM={number_of_nrr}\n'.format(number_of_nrr=number_of_nrr))
        f.close()
        cmd = ''.join(['groupwise_niftyreg_run.sh ', directory_reg, 'groupwise_niftyreg_params.sh'])
        call(cmd, shell=True)

        # downsample template
        cmd = ''.join(['reg_tools -chgres 3 3 3 -in ', directory_reg, 'nrr_{number_of_nrr}'.format(number_of_nrr=number_of_nrr), '/', 'average_nonrigid_it_{number_of_nrr}.nii.gz'.format(number_of_nrr=number_of_nrr), ' -out ', directory_reg, '/average_nonrigid_it_{number_of_nrr}_fmri.nii.gz'.format(number_of_nrr=number_of_nrr)])
        call(cmd, shell=True)

    # (iiii) build final registration
    # reg_transform -ref $group_avg -ref2 $anat_img -comp ${PP_DIR}/fmri2anat.txt ${PP_DIR}/anat2group.nii.gz ${PP_DIR}/fmri2group.nii.gz
    for index, fmri_scan in enumerate(fmri_collection):
        id_start = fmri_scan.rfind('/') + 1
        id_end = fmri_scan.find('.')
        directory = fmri_scan[0:id_start]
        identifier = fmri_scan[id_start:id_end]
        if not isfile('{directory}{fmri_id}.fmri.despike.volreg.deconvolve.err.group.nii.gz'.format(fmri_id=identifier, directory=directory)):
            cmd = 'reg_transform -ref {directory_reg}/average_nonrigid_it_{number_of_nrr}_fmri.nii.gz -ref2 {t1} -comp {directory_reg}{fmri_id}.fmri.despike.volreg.vol4__2__{fmri_id}.t1.txt {directory_reg}/nrr_{number_of_nrr}/nrr_cpp_{fmri_id}.t1_it{number_of_nrr}.nii.gz {directory_reg}/{fmri_id}.fmri.despike.volreg.vol4__2__average_nonrigid_it_{number_of_nrr}_fmri.nii.gz'.format(fmri_id=identifier, t1=t1_collection[index], directory_reg=directory_reg, number_of_nrr=number_of_nrr)
            call(cmd, shell=True)
            cmd = 'reg_resample -ref {directory_reg}/average_nonrigid_it_{number_of_nrr}_fmri.nii.gz -flo {directory}{fmri_id}.fmri.despike.volreg.vol4.nii.gz -trans {directory_reg}/{fmri_id}.fmri.despike.volreg.vol4__2__average_nonrigid_it_{number_of_nrr}_fmri.nii.gz -res {directory}{fmri_id}.fmri.despike.volreg.vol4.group.nii.gz'.format(fmri_id=identifier, t1=t1_collection[index], directory_reg=directory_reg, number_of_nrr=number_of_nrr, directory=directory)
            call(cmd, shell=True)
            cmd = 'reg_resample -ref {directory_reg}/average_nonrigid_it_{number_of_nrr}_fmri.nii.gz -flo {directory}{fmri_id}.fmri.despike.volreg.deconvolve.err.nii.gz -trans {directory_reg}/{fmri_id}.fmri.despike.volreg.vol4__2__average_nonrigid_it_{number_of_nrr}_fmri.nii.gz -res {directory}{fmri_id}.fmri.despike.volreg.deconvolve.err.group.nii.gz'.format(fmri_id=identifier, t1=t1_collection[index], directory_reg=directory_reg, number_of_nrr=number_of_nrr, directory=directory)
            call(cmd, shell=True)


parser = argparse.ArgumentParser(description='fMRI preprocessing and analysis')
parser.add_argument('-fmri', metavar='fmri', type=str, nargs='+', required=True)
parser.add_argument('-t1', metavar='t1', type=str, nargs='+', required=False)
parser.add_argument('-seg', metavar='seg', type=str, nargs='+', required=False)
parser.add_argument('-atlas', metavar='atlas', type=str, nargs='+', required=False)
args = parser.parse_args()

t1_template = check_output('echo $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz', shell=True).rstrip()

fmri(args.fmri, args.t1, args.seg, args.atlas)
relate_scans(args.fmri, args.t1, t1_template)
