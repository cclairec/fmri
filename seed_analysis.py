import argparse
from subprocess import call,check_output
import numpy as np
import sys
from os.path import isfile
# test case epicure: %run seed_analysis -fmri /Volumes/VAULT/epicure_test/*err.group.nii.gz -fmri_groupspace /Volumes/VAULT/epicure_test/reg/average_nonrigid_it_10_fmri.nii.gz -fmri_mask /Volumes/VAULT/epicure_test/reg/average_nonrigid_it_10_fmri_mask.nii.gz -seed_name PCC -seed_mask /Volumes/VAULT/epicure_test/test_seed.nii.gz -seed_coord "-2 40 -2" -seed_radius 5
# test case FTD: %run seed_analysis -fmri /Users/me/_WORKING_MEMORY/FTD/all/*.err.group.nii.gz -fmri_groupspace /Users/me/_WORKING_MEMORY/FTD/reg/average_nonrigid_it_10_fmri.nii.gz -fmri_mask /Users/me/_WORKING_MEMORY/FTD/reg/average_nonrigid_it_10_fmri_mask.nii.gz -seed_name PCC -seed_mask /Users/me/_WORKING_MEMORY/FTD/seed/pcc_seed.nii.gz -seed_coord "30 23 36" -seed_radius 5
# be careful with the sign of the seed coordinates (depending on orientation) abd double check seed in group space
parser = argparse.ArgumentParser(description='fMRI group seed analyis')
parser.add_argument('-fmri', metavar='fmri', type=str, nargs='+', required=True)
parser.add_argument('-fmri_mask', metavar='fmri_mask', type=str, required=True)
parser.add_argument('-fmri_groupspace', metavar='fmri_groupspace', type=str, required=True)
parser.add_argument('-seed_coord', metavar='seed_coord', type=str)
parser.add_argument('-seed_radius', metavar='seed_radius', type=str)
parser.add_argument('-seed_mask', metavar='seed_mask', type=str, required=True)
parser.add_argument('-seed_name', metavar='seed_name', type=str, required=True)
parser.add_argument('-groupA', metavar='groupA', type=str, required=True)
parser.add_argument('-groupB', metavar='groupB', type=str, required=True)
parser.add_argument('-stats', metavar='stats', type=str, required=True)
args = parser.parse_args()

# create ROI in MNI space or provide file with ROI mask
seed_mask = args.seed_mask
if args.seed_radius and args.seed_coord:
    seed_mask_coord_file = "{prefix}.xyz.txt".format(prefix=seed_mask[:-7])
    cmd = "echo {seed_coord} > {file_}".format(seed_coord=args.seed_coord, file_=seed_mask_coord_file)
    print cmd
    call(cmd, shell=True)
    cmd = "3dUndump -overwrite -prefix {out_} -master {in_} -srad {seed_radius} -xyz {file_}".format(out_=seed_mask, in_=args.fmri_groupspace, seed_radius=args.seed_radius, file_=seed_mask_coord_file)
    print cmd
    call(cmd, shell=True)

    # double check if the seed is in the right place before doing the analysis
    cmd = "fslview {anat} {overlay}".format(anat=args.fmri_groupspace, overlay=seed_mask)
    print cmd
    call(cmd, shell=True)
    answer = raw_input("Would you like to continue (yes or no): ")
    if not (answer.startswith( 'y' ) or answer.startswith( 'yes' )):
        sys.exit() # stop programm if you would like to redo your seed mask

for index, fmri_scan in enumerate(args.fmri):
    id_start = fmri_scan.rfind('/') + 1
    id_end = fmri_scan.find('.')
    directory = fmri_scan[0:id_start]
    identifier = fmri_scan[id_start:id_end]

    out_3dmaskave = "{prefix}.{seed_name}.timecourse.txt".format(prefix=fmri_scan[:-7], seed_name=args.seed_name)
    out_3dfim = "{prefix}.{seed_name}.corr.nii.gz".format(prefix=fmri_scan[:-7], seed_name=args.seed_name)
    out_3dz = "{prefix}.{seed_name}.corr.z.nii.gz".format(prefix=fmri_scan[:-7], seed_name=args.seed_name)

    # create correlation maps
    if not isfile(out_3dmaskave):
        cmd = "3dmaskave -overwrite -quiet -mask {mask} {fmri} > {timecourse}".format(mask=seed_mask, fmri=fmri_scan, timecourse=out_3dmaskave)
        print cmd
        call(cmd, shell=True)

    if not isfile(out_3dfim):
        cmd = "3dfim+ -overwrite -input {fmri} -polort 2 -ideal_file {timecourse} -out Correlation -bucket {out_}".format(fmri=fmri_scan, timecourse=out_3dmaskave, out_=out_3dfim)
        print cmd
        call(cmd, shell=True)

    # transform R to Z-score
    if not isfile(out_3dz):
        cmd = "3dcalc -overwrite -a {in_} -expr 'log((1+a)/(1-a))/2' -prefix {out_}".format(in_=out_3dfim, out_=out_3dz)
        print cmd
        call(cmd, shell=True)

if args.groupA and args.groupB:
    f = open(args.groupA, "r")
    groupA = f.readlines()
    f.close()
    f = open(args.groupB, "r")
    groupB = f.readlines()
    f.close()

    groupA_str=''
    for subject in groupA:
        groupA_str = ' '.join([groupA_str,subject.rstrip()])

    groupB_str=''
    for subject in groupB:
        groupB_str = ' '.join([groupB_str,subject.rstrip()])

    cmd = "3dttest++ -prefix {result} -AminusB -unpooled -setA {groupA} -setB {groupB}".format(result=args.stats, groupA=groupA_str, groupB=groupB_str)
    print cmd
    call(cmd, shell=True)
