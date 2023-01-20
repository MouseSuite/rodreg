
import nilearn as nl
import os
import nilearn.image as ni
from nilearn.image.image import load_img
from nilearn.plotting import plot_anat, plot_img
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from nilearn.image.resampling import reorder_img
from nilearn.masking import apply_mask
from glob import glob

subdir_corr = '/deneb_disk/RodentTools/data/EAE/EAE28_biascor'
subdir = '/deneb_disk/RodentTools/data/EAE/Uncorrected'
mask_dir = '/deneb_disk/RodentTools/data/EAE/EAE28_masks'

outdir = '/deneb_disk/RodentTools/data/EAE/data_nii'

flist = [os.path.basename(x) for x in glob(subdir + '/*.img')]
# Usage: fslchfiletype[_exe] <filetype> <filename> [filename2]

for s in flist:
    subid = s[:-4]

    subfile_corr = os.path.join(subdir_corr, 'm'+subid + '.hdr')
    img_corr = ni.swap_img_hemispheres(subfile_corr)
    img_corr = ni.new_img_like(img_corr, np.float32(img_corr.get_fdata()))
    img_corr.to_filename(os.path.join(outdir, subid + '_corr.nii.gz'))

    subfile = os.path.join(subdir, subid + '.img')
    img_src = nb.load(subfile)
    img_src = ni.new_img_like(img_corr, np.float32(img_src.get_fdata()))
    nb.save(img_src, os.path.join(outdir, subid + '_uncorr.nii.gz'))

    mask = os.path.join(mask_dir, subid + '.mask.hdr')
    img_msk = nb.load(mask)
    img_msk = ni.new_img_like(img_corr, np.uint8(img_msk.get_fdata()).squeeze())
    nb.save(img_msk, os.path.join(outdir, subid + '.mask.nii.gz'))

    # apply mask
    img_corr = ni.new_img_like(img_corr,np.float32(img_corr.get_fdata() * (img_msk.get_fdata()>0).squeeze()))
    img_corr.to_filename(os.path.join(outdir, subid + '.bfc.nii.gz'))

