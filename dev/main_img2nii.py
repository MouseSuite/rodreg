
import nilearn as nl
import os
import nilearn.image as ni
from nilearn.image.image import load_img
from nilearn.plotting import plot_anat, plot_img
import matplotlib.pyplot as plt
import nibabel as nb

from glob import glob 

subdir_corr = '/ImagePTE1/ajoshi/mouse_ucla_data/EAE28_cp/Bias_Corrected'
subdir = '/ImagePTE1/ajoshi/mouse_ucla_data/EAE28_cp/Uncorrected'
outdir = '/ImagePTE1/ajoshi/mouse_ucla_data/data4ML'

flist = [os.path.basename(x) for x in glob(subdir + '/*.img')]

for s in flist:
    subid = s[:-4]
    subfile = os.path.join(subdir, subid + '.img')
    img = nb.load(subfile)
    nb.save(img,os.path.join(outdir,subid[1:] + '_uncorr.nii.gz'))

    subfile_corr = os.path.join(subdir_corr,'m'+subid + '.hdr')
    img = ni.swap_img_hemispheres(subfile_corr)
    img.to_filename(os.path.join(outdir,subid[1:] + '_corr.nii.gz'))
