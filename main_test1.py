import nilearn.image as ni
import nibabel as nb
from nilearn.plotting import plot_anat, plot_prob_atlas, show, plot_stat_map
import SimpleITK as sitk
from utils import pad_nifti_image, multires_registration, interpolate_zeros
from aligner import Aligner
from warp_utils import apply_warp
import numpy as np
from monai.transforms import LoadImage, EnsureChannelFirst
from warper import Warper
from composedeformations import composedeformation
from applydeformation import applydeformation


# %matplotlib notebook
# import gui
subbase = "/deneb_disk/RodentTools/data/test4/29408.native"#
"/deneb_disk/RodentTools/data/test_case/M2_LCRP"

sub_bse_t2 = subbase+".bse.nii.gz"

atlas_bse_t2 = "/deneb_disk/RodentTools/data/MSA100/MSA100/MSA100.bfc.nii.gz"
atlas_labels = "/deneb_disk/RodentTools/data/MSA100/MSA100/MSA100.label.nii.gz"

centered_atlas = subbase+".atlas.cent.nii.gz"
centered_atlas_labels = subbase+".atlas.cent.label.nii.gz"

centered_atlas_linreg = subbase+".atlas.lin.nii.gz"
centered_atlas_linreg_labels = subbase+".atlas.lin.label.nii.gz"
lin_reg_map_file = subbase+".lin_ddf.map.nii.gz"


nonlin_reg_map_file = subbase+".nonlin_ddf.map.nii.gz"
inv_nonlin_reg_map_file = subbase+".inv.nonlin_ddf.map.nii.gz"
centered_atlas_nonlinreg = subbase+".atlas.nonlin.nii.gz"
centered_atlas_nonlinreg_labels = subbase+".atlas.nonlin.label.nii.gz"
jac_det_file = subbase+".jacobian_det.nii.gz"
inv_jac_det_file = subbase+".inv.jacobian_det.nii.gz"
composed_ddf_file = subbase+".composed_ddf.map.nii.gz"
composed_jac_det_file = subbase+".composed_jacobian_det.nii.gz"
deformed_atlas = subbase+".deformed_atlas.nii.gz"

#composedeformation(nonlin_reg_map_file, lin_reg_map_file, composed_ddf_file)

applydeformation(atlas_bse_t2, lin_reg_map_file, deformed_atlas)

