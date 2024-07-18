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
from invertdeformationfield import invertdeformationfield
from jacobian import jacobian

subbase = "/deneb_disk/RodentTools/data/test4/29408.native"#"/deneb_disk/RodentTools/data/test_case/M2_LCRP"

sub_bse_t2 = subbase+".bse.nii.gz"

atlas_bse_t2 = "/deneb_disk/RodentTools/data/MSA100/MSA100/MSA100.bfc.nii.gz"
atlas_labels = "/deneb_disk/RodentTools/data/MSA100/MSA100/MSA100.label.nii.gz"

centered_atlas = subbase+".atlas.cent.nii.gz"
centered_subject = subbase+".cent.nii.gz"

centered_atlas_labels = subbase+".atlas.cent.label.nii.gz"
cent_transform_file = subbase+".cent.reg.tfm"
inv_cent_transform_file = subbase+".cent.reg.inv.tfm"
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
inv_composed_ddf_file = subbase+".inv.composed_ddf.map.nii.gz"

lin_deformed_atlas = subbase+".atlas.lin.deformed.nii.gz"
full_deformed_atlas = subbase+".atlas.full.deformed.nii.gz"
full_deformed_subject = subbase+".full.deformed.nii.gz"
subject_deformed2_atlas = subbase+".deformed2.atlas.nii.gz"


jacobian_full_det_file = subbase+".jacobian_det.nii.gz"
inv_jacobian_full_det_file = subbase+".inv.jacobian_det.nii.gz"
inv_jacobian_full_atlas_det_file = subbase+".inv.jacobian_atlas_det.nii.gz"


composedeformation(nonlin_reg_map_file, lin_reg_map_file, composed_ddf_file)

#composed_ddf_file is the map that is combination of linear and non-linear deformation fields. Th ecentering is to be applied separately

fixed_image = sitk.ReadImage(sub_bse_t2, sitk.sitkFloat32)
moving_image = sitk.ReadImage(atlas_bse_t2, sitk.sitkFloat32)
cent_transform = sitk.ReadTransform(cent_transform_file)
moved_image = sitk.Resample(moving_image, fixed_image, cent_transform)
sitk.WriteImage(moved_image, centered_atlas)

applydeformation(centered_atlas, composed_ddf_file, full_deformed_atlas)

# Jacobian of the forward field
jacobian(composed_ddf_file, jacobian_full_det_file)



# Invert the composed deformation field this takes about 15 min
invertdeformationfield(composed_ddf_file, inv_composed_ddf_file)
applydeformation(sub_bse_t2, inv_composed_ddf_file, full_deformed_subject) # subject moved to atlas space (without centering)

# apply centering
moving_image = sitk.ReadImage(full_deformed_subject, sitk.sitkFloat32)
fixed_image = sitk.ReadImage(atlas_bse_t2, sitk.sitkFloat32)
inv_cent_transform = sitk.ReadTransform(inv_cent_transform_file)
moved_image = sitk.Resample(moving_image, fixed_image, inv_cent_transform)
sitk.WriteImage(moved_image, subject_deformed2_atlas)


# Calculate jacobian of the deformation field
jacobian(inv_composed_ddf_file, inv_jacobian_full_det_file)

# apply centering to the Jacobian
moving_image = sitk.ReadImage(inv_jacobian_full_det_file, sitk.sitkFloat32)
fixed_image = sitk.ReadImage(atlas_bse_t2, sitk.sitkFloat32)
inv_cent_transform = sitk.ReadTransform(inv_cent_transform_file)
moved_image = sitk.Resample(moving_image, fixed_image, inv_cent_transform)
sitk.WriteImage(moved_image, inv_jacobian_full_atlas_det_file)

