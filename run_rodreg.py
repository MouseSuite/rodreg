#!/home/ajoshi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import argparse
import sys
import os
from os.path import join
import nilearn.image as ni
import nibabel as nb
import SimpleITK as sitk
RODREG = os.environ['RODREG']
sys.path.append(RODREG)
from utils import pad_nifti_image, multires_registration, interpolate_zeros
from aligner import Aligner
from warp_utils import apply_warp
from monai.transforms import LoadImage, EnsureChannelFirst
from warper import Warper


def main():
    parser = argparse.ArgumentParser(description='Runs rodreg full registration pipeline.')
    parser.add_argument('-s', type=str, help='Input subject file.', required=True)
    parser.add_argument('-a', type=str, help='Reference image file prefix.', required=True)
    parser.add_argument('--o', type=str, help='Output file name (non-linearly warped image).', required=True)
    parser.add_argument('--l', type=str, help='Output label file name.', required=False)
    # parser.add_argument('--j', type=str, help='Output jacobian file name.', required=False)
    parser.add_argument('--lj', type=str, help='Output log jacobian file name.', required=False)
    parser.add_argument('--linloss', type=str, help='Type of loss function for linear registration.', 
                        default = 'cc', choices=['mse', 'cc', 'mi'], required=False)
    parser.add_argument('--nonlinloss', type=str, help='Type of loss function for non-linear registration.', 
                        default = 'cc', choices=['mse', 'cc', 'mi'], required=False)
    parser.add_argument(
        "--le", type=int, default=5000, help="Maximum interations for linear registration"
    )
    parser.add_argument(
        "--ne", type=int, default=5000, help="Maximum interations for non-linear registration"
    )
    parser.add_argument(
        "--d", "--device", type=str, default="cuda", help="device: cuda, cpu, etc."
    )
        
    args = parser.parse_args()

    device = args.d

    inputT2 = args.s
    atlas_brain = args.a + '.brain.nii.gz'
    atlas_label = args.a + '.label.nii.gz'

    if not os.path.exists(inputT2):
        print('ERROR: file', inputT2, 'does not exist.')
        sys.exit(2)

    if not os.path.exists(atlas_brain):
        print('ERROR: file', atlas_brain, 'does not exist.')
        sys.exit(2)

    if not os.path.exists(atlas_label):
        print('ERROR: file', atlas_label, 'does not exist.')
        sys.exit(2)

    subbase = inputT2.split('.')[0] + '.rodreg'


    centered_atlas = subbase+".atlas.cent.nii.gz"
    centered_atlas_labels = subbase+".atlas.cent.label.nii.gz"

    centered_atlas_linreg = subbase+".atlas.lin.nii.gz"
    centered_atlas_linreg_labels = subbase+".atlas.lin.label.nii.gz"
    lin_reg_map_file = subbase+".lin_ddf.map.nii.gz"

    nonlin_reg_map_file = subbase+".nonlin_ddf.map.nii.gz"
    inv_nonlin_reg_map_file = subbase+".inv.nonlin_ddf.map.nii.gz"
    centered_atlas_nonlinreg = subbase+".nonlin.warped.nii.gz"
    centered_atlas_nonlinreg_labels = subbase+".label.nii.gz"
    jac_det_file = subbase+".warp-Jacobian.nii.gz"
    logjac_det_file = subbase+".warp-logJacobian.nii.gz"

    if args.l:
        centered_atlas_nonlinreg_labels = args.l

    if args.lj:
        logjac_det_file = args.lj

    if args.o:
        centered_atlas_nonlinreg = args.o

    fixed_image = sitk.ReadImage(inputT2, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(atlas_brain, sitk.sitkFloat32)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    final_transform, _ = multires_registration(
        fixed_image, moving_image, initial_transform)

    moved_image = sitk.Resample(moving_image, fixed_image, final_transform)

    sitk.WriteImage(moved_image, centered_atlas)

    moving_image = sitk.ReadImage(atlas_label, sitk.sitkUInt16)
    moved_image = sitk.Resample(
        moving_image,
        fixed_image,
        transform=final_transform,
        interpolator=sitk.sitkNearestNeighbor,
    )
    sitk.WriteImage(moved_image, centered_atlas_labels)

    aligner = Aligner()
    aligner.affine_reg(
        fixed_file=inputT2,
        moving_file=centered_atlas,
        output_file=centered_atlas_linreg,
        ddf_file=lin_reg_map_file,
        loss=args.linloss,
        device=device,
        max_epochs=args.le
    )

    disp_field, meta = LoadImage(image_only=False)(lin_reg_map_file)
    disp_field = EnsureChannelFirst()(disp_field)

    at1, meta = LoadImage(image_only=False)(centered_atlas_labels)
    at_lab = EnsureChannelFirst()(at1)

    warped_lab = apply_warp(
        disp_field[None,], at_lab[None,], at_lab[None,], interp_mode="nearest"
    )
    nb.save(
        nb.Nifti1Image(warped_lab[0, 0].detach().cpu().numpy(), at_lab.affine),
        centered_atlas_linreg_labels,
    )

    nonlin_reg = Warper()
    nonlin_reg.nonlinear_reg(
        target_file=inputT2,
        moving_file=centered_atlas_linreg,
        output_file=centered_atlas_nonlinreg,
        ddf_file=nonlin_reg_map_file,
        inv_ddf_file=inv_nonlin_reg_map_file,
        reg_penalty=1,
        nn_input_size=64,
        lr=1e-4,
        max_epochs=args.ne,
        loss=args.nonlinloss,
        jacobian_determinant_file=jac_det_file,
        device=device,
    )

    disp_field, meta = LoadImage(image_only=False)(nonlin_reg_map_file)
    disp_field = EnsureChannelFirst()(disp_field)

    at1, meta = LoadImage(image_only=False)(centered_atlas_linreg_labels)
    at_lab = EnsureChannelFirst()(at1)

    warped_lab = apply_warp(
        disp_field[None,], at_lab[None,], at_lab[None,], interp_mode="nearest"
    )
    nb.save(
        nb.Nifti1Image(
            np.uint16(warped_lab[0, 0].detach().cpu().numpy()), at_lab.affine),
        centered_atlas_nonlinreg_labels,
    )

    jacdet, meta = LoadImage(image_only=False)(jac_det_file)
    logdata = np.log(jacdet)
    
    recon = nib.Nifti1Image(logdata, jacdet.affine)
    nib.save(recon, logjac_det_file)
    

if __name__ == "__main__":
    main()