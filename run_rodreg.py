#!python
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
from pathlib import Path
codedir = str(Path(__file__).parents[1])
sys.path.append(codedir)
#from utilities.fileNames import FileOutputs
if 'RODREG' in os.environ:
    RODREG = os.environ['RODREG']
    sys.path.append(RODREG)
from utils import pad_nifti_image, multires_registration, interpolate_zeros
from aligner import Aligner
from warp_utils import apply_warp
from monai.transforms import LoadImage, EnsureChannelFirst
from warper import Warper
from composedeformations import composedeformation
from applydeformation import applydeformation
from invertdeformationfield import invertdeformationfield
from jacobian import jacobian


def run_rodreg(
    input_file,
    reference_prefix,
    output_file,
    label_file,
    inverse_jacobian_file=None,
    linloss='cc',
    nonlinloss='cc',
    linear_epochs=1500,
    nonlinear_epochs=5000,
    device='cuda'
):
    """
    Run the rodreg full registration pipeline.
    
    Args:
        input_file: Input subject file path
        reference_prefix: Reference image file prefix (without .brain.nii.gz or .label.nii.gz)
        output_file: Output file name for non-linearly warped image
        label_file: Output label file name
        inverse_jacobian_file: Output inverse jacobian file name (optional)
        linloss: Loss function for linear registration ('mse', 'cc', 'mi')
        nonlinloss: Loss function for non-linear registration ('mse', 'cc', 'mi')
        linear_epochs: Maximum iterations for linear registration
        nonlinear_epochs: Maximum iterations for non-linear registration
        device: Device to use ('cuda', 'cpu', etc.)
    """
    inputT2 = input_file
    atlas_brain = reference_prefix + '.brain.nii.gz'
    atlas_label = reference_prefix + '.label.nii.gz'

    if not os.path.exists(inputT2):
        print('ERROR: file', inputT2, 'does not exist.')
        raise FileNotFoundError(inputT2)

    if not os.path.exists(atlas_brain):
        print('ERROR: file', atlas_brain, 'does not exist.')
        raise FileNotFoundError(atlas_brain)

    if not os.path.exists(atlas_label):
        print('ERROR: file', atlas_label, 'does not exist.')
        raise FileNotFoundError(atlas_label)

    # Determine subject name and intermediate directory
    input_path = Path(inputT2)
    name = input_path.name
    if name.endswith('.nii.gz'):
        subject_name = name[:-7]
    elif name.endswith('.nii'):
        subject_name = name[:-4]
    else:
        subject_name = input_path.stem

    output_dir = os.path.dirname(output_file)
    intermediate_dir = os.path.join(output_dir, subject_name + "_intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)

    subbase = os.path.join(intermediate_dir, subject_name + '.rodreg')

    centered_atlas = subbase+".atlas.cent.nii.gz"
    centered_atlas_labels = subbase+".atlas.cent.label.nii.gz"
    cent_transform_file = subbase+".cent.reg.tfm"
    inv_cent_transform_file = subbase+".cent.reg.inv.tfm"

    centered_atlas_linreg = subbase+".atlas.lin.nii.gz"
    centered_atlas_linreg_labels = subbase+".atlas.lin.label.nii.gz"
    lin_reg_map_file = subbase+".lin_ddf.map.nii.gz"

    nonlin_reg_map_file = subbase+".nonlin_ddf.map.nii.gz"
    inv_nonlin_reg_map_file = subbase+".inv.nonlin_ddf.map.nii.gz"
    centered_atlas_nonlinreg = output_file
    centered_atlas_nonlinreg_labels = label_file
    jac_det_file = subbase+".warp-Jacobian.nii.gz"
    inv_jac_det_file = subbase+".inv.warp-Jacobian.subj_space.nii.gz"

    composed_ddf_file = subbase+".composed_ddf.map.nii.gz"
    inv_composed_ddf_file = subbase+".inv.composed_ddf.map.nii.gz"
    full_deformed_atlas = subbase+".atlas.full.deformed.nii.gz"
    full_deformed_subject = subbase+".full.deformed.nii.gz"
    subject_deformed2_atlas = subbase+".deformed2.atlas.nii.gz"

    jacobian_full_det_file = subbase+".jacobian_det.nii.gz"
    inv_jacobian_full_det_file = subbase+".inv.jacobian_det.nii.gz"
    inv_jacobian_full_atlas_det_file = inverse_jacobian_file

    # Centering registration
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

    sitk.WriteTransform(final_transform, cent_transform_file)
    inv_transform = final_transform.GetInverse()
    sitk.WriteTransform(inv_transform, inv_cent_transform_file)

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

    # Linear affine registration
    aligner = Aligner()
    aligner.affine_reg(
        fixed_file=inputT2,
        moving_file=centered_atlas,
        output_file=centered_atlas_linreg,
        ddf_file=lin_reg_map_file,
        loss=linloss,
        device=device,
        max_epochs=linear_epochs
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

    # Non-linear registration
    nonlin_reg = Warper()
    nonlin_reg.nonlinear_reg(
        target_file=inputT2,
        moving_file=centered_atlas_linreg,
        output_file=centered_atlas_nonlinreg,
        ddf_file=nonlin_reg_map_file,
        inv_ddf_file=inv_nonlin_reg_map_file,
        reg_penalty=1e-5,
        nn_input_size=128,
        lr=0.01,
        use_diffusion_reg=False,
        kernel_size=15,
        max_epochs=nonlinear_epochs,
        loss=nonlinloss,
        jacobian_determinant_file=jac_det_file,
        inv_jacobian_determinant_file=inv_jac_det_file,
        device=device,
    )

    disp_field, meta = LoadImage(image_only=False)(nonlin_reg_map_file)
    disp_field = EnsureChannelFirst()(disp_field)

    print(f"Loading linear registered labels from: {centered_atlas_linreg_labels}")
    try:
        at1, meta = LoadImage(image_only=False)(centered_atlas_linreg_labels)
    except Exception as e:
        print(f"Error loading {centered_atlas_linreg_labels}: {e}")
        print("Regenerating...")
        disp_field_lin, _ = LoadImage(image_only=False)(lin_reg_map_file)
        disp_field_lin = EnsureChannelFirst()(disp_field_lin)
        at1_cent, _ = LoadImage(image_only=False)(centered_atlas_labels)
        at_lab_cent = EnsureChannelFirst()(at1_cent)
        warped_lab_lin = apply_warp(
            disp_field_lin[None,], at_lab_cent[None,], at_lab_cent[None,], interp_mode="nearest"
        )
        nb.save(
            nb.Nifti1Image(warped_lab_lin[0, 0].detach().cpu().numpy(), at_lab_cent.affine),
            centered_atlas_linreg_labels,
        )
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

    # Compose deformations
    composedeformation(nonlin_reg_map_file, lin_reg_map_file, composed_ddf_file)

    cent_transform = sitk.ReadTransform(cent_transform_file)
    atlas = sitk.ReadImage(atlas_brain, sitk.sitkFloat32)
    moved_image = sitk.Resample(atlas, fixed_image, cent_transform)
    sitk.WriteImage(moved_image, centered_atlas)

    applydeformation(centered_atlas, composed_ddf_file, full_deformed_atlas)
    jacobian(composed_ddf_file, jacobian_full_det_file)

    # Invert deformations and apply centering
    invertdeformationfield(composed_ddf_file, inv_composed_ddf_file)
    applydeformation(inputT2, inv_composed_ddf_file, full_deformed_subject)

    moving_image = sitk.ReadImage(full_deformed_subject, sitk.sitkFloat32)
    fixed_image = sitk.ReadImage(atlas_brain, sitk.sitkFloat32)
    inv_cent_transform = sitk.ReadTransform(inv_cent_transform_file)
    moved_image = sitk.Resample(moving_image, fixed_image, inv_cent_transform)
    sitk.WriteImage(moved_image, subject_deformed2_atlas)

    # Calculate jacobian of the deformation field
    jacobian(inv_composed_ddf_file, inv_jacobian_full_det_file)

    if inverse_jacobian_file:
        # Apply centering to the Jacobian
        moving_image = sitk.ReadImage(inv_jacobian_full_det_file, sitk.sitkFloat32)
        fixed_image = sitk.ReadImage(atlas_brain, sitk.sitkFloat32)
        inv_cent_transform = sitk.ReadTransform(inv_cent_transform_file)
        moved_image = sitk.Resample(moving_image, fixed_image, inv_cent_transform)
        sitk.WriteImage(moved_image, inverse_jacobian_file)


def main():
    parser = argparse.ArgumentParser(description='Runs rodreg full registration pipeline.')
    parser.add_argument('-i', type=str, help='Input subject file.', required=True)
    parser.add_argument('-r', type=str, help='Reference image file prefix.', required=True)
    parser.add_argument('--o', type=str, help='Output file name (non-linearly warped image).', required=True)
    parser.add_argument('--l', type=str, help='Output label file name.', required=True)
    parser.add_argument('--j', type=str, help='Output inverse jacobian (in the reference image dimension) file name.', required=False)
    parser.add_argument('--linloss', type=str, help='Type of loss function for linear registration.', 
                        default = 'cc', choices=['mse', 'cc', 'mi'], required=False)
    parser.add_argument('--nonlinloss', type=str, help='Type of loss function for non-linear registration.', 
                        default = 'cc', choices=['mse', 'cc', 'mi'], required=False)
    parser.add_argument(
        "--le", type=int, default=1500, help="Maximum interations for linear registration"
    )
    parser.add_argument(
        "--ne", type=int, default=5000, help="Maximum interations for non-linear registration"
    )
    parser.add_argument(
        "--d", "--device", type=str, default="cuda", help="device: cuda, cpu, etc."
    )
        
    args = parser.parse_args()

    run_rodreg(
        input_file=args.i,
        reference_prefix=args.r,
        output_file=args.o,
        label_file=args.l,
        inverse_jacobian_file=args.j,
        linloss=args.linloss,
        nonlinloss=args.nonlinloss,
        linear_epochs=args.le,
        nonlinear_epochs=args.ne,
        device=args.d
    )


if __name__ == "__main__":
    main()