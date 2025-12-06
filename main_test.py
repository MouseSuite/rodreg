# %% [markdown]
# # Run Rodreg on David's Data
# 
# This notebook demonstrates how to run the `rodreg` registration pipeline on a dataset from `/home/ajoshi/Desktop/David_reg/` using the atlas from `/home/ajoshi/Desktop/David_reg/MSA50/`.

# %%
import os
import sys
from pathlib import Path

# Define paths
rodreg_path = "/home/ajoshi/Projects/rodreg/"
data_path = "/home/ajoshi/Desktop/David_reg/"
atlas_path = "/home/ajoshi/Desktop/David_reg/MSA50/"
output_path = "/home/ajoshi/Desktop/David_reg/rodreg_output/"

# Set environment variable for RODREG before importing
os.environ['RODREG'] = rodreg_path
sys.path.append(rodreg_path)

# Import after setting environment variable
from run_rodreg import run_rodreg

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# %%
# Define input and output files
subject_file = os.path.join(data_path, "AgingApoE_F10LC.masked.nii.gz")
atlas_prefix = os.path.join(atlas_path, "MSA50") # The script appends .brain.nii.gz and .label.nii.gz

# Output filenames
output_warped = os.path.join(output_path, "AgingApoE_F10LC_warped.nii.gz")
output_label = os.path.join(output_path, "AgingApoE_F10LC_warped_label.nii.gz")
output_inv_jac = os.path.join(output_path, "AgingApoE_F10LC_inv_jac.nii.gz")


# %%

# Run the registration directly as a function call
run_rodreg(
    input_file=subject_file,
    reference_prefix=atlas_prefix,
    output_file=output_warped,
    label_file=output_label,
    inverse_jacobian_file=output_inv_jac,
    linloss="cc",
    nonlinloss="cc",
    linear_epochs=1500,
    nonlinear_epochs=5000
)
print("Registration completed successfully!")


# %%
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices), figsize=(15, 5))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[i].axis('off')
    plt.show()

if os.path.exists(output_warped):
    # Load images
    img_subj = nib.load(subject_file).get_fdata()
    img_atlas = nib.load(atlas_prefix + ".brain.nii.gz").get_fdata()
    img_warped = nib.load(output_warped).get_fdata()

    # Pick a middle slice
    slice_idx = img_subj.shape[2] // 2
    
    slice_subj = img_subj[:, :, slice_idx]
    slice_atlas = img_atlas[:, :, slice_idx]
    slice_warped = img_warped[:, :, slice_idx]

    print("Displaying middle axial slices:")
    print("Left: Subject, Middle: Atlas, Right: Warped Subject")
    show_slices([slice_subj, slice_atlas, slice_warped])
else:
    print("Output file not found. Registration might have failed.")


