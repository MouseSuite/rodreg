import os
import ants
import nibabel as nib
import numpy as np
import time
from datetime import datetime

# Define paths (consistent with main_test_irina.py)
data_path = "/home/ajoshi/Desktop/rodreg/test/"
atlas_path = "/home/ajoshi/Desktop/rodreg/MSA50/"
output_path = "/home/ajoshi/Desktop/rodreg/test/ants_output/"

# Create output directory
os.makedirs(output_path, exist_ok=True)

# Log file path
log_file = os.path.join(output_path, "registration_log.txt")

def log_message(message, file_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(message)
    with open(file_path, "a") as f:
        f.write(formatted_message + "\n")

# Start timing
start_total = time.time()

log_message("Starting ANTs SyN registration script", log_file)

# Define input files
subject_file = os.path.join(data_path, "ApoE071524_C12L.RodReg_new.masked.nii.gz")
atlas_brain = os.path.join(atlas_path, "MSA50.brain.nii.gz")
atlas_label = os.path.join(atlas_path, "MSA50.label.nii.gz")

# Output filenames
output_warped = os.path.join(output_path, "ApoE071524_C12L_ants_warped.nii.gz")
output_label = os.path.join(output_path, "ApoE071524_C12L_ants_warped_label.nii.gz")
output_jac = os.path.join(output_path, "ApoE071524_C12L_ants_jac.nii.gz")
output_inv_jac = os.path.join(output_path, "ApoE071524_C12L_ants_inv_jac.nii.gz")

log_message("Loading images...", log_file)
start_load = time.time()
# Load images using ANTs
fixed = ants.image_read(subject_file)
moving = ants.image_read(atlas_brain)
moving_label = ants.image_read(atlas_label)
end_load = time.time()
log_message(f"Images loaded in {end_load - start_load:.2f} seconds", log_file)

log_message("Starting ANTs SyN registration...", log_file)
start_reg = time.time()
# Perform SyN registration
# 'SyN' is the symmetric normalization (deformable)
# 'Rigid', 'Affine' are usually performed before SyN
registration = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform='SyN',
    outprefix=os.path.join(output_path, 'ants_')
)
end_reg = time.time()
log_message(f"Registration finished in {end_reg - start_reg:.2f} seconds", log_file)

log_message("Warping images...", log_file)
start_warp = time.time()
# The warped image is already in registration['warpedmovout']
ants.image_write(registration['warpedmovout'], output_warped)

# Warp the labels using the calculated transforms
# We use 'genericLabel' interpolation for labels
warped_label = ants.apply_transforms(
    fixed=fixed,
    moving=moving_label,
    transformlist=registration['fwdtransforms'],
    interpolator='genericLabel'
)

ants.image_write(warped_label, output_label)
end_warp = time.time()
log_message(f"Warping finished in {end_warp - start_warp:.2f} seconds", log_file)

log_message("Calculating Jacobian maps...", log_file)
start_jac = time.time()
# The warp field is the first element in fwdtransforms for SyN
# It maps moving to fixed
jac = ants.create_jacobian_determinant_image(fixed, registration['fwdtransforms'][0])
ants.image_write(jac, output_jac)

# The inverse warp field is the second element in invtransforms for SyN
# It maps fixed to moving
inv_jac = ants.create_jacobian_determinant_image(moving, registration['invtransforms'][1])
ants.image_write(inv_jac, output_inv_jac)
end_jac = time.time()
log_message(f"Jacobian calculation finished in {end_jac - start_jac:.2f} seconds", log_file)

end_total = time.time()
log_message(f"ANTs registration completed successfully in {end_total - start_total:.2f} seconds total!", log_file)
log_message(f"Warped image saved to: {output_warped}", log_file)
log_message(f"Warped labels saved to: {output_label}", log_file)
log_message(f"Jacobian map saved to: {output_jac}", log_file)
log_message(f"Inverse Jacobian map saved to: {output_inv_jac}", log_file)
