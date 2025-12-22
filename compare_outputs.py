import os
import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
from datetime import datetime

def log_message(message, log_file=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(message)
    if log_file:
        with open(log_file, "a") as f:
            f.write(formatted_message + "\n")

def normalize_image(data):
    """Normalize image to [0, 1] range for SSIM calculation."""
    data = data.astype(np.float32)
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min > 0:
        return (data - data_min) / (data_max - data_min)
    return data

def calculate_ssim(img1_path, img2_path):
    """Calculate SSIM between two NIfTI images."""
    img1 = nib.load(img1_path).get_fdata()
    img2 = nib.load(img2_path).get_fdata()
    
    # Ensure images are the same size (they should be if registration worked)
    if img1.shape != img2.shape:
        return None, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    
    # Normalize
    img1_norm = normalize_image(img1)
    img2_norm = normalize_image(img2)
    
    # Calculate SSIM
    # data_range is 1.0 because we normalized to [0, 1]
    score = ssim(img1_norm, img2_norm, data_range=1.0)
    return score, None

def main():
    # Define paths (consistent with previous scripts)
    subject_file = "/home/ajoshi/Desktop/rodreg/test/ApoE071524_C12L.RodReg_new.masked.nii.gz"
    rodreg_output = "/home/ajoshi/Desktop/rodreg/test/rodreg_output/ApoE071524_C12L_warped.nii.gz"
    ants_output = "/home/ajoshi/Desktop/rodreg/test/ants_output/ApoE071524_C12L_ants_warped.nii.gz"
    
    comparison_log = "/home/ajoshi/Desktop/rodreg/test/registration_comparison.txt"
    
    log_message("Starting Registration Comparison (SSIM)", comparison_log)
    
    # Compare RodReg
    log_message(f"Comparing RodReg output to Subject...", comparison_log)
    if os.path.exists(rodreg_output):
        score, err = calculate_ssim(subject_file, rodreg_output)
        if err:
            log_message(f"RodReg SSIM Error: {err}", comparison_log)
        else:
            log_message(f"RodReg vs Subject SSIM: {score:.4f}", comparison_log)
    else:
        log_message("RodReg output file not found.", comparison_log)
        
    # Compare ANTs
    log_message(f"Comparing ANTs output to Subject...", comparison_log)
    if os.path.exists(ants_output):
        score, err = calculate_ssim(subject_file, ants_output)
        if err:
            log_message(f"ANTs SSIM Error: {err}", comparison_log)
        else:
            log_message(f"ANTs vs Subject SSIM: {score:.4f}", comparison_log)
    else:
        log_message("ANTs output file not found.", comparison_log)

if __name__ == "__main__":
    main()
