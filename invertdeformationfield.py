import nibabel as nib
import numpy as np
from scipy.interpolate import griddata
import argparse
from tqdm import tqdm

def load_nifti(file_path):
    """Load a NIfTI file and return the image data as a numpy array along with the affine matrix."""
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine

def save_nifti(data, affine, output_path):
    """Save a numpy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

def invert_deformation_field(def_field, affine):
    """
    Invert a deformation field using scattered interpolation.
    def_field should be a numpy array of shape (x, y, z, 3).
    """
    shape = def_field.shape[:-1]
    
    # Create a meshgrid of coordinates
    coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    coords = np.stack(coords, axis=-1).reshape(-1, 3)
    
    # Get voxel dimensions from the affine matrix
    voxel_dim = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    def_field *= voxel_dim
    # Apply voxel dimensions to get real-world coordinates
    real_coords = coords * voxel_dim
    
    # Compute the displaced coordinates
    displaced_coords = real_coords + def_field.reshape(-1, 3)
    
    # Initialize the inverse deformation field
    inv_def = np.zeros_like(def_field).reshape(-1, 3)
    
    # Perform scattered interpolation for each displacement component
    for i in tqdm(range(3)):
        inv_def[:, i] = griddata(displaced_coords, real_coords[:, i], real_coords, method='linear', fill_value=0)
    
    inv_def -= real_coords
    
    return inv_def.reshape(def_field.shape) / voxel_dim

def invertdeformationfield(def_path, output_path):
    # Load deformation field
    def_field, affine = load_nifti(def_path)
    
    # Invert the deformation field
    inv_def = invert_deformation_field(def_field, affine)
    
    # Save the inverted deformation field
    save_nifti(inv_def, affine, output_path)
    print(f"Inverted deformation field saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert a 3D deformation field in NIfTI format.")
    parser.add_argument('def_path', type=str, help='Path to the deformation field NIfTI file')
    parser.add_argument('output_path', type=str, help='Path to save the inverted deformation field NIfTI file')

    args = parser.parse_args()
    invertdeformationfield(args.def_path, args.output_path)
