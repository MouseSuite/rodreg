import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
import argparse

def load_nifti(file_path):
    """Load a NIfTI file and return the image data as a numpy array along with the affine matrix."""
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine

def save_nifti(data, affine, output_path):
    """Save a numpy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

def compose_deformation_fields(def1, def2, affine1, affine2):
    """
    Compose two deformation fields.
    def1 and def2 should be numpy arrays of shape (x, y, z, 3),
    where the last dimension contains the displacements.
    """
    # Get voxel dimensions from the affine matrices
    voxel_dim1 = np.sqrt(np.sum(affine1[:3, :3] ** 2, axis=0))
    voxel_dim2 = np.sqrt(np.sum(affine2[:3, :3] ** 2, axis=0))
    
    shape = def1.shape[:-1]
    
    # Create a meshgrid of coordinates
    coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    coords = np.stack(coords, axis=-1).astype(np.float64)

    def1 = def1 * voxel_dim1
    def2 = def2 * voxel_dim2

    # Apply voxel dimensions to get real-world coordinates
    real_coords1 = coords * voxel_dim1
    real_coords2 = coords * voxel_dim2
    
    # Apply the first deformation field
    new_real_coords1 = real_coords1 + def1
    
    # Convert real-world coordinates back to voxel coordinates for def2
    new_voxel_coords2 = new_real_coords1 / voxel_dim2
    
    # Interpolate the second deformation field at the new voxel coordinates
    composed_def = np.zeros_like(def1)
    for i in range(3):
        #composed_def[..., i] = map_coordinates(def2[..., i]*voxel_dim2[i], new_voxel_coords2[..., [2, 1, 0]].transpose(3, 0, 1, 2), order=1)
        composed_def[..., i] = map_coordinates(def2[..., i], new_voxel_coords2.transpose(3, 0, 1, 2), order=1)

    # Add the interpolated displacement to the new voxel coordinates
    composed_def += def1
    
    return composed_def/voxel_dim1

def composedeformation(def1_path, def2_path, output_path):
    # Load deformation fields
    def1, affine1 = load_nifti(def1_path)
    def2, affine2 = load_nifti(def2_path)

    # Check if the shapes of the deformation fields match
    if def1.shape != def2.shape:
        raise ValueError("Deformation fields must have the same shape")

    # Compose the deformation fields
    composed_def = compose_deformation_fields(def1, def2, affine1, affine2)

    # Save the composed deformation field
    save_nifti(composed_def, affine1, output_path)
    print(f"Composed deformation field saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose two 3D deformation fields in NIfTI format.")
    parser.add_argument('def1_path', type=str, help='Path to the first deformation field NIfTI file')
    parser.add_argument('def2_path', type=str, help='Path to the second deformation field NIfTI file')
    parser.add_argument('output_path', type=str, help='Path to save the composed deformation field NIfTI file')

    args = parser.parse_args()
    composedeformation(args.def1_path, args.def2_path, args.output_path)
