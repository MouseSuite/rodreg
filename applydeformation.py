import nibabel as nib
import numpy as np
import argparse
from scipy.ndimage import map_coordinates

def load_nifti(file_path):
    """Load a NIfTI file and return the image data as a numpy array along with the affine matrix."""
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine

def save_nifti(data, affine, output_path):
    """Save a numpy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

def apply_deformation(image, def_field, affine,order=1):
    """
    Apply a deformation field to an image using nibabel and scipy.
    image should be a numpy array.
    def_field should be a numpy array of shape (x, y, z, 3).
    affine should be the affine matrix of the image.
    """
    shape = image.shape
    
    # Create a meshgrid of coordinates
    coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    coords = np.stack(coords, axis=-1).astype(np.float64)
    
    # Get voxel dimensions from the affine matrix
    voxel_dim = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    
    ## Apply voxel dimensions to get real-world coordinates
    #real_coords = coords * voxel_dim
    
    # Compute the displaced coordinates
    displaced_coords = (coords + def_field) * voxel_dim
    
    # Convert real-world coordinates back to voxel coordinates
    voxel_coords = displaced_coords / voxel_dim
    
    # Interpolate the image at the displaced coordinates
    deformed_image = np.zeros_like(image)
    #for i in range(shape[2]):
    deformed_image = map_coordinates(image, voxel_coords.transpose(3, 0, 1, 2), order=order)
    #[..., [2, 1, 0]]
    return deformed_image

def applydeformation(image_path, def_path, output_path, order=1):
    # Load image and deformation field
    image, image_affine = load_nifti(image_path)
    def_field, def_affine = load_nifti(def_path)
    
    # Check if the deformation field is a vector field
    if def_field.shape[-1] != 3:
        raise ValueError("Deformation field must have 3 components per pixel.")
    
    # Apply the deformation field to the image
    deformed_image = apply_deformation(image, def_field, image_affine, order=order)
    
    # Save the deformed image
    save_nifti(deformed_image, image_affine, output_path)
    print(f"Deformed image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a 3D deformation field to a NIfTI image using nibabel.")
    parser.add_argument('image_path', type=str, help='Path to the input image NIfTI file')
    parser.add_argument('def_path', type=str, help='Path to the deformation field NIfTI file')
    parser.add_argument('output_path', type=str, help='Path to save the deformed image NIfTI file')

    args = parser.parse_args()
    applydeformation(args.image_path, args.def_path, args.output_path)
