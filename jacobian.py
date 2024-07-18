import nibabel as nib
import numpy as np
import argparse

def load_nifti(file_path):
    """Load a NIfTI file and return the image data as a numpy array along with the affine matrix."""
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine

def save_nifti(data, affine, output_path):
    """Save a numpy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

def compute_jacobian_determinant2(def_field):
    """
    Compute the Jacobian determinant of a 4D deformation field.
    def_field should be a numpy array of shape (x, y, z, 3).
    """
    gradients = np.gradient(def_field, axis=(0, 1, 2))
    
    # Jacobian matrix is 3x3 for each voxel
    jacobian = np.zeros(def_field.shape[:-1] + (3, 3))
    
    for i in range(3):
        for j in range(3):
            jacobian[..., i, j] = gradients[j][..., i]
    
    # Compute the determinant of the Jacobian matrix
    jacobian_determinant = np.linalg.det(jacobian)
    
    return jacobian_determinant


def compute_jacobian_determinant(vf):
    """
    Given a displacement vector field vf, compute the jacobian determinant scalar field.
    vf is assumed to be a vector field of shape (3,H,W,D),
    and it is interpreted as the displacement field.
    So it is defining a discretely sampled map from a subset of 3-space into 3-space,
    namely the map that sends point (x,y,z) to the point (x,y,z)+vf[:,x,y,z].
    This function computes a jacobian determinant by taking discrete differences in each spatial direction.
    Returns a numpy array of shape (H-1,W-1,D-1).
    """

    _, H, W, D = vf.shape

    # Compute discrete spatial derivatives
    def diff_and_trim(array, axis): return np.diff(
        array, axis=axis)[:, :(H-1), :(W-1), :(D-1)]
    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)
    dz = diff_and_trim(vf, 3)

    # Add derivative of identity map
    dx[0] += 1
    dy[1] += 1
    dz[2] += 1

    # Compute determinant at each spatial location
    det = dx[0]*(dy[1]*dz[2]-dz[1]*dy[2]) - dy[0]*(dx[1]*dz[2] -
                                                   dz[1]*dx[2]) + dz[0]*(dx[1]*dy[2]-dy[1]*dx[2])

    return det


def jacobian(def_path, output_path):
    # Load deformation field
    def_field, affine = load_nifti(def_path)
    
    # Check if the deformation field has 3 components per voxel
    if def_field.shape[-1] != 3:
        raise ValueError("Deformation field must have 3 components per voxel.")
    
    # Compute the Jacobian determinant
    jacobian_determinant = compute_jacobian_determinant(def_field.transpose(3, 0, 1, 2))
    
    # Save the Jacobian determinant
    save_nifti(jacobian_determinant, affine, output_path)
    print(f"Jacobian determinant saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the Jacobian determinant of a 4D deformation field using nibabel.")
    parser.add_argument('def_path', type=str, help='Path to the deformation field NIfTI file')
    parser.add_argument('output_path', type=str, help='Path to save the Jacobian determinant NIfTI file')

    args = parser.parse_args()
    jacobian(args.def_path, args.output_path)
