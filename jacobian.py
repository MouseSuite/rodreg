import nibabel as nib
import numpy as np
import argparse
from nilearn.image import resample_to_img

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


def jacobian(def_path, output_path, mask_path=None, edge_margin=3):
    """
    Compute the jacobian determinant for a deformation field and save it.

    By default this will zero-out Jacobian values within `edge_margin` voxels
    from the image boundaries to avoid edge artifacts. If `mask_path` is
    provided it will still be applied after the edge-zeroing (masking is
    optional and preserved for backward-compatibility).
    """
    # Load deformation field
    def_field, affine = load_nifti(def_path)

    # Check if the deformation field has 3 components per voxel
    if def_field.shape[-1] != 3:
        raise ValueError("Deformation field must have 3 components per voxel.")

    # Compute the Jacobian determinant
    jacobian_determinant = compute_jacobian_determinant(def_field.transpose(3, 0, 1, 2))

    # Fill near edges with identity value (1) to avoid spurious extreme values
    # from finite differences/resampling. A value of 1 means 'no local volume change'.
    if edge_margin and edge_margin > 0:
        h, w, d = jacobian_determinant.shape
        m = int(edge_margin)
        if m * 2 >= min(h, w, d):
            # margin too large, set whole image to identity
            jacobian_determinant[:] = 1
        else:
            jacobian_determinant[:m, :, :] = 1
            jacobian_determinant[h - m :, :, :] = 1
            jacobian_determinant[:, :m, :] = 1
            jacobian_determinant[:, w - m :, :] = 1
            jacobian_determinant[:, :, :m] = 1
            jacobian_determinant[:, :, d - m :] = 1

    # If a mask is provided, resample it to the jacobian image and zero outside mask
    if mask_path is not None and mask_path != "":
        try:
            mask_img = nib.load(mask_path)
            # Create a nifti image from jacobian for resampling reference
            jac_img = nib.Nifti1Image(jacobian_determinant, affine)
            resampled_mask = resample_to_img(mask_img, jac_img, interpolation="nearest")
            mask_data = resampled_mask.get_fdata()
            # Treat non-zero as inside mask
            jacobian_determinant[mask_data == 0] = 0
        except Exception as e:
            print(f"Warning: could not apply mask {mask_path}: {e}")

    # Save the Jacobian determinant
    save_nifti(jacobian_determinant, affine, output_path)
    print(f"Jacobian determinant saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the Jacobian determinant of a 4D deformation field using nibabel.")
    parser.add_argument('def_path', type=str, help='Path to the deformation field NIfTI file')
    parser.add_argument('output_path', type=str, help='Path to save the Jacobian determinant NIfTI file')
    parser.add_argument('--mask', type=str, default=None, help='Optional mask NIfTI file; zeros jacobian outside mask')
    parser.add_argument('--edge-margin', type=int, default=3, help='Number of voxels from image border to zero jacobian (default: 3)')

    args = parser.parse_args()
    jacobian(args.def_path, args.output_path, mask_path=args.mask, edge_margin=args.edge_margin)
