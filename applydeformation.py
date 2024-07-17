import SimpleITK as sitk
import argparse

def load_nifti(file_path):
    """Load a NIfTI file and return the SimpleITK image."""
    return sitk.ReadImage(file_path)

def save_nifti(image, output_path):
    """Save a SimpleITK image as a NIfTI file."""
    sitk.WriteImage(image, output_path)

def apply_deformation(image, def_field):
    """
    Apply a deformation field to an image using SimpleITK.
    image should be a SimpleITK image.
    def_field should be a SimpleITK image with vector pixel type.
    """
    # Create a displacement field transform
    displacement_transform = sitk.DisplacementFieldTransform(def_field)
    
    # Apply the transform to the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(displacement_transform)
    
    deformed_image = resampler.Execute(image)
    return deformed_image

def applydeformation(image_path, def_path, output_path):
    # Load image and deformation field
    image = load_nifti(image_path)
    def_field = load_nifti(def_path)
    
    # Check if the deformation field is a vector field
    if def_field.GetNumberOfComponentsPerPixel() != 3:
        raise ValueError("Deformation field must have 3 components per pixel.")
    
    # Apply the deformation field to the image
    deformed_image = apply_deformation(image, def_field)
    
    # Save the deformed image
    save_nifti(deformed_image, output_path)
    print(f"Deformed image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a 3D deformation field to a NIfTI image using SimpleITK.")
    parser.add_argument('image_path', type=str, help='Path to the input image NIfTI file')
    parser.add_argument('def_path', type=str, help='Path to the deformation field NIfTI file')
    parser.add_argument('output_path', type=str, help='Path to save the deformed image NIfTI file')

    args = parser.parse_args()
    applydeformation(args.image_path, args.def_path, args.output_path)
