#!/usr/bin/env python3
import os
import sys
import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch run rodreg registration on multiple subjects.")
    parser.add_argument('--data_dir', type=str, default='/home/ajoshi/Desktop/David_reg/', help='Directory containing subject files')
    parser.add_argument('--atlas_prefix', type=str, default='/home/ajoshi/Desktop/David_reg/MSA50/MSA50', help='Atlas prefix (without .brain.nii.gz)')
    parser.add_argument('--output_dir', type=str, default='/home/ajoshi/Desktop/David_reg/rodreg_output/', help='Directory to save outputs')
    parser.add_argument('--pattern', type=str, default='*.nii.gz', help='Glob pattern for subject files')
    parser.add_argument('--label_ext', type=str, default='_warped_label.nii.gz', help='Output label file extension')
    parser.add_argument('--inv_jac_ext', type=str, default='_inv_jac.nii.gz', help='Output inverse jacobian file extension')
    parser.add_argument('--linear_epochs', type=int, default=1500)
    parser.add_argument('--nonlinear_epochs', type=int, default=5000)
    parser.add_argument('--linloss', type=str, default='cc')
    parser.add_argument('--nonlinloss', type=str, default='cc')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    subject_files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if not subject_files:
        print(f"No subject files found in {args.data_dir} with pattern {args.pattern}")
        sys.exit(1)

    import subprocess
    for subject_file in subject_files:
        subject_base = os.path.splitext(os.path.basename(subject_file))[0]
        output_warped = os.path.join(args.output_dir, f"{subject_base}_warped.nii.gz")
        output_label = os.path.join(args.output_dir, f"{subject_base}{args.label_ext}")
        output_inv_jac = os.path.join(args.output_dir, f"{subject_base}{args.inv_jac_ext}")
        print(f"\nProcessing {subject_file} ...")
        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "run_rodreg.py"),
            "-i", subject_file,
            "-r", args.atlas_prefix,
            "--o", output_warped,
            "--l", output_label,
            "--j", output_inv_jac,
            "--linloss", args.linloss,
            "--nonlinloss", args.nonlinloss,
            "--le", str(args.linear_epochs),
            "--ne", str(args.nonlinear_epochs),
            "--d", args.device
        ]
        print("Running command:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"rodreg failed for {subject_file}")
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
        else:
            print(f"Completed {subject_file}\n")
