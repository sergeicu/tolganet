#!/usr/bin/env python

import argparse
import os
from glob import glob

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from nibabel.nifti1 import Nifti1Image
from nilearn.image import resample_img
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Converts converts 2D output from FD-net back to original space")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to a directory containing the input nifti files",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        required=True,
        help="Path to the original nifti file",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*sv-*_network_image.nii.gz",
        help="Pattern to search for in the input directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="name to save the output nifti file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir: str = args.input
    reference_path: str = args.reference
    pattern: str = args.pattern
    output_path: str = args.output
    ref_nii = nib.load(reference_path)
    ref_img: NDArray = ref_nii.get_fdata()
    ref_shape = ref_img.shape
    ref_affine = ref_nii.affine
    ref_header = ref_nii.header
    input_files = glob(os.path.join(input_dir, pattern))
    input_files.sort(key=lambda x: int(x.split("sv-")[1].split("_")[0]))
    img3D = np.stack([nib.load(f).get_fdata() for f in input_files], axis=-1)
    img3D = np.rot90(img3D, axes=(0, 1), k=3)
    img3D = np.flip(img3D, axis=0)
    input_nii = nib.load(input_files[0])
    input_affine = input_nii.affine
    input_header = input_nii.header
    new_header = ref_header.copy()
    new_header["dim"][1:4] = img3D.shape
    new_header["pixdim"][1] = input_header["pixdim"][1]
    new_header["pixdim"][2] = input_header["pixdim"][2]
    print(f"Reference header: {ref_header}")
    print(f"New header: {new_header}")
    new_nii = Nifti1Image(img3D, ref_affine, new_header)
    nib.save(new_nii, output_path)


if __name__ == "__main__":
    main()
