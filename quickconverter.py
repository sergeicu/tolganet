#!/usr/bin/env python

import argparse
import os

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
        "Converts nifti files from input to FSL TOPUP format to input to FD-net format"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input nifti file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="directory to save the output nifti file",
    )
    parser.add_argument(
        "-s",
        "--slices",
        action="store_true",
        help="Whether the nifti files were designed for single slice TOPUP",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: str = args.input
    output_path: str = args.output
    slice: bool = args.slices
    nii = nib.load(input_path)
    img: NDArray = nii.get_fdata()
    target_shape = np.array(
        [168, 144, 6, 2]
    )  # careful to set this with the understanding we will rotate after interpolation
    plot = False
    if plot:
        plt.imshow(img[:, :, img.shape[2] // 2, 0].T, cmap="gray")
        center = (img.shape[0] // 2, img.shape[1] // 2)
        rec = Rectangle(
            (center[0] - target_shape[0] // 2,
             center[1] - target_shape[1] // 2),
            target_shape[0] - 1,
            target_shape[1] - 1,
            edgecolor="r",
            facecolor="none",
        )
        plt.gca().add_patch(rec)
        plt.show()
    print(img.shape)
    print(nii.affine)
    if img.ndim < 4:
        raise ValueError("Input image does not have 4 dimensions")
    if img.shape[0] < target_shape[0] or img.shape[1] < target_shape[1]:
        raise ValueError(
            "Image is too small for target shape, think about this")
    else:
        voxel_size = nii.header.get_zooms()
        print(voxel_size)
        print(nii.header)
        print(img.shape)
        print(target_shape)
        target_voxel_size = (
            voxel_size[0] * (img.shape[0] / target_shape[0]) + 0.01,
            voxel_size[1] * (img.shape[1] / target_shape[1]) + 0.01,
            voxel_size[2],
        )
        print(target_voxel_size)
        target_affine = np.diag(target_voxel_size)

    nilearn = True
    if nilearn:
        interp_nii = resample_img(
            nii,
            target_affine=target_affine,
            # target_shape=target_shape[:3],
            interpolation="nearest",
        )
        # nib.save(interp_nii,
        # os.path.join(os.path.dirname(output_path), "interp.nii.gz"))
        interp_img = interp_nii.get_fdata()
    else:
        # interpolate in numpy/scipy
        # create grid of points to interpolate
        x = np.linspace(0, img.shape[0] - 1, target_shape[0])
        y = np.linspace(0, img.shape[1] - 1, target_shape[1])

    img = np.rot90(interp_img, axes=(0, 1))
    print(interp_img.shape)
    middle_idx: int = img.shape[2] // 2
    slice_img = img[:, :, middle_idx, :]
    slice_img = slice_img[:, :, np.newaxis, :]  # maintain 4D shape

    print(slice_img.shape)
    header = nii.header
    header["pixdim"][1:4] = target_voxel_size
    header["dim"][1:4] = target_shape[:3]

    new_nii: Nifti1Image = nib.Nifti1Image(slice_img,
                                           affine=nii.affine,
                                           header=header)
    nib.save(new_nii, output_path)


if __name__ == "__main__":
    main()
