from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import yaml
from tqdm import tqdm

import imcurvpy.io as io

from .single_stack import (
    read_and_compute_principal_curvatures_single_stack_multi_nuclei,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _write_out_settings(
    path: Path,
    settings: dict,
) -> None:
    """
    Write analysis settings to a YAML file.

    Parameters
    ----------
    path : Path
        The full path (including filename) where the YAML file will be written,
        typically something like `output_dir / "settings.yaml"`.
    settings : dict
        A dictionary containing all parameter names and values used for analysis.
        This dictionary will be serialized in YAML format for reproducibility and
        logging.

    Returns
    -------
    None
        This function does not return a value. It writes the file to disk.

    Notes
    -----
    Uses PyYAML to serialize the dictionary. Ensures readable formatting by disabling
    flow style and preserving key order.
    """
    with path.open("w") as f:
        yaml.dump(settings, f, default_flow_style=False, sort_keys=False)


def process_directory_principal_curvatures_multi_nuclei(
    input_dir: str | Path,
    channel_indices: int | Sequence[int],
    wavelengths: float | Sequence[float],
    na: float,
    magnification: float,
    ri_sample: float,
    ri_coverslip: float,
    ri_immersion: float,
    confocal: bool,
    rough_lamin_mask_sigma: float,
    rough_lamin_mask_ic_factor: float,
    rough_lamin_mask_alpha: float,
    fine_lamin_mask_sigma: float,
    pc_scale: float,
    expansion_distance: float,
    min_nucleus_area: float,
    axis_order: str = "zyx",
    reduction_mode: Literal["max", "mean", "min"] = "max",
    sign_estimation_mode: Literal["dapi", "image"] | None = None,
    channel_idx_dapi: int | None = None,
    wavelength_dapi: float | None = None,
    spacing_goal: float | tuple[float, float, float] | Literal["min", "max"] = "max",
    order: int = 1,
    downsample_mode: str = "sum",
    pad_mode: str = "constant",
    ball_radius: float = 0.5,
    ball_stat: str = "mean",
    ball_pad_mode: str = "reflect",
    labels_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    """
    Process TIFF image stacks in a directory, segment and compute principal curvatures.

    This function scans a directory for multi-channel TIFF files, processes each stack
    to segment nuclei, compute normalized lamin intensities, and calculate principal
    curvature components of the nuclear lamina for each nucleus detected. Results,
    including cropped images and curvature maps, are saved to an output directory.

    Parameters
    ----------
    input_dir : str | Path
        Path to the directory containing multi-channel TIFF files to process.
    channel_indices : int | Sequence[int]
        Index or indices of lamin channels in the TIFF stacks to analyze.
    wavelengths : float | Sequence[float]
        Emission wavelength(s) of the lamin channels, in micrometers.
    na : float
        Numerical aperture of the microscope objective.
    magnification : float
        Microscope magnification factor.
    ri_sample : float
        Refractive index of the sample.
    ri_coverslip : float
        Refractive index of the coverslip.
    ri_immersion : float
        Refractive index of the immersion medium.
    confocal : bool
        Whether the imaging system is confocal.
    rough_lamin_mask_sigma : float
        Gaussian smoothing sigma (µm) for rough lamin mask segmentation.
    rough_lamin_mask_ic_factor : float
        Intensity correction factor for rough lamin mask segmentation.
    rough_lamin_mask_alpha : float
        Threshold alpha for rough lamin mask generation.
    fine_lamin_mask_sigma : float
        Gaussian smoothing sigma (µm) for fine lamin mask segmentation.
    pc_scale : float
        Physical scale (µm) used for curvature estimation.
    expansion_distance : float
        Distance (µm) for Voronoi expansion of nucleus labels.
    min_nucleus_area : float
        Minimum 2D area (µm²) for nuclei to be included in the analysis.
    axis_order : str, default "zyx"
        Axis order of image data (permutation of 'x', 'y', 'z').
    reduction_mode : {"max", "mean", "min"}, default "max"
        Method to combine multiple lamin channels into a single volume.
    sign_estimation_mode : {"dapi", "image"} | None, optional
        Mode to estimate curvature sign:
        - "dapi": use DAPI channel
        - "image": use image intensity
        - None: disable sign estimation
    channel_idx_dapi : int | None, optional
        Index of DAPI channel if used for sign estimation.
    wavelength_dapi : float | None, optional
        Emission wavelength of the DAPI channel, in micrometers.
    spacing_goal : float | tuple[float, float, float] | {"min", "max"}, default "max"
        Target isotropic voxel spacing after resampling. Can be a numeric value or
        "min"/"max" to use the minimum/maximum native spacing.
    order : int, default 1
        Interpolation order for resampling (0=nearest, 1=linear, 3=cubic, etc.).
    downsample_mode : {"sum", "mean"}, default "sum"
        Mode for intensity downsampling when adjusting voxel spacing.
    pad_mode : str, default "constant"
        How to handle image edges during filtering or resampling.
    ball_radius : float, default 0.5
        Radius (µm) of spherical neighborhood for local intensity statistics.
    ball_stat : {"min", "mean", "max", "median"}, default "mean"
        Statistic to compute in local spherical neighborhood.
    ball_pad_mode : str, default "reflect"
        Padding mode for local spherical statistics.
    labels_dir : str | Path | None, optional
        Directory where labels TIFFs are saved to use for cropping. If None,
        segmentation will be performed on the intensity channels.
    output_dir : str | Path | None, optional
        Directory where output TIFFs and results will be saved.
        If None, a subdirectory named 'analysis' will be created in input_dir.
    overwrite : bool, default False
        If True, existing output directory will be removed and recreated.
        If False, raises an error if output_dir exists.

    Raises
    ------
    ValueError
        If `input_dir` is not a directory.
    FileExistsError
        If `output_dir` exists and `overwrite` is False.

    Notes
    -----
    - TIFF files in the input directory are assumed to contain multi-channel stacks with
    lamin channels.
    - Output files include cropped normalized lamin images and principal curvature maps
    (k1, k2).
    - Nuclei IDs are offset across images to ensure uniqueness.
    - Function uses `read_and_compute_principal_curvatures_single_stack_multi_nuclei`
      internally to process each image.
    """
    # Ensure channel_indices is a list
    if isinstance(channel_indices, int):
        channel_indices = [channel_indices]

    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise ValueError(f"input_dir ('{input_dir}') is not a directory.")

    image_files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in {".tif", ".tiff"}
    )
    if not image_files:
        raise FileNotFoundError(f"No TIFF files found in directory: {input_dir}")

    if labels_dir is not None:
        labels_dir = Path(labels_dir)
        for image_file in image_files:
            labels_path = labels_dir / f"{image_file.stem}_labels.tif"
            if not labels_path.exists():
                raise FileNotFoundError(f"The file at: {labels_path} does not exist.")

    if output_dir is None:
        output_dir = input_dir / "analysis"
    else:
        output_dir = Path(output_dir)

    if output_dir.exists():
        if overwrite:
            print(
                f"output_dir ('{output_dir}') exists, but overwrite=True — removing it."
            )
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise FileExistsError(
                f"output_dir ('{output_dir}') exists and overwrite=False."
            )
    else:
        print(f"output_dir ('{output_dir}') does not exist — creating directory.")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save all input parameters to settings.yaml for reproducibility
    settings_dict = {
        "input_dir": str(input_dir),
        "channel_indices": channel_indices,
        "wavelengths": wavelengths,
        "na": na,
        "magnification": magnification,
        "ri_sample": ri_sample,
        "ri_coverslip": ri_coverslip,
        "ri_immersion": ri_immersion,
        "confocal": confocal,
        "rough_lamin_mask_sigma": rough_lamin_mask_sigma,
        "rough_lamin_mask_ic_factor": rough_lamin_mask_ic_factor,
        "rough_lamin_mask_alpha": rough_lamin_mask_alpha,
        "fine_lamin_mask_sigma": fine_lamin_mask_sigma,
        "pc_scale": pc_scale,
        "expansion_distance": expansion_distance,
        "min_nucleus_area": min_nucleus_area,
        "axis_order": axis_order,
        "reduction_mode": reduction_mode,
        "sign_estimation_mode": sign_estimation_mode,
        "channel_idx_dapi": channel_idx_dapi,
        "wavelength_dapi": wavelength_dapi,
        "spacing_goal": spacing_goal,
        "order": order,
        "downsample_mode": downsample_mode,
        "pad_mode": pad_mode,
        "ball_radius": ball_radius,
        "ball_stat": ball_stat,
        "ball_pad_mode": ball_pad_mode,
        "labels_dir": str(labels_dir) if labels_dir is not None else None,
        "output_dir": str(output_dir),
        "overwrite": overwrite,
    }
    _write_out_settings(output_dir / "settings.yaml", settings_dict)

    csv_path = output_dir / "all_data.csv"
    first_write = True

    print(f"Start processing of the directory: {input_dir}")
    num_nuclei = 0
    for image_file in tqdm(image_files, desc="Processing image files: "):
        data = {
            "nucleus_id": [],
            "k1": [],
            "k2": [],
        }
        for ch in channel_indices:
            data[f"norm_voxel_intensity_ch{ch}"] = []
            data[f"norm_local_intensity_ch{ch}"] = []

        print(f"\nStart processing: {image_file}")

        labels_path = None
        if labels_dir is not None:
            labels_path = labels_dir / f"{image_file.stem}_labels.tif"
            print(f"Using labels at: {labels_path}")

        # Updated call including new parameters
        (
            nucleus_ids,
            images_cropped,
            local_intensities,
            k1,
            k2,
            spacing,
        ) = read_and_compute_principal_curvatures_single_stack_multi_nuclei(
            image_file,
            channel_indices,
            wavelengths,
            na,
            magnification,
            ri_sample,
            ri_coverslip,
            ri_immersion,
            confocal,
            rough_lamin_mask_sigma,
            rough_lamin_mask_ic_factor,
            rough_lamin_mask_alpha,
            fine_lamin_mask_sigma,
            pc_scale,
            expansion_distance,
            min_nucleus_area,
            axis_order=axis_order,
            reduction_mode=reduction_mode,
            sign_estimation_mode=sign_estimation_mode,
            channel_idx_dapi=channel_idx_dapi,
            wavelength_dapi=wavelength_dapi,
            spacing_goal=spacing_goal,
            order=order,
            downsample_mode=downsample_mode,
            pad_mode=pad_mode,
            ball_radius=ball_radius,
            ball_stat=ball_stat,
            ball_pad_mode=ball_pad_mode,
            labels_path=labels_path,
        )

        print("Finished analysis of image file.")

        # Offset nucleus IDs to keep them unique across images
        nucleus_ids = [nucleus_id + num_nuclei for nucleus_id in nucleus_ids]

        for j, nucleus_id in enumerate(nucleus_ids):
            id_tag = f"id{nucleus_id}"
            print(f"Writing out data for nucleus ID {nucleus_id}...")

            k1_j = np.asarray(k1[j], dtype=np.float32)
            k2_j = np.asarray(k2[j], dtype=np.float32)
            mask = (
                (~np.isnan(k1_j))
                & (np.abs(k1_j) > 0)
                & (~np.isnan(k2_j))
                & (np.abs(k2_j) > 0)
            )
            num_voxels = np.sum(mask)

            data["nucleus_id"].extend([nucleus_id] * num_voxels)
            data["k1"].extend(k1_j[mask].flatten())
            data["k2"].extend(k2_j[mask].flatten())

            for k, ch in enumerate(channel_indices):
                # --- Normalized voxel intensity ---
                img_voxel = np.asarray(images_cropped[j][k], dtype=np.float32)
                path_out = output_dir / f"{id_tag}_channel{ch}_{image_file.stem}.tif"
                io.write_tiff_with_spacing(
                    path_out, img_voxel, spacing, axis_order.upper()
                )
                data[f"norm_voxel_intensity_ch{ch}"].extend(img_voxel[mask].flatten())

                # --- Local (ball-averaged) normalized intensity ---
                img_local = np.asarray(local_intensities[j][k], dtype=np.float32)
                path_out = (
                    output_dir
                    / f"{id_tag}_channel{ch}_norm_local_intensity_{image_file.stem}.tif"
                )
                io.write_tiff_with_spacing(
                    path_out, img_local, spacing, axis_order.upper()
                )
                data[f"norm_local_intensity_ch{ch}"].extend(img_local[mask].flatten())

            for label, arr in [("k1", k1_j), ("k2", k2_j)]:
                path_out = output_dir / f"{id_tag}_{label}_{image_file.stem}.tif"
                io.write_tiff_with_spacing(path_out, arr, spacing, axis_order.upper())

        num_nuclei += len(nucleus_ids)

        df = pl.DataFrame(data)
        if first_write:
            df.write_csv(csv_path, include_header=True)
            first_write = False
        else:
            with open(csv_path, mode="a", encoding="utf-8") as f:
                df.write_csv(f, include_header=False)

    print(f"\nFinished processing. Output written to: {output_dir}")
