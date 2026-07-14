from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.ndimage import gaussian_filter

import imcurvpy.analysis as analysis
import imcurvpy.curvature as curvature
import imcurvpy.io as io
import imcurvpy.preprocessing as preprocessing
import imcurvpy.segment as segment
import imcurvpy.utils as utils

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_principal_curvatures_single_stack_single_nucleus(
    images: np.ndarray | Sequence[np.ndarray],
    spacings: Sequence[float, float, float] | Sequence[Sequence[float, float, float]],
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
    axis_order: str = "zyx",
    reduction_mode: Literal["max", "mean", "min"] = "max",
    sign_estimation_mode: Literal["dapi", "image"] | None = None,
    image_dapi: np.ndarray | None = None,
    spacing_dapi: tuple[float, float, float] | None = None,
    wavelength_dapi: float | None = None,
    spacing_goal: float | tuple[float, float, float] | Literal["min", "max"] = "max",
    order: int = 1,
    downsample_mode: str = "sum",
    pad_mode: str = "constant",
    ball_radius: float = 0.5,
    ball_stat: str = "mean",
    ball_pad_mode: str = "reflect",
) -> tuple[
    Sequence[np.ndarray],  # normalized images
    Sequence[np.ndarray],  # local_intensities
    np.ndarray,  # k1
    np.ndarray,  # k2
    tuple[float, float, float],  # spacing_goal
]:
    """
    Compute principal curvature components of the nuclear lamina for a single nucleus.

    This function normalizes and resamples multiple input fluorescence image stacks
    to isotropic voxel spacing, combines them via a specified reduction mode,
    generates lamina masks, and computes principal curvatures of the lamina surface.
    Optionally, a DAPI channel can be used for curvature sign estimation.

    Parameters
    ----------
    images : np.ndarray or Sequence[np.ndarray]
        One or more 3D fluorescence image arrays (e.g., Lamin channels).
        If a single ndarray is given, it will be wrapped into a list internally.
    spacings : Sequence[float, float, float] or Sequence[Sequence[float, float, float]]
        Voxel spacings (z, y, x) for each image stack in micrometers.
        A single spacing can be provided for all images.
    wavelengths : float or Sequence[float]
        Emission wavelength(s) for each image stack in micrometers.
        A single wavelength can be provided for all images.
    na : float
        Numerical aperture of the objective lens.
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
        Gaussian sigma (in micrometers) for smoothing the combined image during rough
        mask creation.
    rough_lamin_mask_ic_factor : float
        Intensity correction factor used in rough lamina segmentation.
    rough_lamin_mask_alpha : float
        Threshold alpha value for rough mask generation.
    fine_lamin_mask_sigma : float
        Gaussian sigma (in micrometers) for smoothing the fine lamina mask.
    pc_scale : float
        Physical scale (in micrometers) used for curvature estimation.
    axis_order : str, default 'zyx'
        Axis ordering of the input images; must be a permutation of 'xyz'.
    reduction_mode : {'max', 'mean', 'min'}, default 'max'
        Mode for combining multiple image stacks into a single composite image.
    sign_estimation_mode : {'dapi', 'image'} or None, optional
        Method to estimate curvature sign:
        - 'dapi': requires DAPI image and spacing to estimate sign.
        - 'image': estimates sign based on intensity.
        - None: no sign estimation.
    image_dapi : np.ndarray or None, optional
        DAPI channel image used if sign_estimation_mode='dapi'.
    spacing_dapi : tuple[float, float, float] or None, optional
        Voxel spacing for DAPI image in micrometers; required if image_dapi is provided.
    wavelength_dapi : float or None, optional
        Emission wavelength for DAPI image in micrometers.
    spacing_goal : float, tuple of float, or {"min", "max"}, default "max"
        Target voxel spacing for isotropization. Options:
            - float: isotropic spacing for all axes.
            - tuple of 3 floats: (Z, Y, X) spacing.
            - "min": smallest spacing among input images.
            - "max": largest spacing among input images.
    order : int, optional, default 1
        Interpolation order for resampling (0=nearest, 1=linear, 3=cubic).
    downsample_mode : str, optional, default "sum"
        Aggregation mode when downsampling: "sum", "mean", "min", "max".
    pad_mode : str, optional, default "constant"
        Padding mode used when downsampling if needed.
    ball_radius : float, default 0.5
        Radius in micrometer of the spherical neighborhood used for local intensity
        analysis.
    ball_stat : {'min', 'mean', 'max', 'median'}, default 'mean'
        Statistic to compute in the spherical neighborhood (see `local_ball_stats`).
    ball_pad_mode : str, default "reflect"
        Padding mode for local intensity computation, passed to `np.pad`.

    Returns
    -------
    images : list of np.ndarray
        Normalized and isotropically resampled fluorescence images corresponding
        to the input channels.
    local_intensities : list of np.ndarray
        Local intensity statistics computed within a spherical neighborhood
        (of radius `ball_radius`) around each voxel inside the fine lamina mask.
        One array per input image channel.
    k1 : np.ndarray
        First principal curvature component.
    k2 : np.ndarray
        Second principal curvature component.
    spacing_goal : tuple of float
        Final isotropic voxel spacing (z, y, x) in micrometers used after resampling.

    Raises
    ------
    ValueError
        If axis_order is invalid, if sign_estimation_mode is invalid,
        if reduction_mode is invalid, if input lengths or shapes do not match,
        or if required parameters are missing.

    Notes
    -----
    - Inputs are automatically normalized and resampled to isotropic resolution.
    - The lamina mask is generated using both rough and fine segmentation steps.
    - Curvatures are computed on a smoothed version of the fine lamina mask.
    - Local intensity features are computed via `analysis.local_ball_stats`,
      using the same mask, to quantify local texture or brightness around the lamina.
    """
    # Validate axis order
    if set(axis_order) != {"x", "y", "z"} or len(axis_order) != 3:
        raise ValueError("axis_order must be a permutation of 'xyz'")

    # Validate reduction mode
    if reduction_mode not in {"max", "mean", "min"}:
        raise ValueError("reduction_mode must be one of {'max', 'mean', 'min'}")

    # Validate sign estimation mode
    if sign_estimation_mode not in {None, "dapi", "image"}:
        raise ValueError("sign_estimation_mode must be one of {None, 'dapi', 'image'}")

    if sign_estimation_mode == "dapi":
        if image_dapi is None:
            raise ValueError(
                "image_dapi must be provided if sign_estimation_mode='dapi'"
            )
        if spacing_dapi is None:
            raise ValueError("spacing_dapi must be provided if image_dapi is given")
        if image_dapi.shape != images[0].shape:
            raise ValueError("image_dapi must have the same shape as images[0]")

    # Coerce single inputs into lists
    if isinstance(images, np.ndarray):
        images = [images]
    if isinstance(spacings[0], (int, float)):
        spacings = [spacings]  # type: ignore
    if isinstance(wavelengths, (int, float)):
        wavelengths = [wavelengths]  # type: ignore

    # Validate lengths
    if not (len(images) == len(spacings) == len(wavelengths)):
        raise ValueError(
            "images, spacings, and wavelengths must all have the same length"
        )

    # Validate shapes
    first_shape = images[0].shape
    for i, image in enumerate(images):
        if image.shape != first_shape:
            raise ValueError(
                f"images[{i}] has shape {image.shape}, expected {first_shape}"
            )

    # Validate spacings
    for i, spacing in enumerate(spacings):
        if len(spacing) != 3:
            raise ValueError(f"spacings[{i}] must have 3 elements")

    # Determine isotropic voxel spacing
    # Normalize spacing_goal to a 3-tuple
    if isinstance(spacing_goal, (int, float)):
        spacing_goal = (spacing_goal,) * 3
    elif isinstance(spacing_goal, str):
        if spacing_goal in ("min", "max"):
            voxel_spacing = (
                min([item for arr in spacings for item in arr])
                if spacing_goal == "min"
                else max([item for arr in spacings for item in arr])
            )
            spacing_goal = (voxel_spacing,) * 3
        else:
            raise ValueError(
                f"spacing_goal string must be 'min' or 'max', got {spacing_goal}"
            )
    elif isinstance(spacing_goal, (tuple, list)):
        if len(spacing_goal) != 3:
            raise ValueError(
                f"spacing_goal tuple/list must have 3 elements, got {spacing_goal}"
            )
        spacing_goal = tuple(spacing_goal)  # ensure it's a tuple
    else:
        raise ValueError(
            "spacing_goal must be a float, a 3-tuple/list, or 'min'/'max', got "
            f"{spacing_goal}"
        )

    # Determine wavelength goal (for PSF matching)
    wavelength_goal = max(wavelengths)

    def isotropize(image, spacing, wavelength):
        image, _ = preprocessing.make_image_fully_isotropic(
            image,
            spacing,
            wavelength,
            na=na,
            magnification=magnification,
            ri_sample=ri_sample,
            ri_coverslip=ri_coverslip,
            ri_immersion=ri_immersion,
            confocal=confocal,
            wavelength_goal=wavelength_goal,
            order=order,
            downsample_mode=downsample_mode,
            pad_mode=pad_mode,
        )
        return image

    print("Start normalizing images and making them isotropic...")
    for i in range(len(images)):
        images[i] = preprocessing.normalize(
            images[i], global_min=None, global_max=None, dtype=np.float32
        )
        images[i] = isotropize(images[i], spacings[i], wavelengths[i])

    dapi_soft_mask = None
    if sign_estimation_mode == "dapi" and image_dapi is not None:
        image_dapi = preprocessing.normalize(
            image_dapi, global_min=None, global_max=None, dtype=np.float32
        )
        image_dapi = isotropize(image_dapi, spacing_dapi, wavelength_dapi)
        dapi_soft_mask = segment.create_soft_dapi_mask(
            image_dapi, pc_scale, voxel_spacing
        )
    print("Finished normalization and making images isotropic.")

    # Combine lamin images
    image_combined = preprocessing.reduce_3d_arrays(images, mode=reduction_mode)

    print("Start masks creation...")
    rough_lamin_mask = segment.create_rough_lamin_mask(
        image=image_combined,
        voxel_spacing=voxel_spacing,
        sigma=rough_lamin_mask_sigma,
        ic_factor=rough_lamin_mask_ic_factor,
        alpha=rough_lamin_mask_alpha,
    )

    fine_lamin_mask = segment.create_fine_lamin_mask(
        image=image_combined,
        rough_mask=rough_lamin_mask,
        voxel_spacing=voxel_spacing,
        sigma_s=fine_lamin_mask_sigma,
    )
    print("Finished creation of masks.")
    print("Start computation of principal components...")

    # Smooth the fine lamin mask with a Gaussian blur of 1 px before computing the
    # curvature
    fine_lamin_mask_blurred = gaussian_filter(fine_lamin_mask.astype(np.float32), 1)

    k1, k2 = curvature.compute_principal_curvatures(
        image=fine_lamin_mask_blurred,  # Use the fine mask to compute the curvature
        # image=image_combined,
        mask=fine_lamin_mask,
        voxel_spacing=voxel_spacing,
        scale=pc_scale,
        axis_order=axis_order,
        sign_estimation_mode=sign_estimation_mode,
        dapi=dapi_soft_mask,
    )

    print("Finished computation of principal components.")

    print("Start local ball intensity statistic computation...")
    ball_radius_px = ball_radius / spacing_goal[0]
    local_intensities = []
    for i in range(len(images)):
        local_intensities.append(
            analysis.local_ball_stats(
                images[i],
                fine_lamin_mask,
                radius=ball_radius_px,
                stat=ball_stat,
                pad_mode=ball_pad_mode,
            )
        )
    print("Finished computation of principal components.")

    return images, local_intensities, k1, k2, spacing_goal


def read_and_compute_principal_curvatures_single_stack_single_nucleus(
    input_path: str | Path,
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
) -> tuple[
    Sequence[np.ndarray],  # normalized images
    Sequence[np.ndarray],  # local_intensities
    np.ndarray,  # k1
    np.ndarray,  # k2
    tuple[float, float, float],  # spacing_goal
]:
    """
    Load TIFF data and compute principal curvatures for a single nucleus.

    This function reads multiple fluorescence channels specified by `channel_indices`
    from a multi-channel TIFF file at `input_path`, along with optional DAPI channel.
    It fetches voxel spacings from image metadata, normalizes and resamples the data,
    combines multi-channel images by the specified `reduction_mode`, and then
    computes principal curvature components of the nuclear lamina.
    Optionally, curvature sign estimation can use DAPI or image intensity.

    Parameters
    ----------
    input_path : str or Path
        Filepath to the multi-channel TIFF file.
    channel_indices : int or Sequence[int]
        Index or list of indices for fluorescence channels to load (e.g., lamin
        channels).
    wavelengths : float or Sequence[float]
        Emission wavelength(s) for each fluorescence channel, in micrometers.
    na : float
        Numerical aperture of the objective lens.
    magnification : float
        Microscope magnification factor.
    ri_sample : float
        Refractive index of the sample.
    ri_coverslip : float
        Refractive index of the coverslip.
    ri_immersion : float
        Refractive index of the immersion medium.
    confocal : bool
        True if the imaging system is confocal.
    rough_lamin_mask_sigma : float
        Gaussian smoothing sigma (micrometers) for rough lamina mask.
    rough_lamin_mask_ic_factor : float
        Intensity correction factor for rough lamina mask.
    rough_lamin_mask_alpha : float
        Threshold alpha value for rough lamina mask.
    fine_lamin_mask_sigma : float
        Gaussian smoothing sigma (micrometers) for fine lamina mask.
    pc_scale : float
        Physical scale (micrometers) for curvature estimation.
    axis_order : str, default "zyx"
        Axis ordering of input image data (permutation of 'xyz').
    reduction_mode : {"max", "mean", "min"}, default "max"
        Method to combine multiple fluorescence channels into one image.
    sign_estimation_mode : {"dapi", "image"} or None, optional
        Method to estimate curvature sign:
        - "dapi": use DAPI channel,
        - "image": use intensity-based estimation,
        - None: skip sign estimation.
    channel_idx_dapi : int or None, optional
        Channel index for DAPI image, if sign_estimation_mode="dapi".
    wavelength_dapi : float or None, optional
        Emission wavelength for DAPI channel in micrometers.
    spacing_goal : float, tuple of float, or {"min", "max"}, default "max"
        Target voxel spacing for isotropization. Options:
            - float: isotropic spacing for all axes.
            - tuple of 3 floats: (Z, Y, X) spacing.
            - "min": smallest spacing among input images.
            - "max": largest spacing among input images.
    order : int, optional, default 1
        Interpolation order for resampling (0=nearest, 1=linear, 3=cubic).
    downsample_mode : str, optional, default "sum"
        Aggregation mode when downsampling: "sum", "mean", "min", "max".
    pad_mode : str, optional, default "constant"
        Padding mode used when downsampling if needed.
    ball_radius : float, default 0.5
        Radius in micrometer of the spherical neighborhood used for local intensity
        analysis.
    ball_stat : {'min', 'mean', 'max', 'median'}, default 'mean'
        Statistic to compute in the spherical neighborhood (see `local_ball_stats`).
    ball_pad_mode : str, default "reflect"
        Padding mode for local intensity computation, passed to `np.pad`.

    Returns
    -------
    images : list of np.ndarray
        Normalized and isotropically resampled fluorescence images corresponding
        to the input channels.
    local_intensities : list of np.ndarray
        Local intensity statistics computed within a spherical neighborhood
        (of radius `ball_radius`) around each voxel inside the fine lamina mask.
        One array per input image channel.
    k1 : np.ndarray
        First principal curvature component.
    k2 : np.ndarray
        Second principal curvature component.
    spacing_goal : tuple of float
        Final isotropic voxel spacing (z, y, x) in micrometers used after resampling.


    Raises
    ------
    ValueError
        If `channel_indices` contains non-integers,
        if axis_order is invalid,
        if reduction_mode is invalid,
        or if sign_estimation_mode is invalid.
    """
    # Coerce single inputs into lists
    if isinstance(channel_indices, int):
        channel_indices = [channel_indices]
    if isinstance(wavelengths, (int, float)):
        wavelengths = [wavelengths]  # type: ignore

    # Validate channel_indices contains only integers
    if not all(isinstance(idx, int) for idx in channel_indices):
        raise ValueError("All elements in channel_indices must be integers")

    images = []
    spacings = []

    for channel_idx in channel_indices:
        image_temp, spacing_temp, _ = io.read_tiff_select_indices(
            input_path, channel_idx=channel_idx
        )
        images.append(image_temp)
        spacings.append(spacing_temp)

    image_dapi = None
    spacing_dapi = None
    if channel_idx_dapi is not None:
        image_dapi, spacing_dapi, _ = io.read_tiff_select_indices(
            input_path, channel_idx=channel_idx_dapi
        )

    images, local_intensities, k1, k2, spacing_goal = (
        compute_principal_curvatures_single_stack_single_nucleus(
            images,
            spacings,
            wavelengths,
            na=na,
            magnification=magnification,
            ri_sample=ri_sample,
            ri_coverslip=ri_coverslip,
            ri_immersion=ri_immersion,
            confocal=confocal,
            rough_lamin_mask_sigma=rough_lamin_mask_sigma,
            rough_lamin_mask_ic_factor=rough_lamin_mask_ic_factor,
            rough_lamin_mask_alpha=rough_lamin_mask_alpha,
            fine_lamin_mask_sigma=fine_lamin_mask_sigma,
            pc_scale=pc_scale,
            axis_order=axis_order,
            reduction_mode=reduction_mode,
            sign_estimation_mode=sign_estimation_mode,
            image_dapi=image_dapi,
            spacing_dapi=spacing_dapi,
            wavelength_dapi=wavelength_dapi,
            spacing_goal=spacing_goal,
            order=order,
            downsample_mode=downsample_mode,
            pad_mode=pad_mode,
            ball_radius=ball_radius,
            ball_stat=ball_stat,
            ball_pad_mode=ball_pad_mode,
        )
    )

    return images, local_intensities, k1, k2, spacing_goal


def compute_principal_curvatures_single_stack_multi_nuclei(
    images: np.ndarray | Sequence[np.ndarray],
    spacings: Sequence[float, float, float] | Sequence[Sequence[float, float, float]],
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
    image_dapi: np.ndarray | None = None,
    spacing_dapi: tuple[float, float, float] | None = None,
    wavelength_dapi: float | None = None,
    spacing_goal: float | tuple[float, float, float] | Literal["min", "max"] = "max",
    order: int = 1,
    downsample_mode: str = "sum",
    pad_mode: str = "constant",
    ball_radius: float = 0.5,
    ball_stat: str = "mean",
    ball_pad_mode: str = "reflect",
    labels: np.ndarray | None = None,
) -> tuple[
    list[int],
    list[list[np.ndarray]],
    list[list[np.ndarray]],
    list[np.ndarray],
    list[np.ndarray],
    tuple[float, float, float],
]:
    """
    Compute the principal curvatures of multiple nuclei within a single 3D FOV.

    This function processes one or more 3D lamin fluorescence images, normalizes and
    resamples them to isotropic resolution, segments nuclei, and computes principal
    curvature components of the nuclear lamina for each segmented nucleus. An optional
    DAPI channel can be used for curvature sign estimation.

    Parameters
    ----------
    images : np.ndarray | Sequence[np.ndarray]
        One or more 3D image volumes of lamin markers. All must have the same shape.
    spacings : Sequence[float, float, float] | Sequence[Sequence[float, float, float]]
        Voxel spacings (z, y, x) for each image, in micrometers.
    wavelengths : float | Sequence[float]
        Emission wavelength(s) of the input images, in micrometers.
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
        Whether imaging was performed using a confocal microscope.
    rough_lamin_mask_sigma : float
        Gaussian smoothing sigma (µm) for rough lamin mask creation.
    rough_lamin_mask_ic_factor : float
        Intensity correction factor for rough lamin mask segmentation.
    rough_lamin_mask_alpha : float
        Threshold alpha for rough lamin mask generation.
    fine_lamin_mask_sigma : float
        Gaussian smoothing sigma (µm) for fine lamin mask creation.
    pc_scale : float
        Physical scale (µm) for curvature estimation.
    expansion_distance : float
        Distance (µm) for Voronoi expansion of nucleus labels.
    min_nucleus_area : float
        Minimum 2D area (µm²) for a nucleus to be included in the analysis.
    axis_order : str, default "zyx"
        Axis order of the input images. Must be a permutation of 'x', 'y', and 'z'.
    reduction_mode : {"max", "mean", "min"}, default "max"
        Method for combining multiple lamin images into a single volume.
    sign_estimation_mode : {"dapi", "image"} | None, optional
        Method for estimating curvature sign:
        - "dapi": Uses the DAPI channel.
        - "image": Uses intensity-based heuristics.
        - None: Disables sign estimation.
    image_dapi : np.ndarray | None, optional
        3D DAPI image for sign estimation. Must match the shape of the first lamin
        image.
    spacing_dapi : tuple[float, float, float] | None, optional
        Voxel spacing (z, y, x) for the DAPI image, in micrometers.
    wavelength_dapi : float | None, optional
        Emission wavelength of the DAPI channel, in micrometers.
    spacing_goal : float, tuple of float, or {"min", "max"}, default "max"
        Target voxel spacing for isotropization. Should be isotropic. Options:
            - float: isotropic spacing for all axes.
            - tuple of 3 floats: (Z, Y, X) spacing.
            - "min": smallest spacing among input images.
            - "max": largest spacing among input images.
    order : int, optional, default 1
        Interpolation order for resampling (0=nearest, 1=linear, 3=cubic).
    downsample_mode : str, optional, default "sum"
        Aggregation mode when downsampling: "sum", "mean", "min", "max".
    pad_mode : str, optional, default "constant"
        Padding mode used when downsampling if needed.
    ball_radius : float, default 0.5
        Radius in micrometer of the spherical neighborhood used for local intensity
        analysis.
    ball_stat : {'min', 'mean', 'max', 'median'}, default 'mean'
        Statistic to compute in the spherical neighborhood (see `local_ball_stats`).
    ball_pad_mode : str, default "reflect"
        Padding mode for local intensity computation, passed to `np.pad`.
    labels : np.ndarray | None
        3D array with labels. Should have the same size as the images. If None, then
        the intensity images will be used for segmentation.

    Returns
    -------
    nucleus_ids : list of int
        Integer labels identifying nuclei included in analysis.
    all_images_cropped : list of list of np.ndarray
        Per-nucleus list of cropped and normalized lamin volumes, one per input image.
    all_local_intensities : list of list of np.ndarray
        Per-nucleus list of local intensity statistic maps for each lamin channel.
        Each map is computed using a spherical neighborhood (`ball_radius`) within
        the fine lamina mask.
    all_k1 : list of np.ndarray
        First principal curvature component per nucleus.
    all_k2 : list of np.ndarray
        Second principal curvature component per nucleus.
    spacing_goal : tuple of float
        Final isotropic voxel spacing (z, y, x), in micrometers.

    Raises
    ------
    ValueError
        If axis_order is invalid.
        If input shapes are inconsistent or required metadata is missing.
        If reduction_mode or sign_estimation_mode is invalid.

    Notes
    -----
    - All input images are normalized to [0, 1] and resampled to isotropic spacing.
    - Nuclei are segmented from the combined lamin image and processed independently.
    - Curvature is estimated using fine lamin masks per nucleus.
    - Curvature sign estimation is optional and based on DAPI or image intensity.
    """
    # Validate axis order
    if set(axis_order) != {"x", "y", "z"} or len(axis_order) != 3:
        raise ValueError("axis_order must be a permutation of 'xyz'")

    # Validate reduction mode
    if reduction_mode not in {"max", "mean", "min"}:
        raise ValueError("reduction_mode must be one of {'max', 'mean', 'min'}")

    # Validate sign estimation mode
    if sign_estimation_mode not in {None, "dapi", "image"}:
        raise ValueError("sign_estimation_mode must be one of {None, 'dapi', 'image'}")

    if sign_estimation_mode == "dapi":
        if image_dapi is None:
            raise ValueError(
                "image_dapi must be provided if sign_estimation_mode='dapi'"
            )
        if spacing_dapi is None:
            raise ValueError("spacing_dapi must be provided if image_dapi is given")
        if image_dapi.shape != images[0].shape:
            raise ValueError("image_dapi must have the same shape as images[0]")

    # Coerce single inputs into lists
    if isinstance(images, np.ndarray):
        images = [images]
    if isinstance(spacings[0], (int, float)):
        spacings = [spacings]  # type: ignore
    if isinstance(wavelengths, (int, float)):
        wavelengths = [wavelengths]  # type: ignore

    # Validate lengths
    if not (len(images) == len(spacings) == len(wavelengths)):
        raise ValueError(
            "images, spacings, and wavelengths must all have the same length"
        )

    # Validate shapes
    first_shape = images[0].shape
    for i, image in enumerate(images):
        if image.shape != first_shape:
            raise ValueError(
                f"images[{i}] has shape {image.shape}, expected {first_shape}"
            )

    # Validate spacings
    for i, spacing in enumerate(spacings):
        if len(spacing) != 3:
            raise ValueError(f"spacings[{i}] must have 3 elements")

    # Determine isotropic voxel spacing
    # Normalize spacing_goal to a 3-tuple
    if isinstance(spacing_goal, (int, float)):
        spacing_goal = (spacing_goal,) * 3
    elif isinstance(spacing_goal, str):
        if spacing_goal in ("min", "max"):
            voxel_spacing = (
                min([item for arr in spacings for item in arr])
                if spacing_goal == "min"
                else max([item for arr in spacings for item in arr])
            )
            spacing_goal = (voxel_spacing,) * 3
        else:
            raise ValueError(
                f"spacing_goal string must be 'min' or 'max', got {spacing_goal}"
            )
    elif isinstance(spacing_goal, (tuple, list)):
        if len(spacing_goal) != 3:
            raise ValueError(
                f"spacing_goal tuple/list must have 3 elements, got {spacing_goal}"
            )
        spacing_goal = tuple(spacing_goal)  # ensure it's a tuple
    else:
        raise ValueError(
            "spacing_goal must be a float, a 3-tuple/list, or 'min'/'max', got "
            f"{spacing_goal}"
        )

    # Validate that spacing_goal is isotropic
    if not (spacing_goal[0] == spacing_goal[1] == spacing_goal[2]):
        raise ValueError(
            f"spacing_goal must be isotropic (all three values equal), got {spacing_goal}"
        )

    # Determine wavelength goal (for PSF matching)
    wavelength_goal = max(wavelengths)

    def isotropize(image, spacing, wavelength):
        image, _ = preprocessing.make_image_fully_isotropic(
            image,
            spacing,
            wavelength,
            na=na,
            magnification=magnification,
            ri_sample=ri_sample,
            ri_coverslip=ri_coverslip,
            ri_immersion=ri_immersion,
            confocal=confocal,
            wavelength_goal=wavelength_goal,
            order=order,
            downsample_mode=downsample_mode,
            pad_mode=pad_mode,
        )
        return image

    print("Start analysis of stack...")
    dapi_soft_mask = None
    if sign_estimation_mode == "dapi" and image_dapi is not None:
        image_dapi = preprocessing.normalize(
            image_dapi, global_min=None, global_max=None, dtype=np.float32
        )
        image_dapi = isotropize(image_dapi, spacing_dapi, wavelength_dapi)
        dapi_soft_mask = segment.create_soft_dapi_mask(
            image_dapi, pc_scale, voxel_spacing
        )

    # Combine lamin images
    image_combined = preprocessing.reduce_3d_arrays(images, mode=reduction_mode)

    if labels is not None:
        # Validate labels
        if not isinstance(labels, np.ndarray):
            raise ValueError("labels must be a numpy array if provided")
        if labels.ndim != 3:
            raise ValueError(f"labels must be 3D, got {labels.ndim}D array")
        if labels.shape != image_combined.shape:
            raise ValueError(
                f"labels shape {labels.shape} must match image_combined shape "
                f"{image_combined.shape}"
            )
    elif labels is None:
        # Perform segmentation because the labels are not provided
        # We do not need to isotropize in the z-direction as we will be taking the
        # maximum intensity projection
        print(
            "Start normalizing images and making them isotropic in the xy-direction..."
        )
        images_temp = []
        for i in range(len(images)):
            images_temp.append(
                preprocessing.normalize(
                    images[i], global_min=None, global_max=None, dtype=np.float32
                )
            )

            if wavelengths[i] < wavelength_goal:
                # Compute the sigmas of the PSF of the provided image
                sigma_lat_current, sigma_ax_current = preprocessing.estimate_psf_sigmas(
                    wavelengths[i],
                    na,
                    magnification,
                    ns=ri_sample,
                    ng=ri_coverslip,
                    ni=ri_immersion,
                    confocal=confocal,
                )

                sigma_lat_goal, sigma_ax_goal = preprocessing.estimate_psf_sigmas(
                    wavelength_goal,
                    na,
                    magnification,
                    ns=ri_sample,
                    ng=ri_coverslip,
                    ni=ri_immersion,
                    confocal=confocal,
                )

                images_temp[i] = preprocessing.blur_xy_to_match_lateral_resolution(
                    images_temp[i], sigma_lat_current, sigma_lat_goal
                )

        print("Finished normalization and making them isotropic in the xy-direction.")

        # Combine lamin images
        image_temp_combined = preprocessing.reduce_3d_arrays(
            images_temp, mode=reduction_mode
        )

        # Labels is already 3d
        labels = segment.segment_and_label_nuclei(
            image_temp_combined,
            spacing_goal[0],
            expansion_distance=expansion_distance,
            min_nucleus_area=min_nucleus_area,
            axis_order=axis_order,
        )

        del images_temp, image_temp_combined

    # Collect valid (non-zero) label IDs (zero is background)
    valid_labels = set(np.unique(labels))
    if 0 in valid_labels:
        valid_labels.remove(0)

    nucleus_ids = []
    all_images_cropped = []
    all_local_intensities = []
    all_k1 = []
    all_k2 = []

    print(f"Start analysis for {len(valid_labels)} nucleus/nuclei...")

    for nucleus_id, label in enumerate(valid_labels):
        print(f"\nAnalyzing nucleus ID {nucleus_id:d}...")

        label_mask = labels == label

        # --- Crop and preprocess per nucleus ---
        images_cropped = []
        for i in range(len(images)):
            image_cropped = segment.crop_image(images[i], label_mask)
            image_cropped = preprocessing.normalize(image_cropped)
            image_cropped = isotropize(image_cropped, spacings[i], wavelengths[i])
            images_cropped.append(image_cropped)

        image_combined_cropped = preprocessing.reduce_3d_arrays(
            images_cropped, mode=reduction_mode
        )

        # --- Mask creation ---
        rough_lamin_mask = segment.create_rough_lamin_mask(
            image_combined_cropped,
            spacing_goal[0],
            rough_lamin_mask_sigma,
            rough_lamin_mask_ic_factor,
            rough_lamin_mask_alpha,
        )

        fine_lamin_mask = segment.create_fine_lamin_mask(
            image_combined_cropped,
            rough_lamin_mask,
            spacing_goal[0],
            fine_lamin_mask_sigma,
        )

        # --- Optional DAPI cropping for curvature sign estimation ---
        if sign_estimation_mode == "dapi" and dapi_soft_mask is not None:
            dapi_cropped = segment.crop_image(dapi_soft_mask, label_mask)
        else:
            dapi_cropped = None

        # --- Principal curvature computation ---
        print("  Computing principal curvatures...")
        fine_lamin_mask_blurred = gaussian_filter(fine_lamin_mask.astype(np.float32), 1)

        k1, k2 = curvature.compute_principal_curvatures(
            image=fine_lamin_mask_blurred,
            mask=fine_lamin_mask,
            voxel_spacing=spacing_goal[0],
            scale=pc_scale,
            axis_order=axis_order,
            sign_estimation_mode=sign_estimation_mode,
            dapi=dapi_cropped,
        )

        # --- Local intensity statistics ---
        print("  Computing local intensity statistics...")
        ball_radius_px = ball_radius / spacing_goal[0]
        local_intensities = []
        for i in range(len(images_cropped)):
            local_map = analysis.local_ball_stats(
                images_cropped[i],
                fine_lamin_mask,
                radius=ball_radius_px,
                stat=ball_stat,
                pad_mode=ball_pad_mode,
            )
            local_intensities.append(local_map)

        # --- Store results ---
        nucleus_ids.append(nucleus_id)
        all_images_cropped.append(images_cropped)
        all_local_intensities.append(local_intensities)
        all_k1.append(k1)
        all_k2.append(k2)

        print(f"  Finished nucleus {nucleus_id:d}.")

    print("Completed analysis for all nuclei.")

    return (
        nucleus_ids,
        all_images_cropped,
        all_local_intensities,
        all_k1,
        all_k2,
        spacing_goal,
    )


def read_and_compute_principal_curvatures_single_stack_multi_nuclei(
    input_path: str | Path,
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
    labels_path: str | Path | None = None,
) -> tuple[
    list[int],
    list[list[np.ndarray]],
    list[list[np.ndarray]],
    list[np.ndarray],
    list[np.ndarray],
    tuple[float, float, float],
]:
    """
    Read TIFF file and compute principal curvatures of nuclei in a single stack.

    This function loads lamin channels (and optionally DAPI) from a multi-channel TIFF
    stack, reads voxel spacing metadata, segments nuclei from the combined lamin image,
    and computes the principal curvature components of the nuclear lamina for each
    nucleus. Curvature signs can optionally be estimated using the DAPI channel.

    Parameters
    ----------
    input_path : str | Path
        Path to the multi-channel TIFF file.
    channel_indices : int | Sequence[int]
        Index or indices of the lamin channels to process (e.g., Lamin A/C, Lamin B1).
    wavelengths : float | Sequence[float]
        Emission wavelength(s) for the lamin channels, in micrometers.
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
        Gaussian smoothing sigma (µm) for rough lamin mask creation.
    rough_lamin_mask_ic_factor : float
        Intensity correction factor for rough lamin mask segmentation.
    rough_lamin_mask_alpha : float
        Threshold alpha for rough lamin mask generation.
    fine_lamin_mask_sigma : float
        Gaussian smoothing sigma (µm) for fine lamin mask creation.
    pc_scale : float
        Physical scale (µm) for curvature estimation.
    expansion_distance : float
        Distance (µm) for Voronoi expansion of nucleus labels.
    min_nucleus_area : float
        Minimum 2D area (µm²) for a nucleus to be included in analysis.
    axis_order : str, default "zyx"
        Axis order of the image data. Must be a permutation of "x", "y", and "z".
    reduction_mode : {"max", "mean", "min"}, default "max"
        Method used to combine multiple lamin channels into a single volume.
    sign_estimation_mode : {"dapi", "image"} | None, optional
        Method for estimating curvature sign:
        - "dapi": Uses DAPI signal.
        - "image": Uses image intensity.
        - None: Disables sign estimation.
    channel_idx_dapi : int | None, optional
        Channel index for the DAPI image, if used for sign estimation.
    wavelength_dapi : float | None, optional
        Emission wavelength of the DAPI channel, in micrometers.
    spacing_goal : float, tuple of float, or {"min", "max"}, default "max"
        Target voxel spacing for isotropization.
    order : int, optional, default 1
        Interpolation order for resampling (0=nearest, 1=linear, 3=cubic).
    downsample_mode : str, optional, default "sum"
        Aggregation mode when downsampling: "sum", "mean", "min", "max".
    pad_mode : str, optional, default "constant"
        Padding mode used when downsampling if needed.
    ball_radius : float, default 0.5
        Radius (µm) of spherical neighborhood for local intensity statistics.
    ball_stat : {'min', 'mean', 'max', 'median'}, default "mean"
        Statistic computed in the spherical neighborhood.
    ball_pad_mode : str, default "reflect"
        Padding mode used for local intensity computation.
    labels_path : str | Path | None
        Path to precomputed labels TIFF for cropping. If None, segmentation is
        performed.

    Returns
    -------
    nucleus_ids : list of int
        Integer labels identifying nuclei included in analysis.
    all_images_cropped : list of list of np.ndarray
        Per-nucleus list of cropped and normalized lamin volumes, one per input image.
    all_local_intensities : list of list of np.ndarray
        Per-nucleus list of local intensity statistic maps for each lamin channel.
    all_k1 : list of np.ndarray
        First principal curvature component per nucleus.
    all_k2 : list of np.ndarray
        Second principal curvature component per nucleus.
    spacing_goal : tuple of float
        Final isotropic voxel spacing (z, y, x), in micrometers.
    """
    # --- Coerce inputs ---
    if isinstance(channel_indices, int):
        channel_indices = [channel_indices]
    if isinstance(wavelengths, (int, float)):
        wavelengths = [wavelengths]  # type: ignore

    if not all(isinstance(idx, int) for idx in channel_indices):
        raise ValueError("All elements in channel_indices must be integers")

    # --- Read lamin channels ---
    images = []
    spacings = []
    for channel_idx in channel_indices:
        image_temp, spacing_temp, axes = io.read_tiff_select_indices(
            input_path, channel_idx=channel_idx
        )
        images.append(image_temp)
        spacings.append(spacing_temp)

    # --- Read optional DAPI channel ---
    image_dapi = None
    spacing_dapi = None
    if channel_idx_dapi is not None:
        image_dapi, spacing_dapi, _ = io.read_tiff_select_indices(
            input_path, channel_idx=channel_idx_dapi
        )

    # --- Load labels if provided ---
    labels = None
    if labels_path is not None:
        labels_path = Path(labels_path)
        if not labels_path.exists():
            raise FileNotFoundError(f"The file at: {labels_path} does not exist.")
        labels, _, labels_axes = io.read_tiff_with_spacing(labels_path)
        labels = utils.broadcast_to_3d(
            labels, labels_axes, axes, images[0].shape[axes.upper().index("Z")]
        )

    # --- Compute curvatures ---
    (
        nucleus_ids,
        all_images_cropped,
        all_local_intensities,
        all_k1,
        all_k2,
        spacing_goal_final,
    ) = compute_principal_curvatures_single_stack_multi_nuclei(
        images=images,
        spacings=spacings,
        wavelengths=wavelengths,
        na=na,
        magnification=magnification,
        ri_sample=ri_sample,
        ri_coverslip=ri_coverslip,
        ri_immersion=ri_immersion,
        confocal=confocal,
        rough_lamin_mask_sigma=rough_lamin_mask_sigma,
        rough_lamin_mask_ic_factor=rough_lamin_mask_ic_factor,
        rough_lamin_mask_alpha=rough_lamin_mask_alpha,
        fine_lamin_mask_sigma=fine_lamin_mask_sigma,
        pc_scale=pc_scale,
        expansion_distance=expansion_distance,
        min_nucleus_area=min_nucleus_area,
        axis_order=axis_order,
        reduction_mode=reduction_mode,
        sign_estimation_mode=sign_estimation_mode,
        image_dapi=image_dapi,
        spacing_dapi=spacing_dapi,
        wavelength_dapi=wavelength_dapi,
        spacing_goal=spacing_goal,
        order=order,
        downsample_mode=downsample_mode,
        pad_mode=pad_mode,
        ball_radius=ball_radius,
        ball_stat=ball_stat,
        ball_pad_mode=ball_pad_mode,
        labels=labels,
    )

    return (
        nucleus_ids,
        all_images_cropped,
        all_local_intensities,
        all_k1,
        all_k2,
        spacing_goal_final,
    )
