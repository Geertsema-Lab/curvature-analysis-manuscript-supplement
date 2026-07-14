"""Collection of preprocessing functions."""

import warnings
from typing import Literal
import math

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter, zoom

from .psf import estimate_psf_sigmas


def reduce_3d_arrays(arrays, mode="mean"):
    """
    Reduce a list of 3D NumPy arrays elementwise using a specified operation.

    Parameters
    ----------
    arrays : list of np.ndarray
        A list of 3D arrays of equal shape to be reduced elementwise.
    mode : {"mean", "max", "min"}, optional
        The reduction operation to apply:
        - "mean": compute the elementwise average
        - "max" : compute the elementwise maximum
        - "min" : compute the elementwise minimum
        Default is "mean".

    Returns
    -------
    np.ndarray
        A 3D array of the same shape as the inputs, containing the reduced values.

    Raises
    ------
    ValueError
        If the input list is empty, contains arrays of mismatched shape or dimension,
        or if an unsupported mode is provided.

    Examples
    --------
    >>> a = np.ones((2, 3, 4))
    >>> b = np.zeros((2, 3, 4))
    >>> reduce_3d_arrays([a, b], mode="max")
    array([[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]],
           ...
    """
    if not arrays:
        raise ValueError("Input list of arrays must not be empty.")

    first_shape = arrays[0].shape
    for i, arr in enumerate(arrays):
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"Item {i} is not a NumPy array.")
        if arr.ndim != 3:
            raise ValueError(f"Array {i} is not 3-dimensional.")
        if arr.shape != first_shape:
            raise ValueError(
                f"Array {i} has shape {arr.shape}, expected {first_shape}."
            )

    if mode == "mean":
        total = np.zeros_like(arrays[0], dtype=np.float64)
        for arr in arrays:
            total += arr
        return total / len(arrays)

    elif mode == "max":
        result = np.copy(arrays[0])
        for arr in arrays[1:]:
            np.maximum(result, arr, out=result)
        return result

    elif mode == "min":
        result = np.copy(arrays[0])
        for arr in arrays[1:]:
            np.minimum(result, arr, out=result)
        return result

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def normalize(
    image: np.ndarray,
    global_min: float | None = None,
    global_max: float | None = None,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
    """
    Normalize image to the [0, 1] range based on specified or inferred min/max values.

    This function scales the input image linearly so that the smallest value maps to 0
    and the largest to 1. Optionally, the output can be cast to a specific
    floating-point type.

    Parameters
    ----------
    image : np.ndarray
        Input image array to normalize (can be any dimensionality).
    global_min : float or None, optional
        Minimum value to use for normalization. If None, uses the minimum of `image`.
    global_max : float or None, optional
        Maximum value to use for normalization. If None, uses the maximum of `image`.
    dtype : numpy.typing.DTypeLike, optional
        The desired floating-point data type of the output. Must be a float type.
        Default is `np.float32`.

    Returns
    -------
    np.ndarray
        Normalized array with values scaled to the range [0, 1] and cast to `dtype`.

    Raises
    ------
    ValueError
        If `global_max` is less than or equal to `global_min`, or if `dtype` is not a
        floating-point type.
    """
    # Validate dtype is a float type
    if not np.issubdtype(np.dtype(dtype), np.floating):
        raise ValueError(f"dtype must be a floating-point type, got {dtype}")

    # Determine min and max values for normalization if not provided
    if global_min is None:
        # global_min = np.min(image)
        global_min = np.iinfo(image.dtype).min
    if global_max is None:
        # global_max = np.max(image)
        global_max = np.iinfo(image.dtype).max

    if global_max <= global_min:
        raise ValueError(
            "global_max must be greater than global_min for normalization."
        )

    # Perform linear normalization to [0, 1]
    img_norm = (image.astype(dtype) - global_min) / (global_max - global_min)

    return img_norm


def downsample_centered(
    arr: np.ndarray,
    factors: tuple[int, ...],
    mode: str = "sum",
    pad_mode: str = "constant",
    **pad_kwargs,
) -> np.ndarray:
    """
    Downsample an N-dimensional array by integer factors, preserving the center.

    The array is symmetrically padded (if needed) so that its shape is divisible by
    the downsampling factors. It is then reshaped into non-overlapping
    blocks and reduced along those blocks according to the specified aggregation mode.

    Parameters
    ----------
    arr : np.ndarray
        Input N-dimensional array to downsample (e.g., 2D image or 3D volume).
    factors : tuple of int
        Downsampling factors for each dimension. Must have the same length as arr.ndim.
    mode : str, optional
        Aggregation mode to apply to each block:
        - "sum" : sum over blocks (preserves total intensity)
        - "mean": average over blocks (preserves mean intensity)
        - "min" : minimum value within each block
        - "max" : maximum value within each block
        Default is "sum".
    pad_mode : str, optional
        Padding mode passed directly to `numpy.pad`. Default is "constant".
        Supports all NumPy padding modes, including:
        {"constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum",
         "reflect", "symmetric", "wrap", "empty"}, or a custom padding function.
    **pad_kwargs : dict, optional
        Additional keyword arguments passed to `numpy.pad`. For example:
        - constant_values=0
        - end_values=1.0
        - stat_length=2

    Returns
    -------
    downsampled : np.ndarray
        Downsampled array after padding and aggregating over integer-sized blocks.

    Raises
    ------
    ValueError
        If the number of factors doesn't match the array dimensions,
        or if `mode` is not one of the allowed aggregation types.

    Notes
    -----
    - Padding is applied symmetrically around each dimension.
    - The spatial center of the original array is preserved in the downsampled result.
    - Using `pad_mode="edge"` or `"reflect"` helps avoid artificial edge effects
      when using "mean", "min", or "max" aggregation.
    """
    if len(factors) != arr.ndim:
        raise ValueError("Number of downsampling factors must match array dimensions")
    if mode not in {"sum", "mean", "min", "max"}:
        raise ValueError("Mode must be one of {'sum', 'mean', 'min', 'max'}")

    # Symmetric padding so shape is divisible by factors
    pads = []
    for size, f in zip(arr.shape, factors, strict=True):
        pad_total = (-size) % f
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        pads.append((pad_before, pad_after))

    arr_padded = np.pad(arr, pads, mode=pad_mode, **pad_kwargs)

    # Reshape into blocks
    # Split each axis into (output_size, block_size), then reduce over the
    # block_size axes to perform block-wise downsampling.
    #
    # Example (1D):
    #   [0, 1, 2, 3, 4, 5]  --factor=2-->
    #   [[0, 1],
    #    [2, 3],
    #    [4, 5]]
    #   sum(axis=1)  -> [1, 5, 9]
    #   mean(axis=1) -> [0.5, 2.5, 4.5]
    new_shape = tuple(arr_padded.shape[i] // factors[i] for i in range(arr.ndim))
    reshape_pattern = []
    for i in range(arr.ndim):
        reshape_pattern.extend([new_shape[i], factors[i]])

    arr_reshaped = arr_padded.reshape(*reshape_pattern)
    axes_to_reduce = tuple(range(1, arr_reshaped.ndim, 2))

    # Apply aggregation
    if mode == "sum":
        out = arr_reshaped.sum(axis=axes_to_reduce)
    elif mode == "mean":
        out = arr_reshaped.mean(axis=axes_to_reduce)
    elif mode == "min":
        out = arr_reshaped.min(axis=axes_to_reduce)
    elif mode == "max":
        out = arr_reshaped.max(axis=axes_to_reduce)

    return out


def downsample_by_spacing(
    arr: np.ndarray,
    orig_spacing: tuple[float, ...],
    new_spacing: tuple[float, ...],
    mode: str = "sum",
    pad_mode: str = "constant",
    **pad_kwargs,
) -> np.ndarray:
    """
    Downsample an array based on physical voxel or pixel spacing, preserving the center.

    Each new spacing must be an integer multiple of the corresponding original spacing.
    The downsampling is performed by symmetric block-wise aggregation using the
    specified mode and padding strategy.

    Parameters
    ----------
    arr : np.ndarray
        Input array (e.g., 2D image or 3D volume).
    orig_spacing : tuple of float
        Original voxel/pixel spacing per dimension (e.g., in micrometers or nanometers).
    new_spacing : tuple of float
        Target voxel/pixel spacing per dimension. Each component must be an
        integer multiple of the corresponding value in `orig_spacing`.
    mode : str, optional
        Aggregation mode used for block reduction:
        - "sum" : sum over blocks (preserves total intensity)
        - "mean": average over blocks (preserves mean intensity)
        - "min" : minimum value within each block
        - "max" : maximum value within each block
        Default is "sum".
    pad_mode : str, optional
        Padding mode to use if array dimensions are not divisible by downsampling
        factors.
        Passed directly to `numpy.pad`. Default is "constant".
        Any valid NumPy pad mode can be used, such as:
        {"constant", "edge", "reflect", "symmetric", "wrap", etc.}
    **pad_kwargs : dict, optional
        Additional keyword arguments passed to `numpy.pad` (e.g., constant_values).

    Returns
    -------
    downsampled : np.ndarray
        Downsampled array corresponding to the new voxel spacing.

    Raises
    ------
    ValueError
        If spacings do not match the number of array dimensions,
        if `new_spacing` is not an integer multiple of `orig_spacing`,
        or if `mode` is invalid.

    Examples
    --------
    >>> voxel_spacing1 = (0.01, 0.01, 0.01)  # 10 nm per voxel
    >>> voxel_spacing2 = (0.16, 0.16, 0.16)  # 160 nm per voxel
    >>> voxel_spacing3 = (0.16, 0.08, 0.08)  # anisotropic spacing
    >>> arr_down1 = downsample_by_spacing(
    ...     arr, voxel_spacing1, voxel_spacing2, mode="sum"
    ... )
    >>> arr_down2 = downsample_by_spacing(
    ...     arr, voxel_spacing1, voxel_spacing3, mode="mean", pad_mode="reflect"
    ... )

    Notes
    -----
    - All spacings **must be in the same physical units** (e.g., µm or nm).
      Mixing units (e.g., µm for one and nm for another) will yield incorrect results.
    - Padding is applied symmetrically to maintain centering.
    - `pad_mode="edge"` or `"reflect"` is often better for "mean", "min", or "max"
      to avoid bias from zero-padding at edges.
    - You can pass extra keyword arguments like `constant_values=1.0` for constant padding.
    """
    if len(orig_spacing) != arr.ndim or len(new_spacing) != arr.ndim:
        raise ValueError("Spacing tuples must match array dimensionality")

    factors = []
    for orig, new in zip(orig_spacing, new_spacing, strict=True):
        factor = new / orig
        if not factor.is_integer():
            raise ValueError(
                f"New spacing {new} is not an integer multiple of original spacing {orig}"
            )
        factors.append(int(factor))

    return downsample_centered(
        arr, tuple(factors), mode=mode, pad_mode=pad_mode, **pad_kwargs
    )


def interpolate_to_spacing(
    image: np.ndarray,
    spacing_current: tuple[float, float, float],
    spacing_goal: tuple[float, float, float],
    order: int = 1,
    clip: bool = True,
) -> np.ndarray:
    """
    Interpolate a 3D image volume to a specified voxel spacing.

    This function rescales a 3D image volume from its current voxel spacing
    to a new target spacing using spline interpolation. It is especially useful
    for standardizing resolution across datasets or preparing image volumes
    for analysis methods requiring consistent voxel sizes.

    Parameters
    ----------
    image : np.ndarray
        Input 3D image volume, ordered as (Z, Y, X).
    spacing_current : tuple of float
        Current voxel spacing in physical units (e.g., micrometers or nanometers),
        ordered as (Z, Y, X).
    spacing_goal : tuple of float
        Desired voxel spacing in the same physical units and axis order.
    order : int, optional
        The order of spline interpolation:
        - 0 : nearest neighbor
        - 1 : linear (default)
        - 3 : cubic
        - 5 : quintic
    clip : bool, optional
        If True, clips output values to [0, 1] range (useful for normalized images).
        Default is True.

    Returns
    -------
    np.ndarray
        Interpolated 3D image with the target voxel spacing.

    Notes
    -----
    - Uses `scipy.ndimage.zoom` for interpolation.
    - Assumes image is in (Z, Y, X) axis order.
    - For label images or masks, use `order=0` to prevent interpolation artifacts.
    - If `clip=False`, output values may slightly exceed [0, 1] due to spline interpolation.

    Examples
    --------
    >>> spacing_current = (0.3, 0.1, 0.1)  # µm per voxel (Z, Y, X)
    >>> spacing_goal = (0.2, 0.2, 0.2)
    >>> img_resampled = interpolate_to_spacing(
    ...     image, spacing_current, spacing_goal, order=3
    ... )
    """
    spacing_current = np.asarray(spacing_current, dtype=np.float32)
    spacing_goal = np.asarray(spacing_goal, dtype=np.float32)

    scale_factors = spacing_current / spacing_goal
    image_resampled = zoom(image, scale_factors, order=order)

    if clip:
        image_resampled = np.clip(image_resampled, 0.0, 1.0)

    return image_resampled


def blur_xy_to_match_lateral_resolution(
    image: np.ndarray,
    sigma_lat_current: float,
    sigma_lat_goal: float,
    axis_order: str = "zyx",
) -> np.ndarray:
    """
    Apply Gaussian blur in the lateral dimension to match a target lateral resolution.

    This function adjusts the lateral (XY) resolution of a 3D image by applying Gaussian
    blur in the Y and X dimensions only. It increases the effective lateral point spread
    function (PSF) from `sigma_lat_current` to `sigma_lat_goal`, simulating a
    lower-resolution imaging system. This is useful when matching datasets with
    different lateral resolutions.

    Parameters
    ----------
    image : np.ndarray
        Input 3D image array. Default axis order is (Z, Y, X).
    sigma_lat_current : float
        Standard deviation of the current lateral PSF in pixels.
    sigma_lat_goal : float
        Standard deviation of the target lateral PSF in pixels.
    axis_order : str, optional
        Order of axes in `image`. Must be a permutation of 'xyz'. Default is 'zyx'.

    Returns
    -------
    np.ndarray
        Image after applying the Gaussian blur to the lateral dimensions.

    Raises
    ------
    ValueError
        If `sigma_lat_goal` is not greater than `sigma_lat_current`, or
        if `axis_order` is not a valid permutation of 'xyz'.

    Notes
    -----
    - No blur is applied along the Z axis.
    - The amount of blur applied is computed as:
        sqrt(sigma_lat_goal² - sigma_lat_current²)
    """
    if set(axis_order) != {"x", "y", "z"} or len(axis_order) != 3:
        raise ValueError("axis_order must be a permutation of 'xyz'.")

    if sigma_lat_goal <= sigma_lat_current:
        raise ValueError(
            "sigma_lat_goal must be greater than sigma_lat_current to apply "
            "compensation."
        )

    sigma_blur = np.sqrt(sigma_lat_goal**2 - sigma_lat_current**2)

    # Build sigma tuple based on axis order
    sigma = tuple(sigma_blur if ax in {"x", "y"} else 0.0 for ax in axis_order)

    return gaussian_filter(image, sigma=sigma)


# def make_image_fully_isotropic(
#     image: np.ndarray,
#     spacing_current: tuple[float, float, float],
#     wavelength_current: float,
#     na: float,
#     magnification: float,
#     ri_sample: float,
#     ri_coverslip: float,
#     ri_immersion: float,
#     confocal: bool,
#     wavelength_goal: float | None = None,
#     spacing_goal: tuple[float, float, float] | None = None,
#     order: int = 1,
# ) -> npt.NDArray:
#     """
#     Resample, and adjust lateral resolution of a 3D fluorescence image.

#     This function prepares a 3D image for further analysis by:
#     1. Resampling to isotropic (or user-defined) voxel spacing.
#     2. Blurring in the XY plane to match the lateral resolution of a target
#        imaging wavelength (if provided).
#     3. Further adjusting XY resolution to match axial resolution, ensuring
#        spatial resolution is isotropic.

#     Parameters
#     ----------
#     image : np.ndarray
#         Input 3D image volume with shape (Z, Y, X).
#     spacing_current : Tuple[float, float, float]
#         Voxel spacing of the input image, in microns (z, y, x).
#     wavelength_current : float
#         Emission wavelength of the current channel, in microns.
#     na : float
#         Numerical aperture of the objective.
#     magnification : float
#         Optical magnification used during acquisition.
#     ri_sample : float
#         Refractive index of the sample.
#     ri_coverslip : float
#         Refractive index of the coverslip.
#     ri_immersion : float
#         Refractive index of the immersion medium (e.g., oil, water).
#     confocal : bool
#         Whether the image was acquired using confocal microscopy.
#     wavelength_goal : float, optional
#         Target wavelength to match lateral resolution against. If provided,
#         the image is blurred in XY to match its resolution.
#     spacing_goal : Tuple[float, float, float], optional
#         Target voxel spacing (z, y, x). If None, uses the smallest spacing
#         from `spacing_current` to enforce isotropy.
#     order : int, optional
#         The order of spline interpolation (0 = nearest, 1 = linear, 3 = cubic, etc.).
#         Default is 1.

#     Returns
#     -------
#     np.ndarray
#         A resampled and resolution-matched 3D image with isotropic spacing and
#         resolution.
#     """
#     if wavelength_goal is not None and wavelength_goal < wavelength_current:
#         raise ValueError(
#             "wavelength_goal should be greater than or equal to wavelength_current"
#         )

#     # Compute the sigmas of the PSF of the provided image
#     sigma_lat_current, sigma_ax_current = estimate_psf_sigmas(
#         wavelength_current,
#         na,
#         magnification,
#         ns=ri_sample,
#         ng=ri_coverslip,
#         ni=ri_immersion,
#         confocal=confocal,
#     )

#     if spacing_goal is None:
#         voxel_spacing = min(spacing_current)
#         spacing_goal = (voxel_spacing, voxel_spacing, voxel_spacing)

#     # Resample to isotropic voxels
#     image = interpolate_to_spacing(
#         image,
#         spacing_current,
#         spacing_goal,
#         order=1,
#     )

#     # Blur in XY to match resolution expected at the longer (goal) wavelength
#     if wavelength_goal is not None and wavelength_current != wavelength_goal:
#         sigma_lat_goal, sigma_ax_goal = estimate_psf_sigmas(
#             wavelength_goal,
#             na,
#             magnification,
#             ns=ri_sample,
#             ng=ri_coverslip,
#             ni=ri_immersion,
#             confocal=confocal,
#         )

#         image = blur_xy_to_match_lateral_resolution(
#             image, sigma_lat_current, sigma_lat_goal
#         )
#         sigma_lat_current = sigma_lat_goal
#     else:
#         sigma_ax_goal = sigma_ax_current

#     # Ensure lateral resolution matches axial resolution
#     image = blur_xy_to_match_lateral_resolution(image, sigma_lat_current, sigma_ax_goal)

#     return image


def make_image_fully_isotropic(
    image: np.ndarray,
    spacing_current: tuple[float, float, float],
    wavelength_current: float,
    na: float,
    magnification: float,
    ri_sample: float,
    ri_coverslip: float,
    ri_immersion: float,
    confocal: bool,
    wavelength_goal: float | None = None,
    spacing_goal: float | tuple[float, float, float] | Literal["min", "max"] = "min",
    order: int = 1,
    downsample_mode: str = "sum",
    pad_mode: str = "constant",
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Resample a 3D image to isotropic voxel spacing and adjust to match optical resolution.

    This function ensures that a 3D image (Z, Y, X) has isotropic voxel spacing while
    preserving or adjusting the optical resolution. It automatically chooses the optimal
    sequence of operations depending on whether the image needs upsampling or downsampling:

        - **Upsampling (scale factor <= 1)**: interpolate to target spacing first, then blur.
        - **Downsampling (scale factor >= 1)**: blur first to avoid aliasing, then downsample.

    Lateral and axial resolutions are matched according to the microscope parameters
    (numerical aperture, magnification, refractive indices) and optionally adjusted
    for a target wavelength.

    Parameters
    ----------
    image : np.ndarray
        Input 3D image array with shape (Z, Y, X).
    spacing_current : tuple of float
        Current voxel spacing in microns along (Z, Y, X) axes.
    wavelength_current : float
        Emission wavelength of the image in microns.
    na : float
        Numerical aperture of the microscope objective.
    magnification : float
        Optical magnification of the imaging system.
    ri_sample : float
        Refractive index of the sample medium.
    ri_coverslip : float
        Refractive index of the coverslip.
    ri_immersion : float
        Refractive index of the immersion medium.
    confocal : bool
        Whether the image was acquired on a confocal microscope.
    wavelength_goal : float, optional
        Target emission wavelength for lateral resolution adjustment. If None, no
        wavelength-based adjustment is performed. Must be >= `wavelength_current`.
    spacing_goal : float, tuple of float, or {"min", "max"}, optional
        Target isotropic voxel spacing:
            - float: use same spacing in all axes.
            - tuple of float: explicit (Z, Y, X) spacing.
            - "min": use the smallest spacing from `spacing_current` for isotropic voxels.
            - "max": use the largest spacing from `spacing_current` for isotropic voxels.
        Default is "min".
    order : int, optional
        Interpolation order for resampling. 0=nearest, 1=linear, 3=cubic. Default=1.
    downsample_mode : str, optional
        Aggregation method for downsampling: "sum", "mean", "min", "max". Default="sum".
    pad_mode : str, optional
        Padding mode for downsampling, passed to `np.pad`. Default="constant".

    Returns
    -------
    tuple
        - `np.ndarray`: The resampled 3D image with isotropic voxel spacing and adjusted
        resolution.
        - `tuple(float, float, float)`: The voxel spacing of the output image (Z, Y, X)
        in microns.

    Raises
    ------
    ValueError
        If `wavelength_goal` < `wavelength_current`.
        If `spacing_goal` is a tuple/list not of length 3.
        If `spacing_goal` is an unsupported string.
    """
    if wavelength_goal is not None and wavelength_goal < wavelength_current:
        raise ValueError(
            "wavelength_goal should be greater than or equal to wavelength_current"
        )
    # Normalize spacing_goal to a 3-tuple
    if isinstance(spacing_goal, (int, float)):
        spacing_goal = (spacing_goal,) * 3
    elif isinstance(spacing_goal, str):
        if spacing_goal in ("min", "max"):
            voxel_spacing = (
                min(spacing_current) if spacing_goal == "min" else max(spacing_current)
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

    # Compute the sigmas of the PSF of the provided image
    sigma_lat_current, sigma_ax_current = estimate_psf_sigmas(
        wavelength_current,
        na,
        magnification,
        ns=ri_sample,
        ng=ri_coverslip,
        ni=ri_immersion,
        confocal=confocal,
    )

    scale_factors = np.array(spacing_goal) / np.array(spacing_current)

    if np.all(scale_factors <= 1):
        # --- Upsampling: interpolate first, then blur ---
        # Resample to isotropic voxels
        image = interpolate_to_spacing(
            image,
            spacing_current,
            spacing_goal,
            order=order,
        )
    # Blur in XY to match resolution expected at the longer (goal) wavelength
    if wavelength_goal is not None and wavelength_current != wavelength_goal:
        sigma_lat_goal, sigma_ax_goal = estimate_psf_sigmas(
            wavelength_goal,
            na,
            magnification,
            ns=ri_sample,
            ng=ri_coverslip,
            ni=ri_immersion,
            confocal=confocal,
        )

        image = blur_xy_to_match_lateral_resolution(
            image, sigma_lat_current, sigma_lat_goal
        )
        sigma_lat_current = sigma_lat_goal
    else:
        sigma_ax_goal = sigma_ax_current

    # Ensure lateral resolution matches axial resolution
    image = blur_xy_to_match_lateral_resolution(image, sigma_lat_current, sigma_ax_goal)

    if np.all(scale_factors >= 1):
        # --- Downsampling: blur first, then downsample ---

        # Compute nearest integer scale factors
        nearest_integer = np.round(scale_factors).astype(int)
        # Avoid division by zero (should not happen, but guard anyway)
        nearest_integer[nearest_integer == 0] = 1

        # Compute percent deviation between true and integer factors
        perc = np.abs((scale_factors - nearest_integer) / scale_factors) * 100

        max_perc = np.max(perc)
        if max_perc > 2.0:
            raise ValueError(
                f"Scale factor rounding exceeds 2% tolerance (max deviation: {max_perc:.2f}%)"
            )
        elif max_perc > 1.0:
            warnings.warn(
                f"Scale factor rounding exceeds 1% (max deviation: {max_perc:.2f}%), "
                "but within 2% tolerance — proceeding with nearest integer factors.",
                UserWarning,
                stacklevel=2,
            )

        # Adjust spacing_current to reflect the effective integer downsampling
        spacing_current = tuple(spacing_goal[i] / nearest_integer[i] for i in range(3))

        # Downsample accordingly
        image = downsample_by_spacing(
            image,
            spacing_current,
            spacing_goal,
            mode=downsample_mode,
            pad_mode=pad_mode,
        )

    return image, spacing_goal
