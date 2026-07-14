"""
Collection of function to create segmentation masks and crop images.

These functions are used to help with the curvature-intensity analysis.
"""

import numpy as np
from scipy.ndimage import (
    binary_closing,
    # binary_dilation,
    binary_fill_holes,
    gaussian_filter,
)
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border, expand_labels

from .steerable3d import steerable_detector_3d


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask (2D or 3D) where nonzero elements represent foreground.

    Returns
    -------
    np.ndarray
        Binary mask with only the largest connected component retained.
    """
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=binary_mask.dtype)

    labeled_mask, num_labels = label(
        binary_mask, return_num=True, connectivity=binary_mask.ndim
    )

    if num_labels == 0:
        return np.zeros_like(binary_mask, dtype=binary_mask.dtype)

    component_sizes = np.bincount(labeled_mask.ravel())
    largest_label = np.argmax(component_sizes[1:]) + 1

    return (labeled_mask == largest_label).astype(binary_mask.dtype)


def segment_and_label_nuclei(
    image: np.ndarray,
    voxel_spacing: float,
    expansion_distance: float,
    min_nucleus_area: float,
    axis_order: str = "zyx",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment and label nuclei in a 3D image.

    This function segments nuclei from a 3D volumetric image by computing a maximum
    intensity projection (MIP) along the Z-axis, applying Otsu thresholding, and
    removing small or border-touching objects. The remaining nuclei regions are labeled
    and expanded using Voronoi tessellation (via `expand_labels`) to better separate
    tightly clustered structures. Labels overlapping with previously removed
    edge-touching regions are excluded from the final result.

    Parameters
    ----------
    image : np.ndarray
        A 3D NumPy array representing the input volumetric image. Its shape should
        correspond to the `axis_order`.
    voxel_spacing : float
        Physical spacing between voxels (assumed isotropic). For example, units can be
        in micrometers.
    expansion_distance : float
        Physical distance (same units as `voxel_spacing`) by which to expand each
        nucleus label using Voronoi tessellation.
    min_nucleus_area : float
        Minimum nucleus area to retain after thresholding, in physical units squared
        (e.g., μm^2 if `voxel_spacing` is in μm).
    axis_order : str, optional
        A permutation of the string "zyx" indicating the axis order in the input image.
        Default is "zyx".

    Returns
    -------
    nuclei_labels_final : np.ndarray
        A 3D array of the same shape as `image`, where each voxel is assigned a
        unique integer label corresponding to a segmented nucleus. Background is 0.

    Notes
    -----
    - Otsu's thresholding is applied to the MIP for initial foreground detection.
    - Objects below the `min_nucleus_area` or touching the MIP border are removed before
    labeling.
    - `expand_labels` from scikit-image is used to grow labels using a Voronoi-like
    expansion in 3D.
    - Labels that intersect with previously removed border-touching objects are excluded
    post-expansion.
    """
    if set(axis_order) != {"x", "y", "z"} or len(axis_order) != 3:
        raise ValueError("axis_order must be a permutation of 'xyz'")

    # Compute maximum intensity projection along z-axis
    max_proj = np.max(image, axis=axis_order.index("z"))

    # Threshold using Otsu's method
    threshold = threshold_otsu(max_proj)
    nuclei_mask = max_proj > threshold

    nuclei_mask = binary_fill_holes(nuclei_mask)

    # Convert minimum area from ..^2 to pixels^2
    min_area_px = min_nucleus_area / voxel_spacing**2

    # Remove small objects
    nuclei_mask = remove_small_objects(nuclei_mask, min_size=min_area_px)

    # Label the regions
    nuclei_labels = label(nuclei_mask)

    # Expand labels using Voronoi tessellation
    # Convert expansion distance from physical units to pixels
    expansion_distance_px = expansion_distance / voxel_spacing
    nuclei_labels = expand_labels(nuclei_labels, distance=expansion_distance_px)

    # Remove objects touching the image border
    mask_no_border = clear_border(nuclei_mask)

    # Collect valid (non-zero) label IDs (zero is background)
    valid_labels = set(np.unique(nuclei_labels[mask_no_border]))
    if 0 in valid_labels:
        valid_labels.remove(0)

    new_labels = list(range(1, len(valid_labels) + 1))

    # Create a new labels array with only the valid labels
    nuclei_labels_final = np.zeros_like(nuclei_labels)
    for valid_label, new_label in zip(valid_labels, new_labels, strict=True):
        nuclei_labels_final[nuclei_labels == valid_label] = new_label

    # Ensure labels has the same shape as input image
    nuclei_labels_final = np.broadcast_to(nuclei_labels_final, image.shape)

    return nuclei_labels_final


def crop_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop a 3D image to the bounding box defined by a binary mask.

    Parameters
    ----------
    image : np.ndarray
        A 3D array (Z, Y, X) representing the image to crop.
    mask : np.ndarray
        A binary 3D array of the same shape as `image`. Nonzero values
        define the region of interest.

    Returns
    -------
    np.ndarray
        A cropped 3D image that includes only the region enclosed by the mask.

    Notes
    -----
    - Pixels outside the mask are set to the minimum of the region inside the mask to
    affect normalization after cropping.
    - If the mask is empty, returns an empty array of shape (0, 0, 0).
    """
    if image.shape != mask.shape:
        raise ValueError("`image` and `mask` must have the same shape")

    if not np.any(mask):
        return np.zeros((0, 0, 0), dtype=image.dtype)

    # Zero out regions outside the mask
    masked_image = np.where(mask, image, image[mask].min())

    # Find bounding box
    coords = np.argwhere(mask)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1  # include max bound

    # Crop using bounding box
    cropped = masked_image[zmin:zmax, ymin:ymax, xmin:xmax]

    return cropped


def create_rough_lamin_mask(
    image: np.ndarray,
    voxel_spacing: float,
    sigma: float,
    ic_factor: float,
    alpha: float,
) -> np.ndarray[bool]:
    """
    Generate a rough binary mask by subtracting a blurred image from the original.

    This function creates a coarse foreground mask from a 3D image volume by enhancing
    local contrast using a subtraction method. A blurred version of the image is
    subtracted from the original, and an intensity-dependent threshold is applied.
    Only the largest connected region is retained to remove noise and small objects.

    Parameters
    ----------
    image : np.ndarray
        3D grayscale image volume from which the binary mask is to be computed.
    voxel_spacing : float
        Physical spacing between voxels (e.g., in microns). Assumed isotropic.
    sigma : float
        Standard deviation of the Gaussian blur applied to the image, in physical units.
        This controls the scale at which background intensity is estimated.
    ic_factor : float
        Scaling factor for the image's maximum intensity. It controls the influence of
        an additive bias term used during thresholding.
    alpha : float
        Weighting factor (between 0 and 1) for the blurred image during subtraction.
        Higher values retain more background structure and reduce sensitivity.

    Returns
    -------
    np.ndarray
        Binary mask of the same shape as the input image, where `True` indicates
        foreground regions and `False` indicates background.

    Notes
    -----
    The rough mask is computed as follows:

    1. A Gaussian blur is applied to the input image with a physical sigma.
    2. A thresholded difference is computed:
           mask = image - (alpha * blur + beta)
       where:
           beta = image.max() * ic_factor * (1 - alpha)

    3. Voxels where the result is greater than zero are kept as foreground.
    4. Small regions are removed by keeping only the largest connected component.

    This method is intended to create a fast and conservative segmentation, useful
    for initializing further image analysis or curvature estimation workflows.
    """
    # Adjust blurring sigma according to image resolution
    sigma /= voxel_spacing

    image_blurred = gaussian_filter(image, sigma=sigma)

    maximum = image_blurred.max()

    beta = ic_factor * maximum * (1 - alpha)

    # Create mask by subtracting weighted blurred image plus offset
    mask = image - (alpha * image_blurred + beta)

    # Threshold mask: pixels > 0 are foreground
    mask = mask > 0

    # Keep only the largest connected component to remove noise
    mask = keep_largest_component(mask)

    return mask


def create_fine_lamin_mask(
    image: np.ndarray,
    rough_mask: np.ndarray,
    voxel_spacing: float,
    sigma_s: float,
    zx_ratio: float = 1.0,
) -> np.ndarray[bool]:
    """
    Generate a refined 3D binary mask using steerable filtering and statistical thresholding.

    This function refines an initial (rough) segmentation by applying a second-order
    3D steerable filter to enhance curvilinear structures, followed by non-maximum
    suppression (NMS). The NMS response is normalized using a locally estimated mean
    and standard deviation, producing a z-score map that is thresholded within the
    rough mask. The largest connected component is retained to suppress noise.

    Parameters
    ----------
    image : np.ndarray
        3D grayscale image volume.
    rough_mask : np.ndarray
        Binary mask restricting processing to a region of interest.
    voxel_spacing : float
        Physical voxel size (e.g., microns). Used to convert filter scale to pixel units.
    sigma_s : float
        Physical scale of the steerable filter (e.g., microns), controlling the size
        of structures to enhance.
    zx_ratio : float, optional
        Anisotropy factor for the z-dimension. The effective sigma in z is
        `sigma_z = sigma_s / zx_ratio`. Default is 1.0 (isotropic filtering).

    Returns
    -------
    np.ndarray of bool
        Refined binary mask of the same shape as `image`.

    Notes
    -----
    - A second-order steerable filter is used to enhance ridge-like structures.
    - Local statistics (mean and variance) are estimated using Gaussian smoothing.
    - The NMS response is converted to a z-score to enable adaptive thresholding.
    - Thresholding is restricted to the rough mask to reduce false positives.
    - Only the largest connected component is retained.
    """
    sigma_s /= voxel_spacing

    response, _, nms = steerable_detector_3d(
        image, m=2, sigma=sigma_s, zx_ratio=zx_ratio
    )

    # Create the z-map
    sigma_z = 2 * sigma_s
    mean = gaussian_filter(response, sigma=sigma_z)
    sq = gaussian_filter(response**2, sigma=sigma_z)
    std = np.sqrt(np.clip(sq - mean**2, 1e-8, None))

    z = (nms - mean) / std

    mask = (z > 1.0) & rough_mask

    mask = keep_largest_component(mask)

    return mask


def create_soft_dapi_mask(
    image: np.ndarray, sigma: float, voxel_spacing: float | int
) -> np.ndarray:
    """
    Create a smooth, filled envelope of DAPI-positive regions from a 3D image.

    This function thresholds the DAPI signal to segment nuclei, fills internal holes,
    keeps the largest connected component, and applies Gaussian smoothing to produce
    a soft spatial mask of the nuclear region.

    Parameters
    ----------
    image : np.ndarray
        3D DAPI image volume.
    sigma : float
        Smoothing scale in physical units for the final Gaussian blur.
    voxel_spacing : float
        Isotropic voxel spacing (same unit as sigma).

    Returns
    -------
    np.ndarray
        Smooth float-valued mask representing the nuclear envelope, with values
        between 0 and 1 and soft transitions at boundaries.
    """
    sigma_px = sigma / voxel_spacing

    binary = image > threshold_otsu(image)
    binary = binary_fill_holes(binary)
    binary = keep_largest_component(binary)

    return gaussian_filter(binary.astype(np.float32), sigma=sigma_px)
