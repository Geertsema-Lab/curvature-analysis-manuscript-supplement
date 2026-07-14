from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def masked_gaussian_convolution(
    data: np.ndarray, mask: np.ndarray, sigma: float
) -> np.ndarray:
    """
    Apply a Gaussian convolution to the values within a 3D mask on a 3D array.

    The convolution is performed only on the region specified by the mask.
    The returned array contains the blurred values inside the mask and zeros
    elsewhere.

    Parameters
    ----------
    data : np.ndarray
        Input 3D array of values to be convolved.
    mask : np.ndarray
        Boolean 3D array with the same shape as `data`. True values specify
        the region to be convolved.
    sigma : float
        Standard deviation for Gaussian kernel.

    Returns
    -------
    np.ndarray
        Array of the same shape as `data` with Gaussian convolution applied
        only inside the mask. Values outside the mask are set to zero.

    Raises
    ------
    ValueError
        If `data` and `mask` have different shapes.
    """
    if data.shape != mask.shape:
        raise ValueError("`data` and `mask` must have the same shape.")

    # Masked data with zeros outside mask
    masked_data = np.where(mask, data, 0)

    # Apply Gaussian filter to masked data
    blurred_full = gaussian_filter(masked_data, sigma=sigma)

    # Create float mask for normalization
    mask_float = mask.astype(float)
    normalization = gaussian_filter(mask_float, sigma=sigma)

    # Prevent division by zero
    normalization[normalization == 0] = 1

    # Normalize blurred data within the mask
    blurred_masked = blurred_full / normalization

    # Create output array initialized to zero
    result = np.zeros_like(data)

    # Assign blurred values inside the mask
    result[mask] = blurred_masked[mask]

    return result


def broadcast_to_3d(
    arr_2d: np.ndarray, axes_2d: str, axes_3d: str, z_slices: int
) -> np.ndarray:
    """
    Broadcast a 2D array to 3D by repeating along the Z axis.

    Parameters
    ----------
    arr_2d : np.ndarray
        Input 2D array.
    axes_2d : str
        Axes of the input array, e.g., "YX" or "XY".
    axes_3d : str
        Desired axes of output array, e.g., "ZYX".
    z_slices : int
        Number of slices along the Z axis.

    Returns
    -------
    np.ndarray
        3D array with shape corresponding to axes_3d.

    Raises
    ------
    ValueError
        If axes_2d is not a permutation of 'XY' or axes_3d is not a permutation of 'XYZ'.

    Examples
    --------
    >>> import numpy as np
    >>> arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    >>> broadcast_to_3d(arr_2d, "YX", "ZYX", 4).shape
    (4, 2, 3)
    """
    # Make the axes consistent uppercase
    axes_2d = axes_2d.upper()
    axes_3d = axes_3d.upper()

    # Validate axes
    if set(axes_2d) != {"X", "Y"}:
        raise ValueError(f"axes_2d must contain exactly 'X' and 'Y', got {axes_2d}")
    if set(axes_3d) != {"X", "Y", "Z"}:
        raise ValueError(
            f"axes_3d must contain exactly 'X', 'Y', and 'Z', got {axes_3d}"
        )

    # Compute the shape needed for reshaping the 2D array to match target axes
    # For Z axis, insert 1 (singleton dimension) so it can be broadcasted later
    expand_shape = [
        1 if ax == "Z" else arr_2d.shape[axes_2d.index(ax)] for ax in axes_3d
    ]
    # Example:
    # arr_2d.shape = (Y, X) = (2, 3)
    # axes_2d = "YX"
    # axes_3d = "ZYX"
    # expand_shape = [1, 2, 3]  -> ready to broadcast along Z

    # Reshape arr_2d to insert singleton for Z axis
    arr_expanded = arr_2d.reshape(expand_shape)
    # arr_expanded now has shape (1, Y, X) for axes "ZYX"

    # Find the index of the Z axis in the target axes
    z_index = axes_3d.index("Z")

    # Repeat the array along the Z axis
    arr_3d = np.repeat(arr_expanded, z_slices, axis=z_index)
    # arr_3d now has shape (Z, Y, X)  -> fully broadcasted

    return arr_3d
