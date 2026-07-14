"""Band-limited ellipsoid shell generator for curvature estimation testing.

This module provides functions to generate 3D band-limited ellipsoid and sphere shells
using the gradient magnitude method described in computer vision literature for creating
test images with shells of constant thickness.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import (
    gaussian_gradient_magnitude,
)

from .utils import calculate_shape_from_ellipsoid_parameters


def create_ellipsoid_shell(
    a: float,
    b: float,
    c: float,
    sigma: float = 1.3,
    center: str | Sequence[int] = "center",
    shape: tuple[int, int, int] | None = None,
    padding_factor: float = 1.2,
    return_coords: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 3D band-limited ellipsoid shell using Gaussian-smoothed gradient magnitude.

    This function constructs a smooth volumetric shell around an ellipsoid with
    semi-axis lengths `a`, `b`, and `c`. It first generates a filled ellipsoid,
    applies Gaussian smoothing, and then computes the squared gradient magnitude
    to approximate a band-limited shell. This representation is useful for
    testing imaging algorithms, curvature estimation, or other analyses where
    smooth, localized boundaries are required.

    Parameters
    ----------
    a, b, c : float
        Semi-axis lengths of the ellipsoid along the x, y, and z axes.
    sigma : float, optional
        Standard deviation of the Gaussian kernel for smoothing. Must be
        >= 1.3 to satisfy band-limitation (default is 1.3).
    center : {'center'} or sequence of 3 floats, optional
        The ellipsoid center position in voxel coordinates. If "center",
        the ellipsoid is centered in the middle of the volume (default).
        Otherwise, provide explicit coordinates as (xc, yc, zc).
    shape : tuple of int, optional
        Desired output volume shape as (nz, ny, nx). If None, it is computed
        automatically from the ellipsoid dimensions and `padding_factor`
        (default is None).
    padding_factor : float, optional
        Factor by which the automatically calculated volume dimensions are
        scaled relative to the ellipsoid size (default is 1.2).
    return_coords : bool, optional
        If True, also return coordinate meshgrids (Z, Y, X) with indexing='ij'
        (default is False).

    Returns
    -------
    shell : np.ndarray
        3D array of normalized shell intensity values in [0, 1].
    Z, Y, X : np.ndarray, optional
        Coordinate meshgrids, returned only if `return_coords=True`.

    Notes
    -----
    - Band-limitation requires sigma >= 0.9 * sqrt(2) ~= 1.3.
    - Edge localization error is approximately -sigma^2 / (2R), where R is the
      local curvature radius. This error is negligible for R > 10.
    - Voxel size is fixed to 1.
    - Coordinate arrays use `indexing='ij'` (z, y, x) convention.

    Examples
    --------
    >>> shell = create_ellipsoid_shell(20, 15, 10, sigma=1.3, shape=(64, 64, 64))
    >>> shell.shape
    (64, 64, 64)

    >>> shell, Z, Y, X = create_ellipsoid_shell(10, 10, 5, return_coords=True)
    >>> shell.max(), Z.shape
    (1.0, shell.shape)
    """
    min_sigma = 0.9 * np.sqrt(2)
    if sigma < min_sigma:
        print(
            f"Warning: sigma={sigma:.2f} < {min_sigma:.2f}, "
            "band-limitation may not be guaranteed."
        )

    min_radius = min(a, b, c)
    if min_radius > 0:
        edge_shift = sigma**2 / (2 * min_radius)
        if min_radius <= 10:
            print(
                f"Warning: minimum radius R={min_radius:.1f} <= 10, "
                f"edge localization error may be significant: {edge_shift:.3f}"
            )

    if shape is None:
        shape = calculate_shape_from_ellipsoid_parameters(a, b, c, padding_factor)

    nz, ny, nx = shape

    if isinstance(center, str):
        if center == "center":
            xc, yc, zc = 0.0, 0.0, 0.0
        else:
            raise ValueError(f"Invalid center string: {center}. Expected 'center'.")
    elif isinstance(center, Sequence) and len(center) == 3:
        xc, yc, zc = center
    else:
        raise ValueError("Center must be 'center' or a sequence of 3 coordinates")

    # Fixed voxel size = 1
    d = 1.0

    # Coordinates arrays: indices centered around zero
    z = (np.arange(nz) - nz // 2) * d
    y = (np.arange(ny) - ny // 2) * d
    x = (np.arange(nx) - nx // 2) * d
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    # Ellipsoid implicit function
    ellipsoid_val = ((X - xc) / a) ** 2 + ((Y - yc) / b) ** 2 + ((Z - zc) / c) ** 2

    # Create filled ellipsoid binary volume
    filled = (ellipsoid_val <= 1).astype(float)

    # Compute squared gradient magnitude with Gaussian smoothing
    grad_mag_sq = gaussian_gradient_magnitude(filled, sigma=sigma) ** 2

    # Normalize shell intensity to [0,1]
    shell = grad_mag_sq / grad_mag_sq.max() if grad_mag_sq.max() > 0 else grad_mag_sq

    if return_coords:
        return shell, Z, Y, X
    else:
        return shell


def create_sphere_shell(
    radius: float,
    sigma: float = 1.3,
    center: str | Sequence[int] = "center",
    shape: tuple[int, int, int] | None = None,
    padding_factor: float = 1.2,
    return_coords: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 3D band-limited spherical shell (special case of ellipsoid).

    Parameters
    ----------
    radius : float
        Sphere radius.
    sigma : float, optional
        Standard deviation for Gaussian smoothing. Must be >= 1.3 for proper
        band-limitation (default 1.3).
    shape : tuple of int or None, optional
        Output volume shape as (nz, ny, nx). If None, calculated automatically
        based on radius and padding_factor (default None).
    padding_factor : float, optional
        Scaling factor for volume size relative to sphere size when shape is None
        (default 1.2).
    return_coords : bool, optional
        If True, also return coordinate meshgrids (Z, Y, X) along with the shell
        (default False).

    Returns
    -------
    shell : np.ndarray
        Normalized shell intensity volume with values in [0, 1].
    Z, Y, X : np.ndarray, optional
        Coordinate meshgrids returned only if `return_coords` is True.

    Examples
    --------
    >>> shell = create_sphere_shell(radius=20, shape=(64, 64, 64))
    >>> shell.shape
    (64, 64, 64)
    """
    if shape is None:
        shape = calculate_shape_from_ellipsoid_parameters(
            radius, radius, radius, padding_factor=padding_factor
        )

    return create_ellipsoid_shell(
        a=radius,
        b=radius,
        c=radius,
        sigma=sigma,
        center=center,
        shape=shape,
        padding_factor=padding_factor,
        return_coords=return_coords,
    )
