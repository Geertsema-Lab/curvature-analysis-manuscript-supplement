from __future__ import annotations

import numpy as np


def extract_ellipsoid_surface_coords(
    a: float,
    b: float,
    c: float,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    center: tuple[float, float, float] = (0, 0, 0),
    tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract coordinates lying on or near the surface of an ellipsoid.

    Given meshgrids of coordinates, this function identifies and returns the
    points whose values satisfy the implicit ellipsoid equation within a
    specified tolerance, effectively extracting a thin shell representing the
    ellipsoid surface.

    Parameters
    ----------
    a, b, c : float
        Semi-axis lengths of the ellipsoid along the x, y, and z axes.
    X, Y, Z : np.ndarray
        3D coordinate meshgrids of the same shape, representing points in space.
    center : tuple of float, optional
        The (x_c, y_c, z_c) coordinates of the ellipsoid center. Defaults to (0, 0, 0).
    tol : float, optional
        Tolerance for determining proximity to the ellipsoid surface. Points with
        implicit function values within this tolerance of 1 are included.

    Returns
    -------
    X_surf, Y_surf, Z_surf : np.ndarray
        1D arrays of x, y, and z coordinates corresponding to points near the
        ellipsoid surface.

    """
    x_c, y_c, z_c = center
    ellipsoid_val = ((X - x_c) / a) ** 2 + ((Y - y_c) / b) ** 2 + ((Z - z_c) / c) ** 2
    mask = np.abs(ellipsoid_val - 1) <= tol
    return X[mask], Y[mask], Z[mask]


def calculate_shape_from_ellipsoid_parameters(
    a: float,
    b: float,
    c: float,
    padding_factor: float = 1.2,
) -> tuple[int, int, int]:
    """
    Calculate the 3D volume shape to contain an ellipsoid with padding.

    This function computes the integer shape (nz, ny, nx) needed for a
    3D volume that fully contains an ellipsoid defined by semi-axis
    lengths a, b, and c, scaled by a padding factor.

    Parameters
    ----------
    a, b, c : float
        Semi-axis lengths of the ellipsoid along the x, y, and z axes.
    padding_factor : float, optional
        Factor to scale the bounding box size around the ellipsoid
        (default is 1.2).

    Returns
    -------
    shape : tuple of int
        The (nz, ny, nx) shape tuple for the 3D volume.
    """
    shape = (
        int(2 * c * padding_factor),  # nz along z-axis length (c)
        int(2 * b * padding_factor),  # ny along y-axis length (b)
        int(2 * a * padding_factor),  # nx along x-axis length (a)
    )
    return shape
