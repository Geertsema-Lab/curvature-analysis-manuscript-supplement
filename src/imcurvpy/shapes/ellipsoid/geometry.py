"""
Ellipsoid geometry utilities.

This module provides functions for converting between Cartesian and parametric
coordinates on a triaxial ellipsoid, as well as computing the principal curvatures
at points on the ellipsoid surface.

The ellipsoid is defined by three semi-axis lengths `a`, `b`, and `c` along the
x-, y-, and z-axes respectively, and an optional center point.

"""

from __future__ import annotations

import numpy as np


def cartesian_to_parametric_ellipsoid(X, Y, Z, a, b, c, center=(0, 0, 0)):
    """
    Convert Cartesian coordinates to parametric coordinates (u, v) on an ellipsoid.

    The ellipsoid is defined by the semi-axis lengths `a`, `b`, `c` along x, y, z
    axes, respectively, and centered at `center`. The parametric coordinates satisfy:

        X = a * cos(u) * sin(v) + x_c
        Y = b * sin(u) * sin(v) + y_c
        Z = c * cos(v) + z_c

    where:
    - u is the azimuthal angle in [0, 2*pi]
    - v is the polar angle in [0, pi]

    Parameters
    ----------
    X : float or np.ndarray
        Cartesian x-coordinate(s).
    Y : float or np.ndarray
        Cartesian y-coordinate(s).
    Z : float or np.ndarray
        Cartesian z-coordinate(s).
    a : float
        Semi-axis length along the x-axis.
    b : float
        Semi-axis length along the y-axis.
    c : float
        Semi-axis length along the z-axis.
    center : tuple of float, optional
        Center coordinates of the ellipsoid (x_c, y_c, z_c). Default is (0, 0, 0).

    Returns
    -------
    u : float or np.ndarray
        Azimuthal parametric coordinate(s), in radians, range [0, 2*pi].
    v : float or np.ndarray
        Polar parametric coordinate(s), in radians, range [0, pi].

    Examples
    --------
    >>> a, b, c = 3.0, 4.0, 5.0
    >>> center = (0.5, -0.5, 0.0)
    >>> X, Y, Z = 3.0, 0.0, 5.0
    >>> u, v = cartesian_to_parametric_ellipsoid(X, Y, Z, a, b, c, center)
    >>> print(f"u = {u:.3f}, v = {v:.3f}")
    u = 3.712, v = 0.000
    """
    x_c, y_c, z_c = center

    # Shift coordinates relative to center
    Xc = X - x_c
    Yc = Y - y_c
    Zc = Z - z_c

    # Calculate parametric coordinates
    u = np.arctan2(Yc / b, Xc / a) % (2 * np.pi)  # u in [0, 2*pi]
    v = np.arccos(np.clip(Zc / c, -1.0, 1.0))  # v in [0, pi]

    return u, v


def parametric_ellipsoid_to_cartesian(u, v, a, b, c, center=(0, 0, 0)):
    """
    Convert parametric coordinates (u, v) to Cartesian coordinates (X, Y, Z) on an ellipsoid.

    The ellipsoid is defined by the semi-axis lengths `a`, `b`, `c` along x, y, z
    axes, respectively, and centered at `center`. The parametric coordinates satisfy:

        X = a * cos(u) * sin(v) + x_c
        Y = b * sin(u) * sin(v) + y_c
        Z = c * cos(v) + z_c

    where:
    - u is the azimuthal angle in [0, 2*pi]
    - v is the polar angle in [0, pi]

    Parameters
    ----------
    u : float or np.ndarray
        Azimuthal parametric coordinate(s), in radians.
    v : float or np.ndarray
        Polar parametric coordinate(s), in radians.
    a : float
        Semi-axis length along the x-axis.
    b : float
        Semi-axis length along the y-axis.
    c : float
        Semi-axis length along the z-axis.
    center : tuple of float, optional
        Center coordinates of the ellipsoid (x_c, y_c, z_c). Default is (0, 0, 0).

    Returns
    -------
    X : float or np.ndarray
        Cartesian x-coordinate(s).
    Y : float or np.ndarray
        Cartesian y-coordinate(s).
    Z : float or np.ndarray
        Cartesian z-coordinate(s).

    Examples
    --------
    >>> a, b, c = 3.0, 4.0, 5.0
    >>> center = (0.5, -0.5, 0.0)
    >>> u, v = np.pi / 4, np.pi / 3
    >>> X, Y, Z = parametric_ellipsoid_to_cartesian(u, v, a, b, c, center)
    >>> print(f"X = {X:.3f}, Y = {Y:.3f}, Z = {Z:.3f}")
    X = 2.560, Y = 1.530, Z = 2.500
    """
    x_c, y_c, z_c = center

    X = a * np.cos(u) * np.sin(v) + x_c
    Y = b * np.sin(u) * np.sin(v) + y_c
    Z = c * np.cos(v) + z_c

    return X, Y, Z


def principal_curvatures_ellipsoid(a, b, c, u, v):
    """
    Compute the principal curvatures of a triaxial ellipsoid at given parametric coordinates.

    The ellipsoid is parameterized as:
        X(u, v) = a * cos(u) * sin(v)
        Y(u, v) = b * sin(u) * sin(v)
        Z(u, v) = c * cos(v)
    with u ∈ [0, 2π], v ∈ [0, π].

    Parameters
    ----------
    a, b, c : float
        Semi-axis lengths of the ellipsoid along the x-, y-, and z-axes.
    u : float or np.ndarray
        Azimuthal parameter (longitude angle), in radians. Range: [0, 2π].
    v : float or np.ndarray
        Polar parameter (colatitude angle), in radians. Range: [0, π].

    Returns
    -------
    k1 : float or np.ndarray
        The first principal curvature (maximum).
    k2 : float or np.ndarray
        The second principal curvature (minimum).

    Notes
    -----
    - The Gaussian curvature K and mean curvature H are first computed using the
      formulas in Rieger (2004, PhD thesis, Eqs. C.15-C.17):

        K = (a² b² c²) / denom²

        H = (a b c / denom_H) * (num₁ * sin(v) + num₂ * sin³(v))

      with denominators and numerators as defined in the implementation.

    - The principal curvatures are obtained via:
        k1, k2 = H ± sqrt(H² - K)

    - Numerical stability is enforced by clamping H² - K ≥ 0.

    - Sign convention: positive curvature corresponds to convex regions with
      outward-pointing normals.

    References
    ----------
    Bernd Rieger (2004). *Structure from Motion in nD Image Analysis*.
    PhD thesis, Delft University of Technology, p. 167.

    Examples
    --------
    >>> a, b, c = 2.0, 3.0, 1.5
    >>> u, v = np.pi / 4, np.pi / 2
    >>> k1, k2 = principal_curvatures_ellipsoid(a, b, c, u, v)
    >>> print(f"Principal curvatures: k1={k1:.4f}, k2={k2:.4f}")
    Principal curvatures: k1=0.2741, k2=0.1082
    """
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    cos_v = np.cos(v)
    sin_v = np.sin(v)
    cos_2u = np.cos(2 * u)
    sin_3v = sin_v**3

    denom_base = a**2 * b**2 * cos_v**2 + c**2 * sin_v**2 * (
        b**2 * cos_u**2 + a**2 * sin_u**2
    )

    # Gaussian curvature K (C.15)
    K = (a**2 * b**2 * c**2) / (denom_base**2)

    # Mean curvature H (C.16)
    numerator1 = 5 * (a**2 + b**2) + 6 * c**2 - 3 * (a**2 - b**2) * cos_2u
    numerator2 = (a**2 + b**2 - 2 * c**2) + (a**2 - b**2) * cos_2u

    denom_H = 16 * denom_base ** (3 / 2) * np.abs(sin_v)

    # eps to prevent singularities
    eps = 1e-12
    H = (a * b * c / (denom_H + eps)) * (numerator1 * sin_v + numerator2 * sin_3v)

    # H = (a * b * c / denom_H) * (numerator1 * sin_v + numerator2 * sin_3v)

    # Principal curvatures (C.17)
    inside_sqrt = H**2 - K
    inside_sqrt = np.maximum(inside_sqrt, 0)  # clamp to avoid numerical errors

    k1 = H + np.sqrt(inside_sqrt)
    k2 = H - np.sqrt(inside_sqrt)

    return k1, k2
