import math
import time

import numpy as np
from numba import njit, prange
from scipy import ndimage


@njit()
def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D vector using Euclidean norm.

    Parameters
    ----------
    v : np.ndarray
        Input vector to normalize. Must be a 1D NumPy array.

    Returns
    -------
    np.ndarray
        A normalized vector of the same shape and dtype as the input.
        If the input vector is a zero vector, returns a zero vector
        of the same shape and dtype.

    Notes
    -----
    This function is implemented using Numba for performance.
    It avoids division by zero by checking the norm before normalizing.
    """
    n = 0.0
    for i in range(v.shape[0]):
        n += v[i] * v[i]
    n = np.sqrt(n)

    out = np.empty_like(v)  # Force return type to be the same as input
    if n != 0.0:
        for i in range(v.shape[0]):  # Do the division manually so
            out[i] = v[i] / n
    else:  # Handle the case of a zero vector to avoid division by zero
        for i in range(v.shape[0]):
            out[i] = 0.0

    return out
    # out = np.zeros(
    #     v.shape, dtype=v.dtype
    # )
    # return out


@njit()
def _cross_product_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute the 3D cross product of two input vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First 3D vector (length-3 array).
    v2 : np.ndarray
        Second 3D vector (length-3 array).

    Returns
    -------
    np.ndarray
        The cross product of `v1` and `v2`, as a new 3D vector
        with the same dtype as the inputs.

    Notes
    -----
    This function assumes both input vectors are 3-dimensional.
    It is implemented using Numba for improved performance.
    """
    r = np.zeros(3, dtype=v1.dtype)
    r[0] = v1[1] * v2[2] - v1[2] * v2[1]
    r[1] = v1[2] * v2[0] - v1[0] * v2[2]
    r[2] = v1[0] * v2[1] - v1[1] * v2[0]
    return r


@njit()
def _mirror(x: int, nx: int):
    """
    Compute the mirrored position for interpolation border conditions.

    Parameters
    ----------
    x : int
        Original position.
    nx : int
        Size of the data array along the relevant axis.

    Returns
    -------
    int
        Mirrored position ensuring the index is within valid bounds.

    Notes
    -----
    This function handles boundary conditions by mirroring the index position.
    If the position is out of bounds, it reflects it back into the valid range.
    """
    if x >= 0 and x < nx:
        return x
    elif x < 0:
        return -x
    else:
        return 2 * nx - 2 - x


@njit()
def _interp_response_3d(response, x: float, y: float, z: float):
    """
    Perform trilinear interpolation on a 3D response array at fractional coordinates.

    Parameters
    ----------
    response : ndarray
        The input 3D data array, indexed as (z, y, x).
    x : float
        The x-coordinate at which to interpolate.
    y : float
        The y-coordinate at which to interpolate.
    z : float
        The z-coordinate at which to interpolate.

    Returns
    -------
    float
        The interpolated value at the specified coordinates.

    Notes
    -----
    - The function uses trilinear interpolation to estimate the value at the given (x, y, z) coordinates.
    - The `_mirror` function is used to handle boundary conditions by reflecting indices that are out of bounds.
    - The coordinates (x, y, z) can be fractional, allowing for smooth interpolation within the 3D data array.
    """
    nz, ny, nx = response.shape

    xi = int(x)
    yi = int(y)
    zi = int(z)

    dx = x - xi
    dy = y - yi
    dz = z - zi

    if x < 0:
        dx = -dx
        x1 = _mirror(xi - 1, nx)
    else:
        x1 = _mirror(xi + 1, nx)

    if y < 0:
        dy = -dy
        y1 = _mirror(yi - 1, ny)
    else:
        y1 = _mirror(yi + 1, ny)

    if z < 0:
        dz = -dz
        z1 = _mirror(zi - 1, nz)
    else:
        z1 = _mirror(zi + 1, nz)

    x0 = _mirror(xi, nx)
    y0 = _mirror(yi, ny)
    z0 = _mirror(zi, nz)

    z00 = (1.0 - dy) * (
        (1.0 - dx) * response[z0, y0, x0] + dx * response[z0, y0, x1]
    ) + dy * ((1.0 - dx) * response[z0, y1, x0] + dx * response[z0, y1, x1])

    z11 = (1.0 - dy) * (
        (1.0 - dx) * response[z1, y0, x0] + dx * response[z1, y0, x1]
    ) + dy * ((1.0 - dx) * response[z1, y1, x0] + dx * response[z1, y1, x1])

    return (1.0 - dz) * z00 + dz * z11


@njit()
def _get_settings_steerable_3d(m: int, sigma_z: float):
    """
    Get parameters for 3D steerable filters based on the order `m` and the standard deviation `sigma_z`.

    Parameters
    ----------
    m : int
        Order of the steerable filter (must be 1 or 2).
    sigma_z : float
        Standard deviation along the z-axis.

    Returns
    -------
    alpha_stg : float
        Scaling parameter for the steerable filter.
    sign_stg : float
        Sign parameter for the steerable filter.
    c_stg : float
        Normalization constant for the steerable filter.

    Raises
    ------
    ValueError
        If the order `m` is not 1 or 2.

    Notes
    -----
    The function returns different parameters based on whether the order `m` is 1 or 2:
    - For `m == 1`: alpha_stg = 2/3, sign_stg = -1, and c_stg = 2 * sqrt(2 * pi) * sigma_z.
    - For `m == 2`: alpha_stg = 4, sign_stg = 1, and c_stg = 8 * pi * sqrt(6) * sigma_z.

    Examples
    --------
    >>> alpha_stg, sign_stg, c_stg = get_settings_steerable_3d(1, 1.0)
    >>> alpha_stg, sign_stg, c_stg = get_settings_steerable_3d(2, 1.0)
    """
    if m == 1:
        alpha_stg = 2.0 / 3.0
        sign_stg = -1.0
        c_stg = 2.0 * math.sqrt(2.0 * math.pi) * sigma_z
    elif m == 2:
        alpha_stg = 4.0
        sign_stg = 1.0
        c_stg = 8.0 * math.pi * math.sqrt(6.0) * sigma_z
    else:
        raise ValueError(f"m = {m}, but should be either 1 or 2")

    return alpha_stg, sign_stg, c_stg


def _calculate_templates_3d(
    img: np.ndarray,
    sigma: float,
    sigma_z: float,
    truncate: float = 4.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate 3D gaussian derivative templates.

    The templates are computed using scipy.ndimage.gaussian_filter with custom magnitude
    scaling to align with steerable filter design conventions.

    This function computes the second-order Gaussian derivatives (Gxx, Gyy, Gzz, Gxy, Gxz, Gyz)
    for a 3D input volume. The output magnitudes are specifically scaled to match common
    conventions in steerable filter implementations (e.g., as described by Aguet, Jacob, and Unser),
    where the underlying 1D Gaussian kernels are often defined without standard probabilistic normalization.

    Parameters
    ----------
    img : ndarray
        The input 3D data array (indexed z, y, x).
    sigma : float
        Standard deviation for the Gaussian kernel in the x and y directions.
    sigma_z : float
        Standard deviation for the Gaussian kernel in the z direction.
    truncate: float, optional
        Truncate the filter at this many standard deviations. For exact replication of
        original steerable filter code, set it to 3.0. Default is 4.0.
    verbose : bool, optional
        If True, prints progress messages and timing information. Defaults to False.

    Returns
    -------
    gxx, gxy, gxz, gyy, gyz, gzz : tuple of ndarrays
        The computed 3D Gaussian derivative templates. The arrays will have
        the same shape and dtype as the input `img`.
    """
    if verbose:
        start_time = time.time()
        print("Calculating Gaussian derivatives with scipy.ndimage...")

    # Gaussian filter parameters for scipy.ndimage.gaussian_filter:
    # - `sigma`: A tuple (sigma_z, sigma_y, sigma_x) specifying standard deviations per axis.
    # - `order`: A tuple specifying the derivative order for each dimension (0 for no derivative).
    # - `mode="_mirror"`: Handles boundary conditions, common in filter implementations.
    # - `truncate=3`: Limits the kernel size to 3 standard deviations from the center,
    #   a standard practical choice for numerical stability and efficiency.

    # Calculate individual second derivatives
    # Gxx: 2nd derivative in x, 0th in y and z
    gxx = ndimage.gaussian_filter(
        img,
        sigma=(sigma_z, sigma, sigma),
        order=(0, 0, 2),
        mode="mirror",
        truncate=truncate,
    )
    # Gyy: 2nd derivative in y, 0th in x and z
    gyy = ndimage.gaussian_filter(
        img,
        sigma=(sigma_z, sigma, sigma),
        order=(0, 2, 0),
        mode="mirror",
        truncate=truncate,
    )
    # Gzz: 2nd derivative in z, 0th in x and y
    gzz = ndimage.gaussian_filter(
        img,
        sigma=(sigma_z, sigma, sigma),
        order=(2, 0, 0),
        mode="mirror",
        truncate=truncate,
    )

    # Calculate mixed derivatives
    # Gxy: 1st derivative in x, 1st in y, 0th in z
    gxy = ndimage.gaussian_filter(
        img,
        sigma=(sigma_z, sigma, sigma),
        order=(0, 1, 1),
        mode="mirror",
        truncate=truncate,
    )
    # Gxz: 1st derivative in x, 1st in z, 0th in y
    gxz = ndimage.gaussian_filter(
        img,
        sigma=(sigma_z, sigma, sigma),
        order=(1, 0, 1),
        mode="mirror",
        truncate=truncate,
    )
    # Gyz: 1st derivative in y, 1st in z, 0th in x
    gyz = ndimage.gaussian_filter(
        img,
        sigma=(sigma_z, sigma, sigma),
        order=(1, 1, 0),
        mode="mirror",
        truncate=truncate,
    )

    # --- Custom Magnitude Scaling to Match Steerable Filter Conventions ---
    # In many steerable filter frameworks (e.g., as derived in F. Aguet et al., 2005),
    # the underlying 1D Gaussian kernels are defined without the standard probabilistic
    # normalization factor (i.e., 1 / (sigma * sqrt(2*pi))). This means those
    # 1D kernels are effectively "scaled up" compared to a unit-area Gaussian.
    #
    # Since scipy.ndimage.gaussian_filter produces standard-normalized outputs
    # (it internally includes these factors), we must *upscale* scipy's results
    # to match the magnitudes produced by convolving with those unnormalized 1D kernels.
    #
    # The total 'scale up' factor for a 3D convolution is the product of these
    # omitted 1D normalizations across the X, Y, and Z dimensions.

    # Calculate the inverse of the standard 1D Gaussian PDF normalization constant.
    # These represent the 'scale-up' factors implicitly present in the 1D kernels
    # defined without standard normalization.
    scale_up_factor_xy_1d = np.sqrt(
        2 * np.pi * sigma**2
    )  # Corresponds to 1 / (1 / sqrt(2*pi*sigma^2))
    scale_up_factor_z_1d = np.sqrt(
        2 * np.pi * sigma_z**2
    )  # Corresponds to 1 / (1 / sqrt(2*pi*sigma_z^2))

    # The combined factor by which the results from unnormalized 1D sequential convolutions
    # are scaled up, relative to a fully normalized 3D Gaussian convolution.
    total_unnormalized_scale_factor = (
        scale_up_factor_xy_1d * scale_up_factor_xy_1d * scale_up_factor_z_1d
    )

    # Apply this scaling to the scipy.ndimage outputs.
    # Note: A further global normalization (e.g., 'c_stg' from _get_settings_steerable_3d)
    # is typically applied at a later stage in the complete steerable filter pipeline.
    gxx *= total_unnormalized_scale_factor
    gxy *= total_unnormalized_scale_factor
    gxz *= total_unnormalized_scale_factor
    gyy *= total_unnormalized_scale_factor
    gyz *= total_unnormalized_scale_factor
    gzz *= total_unnormalized_scale_factor

    if verbose:
        end_time = time.time()
        print(
            f"Gaussian derivative calculation finished in {end_time - start_time:.2f} seconds."
        )

    return gxx, gxy, gxz, gyy, gyz, gzz


@njit(parallel=True)
def _calculate_steerable_3d_response_orientation(
    gxx: np.ndarray,
    gxy: np.ndarray,
    gxz: np.ndarray,
    gyy: np.ndarray,
    gyz: np.ndarray,
    gzz: np.ndarray,
    m: int,
    sigma_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the steerable response and orientation for each voxel using Numba.

    This function iterates through each voxel, constructs a Hessian-like matrix
    from the Gaussian derivatives, computes its eigenvalues and eigenvectors,
    and determines the steerable response and primary orientation.

    Parameters
    ----------
    gxx, gxy, gxz, gyy, gyz, gzz : ndarray
        The computed 3D Gaussian derivative templates (shape: nz, ny, nx).
    m : int
        The order of the steerable filter (1 for curve, 2 for surface).
    sigma_z : float
        Standard deviation along the z-axis, used for normalization.

    Returns
    -------
    response : ndarray
        The steerable filter response volume (shape: nz, ny, nx).
    orientation : ndarray
        The primary orientation vector for each voxel (shape: nz, ny, nx, 3).
    """
    nz, ny, nx = gxx.shape

    response = np.zeros((nz, ny, nx), dtype=gxx.dtype)
    orientation = np.zeros((nz, ny, nx, 3), dtype=gxx.dtype)

    alpha_stg, sign_stg, c_stg = _get_settings_steerable_3d(m, sigma_z)

    # Iterate over all voxels
    for z in prange(nz):  # prange enables parallel execution here
        for y in range(ny):
            for x in range(nx):
                # Extract derivative values for the current voxel
                gxx_temp = gxx[z, y, x]
                gxy_temp = gxy[z, y, x]
                gxz_temp = gxz[z, y, x]
                gyy_temp = gyy[z, y, x]
                gyz_temp = gyz[z, y, x]
                gzz_temp = gzz[z, y, x]

                # Construct the matrix A based on steerable filter formulation
                a = sign_stg * (gyy_temp + gzz_temp - alpha_stg * gxx_temp)
                b = sign_stg * (gxx_temp + gzz_temp - alpha_stg * gyy_temp)
                c = sign_stg * (gxx_temp + gyy_temp - alpha_stg * gzz_temp)

                d = -sign_stg * (1 + alpha_stg) * gxy_temp
                e = -sign_stg * (1 + alpha_stg) * gxz_temp
                f = -sign_stg * (1 + alpha_stg) * gyz_temp

                # Check for flat/uniform region, for example region of all 0's
                # This will lead to a degenerate Hessian, so skip eigen-decomposition
                if max(abs(a), abs(b), abs(c), abs(d), abs(e), abs(f)) < 1e-6:
                    response[z, y, x] = 0.0  # No response
                    orientation[z, y, x, 0] = 1.0  # arbitrary unit vector
                    orientation[z, y, x, 1] = 0.0
                    orientation[z, y, x, 2] = 0.0
                    continue

                A = np.array([[a, d, f], [d, b, e], [f, e, c]], dtype="<f8")
                eigenvals, eigenvecs = np.linalg.eigh(A)

                # Get the largest eigenvalue and compute response value
                # eigh returns the eigenvalues in ascending order
                response[z, y, x] = eigenvals[-1] / c_stg

                # Store the eigenvector corresponding to the largest eigenvalue (x, y, z)
                orientation[z, y, x, :] = eigenvecs[:, -1]

    return response, orientation


@njit(parallel=True)
def _compute_curve_nms(response: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on a 3D response volume along curves.

    This function identifies local maxima in a 3D volume `response` by suppressing
    non-maximum values along curves that are locally perpendicular to the orientation
    vectors. Interpolation is used to sample points along a circular cross-section
    orthogonal to each local orientation.

    Parameters
    ----------
    response : np.ndarray
        3D array of shape (nz, ny, nx) representing the response volume (e.g., from a
        filter bank).
    orientation : np.ndarray
        4D array of shape (nz, ny, nx, 3), where each element is a 3D orientation vector
        associated with the corresponding voxel in `response`.

    Returns
    -------
    nms : np.ndarray
        3D array of shape (nz, ny, nx), same as `response`, containing only the local
        maxima
        along curves defined by the orientation vectors. All other values are set to
        zero.

    Notes
    -----
    - The orientation vectors are expected to be unit vectors or will be normalized
    internally.
    - A small circle of 10 points is interpolated per voxel, perpendicular to its
    orientation.
    - Voxels with undefined orientation vectors (i.e., [0, 0, 0]) are skipped.
    - Uses `_interp_response_3d(response, x_pos, y_pos, z_pos)` for linear
    interpolation, compatible with Numba JIT.
    - Parallelized over the z-axis using `prange` for performance.

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter
    >>> response = gaussian_filter(
    ...     np.random.rand(32, 32, 32).astype(np.float32), sigma=1
    ... )
    >>> orientation = np.random.randn(32, 32, 32, 3).astype(np.float32)
    >>> nms_result = _compute_curve_nms(response, orientation)
    """
    nz, ny, nx = response.shape
    nms = np.zeros_like(response)

    # Unit vectors for cross product base
    iv = np.array([1.0, 0.0, 0.0], dtype=response.dtype)
    jv = np.array([0.0, 1.0, 0.0], dtype=response.dtype)

    # Interpolate X points uniformly distributed on unit circle perpendicular to orientation
    nt = 10  # Number of points on the circle
    theta = np.linspace(0, 2.0 * math.pi, nt)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    null_direction = np.zeros(3, dtype=orientation.dtype)  # (0, 0, 0) vector

    for z in prange(nz):  # prange enables parallel execution
        for y in range(ny):
            for x in range(nx):
                current_orientation = orientation[z, y, x, :]

                # Skip if orientation is a zero vector (no defined direction)
                if np.allclose(current_orientation, null_direction):
                    continue

                # Ensure orientation is normalized for correct vector arithmetic
                current_orientation = _normalize_vector(current_orientation)

                # Calculate two orthogonal vectors (u, v) that are perpendicular to
                # `current_orientation`
                if not np.isclose(current_orientation[0], 1.0) and not np.isclose(
                    current_orientation[0], -1.0
                ):
                    u = _cross_product_3d(iv, current_orientation)
                else:
                    u = _cross_product_3d(
                        jv, current_orientation
                    )  # Use jv if iv is collinear with orientation

                u = _normalize_vector(u)  # Ensure u is a unit vector
                v = _cross_product_3d(
                    current_orientation, u
                )  # v is orthogonal to both orientation and u

                current_response_val = response[z, y, x]
                is_local_maximum = True

                # Sample points on a unit circle perpendicular to the orientation
                for t in range(nt):
                    # Compute interpolated point coordinates (z, y, x)
                    # Note: orientation vector components are (ox, oy, oz).
                    # Here u[0], v[0] are x-components; u[1], v[1] are y-components;
                    # u[2], v[2] are z-components.
                    z_pos = z + cos_theta[t] * u[2] + sin_theta[t] * v[2]
                    y_pos = y + cos_theta[t] * u[1] + sin_theta[t] * v[1]
                    x_pos = x + cos_theta[t] * u[0] + sin_theta[t] * v[0]

                    # Interpolate the response value at the current point
                    ival = _interp_response_3d(response, x_pos, y_pos, z_pos)

                    if ival >= current_response_val:
                        is_local_maximum = False
                        break  # Not a local maximum in this direction

                if is_local_maximum:
                    nms[z, y, x] = current_response_val
    return nms


@njit(parallel=True)
def _compute_surface_nms(response: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on a 3D response volume along surface normals.

    This function identifies local maxima in a 3D `response` volume by suppressing
    values that are not maximal along the direction defined by the local `orientation`
    vector. The check is performed by interpolating the response one step forward and
    backward along the orientation vector, and comparing the current voxel's value
    against them.

    Parameters
    ----------
    response : np.ndarray
        3D array of shape (nz, ny, nx) containing the response volume.
    orientation : np.ndarray
        4D array of shape (nz, ny, nx, 3), where each element is a 3D orientation vector
        associated
        with the corresponding voxel in `response`. Represents the direction of interest
        (e.g., normal vector).

    Returns
    -------
    nms : np.ndarray
        3D array of shape (nz, ny, nx), same as `response`, where only local maxima
        along the orientation vector are preserved. All other values are set to zero.

    Notes
    -----
    - A voxel is considered a local maximum if its response is greater than its
    immediate neighbors in both directions along the orientation vector.
    - Orientation vectors are normalized before use.
    - Voxels with zero orientation vectors ([0, 0, 0]) are ignored.
    - Interpolation is performed using
    `_interp_response_3d(response, x_pos, y_pos, z_pos)` with linear interpolation and
    mirror boundary mode.
    - Parallelized over the z-axis using `prange` for performance with Numba.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.ndimage import gaussian_filter
    >>> response = gaussian_filter(
    ...     np.random.rand(32, 32, 32).astype(np.float32), sigma=1
    ... )
    >>> orientation = np.random.randn(32, 32, 32, 3).astype(np.float32)
    >>> nms_result = _compute_surface_nms(response, orientation)
    """
    nz, ny, nx = response.shape
    nms = np.zeros_like(response)

    null_direction = np.zeros(3, dtype=orientation.dtype)  # (0, 0, 0) vector

    for z in prange(nz):  # prange enables parallel execution
        for y in range(ny):
            for x in range(nx):
                current_orientation = orientation[z, y, x, :]

                # Skip if orientation is a zero vector
                if np.allclose(current_orientation, null_direction):
                    continue

                # Ensure orientation is normalized, though `map_coordinates` will handle
                # magnitude
                # A unit step along the orientation is typically used for NMS
                current_orientation = _normalize_vector(current_orientation)

                current_response_val = response[z, y, x]

                # Point 1: One step along the orientation vector
                z1_pos = z + current_orientation[2]  # z-component of orientation
                y1_pos = y + current_orientation[1]  # y-component of orientation
                x1_pos = x + current_orientation[0]  # x-component of orientation
                A1 = _interp_response_3d(response, x1_pos, y1_pos, z1_pos)

                # Point 2: One step in the opposite direction
                z2_pos = z - current_orientation[2]
                y2_pos = y - current_orientation[1]
                x2_pos = x - current_orientation[0]
                A2 = _interp_response_3d(response, x2_pos, y2_pos, z2_pos)

                # If the current response is greater than both neighbors along the
                # orientation, it's a local maximum
                if current_response_val > A1 and current_response_val > A2:
                    nms[z, y, x] = current_response_val
    return nms


def steerable_detector_3d(
    img: np.ndarray,
    m: int,
    sigma: float,
    zx_ratio: float = 1.0,
    truncate: float = 3.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform 3D steerable detection and non-maximum suppression (NMS) on a 3D volume.

    This function applies a steerable detector to a 3D volume to identify features based
    on the specified filter type and then applies NMS to find local maxima.

    Parameters
    ----------
    img : ndarray
        A 3D volume array (shape: (nz, ny, nx)) to process. The input array must be
        floating-point (e.g., float32, float64) and must not contain NaNs.
    m : int
        The filter type. Must be either 1 (curve detector) or 2 (surface detector).
    sigma : float
        The standard deviation of the Gaussian kernel used in the steerable detector
        for the x and y directions. Must be positive. Unit is pixels.
    zx_ratio : float, optional
        The ratio of sigma in the z-direction to sigma in the x/y directions.
        `sigma_z = sigma / zx_ratio`. Must be positive. Default is 1.0 (isotropic).
    truncate: float, optional
        Truncate the filter at this many standard deviations. For exact replication of
        original steerable filter code, set it to 3.0. Default is 4.0.
    verbose : bool, optional
        If True, prints progress messages and timing information. Default is False.

    Returns
    -------
    response : ndarray
        The response volume after applying the steerable detector, with shape
        (nz, ny, nx).
    orientation : ndarray
        The orientation vectors for each voxel, with shape (nz, ny, nx, 3).
        Each vector is normalized.
    nms : ndarray
        The result of non-maximum suppression, with the same shape as the input volume
        (nz, ny, nx). Only local maxima in the response are preserved.

    Raises
    ------
    ValueError
        If the input volume is not 3D, contains NaNs, has an invalid dtype,
        or if any of the input parameters are invalid
        (e.g., sigma or zx_ratio are non-positive, or the filter support exceeds the
        volume dimensions).
    """
    if not isinstance(img, np.ndarray):
        raise ValueError(f"img is not a NumPy array, but of type={type(img)}")

    if img.ndim != 3:
        raise ValueError(
            f"Input must be a 3D NumPy array, but is a {len(img.ndim)}D array."
        )
    if not np.issubdtype(img.dtype, np.floating):
        # Convert to float to ensure compatibility with scientific computations
        if verbose:
            print(f"Converting input volume from {img.dtype} to float32...")
        img = img.astype(np.float32)

    if np.isnan(img).any():
        raise ValueError(
            "Input volume contains NaN values. Please handle them before processing."
        )

    if not isinstance(m, int) or m not in [1, 2]:
        raise ValueError(
            "Invalid filter type 'm'. Must be 1 (curve detector) or 2 "
            f"(surface detector). Got {m}."
        )

    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError(f"Invalid sigma value. Must be a positive float. Got {sigma}.")

    if not isinstance(zx_ratio, (int, float)) or zx_ratio <= 0:
        raise ValueError(f"Invalid zx_ratio. Must be a positive float. Got {zx_ratio}.")

    sigma_z = sigma / zx_ratio
    nz, ny, nx = img.shape

    # Heuristic: Gaussian filters extend roughly 3-4 standard deviations.
    # For derivatives, it's safer to consider 4*sigma.
    # The filter needs to be fully contained within the volume for accurate mirror
    # boundary conditions.
    min_dim_size = 4 * max(sigma, sigma_z)  # roughly
    if nx < min_dim_size or ny < min_dim_size or nz < min_dim_size:
        raise ValueError(
            f"Volume dimensions ({nx}x{ny}x{nz}) are too small for sigma={sigma}, "
            f"sigma_z={sigma_z}. Minimum suggested dimension size is approximately "
            f"{int(np.ceil(min_dim_size))}. Consider reducing sigma/sigma_z or "
            "providing a larger volume. This can lead to inaccurate boundary effects."
        )

    # --- Core Processing ---

    # 1. Calculate the Gaussian derivative templates
    gxx, gxy, gxz, gyy, gyz, gzz = _calculate_templates_3d(
        img, sigma, sigma_z, truncate=truncate, verbose=verbose
    )

    # 2. Calculate steerable response and orientation using numba
    if verbose:
        start_time = time.time()
        print("Calculating steerable response and orientation with numba...")
    response, orientation = _calculate_steerable_3d_response_orientation(
        gxx, gxy, gxz, gyy, gyz, gzz, m, sigma_z
    )
    if verbose:
        end_time = time.time()
        print(
            "Steerable response and orientation calculation finished in "
            f"{end_time - start_time:.2f} seconds."
        )

    # 3. Perform non-maximum suppression (NMS)
    if verbose:
        start_time = time.time()
        print("Performing non-maximum suppression...")
    if m == 1:  # Curve detector NMS
        nms = _compute_curve_nms(response, orientation)
    elif m == 2:  # Surface detector NMS
        nms = _compute_surface_nms(response, orientation)

    if verbose:
        end_time = time.time()
        print(
            f"Non-maximum suppression finished in {end_time - start_time:.2f} seconds."
        )

    return response, orientation, nms
