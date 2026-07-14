import warnings

import numpy as np
import scipy
from numba import njit
from numpy.fft import fftfreq, fftn, ifftn


@njit(parallel=False, cache=False)
def eigh_3d_batch(a, b, c, d, e, f, sort_descending: bool = True):
    """
    Compute eigenvalues and eigenvectors of symmetric 3x3 matrices in a 3D volume.

    Each 3x3 matrix is defined at a voxel (i, j, k) using the six unique components
    of a symmetric matrix:
        [[a, d, f],
         [d, b, e],
         [f, e, c]]

    Parameters
    ----------
    a, b, c : ndarray of float32, shape (N, M, L)
        Diagonal components of the symmetric matrix at each voxel.
    d, e, f : ndarray of float32, shape (N, M, L)
        Off-diagonal components:
        - d = M[0,1] = M[1,0]
        - e = M[1,2] = M[2,1]
        - f = M[0,2] = M[2,0]
    sort_descending : bool, optional
        Whether to sort eigenvalues in descending (default) or ascending order.
        np.linalg.eigh always returns ascending order; this flag reverses the result.

    Returns
    -------
    eigenvalues : ndarray of float32, shape (N, M, L, 3)
        The eigenvalues at each voxel.
    eigenvectors : ndarray of float32, shape (N, M, L, 3, 3)
        The corresponding unit-norm eigenvectors. For voxel (i, j, k),
        `eigenvectors[i, j, k, :, n]` is the eigenvector for `eigenvalues[i, j, k, n]`.

    Examples
    --------
    >>> mu, vecs = eigh_3d_batch(a, b, c, d, e, f)
    >>> mu.shape
    (N, M, L, 3)
    >>> vecs.shape
    (N, M, L, 3, 3)
    """
    shape = a.shape

    eigenvalues = np.zeros(shape + (3,), dtype=np.float32)  # noqa: RUF005
    eigenvectors = np.zeros(shape + (3, 3), dtype=np.float32)  # noqa: RUF005

    M = np.zeros((3, 3), dtype=np.float32)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                M[0, 0] = a[i, j, k]
                M[0, 1] = d[i, j, k]
                M[0, 2] = f[i, j, k]
                M[1, 0] = d[i, j, k]
                M[1, 1] = b[i, j, k]
                M[1, 2] = e[i, j, k]
                M[2, 0] = f[i, j, k]
                M[2, 1] = e[i, j, k]
                M[2, 2] = c[i, j, k]

                vals, vecs = np.linalg.eigh(M)  # ascending order

                if sort_descending:
                    vals = vals[::-1]
                    vecs = vecs[:, ::-1]

                eigenvalues[i, j, k, :] = vals
                eigenvectors[i, j, k, :, :] = vecs

    return eigenvalues, eigenvectors


def max_principal_curvature_limit(scale: float, coefficient: float = 1.0) -> float:
    """
    Compute the theoretical maximum principal curvature measurable at a spatial scale.

    Curvature has units of inverse length squared (e.g., um^-2). When curvature is
    measured on smoothed data, small-scale variations below that smoothing scale are
    filtered out.

    The smoothing scale `scale` sets a resolution limit. The finest curvature detail
    detectable is approximately limited by the inverse square of this scale.

    Parameters
    ----------
    scale : float
        Effective smoothing scale (same length units as input data).
    coefficient : float, default 0.5
        Proportionality constant. The minimum radius of curvature is taken as
        coefficient^(-0.5) * scale, giving kappa_max = coefficient / scale^2.

    Returns
    -------
    float
        Maximum measurable principal curvature (inverse length squared).
    """
    if scale <= 0:
        raise ValueError("scale must be positive.")

    max_principal_curvature = coefficient / (scale**2)
    return max_principal_curvature


def compute_sigma_t(scale: float, sigma_g: float, sigma_k: float) -> float:
    """
    Compute the tensor smoothing scale (sigma_t) based on other provided sigmas.

    Compute the required tensor smoothing scale (sigma_t) such that the total
    effective smoothing scale (`scale`) is achieved when combined with
    gradient scale (`sigma_k`) and Gaussian smoothing scale (`sigma_g`).

    Gaussian smoothing scales combine by adding their variances (sigma^2) in quadrature.

    Parameters
    ----------
    scale : float
        Desired total effective scale for curvature measurement (e.g., Hessian scale).
    sigma_g : float
        Scale parameter for Gaussian smoothing applied during gradient computation
        (i.e., sigma of Gaussian filter used to compute image derivatives).
    sigma_k : float
        Scale parameter representing the gradient computation smoothing scale.

    Returns
    -------
    sigma_t : float
        Tensor smoothing scale required to achieve total scale `scale`.

    Raises
    ------
    ValueError
        If the desired total scale `scale` is smaller than the combined input scales,
        making it impossible to achieve with positive `sigma_t`.
    """
    # Compute squared scales since variances add
    scale_sq = scale**2
    sigma_k_sq = sigma_k**2
    sigma_g_sq = sigma_g**2

    # Calculate sigma_t squared by subtracting known variances from total variance
    sigma_t_sq = scale_sq - sigma_k_sq - sigma_g_sq

    if sigma_t_sq < 0:
        raise ValueError(
            "scale must be greater than or equal to sqrt(sigma_k^2 + sigma_g^2). "
            f"Got scale={scale}, sigma_k={sigma_k}, sigma_g={sigma_g}."
        )

    # Return the positive square root
    sigma_t = np.sqrt(sigma_t_sq)
    return sigma_t


def compute_structure_tensor_3d(
    image: np.ndarray,
    sigma_g: float,
    sigma_t: float,
    axis_order="zyx",
    return_full: bool = False,
):
    """
    Calculate the 3D structure tensor of a volumetric image.

    The structure tensor summarizes local image gradient orientations by computing
    gradients smoothed by a Gaussian (scale `sigma_g`), forming outer products of
    these gradients, and then smoothing these products with another Gaussian
    (scale `sigma_t`). This produces a tensor field describing local directional
    structure.

    Parameters
    ----------
    image : ndarray
        Input 3D image with spatial axes ordered as specified by `axis_order`.
    sigma_g : float
        Gaussian smoothing sigma for gradient computation.
    sigma_t : float
        Gaussian smoothing sigma for structure tensor component smoothing.
    axis_order : str, optional
        String indicating spatial axis order of `image` dimensions (default "zyx").
    return_full : bool, optional
        If True, return the full 3x3 structure tensor at each voxel as an array.
        If False, return individual tensor components separately.

    Returns
    -------
    Axx, Ayy, Azz, Axy, Ayz, Axz : ndarray or
    gst : ndarray
        If `return_full` is False, returns the six unique components of the symmetric
        3D structure tensor as separate arrays, each matching `image` shape.
        If True, returns the full tensor array with shape (*image.shape, 3, 3).

    Raises
    ------
    ValueError
        If `axis_order` is invalid or if `image` is not 3D.
    """
    # Verify input is 3D
    if image.ndim != 3:
        raise ValueError("Input image must be 3D")

    # Check axis_order validity
    if set(axis_order) != {"x", "y", "z"} or len(axis_order) != 3:
        raise ValueError("axis_order must be a permutation of 'xyz'")

    # Map axis letters to indices
    axis_map = {axis: i for i, axis in enumerate(axis_order)}

    # Compute Gaussian-smoothed image gradients along each axis
    gx = scipy.ndimage.gaussian_filter(
        image, sigma_g, order=[1 if i == axis_map["x"] else 0 for i in range(3)]
    )
    gy = scipy.ndimage.gaussian_filter(
        image, sigma_g, order=[1 if i == axis_map["y"] else 0 for i in range(3)]
    )
    gz = scipy.ndimage.gaussian_filter(
        image, sigma_g, order=[1 if i == axis_map["z"] else 0 for i in range(3)]
    )

    if return_full:
        # Initialize full tensor array
        gst = np.zeros((*image.shape, 3, 3), dtype=np.float32)

        # Compute smoothed outer products of gradients
        gst[..., 0, 0] = scipy.ndimage.gaussian_filter(gx * gx, sigma_t)
        gst[..., 0, 1] = scipy.ndimage.gaussian_filter(gx * gy, sigma_t)
        gst[..., 0, 2] = scipy.ndimage.gaussian_filter(gx * gz, sigma_t)
        del gx  # Release memory for gx

        gst[..., 1, 1] = scipy.ndimage.gaussian_filter(gy * gy, sigma_t)
        gst[..., 1, 2] = scipy.ndimage.gaussian_filter(gy * gz, sigma_t)
        del gy  # Release memory for gy

        gst[..., 2, 2] = scipy.ndimage.gaussian_filter(gz * gz, sigma_t)
        del gz  # Release memory for gz

        # Symmetrize the tensor
        gst[..., 1, 0] = gst[..., 0, 1]
        gst[..., 2, 1] = gst[..., 1, 2]
        gst[..., 2, 0] = gst[..., 0, 2]

        return gst

    else:
        # Compute individual tensor components with smoothing
        Axx = scipy.ndimage.gaussian_filter(gx * gx, sigma_t)
        Axy = scipy.ndimage.gaussian_filter(gx * gy, sigma_t)
        Axz = scipy.ndimage.gaussian_filter(gx * gz, sigma_t)
        del gx

        Ayy = scipy.ndimage.gaussian_filter(gy * gy, sigma_t)
        Ayz = scipy.ndimage.gaussian_filter(gy * gz, sigma_t)
        del gy

        Azz = scipy.ndimage.gaussian_filter(gz * gz, sigma_t)
        del gz

        # Return unique components only to save memory
        return Axx, Ayy, Azz, Axy, Ayz, Axz


def compute_hessian_3d(
    image, sigma, axis_order: str = "zyx", return_full: bool = False
):
    """
    Compute the 3D Hessian matrix of an image via Gaussian second derivatives.

    The Hessian matrix contains all second-order partial derivatives, capturing
    local curvature information useful for detecting edges, blobs, and ridges.

    Parameters
    ----------
    image : ndarray
        3D scalar image.
    sigma : float
        Gaussian smoothing scale for second derivative computation.
    axis_order : str, optional
        String specifying spatial axis order in `image` (default "zyx").
    return_full : bool, optional
        If True, returns full 3x3 Hessian matrix at each voxel as an array.
        Otherwise returns the six unique second derivatives separately.

    Returns
    -------
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz : ndarray or
    hessian : ndarray
        If `return_full` is False, returns six arrays of second derivatives.
        If True, returns an array with shape (*image.shape, 3, 3) representing
        the full symmetric Hessian matrix.

    Raises
    ------
    ValueError
        If `axis_order` is invalid or input is not 3D.

    Notes
    -----
    - Mixed partial derivatives are symmetric, e.g., Hxy == Hyx.
    - Gaussian smoothing regularizes derivative estimates.
    """
    if image.ndim != 3:
        raise ValueError("Input image must be a 3D array.")

    if set(axis_order) != {"x", "y", "z"} or len(axis_order) != 3:
        raise ValueError("axis_order must be a permutation of 'xyz'")

    # Map axis letters to indices
    axis_map = {axis: i for i, axis in enumerate(axis_order)}

    def second_derivative(order_axes):
        order = [0, 0, 0]
        order[axis_map[order_axes[0]]] += 1
        order[axis_map[order_axes[1]]] += 1
        return scipy.ndimage.gaussian_filter(image, sigma=sigma, order=order)

    if return_full:
        # Initialize full Hessian matrix array
        hessian = np.zeros((*image.shape, 3, 3), dtype=np.float32)

        # Fill diagonal second derivatives
        hessian[..., 0, 0] = second_derivative("xx")
        hessian[..., 1, 1] = second_derivative("yy")
        hessian[..., 2, 2] = second_derivative("zz")

        # Fill off-diagonal mixed partials
        hessian[..., 0, 1] = second_derivative("xy")
        hessian[..., 1, 2] = second_derivative("yz")
        hessian[..., 0, 2] = second_derivative("xz")

        # Symmetrize Hessian matrix
        hessian[..., 1, 0] = hessian[..., 0, 1]
        hessian[..., 2, 1] = hessian[..., 1, 2]
        hessian[..., 2, 0] = hessian[..., 0, 2]

        return hessian

    else:
        # Compute unique second derivatives separately to save memory
        Hxx = second_derivative("xx")
        Hxy = second_derivative("xy")
        Hxz = second_derivative("xz")
        Hyy = second_derivative("yy")
        Hyz = second_derivative("yz")
        Hzz = second_derivative("zz")

        return Hxx, Hxy, Hxz, Hyy, Hyz, Hzz


def _dot_gradient_square(
    v: list[np.ndarray] | np.ndarray,
    t: np.ndarray,
    sigma_k: float = 1.0,
    axis_order: str = "zyx",
):
    """
    Calculate squared dot product between vector fields and scalar gradient.

    Computes (v . grad(t))^2 where grad(t) is the gradient of scalar field `t`
    computed via Gaussian derivatives. Supports a single vector field or a list
    of vector fields.

    Parameters
    ----------
    v : list of ndarray or ndarray
        Vector field(s) with shape (..., 3), where last axis corresponds to (x, y, z).
    t : ndarray
        Scalar 3D field over which the gradient is computed.
    sigma_k : float, optional
        Gaussian smoothing scale used when computing gradients of `t`.
    axis_order : str, optional
        Spatial axis order in `t` (default "zyx").

    Returns
    -------
    out : list of ndarray or ndarray
        Squared dot products of shape matching `t` spatial dimensions.
        Returns a list if `v` is a list, else a single array.

    Raises
    ------
    ValueError
        If `axis_order` is not a permutation of 'xyz'.
    """
    if set(axis_order) != {"x", "y", "z"} or len(axis_order) != 3:
        raise ValueError("axis_order must be a permutation of 'xyz'")

    # Map axes to indices
    axis_map = {axis: i for i, axis in enumerate(axis_order)}

    # Compute gradient components of t in x,y,z order
    grad = {}
    for axis in "xyz":
        order_full = [0, 0, 0]
        order_full[axis_map[axis]] = 1
        grad[axis] = scipy.ndimage.gaussian_filter(t, sigma=sigma_k, order=order_full)

    # Stack gradients to form vector field with last axis (x,y,z)
    grad_t = np.stack([grad["x"], grad["y"], grad["z"]], axis=-1)

    del grad  # Release memory

    # Compute squared dot product (dot(v, grad(t)))^2
    if isinstance(v, list):
        return [np.sum(vi * grad_t, axis=-1) ** 2 for vi in v]
    else:
        return np.sum(v * grad_t, axis=-1) ** 2


def _add_elements(s, n):
    """
    Element-wise addition of two arrays or lists.

    Supports inputs that are either both lists or both NumPy arrays,
    returning the element-wise sum in the same format as inputs.

    Parameters
    ----------
    s : list or ndarray
        First operand.
    n : list or ndarray
        Second operand, same type and shape as `s`.

    Returns
    -------
    out : list or ndarray
        Element-wise sum of `s` and `n`.
    """
    # Add element-wise if both inputs are lists
    if isinstance(s, list) and isinstance(n, list):
        return [a + b for a, b in zip(s, n, strict=False)]

    # Add element-wise if both inputs are NumPy arrays
    elif isinstance(s, np.ndarray) and isinstance(n, np.ndarray):
        return s + n


def _truncated_inverse_vector_gradient_nd(
    gradients: list[np.ndarray], p: float = 1, sigma: float = 10, dx: float = 1.0
):
    """
    Reconstruct a scalar field from its gradient components.

    Reconstruct a scalar field from its gradient components using a truncated
    inverse vector-gradient operator in N dimensions.

    Parameters
    ----------
    gradients : list of ndarray
        List of N gradient components [g1, g2, ..., gN], each an ndarray representing
        the gradient along one axis. All arrays must have the same shape.
    p : float, optional
        Power in the generalized vector-gradient model (default is 1).
    sigma : float, optional
        Scale of the Gaussian truncation filter in the frequency domain,
        in pixel units (default is 10).
    dx : float, optional
        Spatial sampling step (assumed equal in all dimensions). Default is 1.0.

    Returns
    -------
    result : ndarray
        Reconstructed scalar field of the same shape as the input gradient components.

    Notes
    -----
    This function implements the generalized inverse vector-gradient operator as
    proposed by Verbeek and Dijk (2003). It reconstructs a scalar field from its
    N-dimensional gradient components using a frequency-domain inversion.

    The inverse operator applies a modulus-frequency term raised to the power `-(p+1)`,
    multiplied by a Gaussian window to truncate long-range integration and reduce
    aliasing in periodic images.

    The Gaussian smoothing emulates limited spatial support and enhances stability. The
    operator's zero crossings remain unaffected by smoothing, making it well-suited for
    practical applications.

    The DC (zero-frequency) component is zeroed to avoid instability due to undefined
    mean values.

    This method supports scale-free enhancement workflows, where modified gradient
    magnitudes are inverted without reference to grey levels or explicit spatial scales.

    References
    ----------
    Verbeek, P.W., & Dijk, J. (2003). The D-Dimensional Inverse Vector-Gradient Operator
    and Its Application for Scale-Free Image Enhancement. In: Petkov, N., Westenberg,
    M.A. (eds) Computer Analysis of Images and Patterns. CAIP 2003. Lecture Notes in
    Computer Science, vol 2756. Springer, Berlin, Heidelberg.
    https://doi.org/10.1007/978-3-540-45179-2_90
    """
    ndim = len(gradients)
    shape = gradients[0].shape
    assert all(g.shape == shape for g in gradients), (
        "All gradient components must have the same shape"
    )

    # Frequency grids
    freq_grids = np.meshgrid(*[fftfreq(n, d=dx) for n in shape], indexing="ij")
    rho_sq = sum(f**2 for f in freq_grids)
    rho = np.sqrt(rho_sq)
    rho[rho == 0] = 1e-8  # avoid divide by zero

    # Gaussian truncation filter in frequency domain
    gaussian = np.exp(-2 * (np.pi**2) * sigma**2 * rho_sq)

    # Apply inverse vector-gradient operator
    F_recon = np.zeros(shape, dtype=np.complex128)
    for gk, fk in zip(gradients, freq_grids, strict=False):
        Gk = fftn(gk)
        multiplier = gaussian * (-1j * fk) / (rho ** (p + 1))
        F_recon += Gk * multiplier

    # Zero out the DC component to avoid instability from undefined mean
    F_recon[tuple([0] * ndim)] = 0

    # Inverse FFT to get reconstructed image
    result = np.real(ifftn(F_recon))

    return result


def _fill_grey_shell(
    image: np.ndarray,
    sigma_g: float,
    sigma_fill: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fill 3D shell via gray-weighted distance transform and vector gradient integration.

    This function performs a fill of a hollow or shell-like object in a 3D volume. It
    uses the grey-weighted distance transform seeded from the outer faces of the image
    domain (excluding any seeds that fall inside a mask). The result is refined using
    smoothed gradient fields derived from the distance transform.

    Parameters
    ----------
    image : np.ndarray
        3D input image array to be filled. Should be of shape corresponding to the given
        `axis_order`.
    sigma_g : float
        Scale of the Gaussian used to compute the image gradients for the structure
        tensor.
    sigma_fill : float
        Scale over which the gradients are integrated, used in the truncated inverse
        vector gradient reconstruction.
    mask : np.ndarray, optional
        Boolean array indicating the structure (inside = True). If None, the mask is
        inferred from the image. In that case, the outermost faces are removed to allow
        for proper seeding.

    Returns
    -------
    image_filled : np.ndarray
        The filled image obtained by gradient-based propagation of the grey-weighted
        distance.

    Raises
    ------
    ValueError
        If the input image is not 3D.
    """
    from .gwdt import grey_weighted_distance_transform

    if image.ndim != 3:
        raise ValueError("Input image must be 3D.")

    if mask is not None:
        mask = mask > 0  # Ensure boolean
    else:
        mask = np.ones_like(image, dtype=bool)
        # Remove faces from the mask to allow for seeds
        mask[0, :, :] = False
        mask[-1, :, :] = False
        mask[:, 0, :] = False
        mask[:, -1, :] = False
        mask[:, :, 0] = False
        mask[:, :, -1] = False

    # Zero out voxels outside the mask
    image_to_fill = np.copy(image)
    image_to_fill[~mask] = 0

    # Create seed array, where 0 indicates seed locations
    seed = np.ones_like(image, dtype=np.int8)

    # Set outermost faces as seeds (value 0)
    seed[0, :, :] = 0
    seed[-1, :, :] = 0
    seed[:, 0, :] = 0
    seed[:, -1, :] = 0
    seed[:, :, 0] = 0
    seed[:, :, -1] = 0

    # Remove seeds that are inside the mask
    seed[mask] = 1

    if np.all(seed):  # No seed locations remain
        warnings.warn(
            "All face-seeds are inside the mask. Randomly selecting a seed voxel "
            "outside the mask.",
            stacklevel=2,
        )
        rng = np.random.default_rng()
        candidates = np.column_stack(np.nonzero(~mask))
        if candidates.size == 0:
            raise ValueError("No available seed voxels outside the mask.")
        idx = tuple(candidates[rng.integers(len(candidates))])
        seed[idx] = 0

    # Compute grey-weighted distance
    grey_dist = grey_weighted_distance_transform(
        seed, image_to_fill, mode="fastmarching"
    )
    del image_to_fill  # Free memory

    # Compute smoothed gradients
    g1 = scipy.ndimage.gaussian_filter(grey_dist, sigma_g, order=[1, 0, 0])
    g2 = scipy.ndimage.gaussian_filter(grey_dist, sigma_g, order=[0, 1, 0])
    g3 = scipy.ndimage.gaussian_filter(grey_dist, sigma_g, order=[0, 0, 1])

    # Fill using truncated inverse gradient
    image_filled = _truncated_inverse_vector_gradient_nd(
        [g1, g2, g3], p=1, sigma=sigma_fill, dx=1
    )

    return image_filled


def _determine_principal_curvature_signs_with_image(
    image: np.ndarray,
    mask: np.ndarray,
    eigenvectors: list[np.ndarray],
    voxel_spacing: float,
    scale: float,
    axis_order: str = "zyx",
    sigma_g: float = 1,
):
    r"""
    Estimate the signs of the principal curvatures using the image intensity structure.

    This function computes the Hessian matrix of a smoothed and filled input image,
    then projects the Hessian onto two principal tangent directions provided by
    `eigenvectors`. The signs of the second directional derivatives (Hessian
    projections) are used to infer the signs of the principal curvatures.

    Parameters
    ----------
    image : np.ndarray
        Raw 3D image volume used for curvature sign estimation.
    mask : np.ndarray
        Binary mask defining the foreground region of interest.
    eigenvectors : list of np.ndarray
        Two orthogonal tangent direction vectors, each with shape (..., 3).
    voxel_spacing : float
        Isotropic voxel spacing in physical units (e.g., microns).
    scale : float
        Spatial scale (in physical units) at which curvature is measured.
    axis_order : str, optional
        Order of spatial axes in the input arrays. Default is 'zyx'.
    sigma_g : float, optional
        Gaussian smoothing scale for derivative computation (in pixel units). Default
        is 1.

    Returns
    -------
    sign1 : np.ndarray
        Sign (+1, -1, or 0) of curvature along the first principal tangent direction.
    sign2 : np.ndarray
        Sign (+1, -1, or 0) of curvature along the second principal tangent direction.

    Notes
    -----
    - The principal curvature sign is determined from the **negative** of the Hessian
    projection onto each tangent direction.

    - Geometrically:

    - **Concave region** (bowl-like):
        - Tangent direction has **positive** second derivative (Hessian projection).
        - Principal curvature is **negative** (surface curves inward).

        Visual:
        ```
        \          /
         \        /
          \      /
             .->      (. = observer, -> = tangent direction)
        ```

    - **Convex region** (dome-like):
        - Tangent direction has **negative** second derivative.
        - Principal curvature is **positive** (surface curves outward).

        Visual:
        ```
             .->
          /     \
         /       \
        /         \
        ```

    - Summary:
    - Positive Hessian projection → Negative principal curvature (concave)
    - Negative Hessian projection → Positive principal curvature (convex)
    - Zero projection → Flat surface
    """
    print("Start image filling...")
    scale /= voxel_spacing
    sigma_fill = np.sqrt(scale**2 - sigma_g**2)

    image_filled = _fill_grey_shell(image, sigma_g, sigma_fill, mask=mask)
    print("Finished image filling.")

    hessian = compute_hessian_3d(
        image_filled, scale, axis_order=axis_order, return_full=True
    )

    del image_filled  # Free up memory

    # Compute quadratic forms for both vector fields
    # v[0].T H v[0]
    result1 = np.einsum(
        "...i,...ij,...j->...", eigenvectors[0], hessian, eigenvectors[0]
    )
    # v[1].T H v[1]
    result2 = np.einsum(
        "...i,...ij,...j->...", eigenvectors[1], hessian, eigenvectors[1]
    )

    del hessian

    # Section 5.1.4: "Structure from Motion in nD Image Analysis" by B. Rieger (2004)
    # Explanation of the relationship between the Hessian matrix and the curvature sign:
    #
    # Concave region:
    #    - In a concave region, the surface curves inward (like the inside of a bowl).
    #    - As you move along the tangent (->) direction, the surface curves **upward** (towards the observer (.)).
    #    - This results in a **positive** second derivative (Hessian), as the surface bends towards the observer.
    #    - The principal curvature, however, is **negative** because it's concave (inward-curving).
    #
    #   \          /
    #    \        /
    #     \      /
    #        .->        (. = observer, -> = tangent direction)
    #
    # Convex region:
    #    - In a convex region, the surface curves outward (like the outside of a sphere).
    #    - As you move along the tangent direction, the surface curves **downward** (away from the observer).
    #    - This results in a **negative** second derivative (Hessian), as the surface bends away from the observer.
    #    - The principal curvature is **positive** because it's convex (outward-curving).
    #
    #         .->       (. = observer, -> = tangent direction)
    #      /     \
    #     /       \
    #    /         \
    #
    # Conclusion:
    #    - The principal curvature sign is determined by the **negative** of the Hessian along the tangent direction:
    #      - **Positive Hessian** = Negative principal curvature (concave, inward-curving).
    #      - **Negative Hessian** = Positive principal curvature (convex, outward-curving).
    #    - Zero = Flat surface (zero second derivative).

    sign1 = np.sign(-result1).astype(
        np.int8
    )  # Sign for curvature along the first principal direction
    sign2 = np.sign(-result2).astype(
        np.int8
    )  # Sign for curvature along the second principal direction

    return sign1, sign2


def _determine_principal_curvature_signs_with_dapi(
    dapi: np.ndarray,
    eigenvectors: list[np.ndarray],
    voxel_spacing: float,
    scale: float,
    axis_order: str = "zyx",
):
    r"""
    Estimate principal curvature signs using the Hessian of a DAPI image.

    This function computes the Hessian matrix of the DAPI channel and projects it onto
    the two principal tangent directions (eigenvectors). Based on the sign of the second
    directional derivatives, the signs of the principal curvatures are inferred. This approach
    distinguishes between convex and concave curvature using local intensity curvature.

    Parameters
    ----------
    dapi : np.ndarray
        3D DAPI image volume used to estimate curvature direction signs.
    eigenvectors : list of np.ndarray
        Two arrays representing the principal tangent directions (orthogonal to the surface normal).
        Each array has shape (..., 3), where the last dimension represents the vector components.
    voxel_spacing : float
        Voxel spacing in physical units (e.g., microns); assumed isotropic.
    scale : float
        Spatial scale (in physical units) at which curvature is measured.
    axis_order : str, optional
        Order of axes in the input image (default is 'zyx').

    Returns
    -------
    sign1 : np.ndarray
        Array of signs (+1, -1, or 0) for the first principal curvature direction.
    sign2 : np.ndarray
        Array of signs (+1, -1, or 0) for the second principal curvature direction.

    Notes
    -----
    This method uses the **negative** of the second directional derivatives from the
    Hessian to infer curvature sign. This is based on the relationship between intensity
    curvature and surface curvature as described in Section 5.1.4 of *Structure from Motion in nD Image Analysis*
    by B. Rieger (2004):

    - **Concave regions** (e.g., bowl-shaped surfaces):
        - The surface bends *toward* the observer.
        - Tangential intensity increases (positive second derivative).
        - Resulting principal curvature is **negative**.

          \          /
           \        /
            \      /
               .->      (. = observer, -> = tangent direction)

    - **Convex regions** (e.g., dome-shaped surfaces):
        - The surface bends *away* from the observer.
        - Tangential intensity decreases (negative second derivative).
        - Resulting principal curvature is **positive**.

               .->
             /     \
            /       \
           /         \

    Therefore, the curvature sign is inferred as:
    - `+1` → convex (negative Hessian projection)
    - `-1` → concave (positive Hessian projection)
    - `0`  → flat (zero second derivative)
    """
    scale /= voxel_spacing
    hessian = compute_hessian_3d(dapi, scale, axis_order=axis_order, return_full=True)

    # Compute quadratic forms for both vector fields
    # v[0].T H v[0]
    result1 = np.einsum(
        "...i,...ij,...j->...", eigenvectors[0], hessian, eigenvectors[0]
    )
    # v[1].T H v[1]
    result2 = np.einsum(
        "...i,...ij,...j->...", eigenvectors[1], hessian, eigenvectors[1]
    )

    del hessian

    # Section 5.1.4 from "Structure from Motion in nD Image Analysis" by B. Rieger (2004)
    # Explanation of the relationship between the Hessian matrix and the curvature sign:
    #
    # Concave region:
    #    - In a concave region, the surface curves inward (like the inside of a bowl).
    #    - As you move along the tangent (->) direction, the surface curves **upward** (towards the observer (.)).
    #    - This results in a **positive** second derivative (Hessian), as the surface bends towards the observer.
    #    - The principal curvature, however, is **negative** because it's concave (inward-curving).
    #
    #   \          /
    #    \        /
    #     \      /
    #        .->        (. = observer, -> = tangent direction)
    #
    # Convex region:
    #    - In a convex region, the surface curves outward (like the outside of a sphere).
    #    - As you move along the tangent direction, the surface curves **downward** (away from the observer).
    #    - This results in a **negative** second derivative (Hessian), as the surface bends away from the observer.
    #    - The principal curvature is **positive** because it's convex (outward-curving).
    #
    #         .->       (. = observer, -> = tangent direction)
    #      /     \
    #     /       \
    #    /         \
    #
    # Conclusion:
    #    - The principal curvature sign is determined by the **negative** of the Hessian along the tangent direction:
    #      - **Positive Hessian** = Negative principal curvature (concave, inward-curving).
    #      - **Negative Hessian** = Positive principal curvature (convex, outward-curving).
    #    - Zero = Flat surface (zero second derivative).

    sign1 = np.sign(-result1).astype(
        np.int8
    )  # Sign for curvature along the first principal direction
    sign2 = np.sign(-result2).astype(
        np.int8
    )  # Sign for curvature along the second principal direction

    return sign1, sign2


def compute_principal_curvatures(
    image: np.ndarray,
    mask: np.ndarray,
    voxel_spacing: float,
    scale: float,
    axis_order: str = "zyx",
    sign_estimation_mode: str | None = None,
    dapi: np.ndarray | None = None,
    sigma_g: float = 1,
    sigma_k: float = 1,
    nan_invalid_pc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the principal curvatures of a grayscale image.

    This function estimates the two principal curvatures of a surface represented in a
    3D image volume at a specified spatial scale by computing the gradient structure
    tensor and applying curvature measures.

    The process involves computing the 3D structure tensor from the input image,
    extracting eigenvectors that represent surface normals and tangent directions, and
    then calculating the principal curvatures using Knutsson's method. Optionally,
    curvature signs can be estimated using either the input image or a DAPI channel
    through Hessian analysis.

    Parameters
    ----------
    image : np.ndarray
        3D input image volume, e.g., from microscopy.
    mask : np.ndarray
        Binary mask defining the region of interest within `image`.
    voxel_spacing : float
        Voxel spacing (in physical units, e.g., microns); assumed isotropic.
    scale : float
        Desired spatial scale at which the curvature is computed (in physical units).
    axis_order : str, optional
        Order of axes in `image`. Must be a permutation of 'xyz'. Default is 'zyx'.
    sign_estimation_mode : {'dapi', 'image'}, optional
        If set, estimate curvature sign using either DAPI ('dapi') or the input image
        ('image').
    dapi : np.ndarray, optional
        3D DAPI image required if `sign_estimation_mode='dapi'`.
    sigma_g : float, optional
        Gaussian smoothing scale for derivative computation (in pixel units). Default is
        1.
    sigma_k : float, optional
        Scale for gradient computation smoothing (in pixel units). Default is 1.
    nan_invalid_pc : bool, optional
        If True (default), principal curvature values outside the valid range defined by
        `max_principal_curvature_limit(scale)` are set to NaN instead of being clipped
        or zeroed. This helps to clearly mark invalid curvature values in the output.

    Returns
    -------
    k1 : np.ndarray
        First principal curvature (in inverse squared physical units, e.g., µm^-2), zero
        outside the mask and NaN where invalid if `nan_invalid_pc` is True.
    k2 : np.ndarray
        Second principal curvature (in inverse squared physical units, e.g., µm^-2), zero
        outside the mask and NaN where invalid if `nan_invalid_pc` is True.

    Raises
    ------
    ValueError
        If `axis_order` is not a permutation of 'xyz'.
        If `sign_estimation_mode` is not one of {None, 'dapi', 'image'}.
        If `sign_estimation_mode='dapi'` and `dapi` is not provided.

    Notes
    -----
    - The total effective tensor smoothing scale (`sigma_t`) is computed automatically
      based on the desired `scale` and the provided `sigma_g` and `sigma_k` using variance
      addition in quadrature:

        scale^2 = sigma_t^2 + sigma_g^2 + sigma_k^2

    - `scale` and `voxel_spacing` are in physical units (e.g., microns), whereas
      `sigma_g` and `sigma_k` are in pixel units.
    - This function assumes isotropic voxels for simplicity.

    References
    ----------
    - Knutsson, H. (1989). "Representing Local Structure Using Tensors." *Scandinavian Conference on Image Analysis*.
    - Rieger, B. (2004). "Structure from Motion in nD Image Analysis." PhD Thesis. Chapter 5
    - DIPLib's curvature implementation: https://github.com/DIPlib/diplib
    """
    if set(axis_order) != {"x", "y", "z"} or len(axis_order) != 3:
        raise ValueError("axis_order must be a permutation of 'xyz'")

    if sign_estimation_mode not in {None, "dapi", "image"}:
        raise ValueError("sign_estimation_mode must be one of {None, 'dapi', 'image'}")

    if sign_estimation_mode == "dapi" and dapi is None:
        raise ValueError("dapi must be provided if sign_estimation_mode='dapi'")

    # sigma_t is in pixel space
    sigma_t = compute_sigma_t(scale / voxel_spacing, sigma_g, sigma_k)

    Axx, Ayy, Azz, Axy, Ayz, Axz = compute_structure_tensor_3d(
        image, sigma_g, sigma_t, axis_order=axis_order
    )

    eigvecs = eigh_3d_batch(Axx, Ayy, Azz, Axy, Ayz, Axz, sort_descending=True)[1]
    del Axx, Ayy, Azz, Axy, Ayz, Axz

    # Largest eigenvector (normal direction)
    n = eigvecs[..., :, 0]
    v = [eigvecs[..., :, 1], eigvecs[..., :, 2]]  # Tangent vectors
    del eigvecs

    # Knutsson surface measure terms
    n0_sq = n[..., 0] ** 2
    n1_sq = n[..., 1] ** 2
    t = n0_sq - n1_sq

    out = _dot_gradient_square(v, t, sigma_k=sigma_k, axis_order=axis_order)

    t = (1 / np.sqrt(3)) * (2 * (n[..., 2] ** 2) - n0_sq - n1_sq)
    del n0_sq, n1_sq
    out = _add_elements(
        out, _dot_gradient_square(v, t, sigma_k=sigma_k, axis_order=axis_order)
    )

    t = 2 * n[..., 0] * n[..., 1]
    out = _add_elements(
        out, _dot_gradient_square(v, t, sigma_k=sigma_k, axis_order=axis_order)
    )

    t = 2 * n[..., 0] * n[..., 2]
    out = _add_elements(
        out, _dot_gradient_square(v, t, sigma_k=sigma_k, axis_order=axis_order)
    )

    t = 2 * n[..., 1] * n[..., 2]
    out = _add_elements(
        out, _dot_gradient_square(v, t, sigma_k=sigma_k, axis_order=axis_order)
    )

    del t, n

    # Final scaling of output curvature components
    k1, k2 = [np.sqrt(a) * 0.5 for a in out]

    # Sign estimation
    if sign_estimation_mode is not None:
        if sign_estimation_mode == "dapi":
            sign1, sign2 = _determine_principal_curvature_signs_with_dapi(
                dapi, v, voxel_spacing, scale, axis_order=axis_order
            )
        elif sign_estimation_mode == "image":
            sign1, sign2 = _determine_principal_curvature_signs_with_image(
                image, mask, v, voxel_spacing, sigma_g, scale, axis_order=axis_order
            )

        k1 *= sign1
        k2 *= sign2

    del v

    # Convert curvature units to physical units (µm^-1 if voxel_spacing in µm)
    k1 = k1 / voxel_spacing
    k2 = k2 / voxel_spacing

    # Apply mask
    k1[~mask] = 0
    k2[~mask] = 0

    if nan_invalid_pc:  # Set principal component values that are invalid to NaN
        max_pc = max_principal_curvature_limit(scale)
        invalid_mask_k1 = (k1 < -max_pc) | (k1 > max_pc)
        invalid_mask_k2 = (k2 < -max_pc) | (k2 > max_pc)
        k1[invalid_mask_k1] = np.nan
        k2[invalid_mask_k2] = np.nan

    return k1, k2
