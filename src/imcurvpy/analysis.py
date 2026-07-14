import math
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy.stats import binned_statistic


def compute_binned_statistics(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    bins: int,
    x_range: tuple[float, float],
    statistics: str | Callable | Sequence[str | Callable] = "median",
    percentiles: float | Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute binned summary statistics for `y` over `x`.

    Parameters
    ----------
    x : array-like
        Independent variable (e.g., Gaussian curvature).
    y : array-like
        Dependent variable (e.g., intensity).
    bins : int
        Number of bins for the histogram.
    x_range : tuple[float, float]
        The (min, max) range over which to compute statistics.
    statistics : str, callable, or sequence of str/callable, default "median"
        The statistic(s) to compute.
    percentiles : float or sequence of floats, optional
        Percentile(s) to compute.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping statistic names to arrays of binned statistics.
        Includes a key "bin_centers" for the centers of each bin.
    """
    # --- Input validation ---
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} and {y.shape}"
        )

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(f"bins must be a positive integer, got {bins}")

    if (
        not isinstance(x_range, tuple)
        or len(x_range) != 2
        or not all(isinstance(v, (int, float)) for v in x_range)
        or x_range[0] >= x_range[1]
    ):
        raise ValueError(
            f"x_range must be a tuple of (min, max) with min < max, got {x_range}"
        )

    if percentiles is not None:
        percentiles_iter = (
            (percentiles,) if isinstance(percentiles, (float, int)) else percentiles
        )
        for p in percentiles_iter:
            if not (0 <= p <= 100):
                raise ValueError(f"Percentiles must be between 0 and 100, got {p}")
    else:
        percentiles_iter = ()

    # --- Helper for percentile computation ---
    def percentile_stat(arr: np.ndarray, p: float) -> float:
        if len(arr) == 0:
            return np.nan
        return np.nanpercentile(arr, p)

    # --- Ensure statistics is a tuple ---
    if isinstance(statistics, (str, Callable)):
        statistics = (statistics,)

    results: dict[str, np.ndarray] = {}

    # --- Compute bin edges once to get bin centers ---
    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    results["bin_centers"] = bin_centers

    # --- Compute each statistic ---
    for stat in statistics:
        if isinstance(stat, str):
            key = stat
            results[key] = binned_statistic(
                x, y, statistic=stat, bins=bins, range=x_range
            ).statistic
        elif callable(stat):
            key = getattr(stat, "__name__", "custom")
            results[key] = binned_statistic(
                x, y, statistic=stat, bins=bins, range=x_range
            ).statistic
        else:
            raise TypeError(f"Invalid statistic type: {type(stat)}")

    # --- Compute percentiles if requested ---
    for p in percentiles_iter:
        key = f"percentile_{int(p)}" if float(p).is_integer() else f"percentile_{p:.3e}"
        stat_func = partial(percentile_stat, p=p)
        results[key] = binned_statistic(
            x, y, statistic=stat_func, bins=bins, range=x_range
        ).statistic

    return results


def local_ball_stats(image, mask, radius, stat="mean", pad_mode="reflect"):
    """
    Compute local statistics in a spherical neighborhood around each voxel in a mask.

    Supports float radius. Uses weighted ball for mean.

    Parameters
    ----------
    image : np.ndarray
        3D image array (e.g., shape (Z, Y, X)).
    mask : np.ndarray
        3D boolean or binary mask (same shape as image).
    radius : int or float
        Radius of the spherical neighborhood (in voxels).
    stat : str, optional
        Which statistic to compute: "min", "mean", "max", or "median".
    pad_mode : str, optional
        How to handle edges. Passed to np.pad. Default is "reflect".

    Returns
    -------
    stats_map : np.ndarray
        3D array (same shape as image), containing the computed statistic
        at masked voxel positions (0 elsewhere).
    """
    if image.shape != mask.shape:
        raise ValueError("Image and mask must have the same shape.")
    if stat not in ("min", "mean", "max", "median"):
        raise ValueError("stat must be one of: 'min', 'mean', 'max', 'median'.")

    # Ceil radius to determine neighborhood size for indexing
    ceil_radius = math.ceil(radius)

    # Create spherical (ball) structuring element
    L = np.arange(-ceil_radius, ceil_radius + 1)
    X, Y, Z = np.meshgrid(L, L, L, indexing="ij")
    ball = X**2 + Y**2 + Z**2 <= radius**2  # still use float radius for precision

    dist = np.sqrt(X**2 + Y**2 + Z**2)

    # Binary ball (for non-mean stats)
    ball = dist <= radius

    # Weighted ball (for mean): weight ~ fraction of voxel inside sphere
    # linear ramp from 1 to 0 over ~1 voxel (radius + 0.5 accounts for voxel size)
    weights = np.clip(radius + 0.5 - dist, 0, 1)

    # Pad image
    padded_img = np.pad(image, ceil_radius, mode=pad_mode)

    stats_map = np.zeros_like(image, dtype=float)
    coords = np.argwhere(mask)

    for z, y, x in coords:
        zp, yp, xp = z + ceil_radius, y + ceil_radius, x + ceil_radius

        zmin, zmax = zp - ceil_radius, zp + ceil_radius + 1
        ymin, ymax = yp - ceil_radius, yp + ceil_radius + 1
        xmin, xmax = xp - ceil_radius, xp + ceil_radius + 1

        local_img = padded_img[zmin:zmax, ymin:ymax, xmin:xmax]

        # Adjust ball mask in case we're near the edges (should match local_img shape)
        bshape = local_img.shape

        if stat == "mean":
            local_w = weights[: bshape[0], : bshape[1], : bshape[2]]
            wsum = np.sum(local_w)

            if wsum > 0:
                val = np.sum(local_img * local_w) / wsum
            else:
                val = 0.0

        else:
            local_ball = ball[: bshape[0], : bshape[1], : bshape[2]]
            local_vals = local_img[local_ball]

            if stat == "min":
                val = np.min(local_vals)
            elif stat == "max":
                val = np.max(local_vals)
            else:  # median
                val = np.median(local_vals)

        stats_map[z, y, x] = val

    return stats_map
