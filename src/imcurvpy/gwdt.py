try:
    import skfmm
except ImportError as e:
    raise ImportError(
        'Install imcurvpy with the [gwdt] extra to use the grey-weighted distance transform feature: pip install "imcurvpy[gwdt]"'
    ) from e

import heapq

import numpy as np
from scipy import ndimage


def chamfer_metric_algorithm(
    weights: np.ndarray,
    gdt: np.ndarray,
    pdt: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Implementation of the Chamfer Metric Algorithm using priority queue.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight values for each pixel.
    gdt : numpy.ndarray
        Grey-weighted distance transform (modified in-place).
    pdt : numpy.ndarray or None, optional
        Physical distance transform (modified in-place).
        Default is None.
    mask : numpy.ndarray or None, optional
        Binary mask - regions with False are excluded.
        Default is None.

    Returns
    -------
    numpy.ndarray
        The modified gdt array.
    """
    # Create neighbor offsets for a city-block neighborhood
    ndim = gdt.shape

    # Create a priority queue
    queue: list[tuple[float, tuple]] = []

    # Initialize priority queue with seed points
    seed_points = np.where(gdt == 0)
    for idx in zip(*seed_points, strict=False):
        heapq.heappush(queue, (0, idx))

    # Create a visited flag array
    visited = np.zeros_like(gdt, dtype=bool)

    # Define neighborhood connectivity (city-block/taxicab)
    offsets: list[tuple[tuple[int, ...], float]] = []
    for dim in range(len(ndim)):
        for direction in [-1, 1]:
            offset = [0] * len(ndim)
            offset[dim] = direction
            offsets.append((tuple(offset), 1.0))  # (offset, distance)

    # Process the queue
    while queue:
        distance, idx = heapq.heappop(queue)

        # Skip if already processed
        if visited[idx]:
            continue

        # Mark as visited
        visited[idx] = True

        # Check all neighbors
        for offset, offset_dist in offsets:
            # Compute neighbor index
            neigh_idx = tuple(i + di for i, di in zip(idx, offset, strict=False))

            # Check if neighbor is in bounds
            if not all(0 <= ni < si for ni, si in zip(neigh_idx, ndim, strict=False)):
                continue

            # Skip if neighbor is masked or already visited
            if (mask is not None and not mask[neigh_idx]) or visited[neigh_idx]:
                continue

            # Calculate new distance value
            weight = weights[neigh_idx]
            value = gdt[idx] + offset_dist * weight

            # Update if the new distance is smaller
            if value < gdt[neigh_idx]:
                gdt[neigh_idx] = value
                if pdt is not None:
                    pdt[neigh_idx] = pdt[idx] + offset_dist
                heapq.heappush(queue, (value, neigh_idx))

    # Set distances for masked regions to 0
    if mask is not None:
        gdt[~mask] = 0
        if pdt is not None:
            pdt[~mask] = 0

    return gdt


def grey_weighted_distance_transform(
    bin_img: np.ndarray,
    grey: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    mode: str = "fastmarching",
    output_distance: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Grey Weighted Distance Transform implementation using scikit-fmm or chamfer transform.

    Parameters
    ----------
    bin_img : numpy.ndarray
        Binary image marking the seeds (0) and target regions (1).
    grey : numpy.ndarray or None
        Image with weights/speed values, must be non-negative.
        If None, uniform weights are used.
    mask : numpy.ndarray or None, optional
        Binary mask - regions with 0 are excluded from computation.
        Default is None.
    mode : str, optional
        Algorithm to use: 'fastmarching' for Fast Marching Method,
        'chamfer' for Chamfer distance.
        Default is 'fastmarching'.
    output_distance : bool, optional
        If True and mode='chamfer', also output the physical distance (not weighted).
        Default is False.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        If output_distance is False: Distance transform image.
        If output_distance is True and mode='chamfer': (weighted_distance, physical_distance).

    Notes
    -----
    Based on:
    https://github.com/DIPlib/diplib/blob/21bb1ac5cce518089ba5a97334bfe340897d1da2/src/distance/gdt.cpp#L317

    Examples
    --------
    >>> import numpy as np
    >>> # Create a binary image with a seed point in the center
    >>> bin_img = np.ones((100, 100), dtype=np.uint8)
    >>> bin_img[45:55, 45:55] = 0
    >>> # Compute distance transform with fast marching
    >>> distance = grey_weighted_distance_transform(bin_img, None)
    """
    if bin_img is None:
        raise ValueError("Binary image must be provided")

    # Basic validation
    if bin_img.ndim < 2:
        raise ValueError("Distance transform requires at least 2 dimensions")

    # Check grey image if provided
    if grey is not None:
        if grey.shape != bin_img.shape:
            raise ValueError("Grey image and binary image must have the same shape")

        # Check for non-negative values
        if np.min(grey) < 0:
            raise ValueError("All input values must be non-negative")

    # Check mask if provided
    if mask is not None:
        # Make sure mask has the same shape (or can be broadcast)
        if mask.shape != bin_img.shape:
            if all(
                m == 1 or m == s
                for m, s in zip(mask.shape, bin_img.shape, strict=False)
            ):
                # Expand singleton dimensions
                mask = np.broadcast_to(mask, bin_img.shape)
            else:
                raise ValueError("Mask shape doesn't match binary image shape")

    # Initialize the distance transform
    gdt = np.zeros_like(bin_img, dtype=np.float32)
    gdt[bin_img > 0] = np.inf  # Set target regions to infinity

    # Create weights array
    if grey is not None:
        weights = grey.astype(np.float32)
    else:
        weights = np.ones_like(bin_img, dtype=np.float32)

    # Apply mask if provided
    if mask is not None:
        mask_array = mask.astype(bool)
    else:
        mask_array = np.ones_like(bin_img, dtype=bool)

    # Choose algorithm
    print("starting")
    if mode.lower() == "fastmarching":
        # Use Fast Marching Method
        # Add small epsilon to avoid division by zero
        # speed = 1.0 / (weights + 1e-4)
        speed = 1.0 / (weights + np.finfo(np.float32).eps)
        speed[mask_array == 0] = 0

        distance = skfmm.travel_time(gdt, speed, narrow=0.0)
        distance[mask_array == 0] = 0

        return distance

    elif mode.lower() == "chamfer":
        # Use Chamfer distance transform
        # First compute standard chamfer distance
        # We'll use scipy's chamfer distance transform as a starting point
        cdt = ndimage.distance_transform_cdt(bin_img == 0, metric="taxicab")

        # If no weights and no output_distance, we're done
        if grey is None and not output_distance:
            cdt[~mask_array] = 0
            return cdt

        # Otherwise, we need to compute the weighted distance
        # For this, we'll implement a custom chamfer algorithm

        # Initialize distances
        gdt = np.ones_like(bin_img, dtype=np.float32) * np.inf
        gdt[bin_img == 0] = 0  # Zero at seed points

        if output_distance:
            pdt = np.ones_like(bin_img, dtype=np.float32) * np.inf
            pdt[bin_img == 0] = 0  # Zero at seed points
        else:
            pdt = None

        # Run the chamfer algorithm
        weighted_dist = chamfer_metric_algorithm(weights, gdt, pdt, mask_array)

        if output_distance:
            return weighted_dist, pdt
        else:
            return weighted_dist
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'fastmarching' or 'chamfer'")


def example_usage() -> dict[str, np.ndarray]:
    """Example of how to use the grey_weighted_distance_transform function.

    Returns
    -------
    dict
        Dictionary containing the results of different distance transform methods.
    """
    # Create sample data
    size = (100, 100)

    # Binary image with seed points (0) and target regions (1)
    bin_img = np.ones(size, dtype=np.uint8)
    bin_img[40:60, 40:60] = 0  # Create a square seed region

    # Grey value image with weights
    grey = np.ones(size, dtype=np.float32)
    # Create a gradient
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
    grey = grey + 2 * x + 3 * y  # Some arbitrary weighting

    # Optional mask (1 = include, 0 = exclude)
    mask = np.ones(size, dtype=np.uint8)
    mask[0:20, 0:20] = 0  # Exclude top-left corner

    # Compute the grey weighted distance transform using fast marching
    fm_result = grey_weighted_distance_transform(
        bin_img, grey, mask, mode="fastmarching"
    )

    # Compute the grey weighted distance transform using chamfer
    ch_result = grey_weighted_distance_transform(bin_img, grey, mask, mode="chamfer")

    # Compute with physical distance output
    ch_weighted, ch_physical = grey_weighted_distance_transform(
        bin_img, grey, mask, mode="chamfer", output_distance=True
    )

    # Visualization (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        axes[0, 0].imshow(bin_img, cmap="gray")
        axes[0, 0].set_title("Binary Image (Seeds)")

        axes[0, 1].imshow(grey, cmap="viridis")
        axes[0, 1].set_title("Grey Values (Weights)")

        axes[0, 2].imshow(mask, cmap="gray")
        axes[0, 2].set_title("Mask")

        axes[1, 0].imshow(fm_result, cmap="inferno")
        axes[1, 0].set_title("Fast Marching Distance")

        axes[1, 1].imshow(ch_result, cmap="inferno")
        axes[1, 1].set_title("Chamfer Weighted Distance")

        axes[1, 2].imshow(ch_physical, cmap="inferno")
        axes[1, 2].set_title("Chamfer Physical Distance")

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization")

    return {
        "fast_marching": fm_result,
        "chamfer_weighted": ch_result,
        "chamfer_physical": ch_physical,
    }


if __name__ == "__main__":
    example_usage()
