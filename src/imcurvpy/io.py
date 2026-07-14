"""Functions to write and read data."""

import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from tifffile import TiffFile, imwrite


def _xy_size(tags, key: str) -> float:
    """
    Calculate physical voxel spacing from TIFF resolution tags.

    Parameters
    ----------
    tags : dict-like
        TIFF image tags dictionary containing metadata.
    key : str
        The key for the resolution tag to read, typically 'XResolution' or
        'YResolution'.

    Returns
    -------
    float
        The physical spacing (size of a voxel) calculated as units per pixel.
        Returns 1.0 if the tag is missing or cannot be read.
    """
    if key in tags:
        num_pixels, units = tags[key].value
        return units / num_pixels
    return 1.0  # fallback spacing if tag missing


def read_tiff_select_indices(
    path: str,
    z_idx: int | Sequence[int] | None = None,
    channel_idx: int | Sequence[int] | None = None,
    time_idx: int | Sequence[int] | None = None,
) -> tuple[np.ndarray, tuple[float, ...], str]:
    """
    Read selected slices from a TIFF file by specifying indices along Z, C, and T.

    Parameters
    ----------
    path : str
        Path to the TIFF file.
    z_idx : int or sequence of int or None, optional
        Index or indices for the Z axis. If None, all Z slices are selected.
    channel_idx : int or sequence of int or None, optional
        Index or indices for the channel axis. If None, all channels are selected.
    time_idx : int or sequence of int or None, optional
        Index or indices for the time axis. If None, all time points are selected.

    Returns
    -------
    image : np.ndarray
        Stacked image array of selected slices. Shape corresponds to selected page
        indices plus spatial dimensions (Y, X). For example, shape might be (T, Z, Y, X)
        if multiple time and Z slices are selected.
    spacing : tuple of float
        Physical voxel spacing for spatial axes in the order they appear in the output.
    out_axes : str
        Axes string describing the order and presence of dimensions in the returned
        image, e.g., 'TZYX', 'ZYX', 'CYX'.

    Notes
    -----
    The TIFF axes string must contain spatial axes ('Y', 'X') and zero or more page
    axes among 'T', 'Z', and 'C' that index TIFF pages.

    Selected page indices for T, Z, and C axes are converted to flat indices to access
    TIFF pages. Selected pages are loaded, stacked, and reshaped to restore the
    multidimensional shape.

    The output axes string includes only page axes where multiple indices were
    selected, followed by spatial axes.

    Examples
    --------
    >>> # Read a single Z slice
    >>> image, spacing, axes = read_tiff_select_indices(path, z_idx=5)
    >>> image.shape
    (Y, X)
    >>> axes
    'YX'

    >>> # Read multiple channels and time points
    >>> image, spacing, axes = read_tiff_select_indices(
    ...     path, channel_idx=[0, 2], time_idx=[0, 1]
    ... )
    >>> image.shape
    (T, C, Y, X)
    >>> axes
    'TCYX'

    ASCII illustration of index conversion and stacking:

    Suppose TIFF axes = 'TZCYX' with shape (10, 5, 2, 100, 100).

        page_axes = ['T', 'Z', 'C']
        page_shape = [10, 5, 2]

    We select:

        T indices: [0, 1]
        Z indices: [2, 3]
        C indices: [0]

    The cartesian product of indices is:

        (0, 2, 0), (0, 3, 0), (1, 2, 0), (1, 3, 0)

    These tuples are converted to flat page indices using strides:

        strides = [5*2, 2, 1] = [10, 2, 1]
        flat indices = [0*10 + 2*2 + 0*1, 0*10 + 3*2 + 0*1,
                        1*10 + 2*2 + 0*1, 1*10 + 3*2 + 0*1]
                     = [4, 6, 14, 16]

    The corresponding pages are loaded and stacked into an array of shape (4, 100, 100).

    Since only the T and Z axes have more than one selected index, the final image is
    reshaped to (2, 2, 100, 100), corresponding to (T, Z, Y, X). The C axis is collapsed
    because only a single index was selected.
    """

    def get_sel(axis: str, sel_idx: int | Sequence[int] | None) -> list[int]:
        """Generate a list of valid indices for a given axis."""
        if axis not in axes:
            # If axis is not in TIFF axes, default to 0 (singleton)
            return [0]
        axis_size = shape[axes.index(axis)]
        if sel_idx is None:
            return list(range(axis_size))
        if isinstance(sel_idx, int):
            if sel_idx >= axis_size or sel_idx < 0:
                raise IndexError(
                    f"{axis}_idx={sel_idx} out of bounds for axis size {axis_size}"
                )
            return [sel_idx]
        for idx in sel_idx:
            if idx >= axis_size or idx < 0:
                raise IndexError(
                    f"{axis}_idx={idx} out of bounds for axis size {axis_size}"
                )
        return list(sel_idx)

    with TiffFile(path) as tiff:
        series = tiff.series[0]
        shape = series.shape
        axes = series.axes

        # Identify page axes (axes indexing TIFF pages)
        page_axes = [ax for ax in axes if ax in ("T", "Z", "C")]
        page_shape = [shape[axes.index(ax)] for ax in page_axes]

        # Compute strides for converting multidim indices to flat page index
        strides = []
        acc = 1
        for size in reversed(page_shape):
            strides.insert(0, acc)
            acc *= size

        sel_indices = {
            ax: get_sel(ax, {"T": time_idx, "Z": z_idx, "C": channel_idx}.get(ax))
            for ax in page_axes
        }

        # Generate all combinations of selected page indices
        all_combinations = list(
            itertools.product(*(sel_indices[ax] for ax in page_axes))
        )

        # Convert multi-dimensional indices to flat page indices
        page_indices = [
            sum(i * s for i, s in zip(idx_tuple, strides, strict=False))
            for idx_tuple in all_combinations
        ]

        # Load selected pages and stack into numpy array
        stack = [series.pages[i].asarray() for i in page_indices]
        image = np.stack(stack)

        # Reshape stack to multidimensional array:
        # Only include page axes with more than one index
        page_dims = [
            len(sel_indices[ax]) for ax in page_axes if len(sel_indices[ax]) > 1
        ]
        # Append spatial dims (Y, X)
        image = image.reshape(*page_dims, *stack[0].shape)

        # Build output axes string:
        out_axes = "".join(ax for ax in page_axes if len(sel_indices[ax]) > 1)
        spatial_axes = [ax for ax in axes if ax not in page_axes]
        out_axes += "".join(spatial_axes)

        # Default physical spacing for spatial axes
        axis_spacing = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Try ImageJ metadata for Z spacing
        ij_metadata = tiff.imagej_metadata
        if ij_metadata and "spacing" in ij_metadata:
            axis_spacing["z"] = ij_metadata["spacing"]

        # Read XY spacing from TIFF tags
        tags = tiff.pages[0].tags
        axis_spacing["y"] = _xy_size(tags, "YResolution")
        axis_spacing["x"] = _xy_size(tags, "XResolution")

        # Extract spacing for spatial axes in output order
        spacing = tuple(
            axis_spacing[ax.lower()] for ax in out_axes if ax.lower() in axis_spacing
        )

    return image, spacing, out_axes


def read_tiff_with_spacing(
    path: str,
) -> tuple[np.ndarray, tuple[float, ...], str | None]:
    """
    Read a TIFF file with tifffile and extract image data along with physical spacings.

    Parameters
    ----------
    path : str
        Path to the TIFF file.

    Returns
    -------
    image : np.ndarray
        The full image data array, shape depends on the TIFF (e.g., ZCYX).
    spacing : tuple of float
        Physical voxel spacing for spatial axes (Z, Y, X) in the order they appear in
        axes.
        If spacing info is missing, defaults to 1.0 for each spatial axis.
    axes : str or None
        String describing the axes order (e.g., "ZCYX"). None if not available.
    """
    with TiffFile(path) as tiff:
        series = tiff.series[0]
        image = series.asarray()
        axes = getattr(
            series, "axes", None
        )  # e.g., "ZCYX", fallback to None if attribute not present

        # Default spacing dictionary with fallback to 1.0
        axis_spacing = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Try to get spacing from ImageJ metadata (common in microscopy TIFFs)
        ij_meta = tiff.imagej_metadata
        if ij_meta is not None and "spacing" in ij_meta:
            axis_spacing["z"] = ij_meta["spacing"]

        # Get X and Y resolution from TIFF tags (if available)
        tags = tiff.pages[0].tags

        axis_spacing["y"] = _xy_size(tags, "YResolution")
        axis_spacing["x"] = _xy_size(tags, "XResolution")

        # Extract spacing in the order of spatial axes as they appear in axes string
        if axes is None:
            spacing = None
        else:
            valid_axes = {"z", "y", "x"}
            spacing = tuple(
                axis_spacing[ax.lower()] for ax in axes if ax.lower() in valid_axes
            )

    return image, spacing, axes


def read_tiff_shape_and_axes(
    path: str,
) -> tuple[tuple[int, ...], str | None]:
    """
    Read a TIFF file using tifffile and extract the image shape and axes metadata.

    Parameters
    ----------
    path : str
        Path to the TIFF file.

    Returns
    -------
    shape : tuple of int
        The shape of the image array, as defined by the TIFF series
        (e.g., (10, 3, 512, 512)).
    axes : str or None
        A string describing the order of axes (e.g., "ZCYX"). Returns None if axes
        metadata is not available.
    """
    with TiffFile(path) as tiff:
        series = tiff.series[0]
        shape = series.shape
        axes = getattr(
            series, "axes", None
        )  # e.g., "ZCYX", fallback to None if attribute not present

    return shape, axes


def write_tiff_with_spacing(
    path: str,
    image: np.ndarray,
    spacing: tuple[float, ...],
    axes: str,
    unit: str = "um",
    fps: float | None = None,
    labels: list[str] | None = None,
    dtype: npt.DTypeLike | None = None,
) -> None:
    """
    Write an ImageJ-compatible TIFF file with voxel spacing and metadata.

    Parameters
    ----------
    path : str
        Output file path.
    image : np.ndarray
        Image data to save (e.g., shape TZYX).
    spacing : tuple of float
        Voxel spacing for spatial axes (Z, Y, X), in the order they appear in `axes`.
    axes : str
        Axis order string, e.g., 'TZYX'. Allowed: T, Z, C, Y, X.
    unit : str
        Physical unit for spacing (e.g., 'um'). Default is 'um'.
    fps : float, optional
        Frames per second for time axis.
    labels : list of str, optional
        Optional string labels for slices/frames.
    dtype : numpy.typing.DTypeLike, optional
        The desired data type of the output. Default is None, so tries the infer the
        datatype of the image.
    """
    axes = axes.upper()
    valid_axes = set("TZCYX")
    if not set(axes).issubset(valid_axes):
        raise ValueError(f"Invalid axes '{axes}'. Allowed axes: {sorted(valid_axes)}")
    if len(set(axes)) != len(axes):
        raise ValueError(f"Duplicate axes found in '{axes}'.")

    if len(image.shape) != len(axes):
        raise ValueError(
            f"Image shape {image.shape} does not match number of axes '{axes}'."
        )

    # Match spacing values to spatial axes present in `axes`
    spatial_axes = "ZYX"
    spatial_found = [ax for ax in axes if ax in spatial_axes]
    if len(spacing) != len(spatial_found):
        raise ValueError(
            f"Expected spacing for {spatial_found}, got {len(spacing)} values."
        )

    spacing_map = dict(zip(spatial_found, spacing, strict=False))
    resolution = (
        1.0 / spacing_map.get("X", 1.0),
        1.0 / spacing_map.get("Y", 1.0),
    )

    metadata = {
        "spacing": spacing_map.get("Z", 1.0),
        "unit": unit,
        "axes": axes,
    }
    if fps:
        metadata.update({"fps": fps, "finterval": 1.0 / fps})
    if labels:
        metadata["Labels"] = labels

    imwrite(
        path,
        image,
        imagej=True,
        resolution=resolution,
        metadata=metadata,
        dtype=dtype,
    )
