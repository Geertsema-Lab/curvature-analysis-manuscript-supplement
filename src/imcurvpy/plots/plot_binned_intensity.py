from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib
import matplotlib.cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import truncnorm

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt


def _get_bin_edges_from_axis(ax: plt.Axes) -> np.ndarray:
    patches = ax.patches
    if not patches:
        raise ValueError("No histogram patches found on the axis")

    # Each patch corresponds to a bar; get left edges and widths
    left_edges = [patch.get_x() for patch in patches]
    widths = [patch.get_width() for patch in patches]

    # Bin edges are left edges plus the last right edge
    bin_edges = np.array([*left_edges, left_edges[-1] + widths[-1]])
    return bin_edges


def _clear_plot_data(ax):
    """
    Remove all plotted data from an Axes while preserving labels, titles, and limits.

    This removes lines, patches, collections, images, containers, and both axes-level
    and figure-level legends, ensuring that no legend handles or labels remain.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object from which to remove plotted data.

    Returns
    -------
    matplotlib.axes.Axes
        The cleared axes.
    """
    # Remove plotted artists
    for line in list(ax.lines):
        line.set_label(s=None)
        line.remove()
    for patch in list(ax.patches):
        patch.set_label(s=None)
        patch.remove()
    for coll in list(ax.collections):
        coll.set_label(s=None)
        coll.remove()
    for img in list(ax.images):
        img.set_label(s=None)
        img.remove()
    for container in list(ax.containers):
        try:
            container.set_label(s=None)
            container.remove()
        except ValueError:
            pass  # Already removed elsewhere

    # Remove axes legend
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    # Remove figure legends
    fig = ax.figure
    if fig is not None:
        for lg in list(fig.legends):
            lg.remove()

    return ax


def plot_binned_intensity_separate(
    curvature: npt.ArrayLike,
    intensity: npt.ArrayLike,
    curvature_ranges: Sequence[tuple[float, float]],
    *,
    stat: Literal[
        "count", "frequency", "probability", "percent", "density"
    ] = "density",
    bins: str | int | Sequence[float] = "auto",
    binrange: Sequence[float] | Sequence[Sequence[float, float]] | None = None,
    element: Literal["bars", "step", "poly"] = "bars",
    kde: bool = False,
    sns_style: str = "darkgrid",
    sns_context: str = "paper",
    figsize: tuple[int, int] | str = "auto",
    save_path: Path | str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot intensity histograms binned by curvature intervals.

    Parameters
    ----------
    curvature : ArrayLike
        Curvature values for each voxel or sample.
    intensity : ArrayLike
        Normalized intensity values in the same order as curvature.
    curvature_ranges : Sequence[tuple[float, float]]
        Curvature intervals to use for binning.
    stat : Literal["count", "frequency", "probability", "percent", "density"], optional
        Aggregate statistic to compute in each bin. Default is "density".
    bins : str, int, or sequence of floats, optional
        Binning strategy passed to seaborn.histplot (also accepted by
        numpy.histogram_bin_edges). Default is "auto".
    binrange : tuple or list of tuples, optional
        The lower and upper range of the bins. Defaults to the min and max of data.
    element : Literal["bars", "step", "poly"], optional
        Style of histogram bars. Default is "bars".
    kde: bool
        If True, compute a kernel density estimate to smooth the distribution and show
        on the plot as (one or more) line(s). Only relevant with univariate data.
    sns_style : str, optional
        Seaborn style to apply (e.g., "darkgrid", "whitegrid"). Default is "darkgrid".
    sns_context : str, optional
        Seaborn context for scaling (e.g., "paper", "talk", "poster"). Default is
        "paper".
    figsize : tuple[int, int] or "auto", optional
        Figure size in inches (width, height). If "auto", scales height by number of
        bins (1.5 inch per bin). Default is "auto".
    save_path : Path or str or None, optional
        Path to save the figure. If None, figure is not saved. Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        Axes for each curvature bin.
    """
    import numbers

    # Validate curvature and intensity are array-like and have same length
    curvature = np.asarray(curvature)
    intensity = np.asarray(intensity)
    if curvature.ndim != 1:
        raise ValueError("`curvature` must be a 1D array-like.")
    if intensity.ndim != 1:
        raise ValueError("`intensity` must be a 1D array-like.")
    if curvature.shape[0] != intensity.shape[0]:
        raise ValueError("`curvature` and `intensity` must have the same length.")

    # Validate curvature_ranges
    if not isinstance(curvature_ranges, (list, tuple)):
        raise TypeError(
            "`curvature_ranges` must be a list or tuple of (float, float) tuples."
        )
    for idx, interval in enumerate(curvature_ranges):
        if (
            not isinstance(interval, tuple)
            or len(interval) != 2
            or not all(isinstance(x, numbers.Real) for x in interval)
        ):
            raise TypeError(
                f"`curvature_ranges[{idx}]` must be a tuple of two real numbers."
            )
        if interval[0] > interval[1]:
            raise ValueError(
                f"`curvature_ranges[{idx}]` interval start must be <= end."
            )

    # Validate sns_style and sns_context
    if not isinstance(sns_style, str):
        raise TypeError("`sns_style` must be a string.")
    if not isinstance(sns_context, str):
        raise TypeError("`sns_context` must be a string.")

    # Validate figsize
    if figsize != "auto":
        if (
            not isinstance(figsize, tuple)
            or len(figsize) != 2
            or not all(isinstance(x, numbers.Real) for x in figsize)
            or not all(x > 0 for x in figsize)
        ):
            raise ValueError(
                "`figsize` must be 'auto' or a tuple of two positive numbers "
                "(width, height)."
            )

    # Validate save_path
    if save_path is not None and not isinstance(save_path, (str, Path)):
        raise TypeError("`save_path` must be None, a string, or a pathlib.Path.")

    # Validate show
    if not isinstance(show, bool):
        raise TypeError("`show` must be a boolean.")

    if binrange is None:
        binrange = (intensity.min(), intensity.max())

    # Set seaborn style and context
    sns.set_style(sns_style)
    sns.set_context(sns_context)

    # Bin the intensity data by curvature intervals
    intensity_binned = [
        intensity[(curvature >= low) & (curvature <= high)]
        for (low, high) in curvature_ranges
    ]

    num_ranges = len(curvature_ranges)

    # Auto figsize calculation
    if figsize == "auto":
        figsize = (8, 1.5 * num_ranges)

    fig, axes = plt.subplots(
        nrows=num_ranges, ncols=1, figsize=figsize, sharex=True, squeeze=False
    )
    axes = axes.flatten()  # flatten for consistent indexing

    # Find the dataset with the least number of data points
    num_data_points = np.array([len(intensities) for intensities in intensity_binned])
    idx_least_num_data_points = np.argmin(num_data_points)

    # Compute bin edges based on that dataset
    bin_edges = np.histogram_bin_edges(
        intensity_binned[idx_least_num_data_points], bins=bins, range=binrange
    )

    if isinstance(bins, str):
        num_bins = len(bin_edges) - 1
        print(f"Number of bins determined using '{bins}': {num_bins:d}")

    for i, ((low, high), data) in enumerate(
        zip(curvature_ranges, intensity_binned, strict=True)
    ):
        sns.histplot(
            data,
            bins=bin_edges,
            binrange=binrange,
            stat=stat,
            ax=axes[i],
            element=element,
            kde=kde,
        )
        axes[i].set_title(
            rf"$\mathrm{{Curvature}}$: {low:.2f} - {high:.2f} "
            r"$\mathrm{\mu m^{-2}}$"
        )
        axes[i].set_xlim(0, 1)
        axes[i].set_ylabel(stat.lower().capitalize())
        if i != num_ranges - 1:
            axes[i].set_xlabel("")
    axes[-1].set_xlabel("Normalized Intensity")

    # Collect all y-limits
    y_mins = []
    y_maxs = []
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        y_mins.append(ymin)
        y_maxs.append(ymax)

    # Determine global min and max y-limits
    global_ymin = min(y_mins)
    global_ymax = max(y_maxs)

    # Apply the global y-limits to all axes
    for ax in axes:
        ax.set_ylim(global_ymin, global_ymax)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_binned_intensity_separate_with_gmm(
    curvature: npt.ArrayLike,
    intensity: npt.ArrayLike,
    curvature_ranges: Sequence[tuple[float, float]],
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    std_devs: npt.ArrayLike,
    *,
    stat: Literal[
        "count", "frequency", "probability", "percent", "density"
    ] = "density",
    bins: str | int | Sequence[float] = "auto",
    binrange: Sequence[float] | Sequence[Sequence[float, float]] | None = None,
    element: Literal["bars", "step", "poly"] = "bars",
    sns_style: str = "darkgrid",
    sns_context: str = "paper",
    figsize: tuple[int, int] | str = "auto",
    plot_data: bool = True,
    plot_single_components: bool = True,
    show_fig_legend: bool = True,
    save_path: Path | str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot histograms of intensity data binned by curvature intervals with overlaid GMM.

    This function bins normalized intensity values according to specified curvature
    ranges, plots histograms of these intensities for each curvature bin, and overlays
    the probability density functions (PDFs) of Gaussian mixture model (GMM) components
    fitted to each bin's data.

    Parameters
    ----------
    curvature : array-like of shape (n_samples,)
        Curvature values corresponding to each intensity measurement.
    intensity : array-like of shape (n_samples,)
        Normalized intensity values, aligned with `curvature`.
    curvature_ranges : sequence of tuple of float
        List or tuple of (low, high) curvature intervals to bin the data.
    weights : sequence of array-like
        Mixture weights for Gaussian components in each curvature bin.
        Each element corresponds to one curvature bin and contains a 1D array of weights.
    means : sequence of array-like
        Means of Gaussian components for each curvature bin.
        Each element corresponds to one curvature bin and contains a 1D array of means.
    std_devs : sequence of array-like
        Standard deviations of Gaussian components for each curvature bin.
        Each element corresponds to one curvature bin and contains a 1D array of
        standard deviations.
    stat : {"count", "frequency", "probability", "percent", "density"}, optional
        Statistic to compute for histogram bins. Default is "density".
    bins : str, int, or sequence of floats, optional
        Binning strategy for histograms, passed to seaborn.histplot. Default is "auto".
    binrange : tuple or sequence of tuples, optional
        Lower and upper range of bins. If None, computed from intensity data.
    element : {"bars", "step", "poly"}, optional
        Style of histogram elements. Default is "bars".
    sns_style : str, optional
        Seaborn style for plot aesthetics (e.g., "darkgrid", "whitegrid"). Default is
        "darkgrid".
    sns_context : str, optional
        Seaborn context scaling (e.g., "paper", "talk", "poster"). Default is "paper".
    figsize : tuple of two positive numbers or "auto", optional
        Figure size (width, height) in inches. If "auto", height scales with number of
        bins. Default is "auto".
    plot_data: bool, optional
        Whether to display the data as histograms. Default is True.
    plot_single_components : bool, optional
        Whether to display individual components for the Gaussian mixture models.
        Default is True.
    show_fig_legend : bool, optional
        Whether to display a combined legend for all Gaussian components
        at the figure level. Default is True.
    save_path : Path, str, or None, optional
        File path to save the figure. If None, the figure is not saved. Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the subplots.
    ax : matplotlib.axes.Axes
        The Axes object containing the histogram and GMM components.

    Raises
    ------
    ValueError
        If input arrays have inconsistent lengths or invalid shapes.
    TypeError
        If input types do not meet the expected formats.

    Notes
    -----
    - Each curvature range defines a subset of the intensity data used to plot a
    histogram.
    - The Gaussian mixture model components are truncated normal distributions
    constrained to the binrange.
    - The function overlays the PDFs of each mixture component on the respective
    histogram.
    - The bin edges for the histogram are automatically extracted from the plotted
    histograms to align GMM component plots precisely.
    """
    import numbers

    # Validate curvature and intensity are array-like and have same length
    curvature = np.asarray(curvature)
    intensity = np.asarray(intensity)
    if curvature.ndim != 1:
        raise ValueError("`curvature` must be a 1D array-like.")
    if intensity.ndim != 1:
        raise ValueError("`intensity` must be a 1D array-like.")
    if curvature.shape[0] != intensity.shape[0]:
        raise ValueError("`curvature` and `intensity` must have the same length.")

    # Validate curvature_ranges
    if not isinstance(curvature_ranges, (list, tuple)):
        raise TypeError(
            "`curvature_ranges` must be a list or tuple of (float, float) tuples."
        )
    for idx, interval in enumerate(curvature_ranges):
        if (
            not isinstance(interval, tuple)
            or len(interval) != 2
            or not all(isinstance(x, numbers.Real) for x in interval)
        ):
            raise TypeError(
                f"`curvature_ranges[{idx}]` must be a tuple of two real numbers."
            )
        if interval[0] > interval[1]:
            raise ValueError(
                f"`curvature_ranges[{idx}]` interval start must be <= end."
            )

    # Validate weights, means, std_devs
    n_ranges = len(curvature_ranges)
    for name, param in zip(
        ["weights", "means", "std_devs"], [weights, means, std_devs], strict=True
    ):
        if not isinstance(param, (list, tuple, np.ndarray)):
            raise TypeError(f"`{name}` must be a sequence (list/tuple/ndarray).")
        if len(param) != n_ranges:
            raise ValueError(
                f"`{name}` must have the same length as `curvature_ranges` ({n_ranges})."
            )
        for idx, arr in enumerate(param):
            arr = np.asarray(arr)
            if arr.ndim != 1:
                raise ValueError(
                    f"Each element of `{name}` must be 1D array-like; "
                    f"found shape {arr.shape} at index {idx}."
                )

    # Check that weights, means, and std_devs shapes match for each bin
    for i in range(n_ranges):
        if not (len(weights[i]) == len(means[i]) == len(std_devs[i])):
            raise ValueError(
                f"At index {i}, lengths of weights, means, and std_devs must match. "
                f"Found lengths: weights={len(weights[i])}, means={len(means[i])}, std_devs={len(std_devs[i])}"
            )

    fig, axes = plot_binned_intensity_separate(
        curvature,
        intensity,
        curvature_ranges,
        stat=stat,
        bins=bins,
        binrange=binrange,
        element=element,
        sns_style=sns_style,
        sns_context=sns_context,
        figsize=figsize,
        save_path=None,  # Don't save yet
        show=False,  # Don't show it yet
    )

    if not plot_data:
        # Make sure to get out the binrange
        bin_edges = _get_bin_edges_from_axis(axes[0])
        binrange = (bin_edges[0], bin_edges[-1])

        # User does not want to data to be plotted, so remove the data but keep
        # labels, ticks, titles, etc.
        for ax in axes:
            ax = _clear_plot_data(ax)

    for i, ax in enumerate(axes):
        x_plot = np.linspace(binrange[0], binrange[1], 500)

        total_pdf = np.zeros_like(x_plot)
        for j, (w, mu, sigma) in enumerate(
            zip(weights[i], means[i], std_devs[i], strict=True)
        ):
            a, b = binrange
            a_trans, b_trans = (a - mu) / sigma, (b - mu) / sigma
            component = truncnorm(a_trans, b_trans, loc=mu, scale=sigma)
            comp_pdf = w * component.pdf(x_plot)
            total_pdf += comp_pdf

            if plot_single_components:
                ax.plot(x_plot, comp_pdf, ls="--", lw=1.5, label=f"Component {j + 1}")

        # Plot sum of all components as solid line
        ax.plot(x_plot, total_pdf, ls="-", lw=1.5, color="black", label="GMM sum")
        # ax.legend()

    if show_fig_legend:
        # After plotting all axes:
        fig.subplots_adjust(
            right=0.8
        )  # Shrinks plot area, leaving room on the right for legend
        handles, labels = axes[0].get_legend_handles_labels()

        # Find index of the "GMM sum" label (assuming you named it exactly)
        sum_idx = labels.index("GMM sum")

        # Move the sum handle and label to the front (top)
        handles = [handles[sum_idx], *handles[:sum_idx], *handles[sum_idx + 1 :]]
        labels = [labels[sum_idx], *labels[:sum_idx], *labels[sum_idx + 1 :]]
        fig.legend(
            handles, labels, loc="center right", title="GMM components"
        )  # fontsize="small")

    # Collect all y-limits
    y_mins = []
    y_maxs = []
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        y_mins.append(ymin)
        y_maxs.append(ymax)

    # Determine global min and max y-limits
    global_ymin = min(y_mins)
    global_ymax = max(y_maxs)

    # Apply the global y-limits to all axes
    for ax in axes:
        ax.set_ylim(global_ymin, global_ymax)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_binned_intensity_overlay(
    curvature: npt.ArrayLike,
    intensity: npt.ArrayLike,
    curvature_ranges: Sequence[tuple[float, float]],
    *,
    stat: Literal[
        "count", "frequency", "probability", "percent", "density"
    ] = "density",
    bins: str | int | Sequence[float] = "auto",
    binrange: Sequence[float] | Sequence[Sequence[float, float]] | None = None,
    element: Literal["bars", "step", "poly"] = "bars",
    kde: bool = False,
    sns_style: str = "darkgrid",
    sns_context: str = "paper",
    figsize: tuple[int, int] | str = "auto",
    save_path: Path | str | None = None,
    show: bool = True,
    num_colors: int | None = None,
    density_range: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot intensity histograms overlaid for multiple curvature intervals.

    Each specified curvature interval is used to select a subset of intensity
    values, and a histogram of those values is plotted on a shared set of axes.
    Colors are assigned from a discrete colormap (default: "viridis").

    Parameters
    ----------
    curvature : ArrayLike
        1D array of curvature values for each voxel or sample.
    intensity : ArrayLike
        1D array of normalized intensity values corresponding to `curvature`.
    curvature_ranges : sequence of (float, float)
        List of (low, high) curvature intervals to bin the intensity data.
    stat : {"count", "frequency", "probability", "percent", "density"}, optional
        Aggregate statistic to compute in each histogram bin. Passed to
        `seaborn.histplot`. Default is "density".
    bins : str, int, or sequence of floats, optional
        Binning strategy. Passed to `numpy.histogram_bin_edges` / `seaborn.histplot`.
        Default is "auto".
    binrange : (float, float) or None, optional
        Lower and upper range of the bins. If None, uses the min and max of
        `intensity`. Default is None.
    element : {"bars", "step", "poly"}, optional
        Style of histogram elements. Default is "bars".
    kde : bool, optional
        If True, overlay kernel density estimate lines for each distribution.
        Default is False.
    sns_style : str, optional
        Seaborn plotting style (e.g., "darkgrid", "whitegrid"). Default is "darkgrid".
    sns_context : str, optional
        Seaborn context scaling (e.g., "paper", "talk", "poster"). Default is "paper".
    figsize : (int, int) or "auto", optional
        Figure size in inches (width, height). If "auto", uses a default size (8, 6).
        Default is "auto".
    save_path : str or Path or None, optional
        If provided, path where the figure will be saved as a PNG. Default is None.
    show : bool, optional
        If True, display the plot interactively. Default is True.
    num_colors : int or None, optional
        Number of discrete colors to sample from the colormap. If None, uses the
        number of curvature intervals. Default is None.
    density_range : (float, float) or None, optional
        y-axis limits for the density/statistic. If None, set automatically.
        Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The Axes object with all histograms overlaid.

    Notes
    -----
    - The bin edges are determined using the curvature interval with the fewest
      samples, ensuring consistent alignment across histograms.
    - Histogram transparency is adjusted to improve visibility of overlaps.
    - Colors are drawn from the "viridis" colormap by default, but the number of
      distinct colors can be overridden with `num_colors`.
    """
    import numbers

    # Validate curvature and intensity are array-like and have same length
    curvature = np.asarray(curvature)
    intensity = np.asarray(intensity)
    if curvature.ndim != 1:
        raise ValueError("`curvature` must be a 1D array-like.")
    if intensity.ndim != 1:
        raise ValueError("`intensity` must be a 1D array-like.")
    if curvature.shape[0] != intensity.shape[0]:
        raise ValueError("`curvature` and `intensity` must have the same length.")

    # Validate curvature_ranges
    if not isinstance(curvature_ranges, (list, tuple)):
        raise TypeError(
            "`curvature_ranges` must be a list or tuple of (float, float) tuples."
        )
    for idx, interval in enumerate(curvature_ranges):
        if (
            not isinstance(interval, tuple)
            or len(interval) != 2
            or not all(isinstance(x, numbers.Real) for x in interval)
        ):
            raise TypeError(
                f"`curvature_ranges[{idx}]` must be a tuple of two real numbers."
            )
        if interval[0] > interval[1]:
            raise ValueError(
                f"`curvature_ranges[{idx}]` interval start must be <= end."
            )

    # Validate sns_style and sns_context
    if not isinstance(sns_style, str):
        raise TypeError("`sns_style` must be a string.")
    if not isinstance(sns_context, str):
        raise TypeError("`sns_context` must be a string.")

    # Validate figsize
    if figsize != "auto":
        if (
            not isinstance(figsize, tuple)
            or len(figsize) != 2
            or not all(isinstance(x, numbers.Real) for x in figsize)
            or not all(x > 0 for x in figsize)
        ):
            raise ValueError(
                "`figsize` must be 'auto' or a tuple of two positive numbers "
                "(width, height)."
            )

    # Validate save_path
    if save_path is not None and not isinstance(save_path, (str, Path)):
        raise TypeError("`save_path` must be None, a string, or a pathlib.Path.")

    # Validate show
    if not isinstance(show, bool):
        raise TypeError("`show` must be a boolean.")

    if binrange is None:
        binrange = (intensity.min(), intensity.max())

    # Set seaborn style and context
    sns.set_style(sns_style)
    sns.set_context(sns_context)

    # Bin the intensity data by curvature intervals
    intensity_binned = [
        intensity[(curvature >= low) & (curvature <= high)]
        for (low, high) in curvature_ranges
    ]

    # Auto figsize calculation
    if figsize == "auto":
        figsize = (8, 6)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=False)
    # Flatten axes array and select the first Axes object for single-axes plotting
    ax = ax.flatten()[0]

    # Find the dataset with the least number of data points
    num_data_points = np.array([len(intensities) for intensities in intensity_binned])
    idx_least_num_data_points = np.argmin(num_data_points)

    # Compute bin edges based on that dataset
    bin_edges = np.histogram_bin_edges(
        intensity_binned[idx_least_num_data_points], bins=bins, range=binrange
    )

    if isinstance(bins, str):
        num_bins = len(bin_edges) - 1
        print(f"Number of bins determined using '{bins}': {num_bins:d}")

    num_curv_bins = len(curvature_ranges)
    if num_colors is not None:
        if not isinstance(num_colors, int):
            raise TypeError("num_colors should be an integer.")
        num_curv_bins = num_colors

    # Get a colormap with n_bins discrete colors
    cmap = matplotlib.cm.get_cmap("viridis", num_curv_bins)

    # Extract colors as a list/array
    colors = [cmap(i) for i in range(num_curv_bins)]

    # Plot all histograms on the same axes with labels
    alpha = (1 / num_curv_bins) * 1.25  # factor 1.25 is for aesthetic reasons
    alpha = np.clip(alpha, 0.0, 1.0)
    for i, ((low, high), data) in enumerate(
        zip(curvature_ranges, intensity_binned, strict=True)
    ):
        color = colors[i]
        sns.histplot(
            data,
            bins=bin_edges,
            binrange=binrange,
            stat=stat,
            ax=ax,
            element=element,
            label=rf"{low:.2f} - {high:.2f} $\mathrm{{\mu m^{{-2}}}}$",
            alpha=alpha,  # Make histograms semi-transparent for overlap visibility
            color=color,
            kde=kde,
        )

    if density_range is not None:
        ax.set_ylim(density_range)

    ax.set_xlabel("Normalized Intensity")
    ax.set_ylabel(stat.lower().capitalize())

    ax.legend(title="Curvature ranges", loc="best")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_binned_intensity_overlay_with_gmm(
    curvature: npt.ArrayLike,
    intensity: npt.ArrayLike,
    curvature_ranges: Sequence[tuple[float, float]],
    weights: npt.ArrayLike,
    means: npt.ArrayLike,
    std_devs: npt.ArrayLike,
    *,
    stat: Literal[
        "count", "frequency", "probability", "percent", "density"
    ] = "density",
    bins: str | int | Sequence[float] = "auto",
    binrange: Sequence[float] | Sequence[Sequence[float, float]] | None = None,
    element: Literal["bars", "step", "poly"] = "bars",
    sns_style: str = "darkgrid",
    sns_context: str = "paper",
    figsize: tuple[int, int] | str = "auto",
    plot_data: bool = True,
    plot_single_components: bool = True,
    show_fig_legend: bool = True,
    save_path: Path | str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot histograms of intensity data binned by curvature intervals with overlaid GMM.

    This function bins normalized intensity values according to specified curvature
    ranges, plots histograms of these intensities for each curvature bin, and overlays
    the probability density functions (PDFs) of Gaussian mixture model (GMM) components
    fitted to each bin's data.

    Parameters
    ----------
    curvature : array-like of shape (n_samples,)
        Curvature values corresponding to each intensity measurement.
    intensity : array-like of shape (n_samples,)
        Normalized intensity values, aligned with `curvature`.
    curvature_ranges : sequence of tuple of float
        List or tuple of (low, high) curvature intervals to bin the data.
    weights : sequence of array-like
        Mixture weights for Gaussian components in each curvature bin.
        Each element corresponds to one curvature bin and contains a 1D array of weights.
    means : sequence of array-like
        Means of Gaussian components for each curvature bin.
        Each element corresponds to one curvature bin and contains a 1D array of means.
    std_devs : sequence of array-like
        Standard deviations of Gaussian components for each curvature bin.
        Each element corresponds to one curvature bin and contains a 1D array of
        standard deviations.
    stat : {"count", "frequency", "probability", "percent", "density"}, optional
        Statistic to compute for histogram bins. Default is "density".
    bins : str, int, or sequence of floats, optional
        Binning strategy for histograms, passed to seaborn.histplot. Default is "auto".
    binrange : tuple or sequence of tuples, optional
        Lower and upper range of bins. If None, computed from intensity data.
    element : {"bars", "step", "poly"}, optional
        Style of histogram elements. Default is "bars".
    sns_style : str, optional
        Seaborn style for plot aesthetics (e.g., "darkgrid", "whitegrid"). Default is
        "darkgrid".
    sns_context : str, optional
        Seaborn context scaling (e.g., "paper", "talk", "poster"). Default is "paper".
    figsize : tuple of two positive numbers or "auto", optional
        Figure size (width, height) in inches. If "auto", height scales with number of
        bins. Default is "auto".
    plot_data: bool, optional
        Whether to display the data as histograms. Default is True.
    plot_single_components : bool, optional
        Whether to display individual components for the Gaussian mixture models.
        Default is True.
    show_fig_legend : bool, optional
        Whether to display a combined legend for all Gaussian components
        at the figure level. Default is True.
    save_path : Path, str, or None, optional
        File path to save the figure. If None, the figure is not saved. Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the subplots.
    ax : matplotlib.axes.Axes
        The Axes object containing the histogram and GMM components.

    Raises
    ------
    ValueError
        If input arrays have inconsistent lengths or invalid shapes.
    TypeError
        If input types do not meet the expected formats.

    Notes
    -----
    - Each curvature range defines a subset of the intensity data used to plot a
    histogram.
    - The Gaussian mixture model components are truncated normal distributions
    constrained to the binrange.
    - The function overlays the PDFs of each mixture component on the respective
    histogram.
    - The bin edges for the histogram are automatically extracted from the plotted
    histograms to align GMM component plots precisely.
    """
    import numbers

    # Validate curvature and intensity are array-like and have same length
    curvature = np.asarray(curvature)
    intensity = np.asarray(intensity)
    if curvature.ndim != 1:
        raise ValueError("`curvature` must be a 1D array-like.")
    if intensity.ndim != 1:
        raise ValueError("`intensity` must be a 1D array-like.")
    if curvature.shape[0] != intensity.shape[0]:
        raise ValueError("`curvature` and `intensity` must have the same length.")

    # Validate curvature_ranges
    if not isinstance(curvature_ranges, (list, tuple)):
        raise TypeError(
            "`curvature_ranges` must be a list or tuple of (float, float) tuples."
        )
    for idx, interval in enumerate(curvature_ranges):
        if (
            not isinstance(interval, tuple)
            or len(interval) != 2
            or not all(isinstance(x, numbers.Real) for x in interval)
        ):
            raise TypeError(
                f"`curvature_ranges[{idx}]` must be a tuple of two real numbers."
            )
        if interval[0] > interval[1]:
            raise ValueError(
                f"`curvature_ranges[{idx}]` interval start must be <= end."
            )

    # Validate weights, means, std_devs
    n_ranges = len(curvature_ranges)
    for name, param in zip(
        ["weights", "means", "std_devs"], [weights, means, std_devs], strict=True
    ):
        if not isinstance(param, (list, tuple, np.ndarray)):
            raise TypeError(f"`{name}` must be a sequence (list/tuple/ndarray).")
        if len(param) != n_ranges:
            raise ValueError(
                f"`{name}` must have the same length as `curvature_ranges` ({n_ranges})."
            )
        for idx, arr in enumerate(param):
            arr = np.asarray(arr)
            if arr.ndim != 1:
                raise ValueError(
                    f"Each element of `{name}` must be 1D array-like; "
                    f"found shape {arr.shape} at index {idx}."
                )

    # Check that weights, means, and std_devs shapes match for each bin
    for i in range(n_ranges):
        if not (len(weights[i]) == len(means[i]) == len(std_devs[i])):
            raise ValueError(
                f"At index {i}, lengths of weights, means, and std_devs must match. "
                f"Found lengths: weights={len(weights[i])}, means={len(means[i])}, std_devs={len(std_devs[i])}"
            )

    fig, ax = plot_binned_intensity_overlay(
        curvature,
        intensity,
        curvature_ranges,
        stat=stat,
        bins=bins,
        binrange=binrange,
        element=element,
        sns_style=sns_style,
        sns_context=sns_context,
        figsize=figsize,
        save_path=None,  # Don't save yet
        show=False,  # Don't show it yet
    )

    # Make sure to get out the binrange
    bin_edges = _get_bin_edges_from_axis(ax)
    binrange = (bin_edges[0], bin_edges[-1])

    if not plot_data:
        # User does not want to data to be plotted, so remove the data but keep
        # labels, ticks, titles, etc.
        ax = _clear_plot_data(ax)

    # Compute mean curvature for each bin
    curvature_means = np.array([(low + high) / 2 for (low, high) in curvature_ranges])
    cmin, cmax = curvature_means.min(), curvature_means.max()

    # Other option, use the complete curvature range to set cmin and cmax
    # all_curvatures = np.array(curvature_ranges).flatten()
    # cmin, cmax = all_curvatures.min(), all_curvatures.max()

    # Normalize curvature means to [0, 1] for colormap
    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    cmap = matplotlib.colormaps.get_cmap("viridis")

    x_plot = np.linspace(binrange[0], binrange[1], 500)
    for i, ((low, high)) in enumerate(curvature_ranges):
        color = cmap(norm(curvature_means[i]))
        total_pdf = np.zeros_like(x_plot)
        for j, (w, mu, sigma) in enumerate(
            zip(weights[i], means[i], std_devs[i], strict=True)
        ):
            a, b = binrange
            a_trans, b_trans = (a - mu) / sigma, (b - mu) / sigma
            component = truncnorm(a_trans, b_trans, loc=mu, scale=sigma)
            comp_pdf = w * component.pdf(x_plot)
            total_pdf += comp_pdf

            if plot_single_components:
                ax.plot(x_plot, comp_pdf, ls="--", lw=1.5, label=f"Component {j + 1}")

        label = None
        if not plot_data:
            # To ensure no double labels
            label = rf"{low:.2f} - {high:.2f} $\mathrm{{\mu m^{{-2}}}}$"

        # Plot sum of all components as solid line
        ax.plot(
            x_plot,
            total_pdf,
            ls="-",
            lw=1.5,
            label=label,
            color=color,
        )

    # # Add colorbar on the right for curvature mapping
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.05)
    # cbar.set_label(r"Curvature ($\mathrm{\mu m^{-2}}$)")

    ax.set_xlabel("Normalized Intensity")
    ax.set_ylabel(stat.lower().capitalize())
    if show_fig_legend:
        ax.legend(title="Curvature ranges", loc="best")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    import polars as pl

    csv_path = r"E:\Myron_confocal_data\Nuclei_imaged_DAPI_LAC_LB1\spheres\analysis\all_data.csv"
    df = pl.read_csv(csv_path)

    curvature = np.asarray(df["k1"]) * np.asarray(df["k2"])
    intensity = np.asarray(df["norm_intensity_ch2"])
    curvature_ranges = [
        (0.00, 0.15),
        # (0.15, 0.30),
        # (0.30, 0.45),
        (0.45, 0.60),
        # (0.60, 0.75),
        (0.75, 0.90),
    ]

    # plot_binned_intensity_separate(
    #     curvature, intensity, curvature_ranges, bins="auto", binrange=(0, 1)
    # )

    # plot_binned_intensity_overlay(
    #     curvature,
    #     intensity,
    #     curvature_ranges,
    #     bins="auto",
    #     binrange=(0, 1),
    #     element="step",
    # )

    weights = [[0.7, 0.3], [0.4, 0.6], [0.3, 0.7]]
    means = [
        [0.2, 0.42],
        [0.2, 0.42],
        [0.2, 0.42],
    ]
    std_devs = [
        [0.1, 0.2],
        [0.1, 0.2],
        [0.1, 0.2],
    ]
    # plot_binned_intensity_separate_with_gmm(
    #     curvature,
    #     intensity,
    #     curvature_ranges,
    #     weights,
    #     means,
    #     std_devs,
    #     plot_data=False,
    # )
    plot_binned_intensity_overlay_with_gmm(
        curvature,
        intensity,
        curvature_ranges,
        weights,
        means,
        std_devs,
        plot_data=True,
        plot_single_components=False,
    )
    plt.show()
