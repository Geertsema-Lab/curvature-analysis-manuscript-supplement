from __future__ import annotations

from typing import TYPE_CHECKING

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import scipy.stats
import seaborn as sns
from matplotlib import cm
from matplotlib import ticker as mticker
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm

import imcurvpy.analysis as analysis

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib


def _validate_inputs(
    intensities: npt.ArrayLike,
    curvatures: npt.ArrayLike,
    min_curv: float,
    max_curv: float,
    bins: int,
    percentiles: float | None,
    curvature_type: str,
) -> None:
    """
    Validate input arrays and plotting parameters.

    Parameters
    ----------
    intensities : array-like
        Array of intensity values. Must have the same shape as `curvatures`.
    curvatures : array-like
        Array of curvature values. Must have the same shape as `intensities`.
    min_curv : float
        Minimum curvature value for filtering. Must be less than `max_curv`.
    max_curv : float
        Maximum curvature value for filtering. Must be greater than `min_curv`.
    bins : int
        Number of histogram bins. Must be a positive integer.
    percentiles : float or None
        Percentile threshold for filtering. Must be between 0 and 100 (inclusive).
    curvature_type : str
        Type of curvature to use. Must be either "gaussian" or "mean"
        (case-insensitive).

    Raises
    ------
    ValueError
        If array shapes mismatch or if any parameter is out of the valid range.
    """
    if intensities.shape != curvatures.shape:
        raise ValueError("Intensities and curvatures must have the same length.")
    if min_curv >= max_curv:
        raise ValueError("min_curvature must be less than max_curvature.")
    if percentiles is not None:
        if isinstance(percentiles, (int, float)):
            percentiles = [percentiles]
        for p in percentiles:
            if not (0 <= p <= 100):
                raise ValueError("percentiles must be between 0 and 100.")
    if bins <= 0:
        raise ValueError("bins must be a positive integer.")
    if curvature_type.lower() not in {"gaussian", "mean"}:
        raise ValueError(
            f"Invalid curvature type: {curvature_type}. Must be 'gaussian' or 'mean'."
        )


def _get_joint_histogram_color(
    ax_joint: matplotlib.axes.Axes, data_size: int
) -> tuple[float, float, float, float]:
    """
    Return the colormap color corresponding to the average non-zero bin proportion.

    Extracts the QuadMesh from a seaborn JointGrid joint axis, computes the
    average of non-zero proportions, and maps it to a color using logarithmic
    normalization with the 'viridis' colormap.

    Parameters
    ----------
    ax_joint : matplotlib.axes.Axes
        Joint axis containing the 2D histogram QuadMesh.
    data_size : int
        Total number of data points, used for log normalization.

    Returns
    -------
    tuple
        RGBA tuple from the 'viridis' colormap.

    Raises
    ------
    RuntimeError
        If no QuadMesh is found in the joint axis.
    """
    for artist in ax_joint.collections:
        if isinstance(artist, QuadMesh):
            proportions = artist.get_array().data
            avg_proportion = np.mean(proportions[proportions > 0])  # ignore zero bins
            norm = LogNorm(vmin=1 / data_size, vmax=proportions.max())
            return cm.get_cmap("viridis")(norm(avg_proportion))
    raise RuntimeError("Could not find QuadMesh for 2D histogram.")


def plot_curv_intensity_joint(
    curvatures: npt.ArrayLike,
    intensities: npt.ArrayLike,
    curvature_range: tuple[float, float] = (0, None),
    intensities_range: tuple[float, float] = (0, None),
    bins: int = 100,
    curvature_type: str = "gaussian",
    statistics: str | Callable | Sequence[str | Callable] | None = "median",
    percentiles: float | Sequence[float] | None = None,
    sns_style: str = "darkgrid",
    sns_context: str = "paper",
    figsize: tuple[int, int] = (8, 6),
    save_path: Path | str | None = None,
    show: bool = True,
    return_stats: bool = False,
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, plt.Axes, dict[str, np.ndarray]]:
    """
    Plot a 2D histogram of normalized intensity versus surface curvature.

    This function creates a seaborn 2D histogram (without marginal plots) of normalized
    intensity against surface curvature (Gaussian or mean). It can optionally overlay
    statistical summaries (mean, median, or custom functions) and percentile curves.

    Parameters
    ----------
    curvatures : array-like
        Curvature values (µm^-2) for each data point.
    intensities : array-like
        Normalized intensity values corresponding to each curvature measurement.
    curvature_range : tuple[float, float], optional
        Tuple specifying (min_curvature, max_curvature). Use None for automatic max.
        Default is (0, None).
    intensities_range : tuple[float, float], optional
        Tuple specifying (min_intensity, max_intensity). Use None for automatic max.
        Default is (0, None).
    bins : int, optional
        Number of bins for the histogram and statistics. Default is 100.
    curvature_type : str, optional
        Type of curvature: "gaussian" or "mean" (case-insensitive). Determines
        axis labeling. Default is "gaussian".
    statistics : str, callable, or sequence of str/callable, optional
        Statistic(s) to compute and overlay on the histogram (e.g., "mean", "median").
        If None, no statistics are plotted. Default is "median".
    percentiles : float or sequence of floats, optional
        Percentile(s) to compute and overlay on the histogram (0 - 100). Can be a single
        float or a sequence. Default is None.
    sns_style : str, optional
        Seaborn style (e.g., "darkgrid", "white", "ticks"). Default is "darkgrid".
    sns_context : str, optional
        Seaborn context for scaling elements (e.g., "paper", "talk"). Default is "paper".
    figsize : tuple[int, int], optional
        Figure size in inches as (width, height). Default is (8, 6).
    save_path : str or Path, optional
        If provided, saves the figure to this path. Default is None (no saving).
    show : bool, optional
        If True, displays the plot immediately. Default is True.
    return_stats : bool, optional
        If True, returns a dictionary of computed statistics along with the figure and axes.
        Default is False.

    Returns
    -------
    tuple
        If return_stats is False:
            (fig, ax) : matplotlib.figure.Figure, matplotlib.axes.Axes
                The figure and axes objects of the plot.
        If return_stats is True:
            (fig, ax, stats_dict) : matplotlib.figure.Figure, matplotlib.axes.Axes, dict
                The figure, axes, and dictionary of computed statistics keyed by name.

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., mismatched array shapes, invalid percentile,
        or unsupported curvature type).

    Examples
    --------
    >>> import numpy as np
    >>> curvatures = np.random.gamma(shape=2, scale=0.5, size=10000)
    >>> intensities = np.exp(-curvatures) + 0.1 * np.random.randn(10000)
    >>> fig, ax, stats = plot_curv_intensity_joint(
    ...     curvatures=curvatures,
    ...     intensities=intensities,
    ...     bins=80,
    ...     statistics=("median", "mean"),
    ...     percentiles=90,
    ...     curvature_type="mean",
    ... )
    """
    curvatures = np.asarray(curvatures)
    intensities = np.asarray(intensities)

    min_curv, max_curv = curvature_range
    if max_curv is None:
        max_curv = curvatures.max()

    min_int, max_int = intensities_range
    if max_int is None:
        max_int = intensities.max()

    _validate_inputs(
        intensities,
        curvatures,
        min_curv,
        max_curv,
        bins,
        percentiles,
        curvature_type,
    )

    # Filter data
    mask = (
        (curvatures >= min_curv)
        & (curvatures <= max_curv)
        & (intensities >= min_int)
        & (intensities <= max_int)
    )
    curvatures = curvatures[mask]
    intensities = intensities[mask]

    sns.set_context(sns_context)
    sns.set_style(sns_style)

    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(
        x=curvatures,
        y=intensities,
        bins=bins,
        cmap="viridis",
        stat="proportion",
        cbar=True,
        cbar_kws={"label": "Proportion"},
        ax=ax,
        pmax=None,
        norm="log",
        vmin=(1 / curvatures.size),
    )

    xarr = np.linspace(min_curv, max_curv, bins)

    stats_dict = {}
    if statistics is not None or percentiles is not None:
        stats_dict = analysis.compute_binned_statistics(
            curvatures,
            intensities,
            bins,
            (min_curv, max_curv),
            statistics=statistics,
            percentiles=percentiles,
        )

        # Matplotlib color cycle
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        stats_labels = {}
        stats_colors = {}

        # Normalize statistics input
        stats_list = (
            [statistics]
            if isinstance(statistics, (str, Callable))
            else statistics or []
        )
        for i, stat in enumerate(stats_list):
            key = stat if isinstance(stat, str) else stat.__name__
            stats_labels[key] = key.capitalize()
            stats_colors[key] = prop_cycle[i % len(prop_cycle)]

        # Handle percentiles
        perc_list = (
            [percentiles]
            if isinstance(percentiles, (int, float))
            else percentiles or []
        )
        for i, p in enumerate(perc_list):
            key = f"percentile_{p:.2f}"
            stats_labels[key] = f"Upper {p:.2f}th percentile"
            stats_colors[key] = prop_cycle[(i + len(stats_list)) % len(prop_cycle)]

        # Plot statistics curves
        for key in stats_dict:
            if key == "bin_centers":
                continue
            ax.plot(
                xarr,
                stats_dict[key],
                label=stats_labels.get(key, key),
                linewidth=2.5,
                color=stats_colors.get(key, "black"),
            )
        ax.legend()

    # Axis labels
    if curvature_type.lower() == "gaussian":
        ax.set_xlabel(r"Gaussian curvature ($\mathrm{\mu m^{-2}}$)")
    elif curvature_type.lower() == "mean":
        ax.set_xlabel(r"Mean curvature ($\mathrm{\mu m^{-2}}$)")

    ax.set_ylabel("Normalized intensity")

    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()

    if return_stats:
        return fig, ax, stats_dict
    return fig, ax


def plot_curv_intensity_joint_marginal(
    curvatures: npt.ArrayLike,
    intensities: npt.ArrayLike,
    curvature_range: tuple[float, float] = (0, None),
    intensities_range: tuple[float, float] = (0, None),
    bins: int = 100,
    curvature_type: str = "gaussian",
    statistics: str | Callable | Sequence[str | Callable] | None = ("median", "mean"),
    percentiles: float | Sequence[float] | None = 95,
    sns_style: str = "darkgrid",
    sns_context: str = "paper",
    figsize: tuple[int, int] = (8, 6),
    save_path: Path | str | None = None,
    show: bool = True,
    return_stats: bool = False,
):
    """
    Visualize the relationship between normalized intensity and surface curvature
    with marginal histograms.

    Generates a seaborn JointGrid 2D histogram of intensity vs. curvature (Gaussian
    or mean), including marginal histograms. Optionally overlays summary statistics
    (percentiles, mean, median) as curves.

    Parameters
    ----------
    curvatures : array-like
        Curvature values (µm^-2) for each data point.
    intensities : array-like
        Normalized intensity values corresponding to each curvature measurement.
    curvature_range : tuple[float, float], optional
        Tuple specifying (min_curvature, max_curvature). Use None for automatic max.
        Default is (0, None).
    intensities_range : tuple[float, float], optional
        Tuple specifying (min_intensity, max_intensity). Use None for automatic max.
        Default is (0, None).
    bins : int, optional
        Number of bins for histograms and statistics. Default is 100.
    curvature_type : str, optional
        Type of curvature to visualize: "gaussian" or "mean" (case-insensitive).
        Default is "gaussian".
    statistics : str, callable, or sequence of str/callable, optional
        Statistic(s) to compute and overlay (e.g., "median", "mean"). Default is
        ("median", "mean").
    percentiles : float or sequence of floats, optional
        Percentile(s) to overlay (0-100). Can be single float or sequence. Default is 95.
    sns_style : str, optional
        Seaborn style (e.g., "darkgrid", "white", "ticks"). Default is "darkgrid".
    sns_context : str, optional
        Seaborn context for scaling elements (e.g., "paper", "talk"). Default is "paper".
    figsize : tuple[int, int], optional
        Figure size in inches as (width, height). Default is (8, 6).
    save_path : str or Path, optional
        If provided, saves the figure. Default is None.
    show : bool, optional
        If True, displays the plot immediately. Default is True.
    return_stats : bool, optional
        If True, returns computed statistics. Default is False.

    Returns
    -------
    tuple
        (fig, ax) if return_stats is False, or (fig, ax, stats_dict) if True.

    """
    curvatures = np.asarray(curvatures)
    intensities = np.asarray(intensities)

    min_curv, max_curv = curvature_range
    if max_curv is None:
        max_curv = curvatures.max()

    min_int, max_int = intensities_range
    if max_int is None:
        max_int = intensities.max()

    _validate_inputs(
        intensities,
        curvatures,
        min_curv,
        max_curv,
        bins,
        percentiles,
        curvature_type,
    )

    # Filter data
    mask = (
        (curvatures >= min_curv)
        & (curvatures <= max_curv)
        & (intensities >= min_int)
        & (intensities <= max_int)
    )
    curvatures = curvatures[mask]
    intensities = intensities[mask]

    df = pl.DataFrame({"curvature": curvatures, "intensity": intensities})
    xarr = np.linspace(min_curv, max_curv, bins)

    sns.set_context(sns_context)
    sns.set_style(sns_style)

    g = sns.JointGrid(
        data=df,
        x="curvature",
        y="intensity",
        space=0,
        height=figsize[1],
        ratio=5,
        marginal_ticks=True,
    )

    # Joint 2D histogram
    g.plot_joint(
        sns.histplot,
        bins=bins,
        cmap="viridis",
        stat="proportion",
        cbar=True,
        cbar_kws={"label": "Proportion"},
        norm="log",
        vmin=1 / intensities.size,
    )

    # Marginal color based on joint histogram
    marginal_color = _get_joint_histogram_color(g.ax_joint, intensities.size)

    # Marginal histograms with KDE
    g.plot_marginals(
        sns.histplot,
        bins=bins,
        color=marginal_color,
        edgecolor="none",
        stat="proportion",
        kde=True,
    )

    # Format axes
    g.ax_marg_y.set_xlim((1e-4, 1))
    g.ax_marg_y.set_xlabel("Proportion")
    g.ax_marg_y.set_xscale("log")
    g.ax_marg_y.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
    g.ax_marg_x.set_ylim((1e-4, 1))
    g.ax_marg_x.set_ylabel("Proportion")
    g.ax_marg_x.set_yscale("log")
    g.ax_marg_x.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
    pos_joint_ax = g.ax_joint.get_position()
    pos_marg_x_ax = g.ax_marg_x.get_position()
    g.ax_joint.set_position(
        [pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height]
    )
    g.figure.axes[-1].set_position([0.87, pos_joint_ax.y0, 0.03, pos_joint_ax.height])

    stats_dict = {}
    if statistics is not None or percentiles is not None:
        stats_dict = analysis.compute_binned_statistics(
            curvatures,
            intensities,
            bins,
            (min_curv, max_curv),
            statistics=statistics,
            percentiles=percentiles,
        )

        # Assign colors and labels
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        stats_labels = {}
        stats_colors = {}

        stats_list = (
            [statistics]
            if isinstance(statistics, (str, Callable))
            else statistics or []
        )
        for i, stat in enumerate(stats_list):
            key = stat if isinstance(stat, str) else stat.__name__
            stats_labels[key] = key.capitalize()
            stats_colors[key] = prop_cycle[i % len(prop_cycle)]

        perc_list = (
            [percentiles]
            if isinstance(percentiles, (int, float))
            else percentiles or []
        )
        for i, p in enumerate(perc_list):
            key = f"percentile_{p:.2f}"
            stats_labels[key] = f"Upper {p:.2f}th percentile"
            stats_colors[key] = prop_cycle[(i + len(stats_list)) % len(prop_cycle)]

        # Plot summary curves
        for key in stats_dict:
            if key == "bin_centers":
                continue
            g.ax_joint.plot(
                xarr,
                stats_dict[key],
                label=stats_labels.get(key, key),
                linewidth=2.5,
                color=stats_colors.get(key, "black"),
            )
        g.ax_joint.legend()

    # Axis labels
    if curvature_type.lower() == "gaussian":
        g.ax_joint.set_xlabel(r"Gaussian curvature ($\mathrm{\mu m^{-2}}$)")
    elif curvature_type.lower() == "mean":
        g.ax_joint.set_xlabel(r"Mean curvature ($\mathrm{\mu m^{-2}}$)")
    g.ax_joint.set_ylabel("Normalized intensity")

    if save_path:
        g.figure.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(g.figure)
    if return_stats:
        return g.figure, g.ax_joint, stats_dict
    return g.figure, g.ax_joint


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    import imcurvpy.curvature as curvature
    import imcurvpy.plots.curv_intensity_hist2d as curv_intensity_hist2d

    data = pl.read_csv("data/test_data.csv")

    print(data)

    gaussian_curvature = data["k1"] * data["k2"]

    max_curv = curvature.max_principal_curvature_limit(0.5)

    curv_intensity_hist2d.plot_curv_intensity_joint(
        gaussian_curvature,
        data["norm_intensity_ch2"],
        curvature_range=(0, max_curv),
        sns_style="darkgrid",
    )
    plt.show()
