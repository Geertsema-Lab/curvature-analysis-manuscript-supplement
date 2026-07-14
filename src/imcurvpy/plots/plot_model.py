from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import truncnorm

from imcurvpy.model import (
    lamin_curvature_intensity_model,
    fit_data_to_model,
    print_fit_results,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt


def plot_data_with_model(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    energy_barrier: float,
    affinity: float,
    min_probability: float,
    max_probability: float,
    persistence_length: float = 0.38,
    filament_length: float = 0.5,
    yerr: npt.ArrayLike | None = None,
    sns_style: str = "darkgrid",
    sns_context: str = "paper",
    figsize: tuple[int, int] | str = "auto",
    save_path: Path | str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the lamin curvature-intensity probability model alongside experimental data.

    Parameters
    ----------
    x : array-like
        Gaussian curvature values (1/μm²), must be non-negative.
    y : array-like
        Observed probability or intensity values corresponding to `x`.
    energy_barrier : float
        Energy barrier parameter for the model.
    affinity : float
        Affinity parameter for the model.
    min_probability : float
        Minimum predicted probability (y-intercept).
    max_probability : float
        Maximum predicted probability.
    persistence_length : float, optional
        Filament persistence length in μm. Default is 0.38.
    filament_length : float, optional
        Filament length in μm. Default is 0.5.
    yerr : array-like or None, optional
        Symmetric y-error values for each data point. If None, no error bars are shown.
    sns_style : str, optional
        Seaborn style (e.g., "darkgrid", "whitegrid"). Default is "darkgrid".
    sns_context : str, optional
        Seaborn scaling context (e.g., "paper", "talk"). Default is "paper".
    figsize : tuple[int, int] or "auto", optional
        Figure size in inches. If "auto", defaults to (8, 6).
    save_path : Path or str or None, optional
        Path to save the figure. If None, figure is not saved. Default is None.
    show : bool, optional
        Whether to display the plot after creation. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object.
    """
    import numbers

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1:
        raise ValueError("`x` must be a 1D array-like.")
    if y.ndim != 1:
        raise ValueError("`y` must be a 1D array-like.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("`x` and `y` must have the same length.")
    if np.any(x < 0):
        raise ValueError("Gaussian curvature values in `x` must be non-negative.")

    if yerr is not None:
        yerr = np.asarray(yerr)
        if yerr.shape != y.shape:
            raise ValueError("`yerr` must have the same shape as `y`.")
        if np.any(yerr < 0):
            raise ValueError("`yerr` values must be non-negative.")

    for param_name, param_val in [
        ("energy_barrier", energy_barrier),
        ("affinity", affinity),
        ("min_probability", min_probability),
        ("max_probability", max_probability),
        ("persistence_length", persistence_length),
        ("filament_length", filament_length),
    ]:
        if not isinstance(param_val, numbers.Real):
            raise TypeError(f"`{param_name}` must be a real number.")

    if not isinstance(sns_style, str):
        raise TypeError("`sns_style` must be a string.")
    if not isinstance(sns_context, str):
        raise TypeError("`sns_context` must be a string.")

    if figsize != "auto":
        if (
            not isinstance(figsize, tuple)
            or len(figsize) != 2
            or not all(isinstance(val, numbers.Real) for val in figsize)
            or not all(val > 0 for val in figsize)
        ):
            raise ValueError(
                "`figsize` must be 'auto' or a tuple of two positive numbers."
            )

    if save_path is not None and not isinstance(save_path, (str, Path)):
        raise TypeError("`save_path` must be None, a string, or a pathlib.Path.")

    if not isinstance(show, bool):
        raise TypeError("`show` must be a boolean.")

    if figsize == "auto":
        figsize = (8, 6)

    sns.set_style(sns_style)
    sns.set_context(sns_context)

    data_marker_size = 3  # same as in the other function

    fig, ax = plt.subplots(figsize=figsize)

    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    x_lower = max(0, x_min - 0.05 * x_range)  # no negative lower bound
    x_upper = x_max + 0.05 * x_range
    x_model = np.linspace(x_lower, x_upper, 200)

    y_model = lamin_curvature_intensity_model(
        x_model,
        energy_barrier,
        affinity,
        min_probability,
        max_probability,
        persistence_length=persistence_length,
        filament_length=filament_length,
    )

    ax.errorbar(
        x, y, yerr=yerr, fmt="o", markersize=data_marker_size, label="Data", zorder=2
    )
    ax.plot(x_model, y_model, label="Fitted probability model", zorder=3)

    ax.set_xlabel("Gaussian curvature (1/μm²)")
    ax.set_ylabel("Intensity probability")
    ax.legend()
    ax.grid(True, alpha=0.5)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_data_with_model_and_residuals(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    energy_barrier: float,
    affinity: float,
    min_probability: float,
    max_probability: float,
    persistence_length: float = 0.38,
    filament_length: float = 0.5,
    yerr: npt.ArrayLike | None = None,
    sns_style: str = "darkgrid",
    sns_context: str = "paper",
    figsize: tuple[int, int] | str = "auto",
    save_path: Path | str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot lamin curvature-intensity model with experimental data and residuals subplot.

    The top subplot shows the experimental data (optionally with error bars) together
    with the fitted probability model curve. The bottom subplot shows the residuals
    (data minus model) along with reference lines at zero and ±2*std.

    Parameters
    ----------
    x : array-like
        Gaussian curvature values (1/μm²). Must be non-negative and 1D.
    y : array-like
        Observed probability or intensity values corresponding to `x`. Must be 1D.
    energy_barrier : float
        Energy barrier parameter for the lamin probability model.
    affinity : float
        Affinity parameter for the lamin probability model.
    min_probability : float
        Minimum predicted probability (y-intercept of the model).
    max_probability : float
        Maximum predicted probability of the model.
    persistence_length : float, optional
        Filament persistence length in μm. Default is 0.38 μm.
    filament_length : float, optional
        Filament length in μm. Default is 0.5 μm.
    yerr : array-like or None, optional
        Symmetric y-error values for each data point. Must have the same shape as `y`.
        If None, no error bars are shown.
    sns_style : str, optional
        Seaborn style to apply (e.g., "darkgrid", "whitegrid"). Default is "darkgrid".
    sns_context : str, optional
        Seaborn scaling context (e.g., "paper", "talk"). Default is "paper".
    figsize : tuple[int, int] or "auto", optional
        Size of the matplotlib figure in inches. If "auto", defaults to (8, 8).
    save_path : Path or str or None, optional
        Path to save the figure. If None, the figure is not saved.
    show : bool, optional
        Whether to display the plot immediately. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plots.
    axes : tuple of matplotlib.axes.Axes
        A tuple `(ax1, ax2)` where:
        - `ax1` is the top subplot (data + model curve)
        - `ax2` is the bottom subplot (residuals plot)

    Notes
    -----
    The residual standard deviation is computed as:

        std_resid = std(y - y_model)

    and ±2*std reference lines are plotted in the residual subplot to help identify
    systematic deviations from the model.

    Examples
    --------
    >>> fig, (ax1, ax2) = plot_data_with_model_and_residuals(
    ...     x_data,
    ...     y_data,
    ...     energy_barrier=5.0,
    ...     affinity=0.3,
    ...     min_probability=0.05,
    ...     max_probability=0.9,
    ...     yerr=y_errors,
    ... )
    """
    import numbers

    # --- Convert to arrays and validate ---
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1:
        raise ValueError("`x` must be a 1D array-like.")
    if y.ndim != 1:
        raise ValueError("`y` must be a 1D array-like.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("`x` and `y` must have the same length.")
    if np.any(x < 0):
        raise ValueError("Gaussian curvature values in `x` must be non-negative.")

    if yerr is not None:
        yerr = np.asarray(yerr)
        if yerr.shape != y.shape:
            raise ValueError("`yerr` must have the same shape as `y`.")
        if np.any(yerr < 0):
            raise ValueError("`yerr` values must be non-negative.")

    for param_name, param_val in [
        ("energy_barrier", energy_barrier),
        ("affinity", affinity),
        ("min_probability", min_probability),
        ("max_probability", max_probability),
        ("persistence_length", persistence_length),
        ("filament_length", filament_length),
    ]:
        if not isinstance(param_val, numbers.Real):
            raise TypeError(f"`{param_name}` must be a real number.")

    if not isinstance(sns_style, str):
        raise TypeError("`sns_style` must be a string.")
    if not isinstance(sns_context, str):
        raise TypeError("`sns_context` must be a string.")

    if figsize != "auto":
        if (
            not isinstance(figsize, tuple)
            or len(figsize) != 2
            or not all(isinstance(val, numbers.Real) for val in figsize)
            or not all(val > 0 for val in figsize)
        ):
            raise ValueError(
                "`figsize` must be 'auto' or a tuple of two positive numbers."
            )

    if save_path is not None and not isinstance(save_path, (str, Path)):
        raise TypeError("`save_path` must be None, a string, or a pathlib.Path.")

    if not isinstance(show, bool):
        raise TypeError("`show` must be a boolean.")

    if figsize == "auto":
        figsize = (8, 6)

    sns.set_style(sns_style)
    sns.set_context(sns_context)

    data_marker_size = 3
    scatter_marker_size = data_marker_size**2

    # --- Create figure with two subplots ---
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        sharex=False,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # --- Model curve ---
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    x_lower = max(0, x_min - 0.05 * x_range)  # ensure >= 0
    x_upper = x_max + 0.05 * x_range
    x_model = np.linspace(x_lower, x_upper, 200)
    y_model = lamin_curvature_intensity_model(
        x_model,
        energy_barrier,
        affinity,
        min_probability,
        max_probability,
        persistence_length=persistence_length,
        filament_length=filament_length,
    )

    # --- Plot data + model (top panel) ---
    ax1.errorbar(
        x, y, yerr=yerr, fmt="o", markersize=data_marker_size, label="Data", zorder=2
    )
    ax1.plot(x_model, y_model, label="Fitted probability model", zorder=3)
    ax1.set_xlabel("Gaussian curvature (1/μm²)")
    ax1.set_ylabel("Intensity probability")
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # --- Residuals (bottom panel) ---
    y_fit = lamin_curvature_intensity_model(
        x,
        energy_barrier,
        affinity,
        min_probability,
        max_probability,
        persistence_length=persistence_length,
        filament_length=filament_length,
    )
    residuals = y - y_fit

    ax2.scatter(x, residuals, alpha=1.0, s=scatter_marker_size)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.7)

    residual_std = np.std(residuals)
    ax2.axhline(
        y=2 * residual_std,
        color="gray",
        linestyle="--",
        alpha=0.8,
        label=f"±2\u03c3 ({2 * residual_std:.4f})",
    )
    ax2.axhline(y=-2 * residual_std, color="gray", linestyle="--", alpha=0.7)
    ylim = ax2.get_ylim()
    yrange = ylim[1] - ylim[0]
    new_ylim = (ylim[0] - 0.15 * yrange, ylim[1] + 0.15 * yrange)
    ax2.set_ylim(new_ylim)

    ax2.set_xlabel("Gaussian curvature (1/μm²)")
    ax2.set_ylabel("Residuals (data - fit)")
    # ax2.set_title("Residuals Plot")
    ax2.grid(True, alpha=0.5)
    ax2.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, (ax1, ax2)


if __name__ == "__main__":
    # Generate some example data for demonstration
    np.random.seed(42)
    x_example = np.linspace(0.01, 3.0, 50)

    # True parameters for generating synthetic data
    true_params = (5.7, 8, 0.1, 0.9)  # energy_barrier, affinity, min_prob, max_prob
    y_true = lamin_curvature_intensity_model(x_example, *true_params)

    # Add some noise
    y_example = y_true + np.random.normal(0, 0.05, len(y_true))

    # Fit the model
    try:
        sigma = np.abs(y_example - y_true)
        # sigma = None
        popt, pcov, fit_info = fit_data_to_model(x_example, y_example, sigma=sigma)
        print_fit_results(popt, pcov, fit_info)

        print(f"\nTrue parameters: {true_params}")
        print(f"Fitted parameters: {popt}")

    except Exception as e:
        print(f"Fitting failed: {e}")

    plot_data_with_model_and_residuals(x_example, y_example, *popt, yerr=sigma)
