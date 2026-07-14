from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit


def lamin_curvature_intensity_model(
    gaussian_curvature: npt.ArrayLike,
    energy_barrier: float,
    affinity: float,
    min_probability: float,
    max_probability: float,
    persistence_length: float = 0.38,
    filament_length: float = 0.5,
):
    r"""
    Theoretical model for intensity probability as a function of Gaussian curvature.

    This model is only valid for zero or positive Gaussian curvature (K >= 0),
    since it includes sqrt(K). Negative values are not allowed.

    All length-related parameters are in micrometers (um), and Gaussian curvature is in
    1/um^2.

    Parameters
    ----------
    gaussian_curvature : array-like
        Gaussian curvature (K), in 1/um^2. Must be zero or positive.
    energy_barrier : float
        Energy barrier parameter (E) (dimensionless).
    affinity : float
        Curvature affinity parameter (a) (dimensionless).
    min_probability : float
        Minimum value of intensity probability (lower plateau).
    max_probability : float
        Maximum value of intensity probability (upper plateau).
    persistence_length : float, optional
        Persistence length of the filament (l_p) in um, default is 0.38.
    filament_length : float, optional
        Length of the filament (L_fil) in um, default is 0.5.

    Returns
    -------
    float or np.ndarray
        Probability of high intensity as a function of Gaussian curvature.

    Raises
    ------
    ValueError
        If any value in gaussian_curvature is less than or equal to zero.

    Notes
    -----
    The default values for the persistence length (l_p) and the filament length are
    taken from Turgay et al. (2017) [1]_.

    Formula is based on Single-filament model of Pfeiffer et al. (2022) [2]_, but
    instead of computing P_detached, we compute the P_attached:

    .. math::

        P_{\\lamin}(K) = \frac{(P_{\\max} - P_{\\min}) \, e^x}{1 + e^x} + P_{\\min}

    where

    .. math::

        x = \\frac{E - a \\sqrt{K}}{k_B T} - \\frac{l_p L_{fil}}{2} K

    Here, :math:`P_{\\min}` and :math:`P_{\\max}` are the lower and upper plateaus of the
    probability, :math:`E` is the energy barrier, :math:`a` is the curvature affinity,
    :math:`K` is the Gaussian curvature, :math:`l_p` is the persistence length,
    and :math:`L_{fil}` is the filament length.

    References
    ----------
    .. [1] Turgay, Y. et al. The molecular architecture of lamins in somatic cells.
        Nature 543, 261-264 (2017).
    .. [2] Pfeifer, C. R. et al. Gaussian curvature dilutes the nuclear lamina,
        favoring nuclear rupture, especially at high strain rate. Nucleus 13,
        130-144 (2022).
    """
    K = np.asarray(gaussian_curvature)

    if np.any(K < 0):
        raise ValueError("Gaussian curvature must be zero or positive (K >= 0).")

    exponent = (
        energy_barrier
        - affinity * np.sqrt(K)
        - 0.5 * persistence_length * filament_length * K
    )
    prob = (
        (max_probability - min_probability) * np.exp(exponent) / (1 + np.exp(exponent))
    )
    result = prob + min_probability

    if np.isscalar(gaussian_curvature):
        return result.item()
    return result


def fit_data_to_model(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    sigma: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    persistence_length: float = 0.38,
    filament_length: float = 0.5,
    initial_guess: tuple[float, float, float, float] | None = None,
    bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    **curve_fit_kwargs,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | list[str] | list[float]]]:
    """
    Weighted fitting of the lamin curvature-intensity model.

    Parameters
    ----------
    x : array-like
        Gaussian curvature values (K), in 1/um^2. Must be >= 0.
    y : array-like
        Probability/intensity values [0, 1].
    sigma : array-like, optional
        Measurement errors (standard deviations) for each y point.
    weights : array-like, optional
        Statistical weights (inverse variance). Ignored if sigma is given.
    persistence_length : float, optional
        Persistence length of the filament (l_p) in um, default is 0.38.
    filament_length : float, optional
        Length of the filament (L_fil) in um, default is 0.5.
    initial_guess : tuple of 4 floats, optional
        Initial guess for (energy_barrier, affinity, min_probability, max_probability).
        If None, reasonable defaults will be used based on the data.
    bounds : tuple of two tuples, optional
        Lower and upper bounds for the parameters in the format:
        ((lower_bounds), (upper_bounds)) where each tuple contains 4 values
        corresponding to (energy_barrier, affinity, min_probability, max_probability).
        If None, reasonable defaults will be used.
    **curve_fit_kwargs
        Additional keyword arguments passed to scipy.optimize.curve_fit.

    Returns
    -------
    popt : np.ndarray
        Optimal parameters (energy_barrier, affinity, min_probability, max_probability).
    pcov : np.ndarray
        Covariance matrix of the parameters.
    fit_info : dict
        Dictionary containing additional fitting information:
        - 'rmse': root mean square error
        - 'mae': mean absolute error
        - 'max_abs_error': maximum absolute error
        - 'aic': Akaike Information Criterion
        - 'bic': Bayesian Information Criterion
        - 'parameter_names': list of parameter names
        - 'fitted_y': model prediction using fitted parameters
        - 'residuals': residuals (y - fitted_y)

    Raises
    ------
    ValueError
        If x contains negative values or if x and y have different lengths.
    RuntimeError
        If the fitting fails to converge.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Validate input lengths
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if np.any(x < 0):
        raise ValueError("Gaussian curvature values must be zero or positive")

    # Handle weighting logic
    if sigma is not None and weights is not None:
        raise ValueError("Provide either sigma or weights, not both")
    if sigma is not None:
        sigma = np.asarray(sigma)
        if sigma.shape != y.shape:
            raise ValueError("sigma must have the same shape as y")
    elif weights is not None:
        weights = np.asarray(weights)
        if weights.shape != y.shape:
            raise ValueError("weights must have the same shape as y")
        sigma = 1.0 / np.sqrt(weights)  # Convert regression weights to sigma

    # Model wrapper
    def model_wrapper(gaussian_curvature, energy_barrier, affinity, min_prob, max_prob):
        return lamin_curvature_intensity_model(
            gaussian_curvature,
            energy_barrier,
            affinity,
            min_prob,
            max_prob,
            persistence_length,
            filament_length,
        )

    # Initial guess defaults
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    if initial_guess is None:
        initial_guess = (1.0, 1.0, y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Bounds defaults
    if bounds is None:
        bounds = ((-30.0, -30.0, 0.0, 0.0), (30.0, 30.0, 1.0, 1.0))

    # Ensure guess within bounds
    lower_bounds, upper_bounds = bounds
    initial_guess = np.clip(initial_guess, lower_bounds, upper_bounds)

    try:
        # Fit
        popt, pcov = curve_fit(
            model_wrapper,
            x,
            y,
            p0=initial_guess,
            bounds=bounds,
            sigma=sigma,
            absolute_sigma=True if sigma is not None else False,
            **curve_fit_kwargs,
        )

        # Predictions & residuals
        fitted_y = model_wrapper(x, *popt)
        residuals = y - fitted_y

        # Weighted error metrics if sigma/weights given
        if sigma is not None:
            w = 1.0 / sigma**2
            rmse = np.sqrt(np.average(residuals**2, weights=w))
            mae = np.average(np.abs(residuals), weights=w)
            ssr = np.sum(w * residuals**2)
        else:
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            ssr = np.sum(residuals**2)

        n, p = len(y), len(popt)
        aic = n * np.log(ssr / n) + 2 * p
        bic = n * np.log(ssr / n) + p * np.log(n)
        max_abs_error = np.max(np.abs(residuals))

        fit_info = {
            "rmse": rmse,
            "mae": mae,
            "max_abs_error": max_abs_error,
            "aic": aic,
            "bic": bic,
            "ssr": ssr,
            "parameter_names": [
                "energy_barrier",
                "affinity",
                "min_probability",
                "max_probability",
            ],
            "fitted_y": fitted_y,
            "residuals": residuals,
            "persistence_length_used": persistence_length,
            "filament_length_used": filament_length,
            "n_points": n,
            "n_parameters": p,
        }

        return popt, pcov, fit_info

    except RuntimeError as e:
        raise RuntimeError(f"Curve fitting failed to converge: {e}") from e


def print_fit_results(
    popt: np.ndarray,
    pcov: np.ndarray,
    fit_info: Mapping[str, float | list[str] | list[float]],
):
    """
    Print formatted fitting results.

    Parameters
    ----------
    popt : np.ndarray
        Optimal parameters from fit_data_to_model.
    pcov : np.ndarray
        Covariance matrix from fit_data_to_model.
    fit_info : dict
        Fit information dictionary from fit_data_to_model.
    """
    param_names = fit_info["parameter_names"]
    param_errors = np.sqrt(np.diag(pcov))

    print("Fitting Results:")
    print("-" * 50)
    for name, value, error in zip(param_names, popt, param_errors, strict=True):
        print(f"{name:20s}: {value:10.4f} ± {error:8.4f}")

    print("\nGoodness of fit:")
    print(f"RMSE = {fit_info['rmse']:.4f}")
    print(f"MAE = {fit_info['mae']:.4f}")
    print(f"Max |error| = {fit_info['max_abs_error']:.4f}")
    print(f"AIC = {fit_info['aic']:.2f}")
    print(f"BIC = {fit_info['bic']:.2f}")

    print("\nData info:")
    print(f"Number of points: {fit_info['n_points']}")
    print(f"Number of parameters: {fit_info['n_parameters']}")

    print("\nFixed parameters:")
    print(f"Persistence length: {fit_info['persistence_length_used']:.3f} μm")
    print(f"Filament length: {fit_info['filament_length_used']:.3f} μm")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gaussian_curvatures = np.linspace(0, 3.0, num=100)

    probability = lamin_curvature_intensity_model(
        gaussian_curvatures,
        energy_barrier=5.0,
        affinity=8.2,
        min_probability=0,
        max_probability=1,
        persistence_length=0.38,
        filament_length=0.5,
    )

    plt.plot(gaussian_curvatures, probability)
    plt.xlabel("Gaussian curvature")
    plt.ylabel("Probability of lamin being present (Normalized intensity)")

    plt.show()
