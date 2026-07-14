"""
Adapted from author: sylvainprigent - https://github.com/sylvainprigent/sdeconv/blob/main/sdeconv/psfs/gibson_lanni.py.

Original notes:
Implementation of Gibson Lanni Point Spread Function model

This implementation is an adaptation of
https://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python/
"""

from math import sqrt

import numpy as np
import scipy.special
from scipy.interpolate import interp1d


class SPSFGibsonLanni:
    """
    Generate a Gibson-Lanni Point Spread Function (PSF) model.

    This class implements a numerical approximation of the Gibson-Lanni PSF,
    modeling optical aberrations in widefield and confocal microscopy.
    Adapted from:
    - https://github.com/sylvainprigent/sdeconv
    - https://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python/

    Parameters
    ----------
    shape : tuple of int
        Size of the PSF array in each dimension: (Z, Y, X) or (Y, X).
    NA : float, optional
        Numerical aperture of the objective lens. Default is 1.4.
    wavelength : float, optional
        Emission wavelength in microns. Default is 0.610.
    M : float, optional
        Magnification. Default is 100.
    ns : float, optional
        Specimen refractive index (RI). Default is 1.33.
    ng0 : float, optional
        Coverslip RI (design value). Default is 1.5.
    ng : float, optional
        Coverslip RI (experimental value). Default is 1.5.
    ni0 : float, optional
        Immersion medium RI (design value). Default is 1.5.
    ni : float, optional
        Immersion medium RI (experimental value). Default is 1.5.
    ti0 : float, optional
        Working distance (immersion medium thickness) in microns (design value). Default is 150.
    tg0 : float, optional
        Coverslip thickness in microns (design value). Default is 170.
    tg : float, optional
        Coverslip thickness in microns (experimental value). Default is 170.
    res_lateral : float, optional
        Lateral (XY) resolution in microns. Default is 0.1.
    res_axial : float, optional
        Axial (Z) resolution in microns. Default is 0.25.
    pZ : float, optional
        Distance from coverslip to particle in microns. Default is 0.
    use_square : bool, optional
        If True, square the resulting PSF to simulate a confocal pinhole effect. Default is False.

    Attributes
    ----------
    psf_ : np.ndarray or None
        Cached PSF volume. Generated after the object is called.
    """

    def __init__(
        self,
        shape: tuple[int, int] | tuple[int, int, int],
        NA: float = 1.4,
        wavelength: float = 0.610,
        M: float = 100,
        ns: float = 1.33,
        ng0: float = 1.5,
        ng: float = 1.5,
        ni0: float = 1.5,
        ni: float = 1.5,
        ti0: float = 150,
        tg0: float = 170,
        tg: float = 170,
        res_lateral: float = 0.1,
        res_axial: float = 0.25,
        pZ: float = 0,
        use_square: bool = False,
    ):
        super().__init__()
        self.shape = shape

        # Microscope parameters
        self.NA = NA
        self.wavelength = wavelength
        self.M = M
        self.ns = ns
        self.ng0 = ng0
        self.ng = ng
        self.ni0 = ni0
        self.ni = ni
        self.ti0 = ti0
        self.tg0 = tg0
        self.tg = tg
        self.res_lateral = res_lateral
        self.res_axial = res_axial
        self.pZ = pZ
        self.use_square = use_square
        # output
        self.psf_ = None

    def __call__(self) -> np.ndarray:
        """
        Compute the Gibson-Lanni PSF.

        Returns
        -------
        np.ndarray
            A 3D array representing the point spread function with shape (Z, Y, X).
        """
        # Precision control
        num_basis = (
            100  # Number of rescaled Bessels that approximate the phase function
        )
        num_samples = 1000  # Number of pupil samples along radial direction
        oversampling = (
            2  # Defines the sampling ratio on the image space grid for computations
        )

        size_x = self.shape[2]
        size_y = self.shape[1]
        size_z = self.shape[0]
        min_wavelength = 0.436  # microns
        scaling_factor = (
            self.NA
            * (3 * np.arange(1, num_basis + 1) - 2)
            * min_wavelength
            / self.wavelength
        )

        # Place the origin at the center of the final PSF array
        x0 = (size_x - 1) / 2
        y0 = (size_y - 1) / 2
        # Find the maximum possible radius coordinate of the PSF array by finding the distance
        # from the center of the array to a corner
        max_radius = (
            round(sqrt((size_x - x0) * (size_x - x0) + (size_y - y0) * (size_y - y0)))
            + 1
        )
        # Radial coordinates, image space
        r = self.res_lateral * np.arange(0, oversampling * max_radius) / oversampling
        # Radial coordinates, pupil space
        a = min([self.NA, self.ns, self.ni, self.ni0, self.ng, self.ng0]) / self.NA
        rho = np.linspace(0, a, num_samples)
        # Stage displacements away from best focus
        z = self.res_axial * np.arange(-size_z / 2, size_z / 2) + self.res_axial / 2

        # Define the wavefront aberration
        OPDs = self.pZ * np.sqrt(self.ns * self.ns - self.NA * self.NA * rho * rho)
        OPDi = (z.reshape(-1, 1) + self.ti0) * np.sqrt(
            self.ni * self.ni - self.NA * self.NA * rho * rho
        ) - self.ti0 * np.sqrt(
            self.ni0 * self.ni0 - self.NA * self.NA * rho * rho
        )  # OPD in the immersion medium
        OPDg = self.tg * np.sqrt(
            self.ng * self.ng - self.NA * self.NA * rho * rho
        ) - self.tg0 * np.sqrt(
            self.ng0 * self.ng0 - self.NA * self.NA * rho * rho
        )  # OPD in the coverslip
        W = 2 * np.pi / self.wavelength * (OPDs + OPDi + OPDg)

        # Sample the phase
        # Shape is (number of z samples by number of rho samples)
        phase = np.cos(W) + 1j * np.sin(W)

        # Define the basis of Bessel functions
        # Shape is (number of basis functions by number of rho samples)
        J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)

        # Compute the approximation to the sampled pupil phase by finding the least squares
        # solution to the complex coefficients of the Fourier-Bessel expansion.
        # Shape of C is (number of basis functions by number of z samples).
        # Note the matrix transposes to get the dimensions correct.
        C, residuals, _, _ = np.linalg.lstsq(J.T, phase.T, rcond=None)

        # compute the PSF
        b = 2 * np.pi * r.reshape(-1, 1) * self.NA / self.wavelength

        # Convenience functions for J0 and J1 Bessel functions
        def J0(x):
            return scipy.special.jv(0, x)

        def J1(x):
            return scipy.special.jv(1, x)

        # See equation 5 in Li, Xue, and Blu
        denom = scaling_factor * scaling_factor - b * b
        R = (
            scaling_factor * J1(scaling_factor * a) * J0(b * a) * a
            - b * J0(scaling_factor * a) * J1(b * a) * a
        )
        R /= denom

        # The transpose places the axial direction along the first dimension of the array, i.e. rows
        # This is only for convenience.
        PSF_rz = (np.abs(R.dot(C)) ** 2).T

        # Normalize to the maximum value
        PSF_rz /= np.max(PSF_rz)

        # cartesian PSF
        # Create the fleshed-out xy grid of radial distances from the center
        xy = np.mgrid[0:size_y, 0:size_x]
        r_pixel = (
            np.sqrt((xy[1] - x0) * (xy[1] - x0) + (xy[0] - y0) * (xy[0] - y0))
            * self.res_lateral
        )

        self.psf_ = np.zeros((size_z, size_y, size_x))

        for z_index in range(size_z):
            # Interpolate the radial PSF function
            PSF_interp = interp1d(r, PSF_rz[z_index, :])

            # Evaluate the PSF at each value of r_pixel
            self.psf_[z_index, :, :] = PSF_interp(r_pixel.ravel()).reshape(
                size_y, size_x
            )

        if self.use_square:
            self.psf_ = np.square(self.psf_)

        return self.psf_
