"""
Lamina mask segmentation and visualization with napari.

This script loads confocal microscopy data (Lamin A/C and Lamin B1 channels),
performs preprocessing including normalization and isotropic resampling, and
computes rough and fine nuclear lamina masks using custom segmentation tools.
A napari viewer is launched to interactively visualize the results and apply
parameterized segmentation through MagicGUI widgets.
"""

from __future__ import annotations

import napari
import napari.layers
import napari.types
import numpy as np
from magicgui import magicgui

import imcurvpy.curvature as curvature
import imcurvpy.io as io
import imcurvpy.preprocessing as preprocessing
import imcurvpy.segment as segment
import imcurvpy.steerable3d as steerable3d

if __name__ == "__main__":
    # Data loading settings
    # path_input = r"E:\Myron_confocal_data\260407_Data\alleen goede\260205_LaminAC_laminB1.lif - Series007.tif"  # "path/to/data/image.tif"
    path_input = r"E:\Myron_confocal_data\260407_Data\alleen goede\260205_LaminAC_laminB1.lif - Series008.tif"
    channel_indices = (
        0,
        1,
    )  # Channel index 0: Lamin B1, 1: Lamin A/C
    time_idx = None
    axis_order = "zyx"  # Only "zyx" is allowed at the moment

    # DAPI options
    sign_estimation_mode = None
    image_dapi = None
    spacing_dapi = None
    wavelength_dapi = None

    # Optics parameters
    wavelengths = (0.488, 0.561)  # um
    na = 1.4  # Numerical aperture
    magnification = 63  # Objective magnification
    ri_sample = 1.33  # Water
    ri_coverslip = 1.52  # Refractive index of coverslip
    ri_immersion = 1.52  # Leica immersion oil (1.516)
    confocal = True

    # Preprocessing parameters
    spacing_goal = "max"  # results in (0.16 um, 0.16 um, 0.16 um) for the paper dataset
    order = 1  # Interpolation order (0=nearest, 1=linear, 3=cubic). Default=1.
    downsample_mode = (
        "sum"  # Aggregation mode when downsampling: "sum", "mean", "min", "max".
    )
    pad_mode = "constant"  # Padding mode for downsampling if necessary.

    # Analysis parameters
    reduction_mode = "max"
    rough_lamin_mask_sigma = 1.0  # um (blurring factor for mask creation)
    rough_lamin_mask_ic_factor = 0.4  # intensity threshold for mask creation
    rough_lamin_mask_alpha = 0.85  # Shape parameter for mask creation
    fine_lamin_mask_sigma = 0.35  # um (scale of steerable filter)
    pc_scale = 0.6  # um (scale over which the principal components are computed)

    if axis_order != "zyx":
        raise NotImplementedError(
            "Any axis order other than 'zyx' is not yet implemented."
        )

    # Coerce single inputs into lists
    if isinstance(channel_indices, int):
        channel_indices = [channel_indices]
    if isinstance(wavelengths, (int, float)):
        wavelengths = [wavelengths]  # type: ignore

    # Validate channel_indices contains only integers
    if not all(isinstance(idx, int) for idx in channel_indices):
        raise ValueError("All elements in channel_indices must be integers")

    print(
        "Maximum recoverable principal curvature: "
        f"{curvature.max_principal_curvature_limit(pc_scale):.3e} um^-1"
    )

    print("Start data loading...")
    images = []
    spacings = []

    for channel_idx in channel_indices:
        image_temp, spacing_temp, _ = io.read_tiff_select_indices(
            path_input,
            channel_idx=channel_idx,
            time_idx=time_idx,
        )
        images.append(image_temp)
        spacings.append(spacing_temp)

    # Determine wavelength goal (for PSF matching)
    wavelength_goal = max(wavelengths)

    def isotropize(image, spacing, wavelength):
        image, spacing_result = preprocessing.make_image_fully_isotropic(
            image,
            spacing,
            wavelength,
            na=na,
            magnification=magnification,
            ri_sample=ri_sample,
            ri_coverslip=ri_coverslip,
            ri_immersion=ri_immersion,
            confocal=confocal,
            wavelength_goal=wavelength_goal,
            spacing_goal=spacing_goal,
            order=order,
            downsample_mode=downsample_mode,
            pad_mode=pad_mode,
        )
        return image, spacing_result

    print("Start normalizing images and making them isotropic...")
    for i in range(len(images)):
        images[i] = preprocessing.normalize(
            images[i], global_min=None, global_max=None, dtype=np.float32
        )
        images[i], spacing_goal = isotropize(images[i], spacings[i], wavelengths[i])
        # NOTE: that the intensity per voxel can now be higher than 1
        # images[i] = preprocessing.normalize(
        #     images[i], global_min=None, global_max=None, dtype=np.float32
        # )

    voxel_spacing = spacing_goal[0]

    print("Finished normalization and making images isotropic.")

    # Combine lamin images
    image_combined = preprocessing.reduce_3d_arrays(images, mode=reduction_mode)

    print("Start masks creation...")
    rough_lamin_mask = segment.create_rough_lamin_mask(
        image=image_combined,
        voxel_spacing=voxel_spacing,
        sigma=rough_lamin_mask_sigma,
        ic_factor=rough_lamin_mask_ic_factor,
        alpha=rough_lamin_mask_alpha,
    )

    #######################################################################
    # Mask creation widgets
    #######################################################################
    @magicgui(
        voxel_spacing={"min": 0.01, "max": 2.0, "step": 0.001},
        sigma={"min": 0.05, "max": 5.0, "step": 0.01},
        ic_factor={"min": 0.0, "max": 1.0, "step": 0.01},
        alpha={"min": 0.0, "max": 1.0, "step": 0.01},
        call_button="Create rough mask",
    )
    def rough_lamin_mask_widget(
        image_layer: napari.layers.Image,
        voxel_spacing: float = voxel_spacing,
        sigma: float = rough_lamin_mask_sigma,
        ic_factor: float = rough_lamin_mask_ic_factor,
        alpha: float = rough_lamin_mask_alpha,
    ) -> napari.types.LayerDataTuple:
        """
        Create a rough binary mask for lamin structures from an input image.

        Parameters
        ----------
        image_layer : napari.layers.Image
            The input image to segment.
        voxel_spacing : float
            Voxel spacing in micrometers; used for Gaussian smoothing.
        sigma : float
            Gaussian filter sigma for smoothing the image.
        ic_factor : float
            Intensity contrast factor used in the mask thresholding.
        alpha : float
            Weighting parameter that adjusts intensity thresholds.

        Returns
        -------
        LayerDataTuple
            A new image layer containing the rough lamin mask, with display settings.
        """
        image = image_layer.data
        print(image_layer)
        print("Starting rough mask creation...")
        mask = segment.create_rough_lamin_mask(
            image, voxel_spacing, sigma, ic_factor, alpha
        )
        print("Finished creating rough mask.")
        # Set all properties to same properties as input mask
        return (
            mask,
            {
                "name": "Rough mask",
                "blending": "additive",
                "opacity": 1.0,
                "scale": spacing_goal,
                "colormap": "green",
            },
            "image",
        )

    @magicgui(
        voxel_spacing={"min": 0.01, "max": 2.0, "step": 0.001},
        sigma={"min": 0.05, "max": 5.0, "step": 0.01},
        zx_ratio={"min": 0.05, "max": 3.0, "step": 0.01},
        call_button="Create fine mask",
    )
    def fine_lamin_mask_widget(
        image_layer: napari.layers.Image,
        rough_mask_layer: napari.layers.Image,
        voxel_spacing: float = voxel_spacing,
        sigma: float = fine_lamin_mask_sigma,
        zx_ratio: float = 1.0,
    ) -> napari.types.LayerDataTuple:
        """
        Refine a rough lamin mask to create a fine, high-accuracy segmentation.

        Parameters
        ----------
        image_layer : napari.layers.Image
            The input image from which to refine the segmentation.
        rough_mask_layer : napari.layers.Image
            An existing rough mask to constrain the fine segmentation.
        voxel_spacing : float
            Voxel spacing in micrometers; used for Gaussian smoothing.
        sigma : float
            Sigma for Gaussian filter applied during fine segmentation.

        Returns
        -------
        LayerDataTuple
            A new image layer containing the fine lamin mask, with display settings.
        """
        image = image_layer.data
        rough_mask = rough_mask_layer.data
        print("Starting fine mask creation...")
        # sigma_px = sigma / voxel_spacing
        # _, _, mask = steerable3d.steerable_detector_3d(
        #     image, m=2, sigma=sigma_px, verbose=True
        # )
        # print(response.shape)
        mask = segment.create_fine_lamin_mask(
            image, rough_mask, voxel_spacing, sigma, zx_ratio=zx_ratio
        )
        print("Finished creating fine mask.")
        return (
            mask,
            {
                "name": "Fine mask",
                "blending": "additive",
                "opacity": 1.0,
                "scale": spacing_goal,
                "colormap": "magenta",
            },
            "image",
        )

    ###################################################################

    viewer = napari.Viewer()
    viewer.add_image(
        image_combined,
        name="Nuclear lamina",
        colormap="inferno",
        blending="additive",
        scale=spacing_goal,
    )
    viewer.add_image(
        rough_lamin_mask,
        name="Rough mask",
        colormap="green",
        blending="additive",
        scale=spacing_goal,
    )
    viewer.window.add_dock_widget(rough_lamin_mask_widget, area="right")
    viewer.window.add_dock_widget(fine_lamin_mask_widget, area="right")
    napari.run()
