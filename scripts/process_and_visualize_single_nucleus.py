from __future__ import annotations

import napari
import numpy as np
from scipy.ndimage import gaussian_filter

import imcurvpy.analysis as analysis
import imcurvpy.curvature as curvature
import imcurvpy.io as io
import imcurvpy.preprocessing as preprocessing
import imcurvpy.segment as segment

if __name__ == "__main__":
    # Data loading settings
    # TODO: please provide the path to the image you would like to process/visualize
    path = "path/to/data/image.tif"

    channel_idx_lac = 0  # Channel index for Lamin A/C
    channel_idx_lb1 = 1  # Channel index for Lamin B1
    axis_order = "zyx"

    # Optics parameters
    wavelength_lac = 0.488  # Excitation wavelength for Lamin A/C (µm)
    wavelength_lb1 = 0.561  # Excitation wavelength for Lamin B1 (µm)
    na = 1.4  # Numerical aperture
    magnification = 63  # Objective magnification
    ri_sample = 1.33  # Water
    ri_coverslip = 1.52  # Refractive index of coverslip
    ri_immersion = 1.52  # Leica immersion oil (1.516)
    confocal = True

    # Preprocessing parameters
    spacing_goal = "max"  # results in (0.16 um, 0.16 um, 0.16 um) for the paper dataset
    order = 1
    downsample_mode = "sum"
    pad_mode = "constant"

    # Analysis parameters
    reduction_mode = "max"  # how to combine the different lamin channels
    rough_lamin_mask_sigma = 1.0  # um (blurring factor for mask creation)
    rough_lamin_mask_ic_factor = 0.4  # intensity threshold for mask creation
    rough_lamin_mask_alpha = 0.85  # Shape parameter for mask creation
    fine_lamin_mask_sigma = 0.35  # um (scale of steerable filter)
    pc_scale = 0.5  # um (scale over which the principal components are computed)

    # Local intensity computation
    ball_radius = 0.5  # um
    ball_stat = (
        "mean"  # what statistic to compute, can be "min", "max", "mean", "median"
    )
    ball_pad_mode = "reflect"

    # DAPI options (not really useable at the moment)
    sign_estimation_mode = None
    image_dapi = None
    spacing_dapi = None
    wavelength_dapi = None

    ######################################################
    print("Start data loading...")
    image_lac, spacing_lac, _ = io.read_tiff_select_indices(
        path, channel_idx=channel_idx_lac
    )

    image_lb1, spacing_lb1, _ = io.read_tiff_select_indices(
        path, channel_idx=channel_idx_lb1
    )

    spacings = (spacing_lac, spacing_lb1)

    # Determine isotropic voxel spacing
    # Normalize spacing_goal to a 3-tuple
    if isinstance(spacing_goal, (int, float)):
        spacing_goal = (spacing_goal,) * 3
    elif isinstance(spacing_goal, str):
        if spacing_goal in ("min", "max"):
            voxel_spacing = (
                min([item for arr in spacings for item in arr])
                if spacing_goal == "min"
                else max([item for arr in spacings for item in arr])
            )
            spacing_goal = (voxel_spacing,) * 3
        else:
            raise ValueError(
                f"spacing_goal string must be 'min' or 'max', got {spacing_goal}"
            )
    elif isinstance(spacing_goal, (tuple, list)):
        if len(spacing_goal) != 3:
            raise ValueError(
                f"spacing_goal tuple/list must have 3 elements, got {spacing_goal}"
            )
        spacing_goal = tuple(spacing_goal)  # ensure it's a tuple
    else:
        raise ValueError(
            "spacing_goal must be a float, a 3-tuple/list, or 'min'/'max', got "
            f"{spacing_goal}"
        )

    voxel_spacing = spacing_goal[0]  # Extract voxel spacing
    wavelength_goal = max(wavelength_lac, wavelength_lb1)

    # First perform normalization
    image_lac = preprocessing.normalize(
        image_lac, global_min=None, global_max=None, dtype=np.float32
    )
    image_lb1 = preprocessing.normalize(
        image_lb1, global_min=None, global_max=None, dtype=np.float32
    )

    print("Start making images isotropic...")
    image_lac, _ = preprocessing.make_image_fully_isotropic(
        image_lac,
        spacing_lac,
        wavelength_lac,
        na,
        magnification,
        ri_sample,
        ri_coverslip,
        ri_immersion,
        confocal,
        wavelength_goal=wavelength_goal,
        spacing_goal=spacing_goal,
        order=order,
        downsample_mode=downsample_mode,
        pad_mode=pad_mode,
    )
    image_lb1, _ = preprocessing.make_image_fully_isotropic(
        image_lb1,
        spacing_lb1,
        wavelength_lb1,
        na,
        magnification,
        ri_sample,
        ri_coverslip,
        ri_immersion,
        confocal,
        wavelength_goal=wavelength_goal,
        spacing_goal=spacing_goal,
        order=order,
        downsample_mode=downsample_mode,
        pad_mode=pad_mode,
    )
    print("Finished making images isotropic.")

    # Find the maximum of both lamin images
    image_combined = np.maximum(image_lac, image_lb1)

    print("Start masks creation...")
    # Create the rough and fine nuclear lamina masks
    rough_lamin_mask = segment.create_rough_lamin_mask(
        image_combined,
        voxel_spacing,
        rough_lamin_mask_sigma,
        rough_lamin_mask_ic_factor,
        rough_lamin_mask_alpha,
    )

    fine_lamin_mask = segment.create_fine_lamin_mask(
        image_combined,
        rough_lamin_mask,
        voxel_spacing,
        fine_lamin_mask_sigma,
    )
    print("Finished creation of masks.")

    if sign_estimation_mode == "dapi" and image_dapi is not None:
        # Normalize
        image_dapi = preprocessing.normalize(
            image_dapi, global_min=None, global_max=None, dtype=np.float32
        )
        # Make it fully isotropic
        image_dapi, _ = preprocessing.make_image_fully_isotropic(
            image_dapi,
            spacing_dapi,
            wavelength_dapi,
            na,
            magnification,
            ri_sample,
            ri_coverslip,
            ri_immersion,
            confocal,
            wavelength_goal=wavelength_goal,
            spacing_goal=spacing_goal,
            order=order,
            downsample_mode=downsample_mode,
            pad_mode=pad_mode,
        )

        # Create the dapi soft mask
        dapi_soft_mask = segment.create_soft_dapi_mask(
            image_dapi, pc_scale, voxel_spacing
        )
    else:
        dapi_soft_mask = None

    print(
        f"fine_lamin_mask limits: {fine_lamin_mask.astype(np.float32).min()} "
        f"{fine_lamin_mask.astype(np.float32).max()}"
    )

    print("Start computation of principal components...")

    fine_lamin_mask_blurred = gaussian_filter(fine_lamin_mask.astype(np.float32), 1)

    k1, k2 = curvature.compute_principal_curvatures(
        fine_lamin_mask_blurred,
        fine_lamin_mask,
        voxel_spacing,
        pc_scale,
        axis_order=axis_order,
        sign_estimation_mode=sign_estimation_mode,
        dapi=dapi_soft_mask,
    )

    print("Finished computation of principal components.")

    gaussian_curvature = k1 * k2  # Compute Gaussian curvature

    ball_radius_px = ball_radius / voxel_spacing
    local_intensity_lac = analysis.local_ball_stats(
        image_lac,
        fine_lamin_mask,
        radius=ball_radius_px,
        stat="mean",
        pad_mode="reflect",
    )
    local_intensity_lb1 = analysis.local_ball_stats(
        image_lb1,
        fine_lamin_mask,
        radius=ball_radius_px,
        stat="mean",
        pad_mode="reflect",
    )

    spacing = spacing_goal
    viewer = napari.Viewer()
    viewer.add_image(
        image_lac,
        name="Lamin AC",
        colormap="green",
        blending="additive",
        scale=spacing,
        visible=False,
    )
    viewer.add_image(
        image_lb1,
        name="Lamin B1",
        colormap="magenta",
        blending="additive",
        scale=spacing,
        visible=False,
    )

    viewer.add_image(
        image_combined,
        name="max(LAC, LB1)",
        colormap="green",
        blending="additive",
        scale=spacing,
        visible=False,
    )
    viewer.add_image(
        gaussian_curvature,
        name="Gaussian curvature",
        colormap="seismic",
        blending="additive",
        scale=spacing,
        visible=False,
    )
    viewer.add_image(
        rough_lamin_mask,
        name="Rough lamin mask",
        colormap="blue",
        blending="additive",
        scale=spacing,
        visible=False,
    )
    viewer.add_image(
        fine_lamin_mask,
        name="Fine lamin mask",
        colormap="magenta",
        blending="additive",
        scale=spacing,
        visible=False,
    )
    viewer.add_image(
        local_intensity_lac,
        name="Local intensity (LAC)",
        colormap="green",
        blending="additive",
        scale=spacing,
        visible=False,
    )
    viewer.add_image(
        local_intensity_lb1,
        name="Local intensity (LB1)",
        colormap="magenta",
        blending="additive",
        scale=spacing,
        visible=False,
    )

    napari.run()
