from __future__ import annotations

import imcurvpy.curvature as curvature
import imcurvpy.workflows as workflows

if __name__ == "__main__":
    # Data loading and saving settings
    # TODO: please provide the paths to the directories
    input_dir = "path/to/dir_data"
    labels_dir = "path/to/dir_data/dir_labels"  # If None, the segmentation is performed on the intensity # noqa: E501
    output_dir = "path/to/dir_data/analysis"  # Can be either None or a path, if None, it is automatically determined
    overwrite = True  # Allow overwriting of output directory

    channel_idx_lac = 0  # Channel index for Lamin A/C
    channel_idx_lb1 = 1  # Channel index for Lamin B1
    axis_order = "zyx"

    # DAPI options (just turned off now)
    sign_estimation_mode = None
    image_dapi = None
    spacing_dapi = None
    wavelength_dapi = None

    # Optics parameters
    wavelength_lac = 0.488  # Excitation wavelength for Lamin A/C (um)
    wavelength_lb1 = 0.561  # Excitation wavelength for Lamin B1 (um)
    na = 1.4  # Numerical aperture
    magnification = 63  # Objective magnification
    ri_sample = 1.33  # water
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
    pc_scale = 0.6  # um (scale over which the principal components are computed)

    # Local intensity computation
    ball_radius = 0.6  # um
    ball_stat = (
        "mean"  # what statistic to compute, can be "min", "max", "mean", "median"
    )
    ball_pad_mode = "reflect"

    # Parameters for nuclei segmentation
    expansion_distance = 4  # um
    min_nucleus_area = 100  # um^2

    if axis_order != "zyx":
        raise NotImplementedError(
            "Any axis order other than 'zyx' is not yet implemented."
        )

    print(
        "Maximum recoverable principal curvature: "
        f"{curvature.max_principal_curvature_limit(pc_scale):.3e} um^-1"
    )

    print("Start processing of all images in directory...")

    workflows.process_directory_principal_curvatures_multi_nuclei(
        input_dir,
        [channel_idx_lac, channel_idx_lb1],
        [wavelength_lac, wavelength_lb1],
        na,
        magnification,
        ri_sample,
        ri_coverslip,
        ri_immersion,
        confocal,
        rough_lamin_mask_sigma,
        rough_lamin_mask_ic_factor,
        rough_lamin_mask_alpha,
        fine_lamin_mask_sigma,
        pc_scale,
        expansion_distance=expansion_distance,
        min_nucleus_area=min_nucleus_area,
        axis_order=axis_order,
        reduction_mode=reduction_mode,
        spacing_goal=spacing_goal,
        order=order,
        downsample_mode=downsample_mode,
        pad_mode=pad_mode,
        ball_radius=ball_radius,
        ball_stat=ball_stat,
        labels_dir=labels_dir,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    print("Finished processing of all images in directory.")
