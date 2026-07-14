"""Nuclear lamina mask segmentation and visualization with napari.

This script loads confocal microscopy data (Lamin A/C and Lamin B1 channels),
performs preprocessing including normalization and PSF matching across channels,
and computes nuclear lamina masks using interactive segmentation tools.
A napari viewer is launched to interactively visualize the results and apply
parameterized segmentation through MagicGUI widgets.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import napari
import napari.layers
import napari.types
import numpy as np
from magicgui import magicgui
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border, expand_labels

import imcurvpy.io as io
import imcurvpy.preprocessing as preprocessing

if TYPE_CHECKING:
    import pathlib


def create_segmentation_widgets(
    voxel_spacing: float,
    labels_axes: str,
    min_nucleus_area: float,
    expansion_distance: float,
    filename: str | pathlib.Path | None = None,
):
    """Create MagicGUI widgets for interactive segmentation.

    Parameters
    ----------
    voxel_spacing : float
        The voxel spacing in micrometers (µm) for the image data.
    min_nucleus_area : float
        Minimum nucleus area in square micrometers (µm²) to include in the segmentation.
    expansion_distance : float
        Distance in micrometers (µm) to expand each segmented label outward.
    filename : str or pathlib.Path or None, optional
        Default output file path for saving segmentation results.
        If None, the home directory will be used as the initial location
        when the save dialog opens.
    """
    global rough_nuclei_segmentation, relabel_labels, expand_current_labels
    global create_valid_mask, save_valid_mask

    if filename is None:
        filename = Path.home()

    @magicgui(
        voxel_spacing={"min": 0.01, "max": 2.0, "step": 0.001},
        min_nucleus_area={"min": 1, "max": 1000, "step": 1},
        call_button="Create rough mask",
    )
    def rough_nuclei_segmentation(
        image_layer: napari.layers.Image,
        voxel_spacing: float = voxel_spacing,
        min_nucleus_area: float = min_nucleus_area,
    ) -> napari.types.LayerDataTuple:
        """Create rough nuclear segmentation using Otsu thresholding.

        Parameters
        ----------
        image_layer : napari.layers.Image
            Input image layer containing nuclear lamina signal.
        voxel_spacing : float, default from global
            Voxel spacing in micrometers for area calculations.
        min_nucleus_area : float, default from global
            Minimum nucleus area in square micrometers to retain.

        Returns
        -------
        napari.types.LayerDataTuple
            Tuple containing (labels_data, layer_properties, layer_type).
        """
        image = image_layer.data

        print("Start rough nuclei segmentation...")
        # Threshold using Otsu's method
        threshold = threshold_otsu(image)
        nuclei_mask = image > threshold

        nuclei_mask = binary_fill_holes(nuclei_mask)

        # Convert minimum area from um^2 to pixels^2
        min_area_px = min_nucleus_area / voxel_spacing**2

        # Remove small objects
        nuclei_mask = remove_small_objects(nuclei_mask, min_size=min_area_px)

        nuclei_labels = label(nuclei_mask)
        print("Finished rough nuclei segmentation.")
        return (
            nuclei_labels,
            {
                "name": "Labels",
                "blending": "additive",
                "opacity": 1.0,
                "scale": (voxel_spacing, voxel_spacing),
            },
            "labels",
        )

    @magicgui(call_button="Re-label labels")
    def relabel_labels(
        labels_layer: napari.layers.Labels,
    ) -> napari.types.LayerDataTuple:
        """Re-label the current labels layer with consecutive integers.

        Parameters
        ----------
        labels_layer : napari.layers.Labels
            Input labels layer to be re-labeled.

        Returns
        -------
        napari.types.LayerDataTuple
            Updated labels layer data tuple.
        """
        # Get the data
        nuclei_labels = labels_layer.data

        print("Start re-labelling of labels layer...")
        nuclei_mask = nuclei_labels > 0

        # Label connected components
        nuclei_labels = label(nuclei_mask)

        labels_layer.data = nuclei_labels
        print("Finished re-labelling.")

    @magicgui(
        voxel_spacing={"min": 0.01, "max": 2.0, "step": 0.001},
        expansion_distance={"min": 0.1, "max": 10.0, "step": 0.1},
        call_button="Expand labels",
    )
    def expand_current_labels(
        labels_layer: napari.layers.Labels,
        voxel_spacing: float = voxel_spacing,
        expansion_distance: float = expansion_distance,
    ) -> napari.types.LayerDataTuple:
        """Expand current labels using Voronoi tessellation.

        Parameters
        ----------
        labels_layer : napari.layers.Labels
            Input labels layer to expand.
        voxel_spacing : float, default from global
            Voxel spacing in micrometers for distance calculations.
        expansion_distance : float, default from global
            Distance to expand labels in micrometers.

        Returns
        -------
        napari.types.LayerDataTuple
            Updated labels layer data tuple.
        """
        nuclei_labels = labels_layer.data

        # Convert expansion distance from physical units to pixels
        expansion_distance_px = expansion_distance / voxel_spacing

        print("Start label expansion...")
        # Expand labels using Voronoi tessellation
        expanded_labels = expand_labels(nuclei_labels, distance=expansion_distance_px)

        print("Finished label expansion.")
        labels_layer.data = expanded_labels

    @magicgui(
        voxel_spacing={"min": 0.01, "max": 2.0, "step": 0.001},
        min_nucleus_area={"min": 1, "max": 1000, "step": 1},
        call_button="Create valid mask",
    )
    def create_valid_mask(
        image_layer: napari.layers.Image,
        labels_layer: napari.layers.Labels,
        voxel_spacing: float = voxel_spacing,
        min_nucleus_area: float = min_nucleus_area,
    ):
        """Create valid mask by filtering labels based on image threshold and borders.

        Parameters
        ----------
        image_layer : napari.layers.Image
            Input image layer for threshold-based filtering.
        labels_layer : napari.layers.Labels
            Input labels layer to filter.
        voxel_spacing : float, default from global
            Voxel spacing in micrometers for area calculations.
        min_nucleus_area : float, default from global
            Minimum nucleus area in square micrometers to retain.
        """
        # Retrieve data
        image = image_layer.data
        nuclei_labels = labels_layer.data

        print("Start rough nuclei segmentation...")

        # Threshold using Otsu's method
        threshold = threshold_otsu(image)
        nuclei_mask = image > threshold

        nuclei_mask = binary_fill_holes(nuclei_mask)

        # Convert minimum area from um^2 to pixels^2
        min_area_px = min_nucleus_area / voxel_spacing**2

        # Remove small objects
        nuclei_mask = remove_small_objects(nuclei_mask, min_size=min_area_px)

        # Remove objects touching the image border
        mask_no_border = clear_border(nuclei_mask)

        # Collect valid (non-zero) label IDs (zero is background)
        valid_labels = set(np.unique(nuclei_labels[mask_no_border]))
        if 0 in valid_labels:
            valid_labels.remove(0)

        new_labels = list(range(1, len(valid_labels) + 1))

        print(f"Current valid labels: {valid_labels}")
        print(f"New labels: {new_labels}")

        # Create a new labels array with only the valid labels
        nuclei_labels_final = np.zeros_like(nuclei_labels)
        for valid_label, new_label in zip(valid_labels, new_labels, strict=True):
            nuclei_labels_final[nuclei_labels == valid_label] = new_label

        print("Finished creating valid mask.")
        labels_layer.data = nuclei_labels_final

    @magicgui(
        filename={"mode": "w", "label": "Choose file save path_input:"},
        call_button="Save mask to file",
    )
    def save_valid_mask(
        labels_layer: napari.layers.Labels,
        filename: pathlib.Path = filename,
    ) -> None:
        """Save labels layer data to TIFF file with proper spacing metadata.

        Parameters
        ----------
        labels_layer : napari.layers.Labels
            Input labels layer to save.
        filename : pathlib.Path
            Output file path_input for saving the mask.
        """
        if not filename:  # Covers None, empty string, etc.
            print("No filename provided. Skipping save.")
            return

        nuclei_labels = labels_layer.data

        if nuclei_labels.min() < 0:
            raise ValueError("Labels lower than 0 are not allowed.")

        print("Update labels...")
        # Collect valid (non-zero) label IDs (zero is background)
        valid_labels = set(np.unique(nuclei_labels))
        if 0 in valid_labels:
            valid_labels.remove(0)

        new_labels = list(range(1, len(valid_labels) + 1))

        print(f"Current valid labels: {valid_labels}")
        print(f"New labels: {new_labels}")

        # Create a new labels array with only the valid labels
        nuclei_labels_final = np.zeros_like(nuclei_labels)
        for valid_label, new_label in zip(valid_labels, new_labels, strict=True):
            nuclei_labels_final[nuclei_labels == valid_label] = new_label

        print("Finished updating to new labels.")
        labels_layer.data = nuclei_labels_final

        filename = Path(filename)

        if not filename.parent.exists():
            print(
                f"Directory:('{filename.parent}') does not exist, creating directory."
            )
            filename.parent.mkdir(parents=True, exist_ok=True)

        print(f"Start saving mask to file: {filename}")
        data = labels_layer.data

        if data.max() < 2**8:
            data = data.astype(np.uint8)
        elif data.max() < 2**16:
            data = data.astype(np.uint16)

        # Note: 'spacing' needs to be available in scope
        spacing_2d = (voxel_spacing, voxel_spacing)
        io.write_tiff_with_spacing(
            filename,
            data,
            spacing_2d,
            labels_axes,
            unit="um",
        )
        print("Finished saving of mask.")


if __name__ == "__main__":
    # Data loading settings
    path_input = "path/to/data/image.tif"  # path to a .tif file
    output_dir = "path/to/data/labels"  # output_dir can also be None, then the default path is the home directory # noqa: E501

    channel_indices = (
        0,
        1,
    )  # Channel index 1: Lamin A/C channel, while 2: Lamin B1
    axis_order = "zyx"

    # DAPI options

    # Optics parameters
    wavelengths = (0.488, 0.561)  # um
    na = 1.4  # Numerical aperture
    magnification = 63  # Objective magnification
    ri_sample = 1.33  # Water
    ri_coverslip = 1.52  # Refractive index glass
    ri_immersion = 1.52  # Leica immersion oil (1.516)
    confocal = True

    # Analysis parameters
    reduction_mode = "max"
    min_nucleus_area = 50  # um^2
    expansion_distance = 4  # um

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

    path_input = Path(path_input)
    path_output = Path.home()
    if output_dir is not None:
        output_dir = Path(output_dir)
        path_output = output_dir / f"{path_input.stem}_labels.tif"

    print("Start data loading...")
    images = []
    spacings = []

    for channel_idx in channel_indices:
        image_temp, spacing_temp, axes = io.read_tiff_select_indices(
            path_input, channel_idx=channel_idx
        )
        images.append(image_temp)
        spacings.append(spacing_temp)

    # Determine isotropic voxel spacing
    voxel_spacing = max([elem for spacing in spacings for elem in spacing])

    # Determine wavelength goal (for PSF matching)
    wavelength_goal = max(wavelengths)

    # We do not need to isotropize in the z-direction as we will be taking the
    # maximum intensity projection
    print("Start normalizing images and making them isotropic in the xy-direction...")
    for i in range(len(images)):
        images[i] = preprocessing.normalize(
            images[i], global_min=None, global_max=None, dtype=np.float32
        )

        if wavelengths[i] < wavelength_goal:
            # Compute the sigmas of the PSF of the provided image
            sigma_lat_current, _sigma_ax_current = preprocessing.estimate_psf_sigmas(
                wavelengths[i],
                na,
                magnification,
                ns=ri_sample,
                ng=ri_coverslip,
                ni=ri_immersion,
                confocal=confocal,
            )

            sigma_lat_goal, _sigma_ax_goal = preprocessing.estimate_psf_sigmas(
                wavelength_goal,
                na,
                magnification,
                ns=ri_sample,
                ng=ri_coverslip,
                ni=ri_immersion,
                confocal=confocal,
            )

            images[i] = preprocessing.blur_xy_to_match_lateral_resolution(
                images[i], sigma_lat_current, sigma_lat_goal
            )

    print("Finished normalization and making them isotropic in the xy-direction.")

    # Combine lamin images
    image_combined = preprocessing.reduce_3d_arrays(images, mode=reduction_mode)

    # Compute maximum intensity projection along z-axis
    z_idx = axes.lower().index("z")
    max_proj = np.max(image_combined, axis=z_idx)

    # Only take yx axes, since it is a maximum intensity projection along the z-axis
    labels_axes = axes[:z_idx] + axes[z_idx + 1 :]

    spacing = (voxel_spacing, voxel_spacing)

    # Create segmentation widgets and launch napari viewer
    create_segmentation_widgets(
        voxel_spacing,
        labels_axes.upper(),
        min_nucleus_area,
        expansion_distance,
        filename=path_output,
    )

    # Launch napari and add the dock widgets
    viewer = napari.Viewer()
    viewer.add_image(
        max_proj,
        name="Nuclear lamina",
        colormap="inferno",
        blending="additive",
        scale=spacing,
    )
    viewer.window.add_dock_widget(rough_nuclei_segmentation, area="right")
    viewer.window.add_dock_widget(relabel_labels, area="right")
    viewer.window.add_dock_widget(expand_current_labels, area="right")
    viewer.window.add_dock_widget(create_valid_mask, area="right")
    viewer.window.add_dock_widget(save_valid_mask, area="right")
    napari.run()
