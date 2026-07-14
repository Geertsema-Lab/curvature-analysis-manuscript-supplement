# README

## General Info

This repository contains the code for performing **3D confocal curvature-intensity analysis** of nuclear lamina data, as used in: **TODO (add paper reference).**

It includes:

- Core analysis modules (`src/imcurvpy`)
- Reproducible processing scripts (`scripts`)
- Interactive Napari-based tools (`widgets`)

---

## Installation & Preparation

### 1. Download code and data

- Download the repository:
  - Either via the green button: **Code -> Download ZIP**
  - Clone with git
- Download the dataset (raw intensity stacks + labels) from: **TODO (add data link)**

### 2. Environment setup (recommended: `uv`)

We recommend using `uv`, a fast Python environment manager.

Install `uv`: <https://docs.astral.sh/uv/getting-started/installation/>

Then, in the **root folder (where `curvature-analysis-manuscript-supplement/` is located)**:

``` bash
uv sync
```

This will:

- Create a virtual environment
- Install exact dependencies used for the paper from the lockfile

> Alternatively, you can use `requirements.txt` with pip, but `uv` is recommended.

---

## Usage

### General Notes

In all scripts/widgets, update paths like:

```python
path = "path/to/data/image.tif"
input_dir = "path/to/data"
labels_dir = "path/to/data/labels"
```

to match your local setup.

---

## Scripts

### 1. Process & visualize a single nucleus

File: `scripts/process_and_visualize_single_nucleus.py`

Update:

```python
path = "path/to/data/image.tif"
```

Run:

```bash
uv run scripts/process_and_visualize_single_nucleus.py
```

This script:

- Loads a single confocal stack (Lamin A/C + Lamin B1)
- Normalizes and isotropizes the data
- Computes:
  - Rough and fine lamina masks
  - Principal curvatures (k1, k2)
  - Gaussian curvature (k1 x k2)
  - Local intensity statistics
- Opens a **Napari viewer** to explore all intermediate and final results

### 2. Process full dataset (batch processing)

File: `scripts/process_data.py`

Update:

```python
input_dir = "path/to/data"
labels_dir = "path/to/data/labels"  # or None to segment automatically
```

Run:

```bash
uv run scripts/process_data.py
```

This script:

- Processes **all images in a directory**
- Optionally uses precomputed nucleus labels
- Otherwise performs segmentation automatically
- Computes:
  - Lamina masks
  - Curvatures
  - Local intensities
- Saves results to an output directory

### 3. Plot processed data

> Note: requires that `process_data.py` has been run first.

TODO: UPDATE THIS PART: ONCE `plot_data.py` has been updated.

```bash
uv run scripts/plot_data.py
```

This script:

- Loads processed output from `process_data.py`
- Generates analysis plots

---

## Interactive Widgets (Napari)

These tools allow interactive exploration, segmentation, and parameter tuning.

---

### 1. Label nuclei

File: `widgets/label_nuclei.py`

This was used to manually label nuclei for rough segmentation in case they were touching or not fully in the FOV.

Update:

```python
path_input = "path/to/data/image.tif"  # path to a .tif file
output_dir = "path/to/data/labels"  # output_dir can also be None, then the default path is the home directory
```

Run:

```bash
uv run widgets/label_nuclei.py
```

Pipeline:

- Loads multi-channel data
- Normalizes and matches resolution
- Computes a **maximum intensity z-projection (3D -> 2D)**
- Performs segmentation

Features:

- Otsu-based nucleus segmentation
- Relabeling of nuclei
- Expansion of labels (Voronoi-based)
- Filtering:
  - Remove small nuclei
  - Remove border-touching nuclei
- Save labels to TIFF

---

### 2. Create lamina masks

File: `widgets/create_nucleus_masks.py`

This was used to find appropriate values for masking parameters.

Update:

```python
path_input = "path/to/data/image.tif"  # path to a .tif file
```

Run:

```bash
uv run widgets/create_nucleus_masks.py
```

Features:

- Load multi-channel confocal data
- Normalization and isotropic resampling
- Create:
  - Rough lamina mask
  - Fine lamina mask (refined segmentation)

Interactive controls allow tuning of:

- Smoothing scale (sigma)
- Intensity thresholds
- Shape parameters

---

### 3. Inspect curvature & intensity values

File: `widgets/inspect_values.py`

This file allows you to easily highlight the voxels that correspond to a certain intensity and/or Gaussian curvature.

You do not need to set a path to run this file, you can select the files easily in the widget.

```bash
uv run widgets/inspect_values.py
```

Features:

- Load:
  - Normalized intensity images (1–2 channels)
  - Principal curvatures (k1, k2)
- Compute Gaussian curvature (k1 x k2)
- Create masks based on:
  - Intensity thresholds
  - Curvature thresholds

Useful for:

- Exploring curvature–intensity relationships
- Defining thresholds for downstream analysis

---

## Notes & Tips

- Only `axis_order = "zyx"` is currently supported\
- The analysis requires for the sampling that the **axial (z) sampling resolution should be an integer multiple of the lateral (x/y) resolution**
  - Example (paper dataset):
    - Lateral sampling (voxel-size in xy-direction): **80 nm**
    - Axial sampling (voxel-size in z-direction): **160 nm** (2× lateral)

---

## TODO

- Add dataset link
- Add paper reference
- Create a `requirements.txt` file (<https://docs.astral.sh/uv/concepts/projects/export/#overview-of-export-formats>)