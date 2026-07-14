import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def make_bin_labels(bins):
    return [f"{bins[i]:.3f} - {bins[i + 1]:.3f}" for i in range(len(bins) - 1)]


def make_bin_centers(bins):
    print(bins)
    return (bins[:-1] + bins[1:]) / 2


def make_bin_center_map(bins):
    return dict(zip(make_bin_labels(bins), make_bin_centers(bins), strict=True))


def process_dataframe(
    df: pd.DataFrame,
    curvature_type: str,
    curvature_bins: list,
    intensity_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()

    # curvature
    if curvature_type == "gaussian":
        df["curvature"] = df["k1"] * df["k2"]
    elif curvature_type == "mean":
        df["curvature"] = (df["k1"] + df["k2"]) / 2
    else:
        raise ValueError(f"Invalid curvature_type: {curvature_type!r}")

    # binning
    labels = make_bin_labels(curvature_bins)
    df["K_bin"] = pd.cut(
        df["curvature"],
        bins=curvature_bins,
        labels=labels,
        include_lowest=True,
    )

    # per-nucleus normalization against the first (lowest) bin
    for col in intensity_cols:
        baseline = df[df["K_bin"] == labels[0]].groupby("nucleus_id")[col].mean()

        df["_baseline"] = df["nucleus_id"].map(baseline)
        df = df.dropna(subset=["_baseline"]).copy()
        df[col] = df[col] / df["_baseline"]
        df = df.drop(columns=["_baseline"])

    # derived columns
    if any("local" in intensity_col for intensity_col in intensity_cols):
        df["ratio_local"] = (
            df["norm_local_intensity_ch1"] / df["norm_local_intensity_ch0"]
        )
    if any("voxel" in intensity_col for intensity_col in intensity_cols):
        df["ratio_voxel"] = (
            df["norm_voxel_intensity_ch1"] / df["norm_voxel_intensity_ch0"]
        )

    return df


def compute_per_nucleus_stats(df, intensity_cols):
    # Compute per nucleus & per bin stats
    per_nucleus_stats = (
        df.groupby(["nucleus_id", "K_bin"], observed=True)[intensity_cols]
        .mean()
        .reset_index()
    )

    # Ratio
    if any("local" in intensity_col for intensity_col in intensity_cols):
        per_nucleus_stats["ratio_local"] = (
            per_nucleus_stats["norm_local_intensity_ch1"]
            / per_nucleus_stats["norm_local_intensity_ch0"]
        )
    if any("voxel" in intensity_col for intensity_col in intensity_cols):
        per_nucleus_stats["ratio_voxel"] = (
            per_nucleus_stats["norm_voxel_intensity_ch1"]
            / per_nucleus_stats["norm_voxel_intensity_ch0"]
        )

    return per_nucleus_stats


if __name__ == "__main__":
    # TODO: please provide the paths to the directories
    path_data = "path/to/dir_data/all_data.csv"
    output_folder = "path/to/dir_data/figures"

    curvature_bins = np.array([0.01, 0.04, 0.07, 0.10, 0.13, 0.16, 0.19])

    curvature_type = "gaussian"  # "gaussian" or "mean"
    intensity_cols = [
        "norm_local_intensity_ch0",
        "norm_local_intensity_ch1",
    ]

    color_lac = "#1f77b4"  # Blue
    color_lb1 = "#d62728"  # Red
    color_ratio = "purple"

    # Do not change things below here...

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load and process the data
    df = pd.read_csv(path_data)

    df = process_dataframe(
        df,
        curvature_type,
        curvature_bins,
        intensity_cols,
    )

    per_nucleus_stats = compute_per_nucleus_stats(df, intensity_cols)

    print("-" * 40)
    print(f"Total number of voxels: {len(df)}")
    print("-" * 40)

    # Preparation for plotting
    labels = make_bin_labels(curvature_bins)
    center_map = make_bin_center_map(curvature_bins)

    # Melt voxel-level data for violin plot
    df_long = df.melt(
        id_vars=["nucleus_id", "K_bin"],
        value_vars=intensity_cols,
        var_name="Channel",
        value_name="intensity",
    )

    # Melt per-nucleus stats for line plot
    per_nucleus_long = per_nucleus_stats.melt(
        id_vars=["nucleus_id", "K_bin"],
        value_vars=intensity_cols,
        var_name="Channel",
        value_name="intensity",
    )

    # Optional: nicer names
    channel_map = {
        "norm_local_intensity_ch0": "Lamin A/C",
        "norm_local_intensity_ch1": "Lamin B1",
    }
    df_long["Channel"] = df_long["Channel"].map(channel_map)
    per_nucleus_long["Channel"] = per_nucleus_long["Channel"].map(channel_map)

    # Colors
    palette = {
        "Lamin A/C": color_lac,
        "Lamin B1": color_lb1,
    }

    # -----------------------------
    # Plotting
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Subplot 1: violin plot
    sns.violinplot(
        data=df_long,
        x="K_bin",
        y="intensity",
        hue="Channel",
        split=True,
        inner="quart",
        palette=palette,
        hue_order=["Lamin B1", "Lamin A/C"],
        ax=ax1,
    )

    # ax1.set_title(f"Voxel Distribution (Total N = {len(df):,} voxels)", fontsize=14)
    ax1.set_ylim(0, 3.5)
    ax1.set_xlabel(r"Gaussian curvature ($\mathrm{\mu m^{-2}}$)")
    ax1.set_ylabel("Normalized intensity\n(Rel. to first bin)")

    # Subplot 2: line-plot
    sns.lineplot(
        data=per_nucleus_long,
        x="K_bin",
        y="intensity",
        hue="Channel",
        style="Channel",
        markers=True,
        dashes=False,
        palette=palette,
        ax=ax2,
        errorbar="se",
        linewidth=2,
    )

    # Ratio line
    summary = per_nucleus_stats.groupby("K_bin", observed=True).mean().reindex(labels)

    if any("local" in intensity_col for intensity_col in intensity_cols):
        ratio_local = (
            summary["norm_local_intensity_ch1"] / summary["norm_local_intensity_ch0"]
        )

        ax2.plot(
            labels,
            ratio_local,
            color=color_ratio,
            linestyle="--",
            marker="D",
            label="B1/AC ratio",
            zorder=10,
        )

    if any("voxel" in intensity_col for intensity_col in intensity_cols):
        summary["ratio_voxel"] = (
            summary["norm_voxel_intensity_ch1"] / summary["norm_voxel_intensity_ch0"]
        )

        ax2.plot(
            labels,
            ratio_local,
            color=color_ratio,
            linestyle="--",
            marker="D",
            label="B1/AC ratio (voxel intensity)",
            zorder=10,
        )

    ax2.set_xlabel(r"Gaussian curvature ($\mathrm{\mu m^{-2}}$)")
    ax2.set_ylabel("Relative protein density")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "curvature_analysis.png"), dpi=300)
    plt.show()

    # Pretty printing of bin counts
    counts_per_bin = (
        df_long.groupby(["K_bin", "Channel"], observed=True)
        .size()
        .unstack()
        .reindex(labels)
    )

    print("\nVoxel count per bin:")
    print(counts_per_bin)
