import os
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
import scipy.io as sio
import matplotlib.pyplot as plt


def normalize_to_255(array):
    """Scale an array so its maximum value maps to 255 (8-bit intensity range)."""
    return array / np.max(array) * 255.0


def save_masked_image(image, mask, filename, cmap, vmin, vmax, colorbar=False):
    """
    Overlay a mask on an image and save it as a transparent PNG.

    Parameters
    ----------
    image : np.ndarray
        The 2D image to display (already normalized/scaled as needed).
    mask : np.ndarray
        Binary mask applied to the (rotated) image.
    filename : Path or str
        Full output path for the saved figure.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float
        Display window limits.
    colorbar : bool
        Whether to add a colorbar (used for the IQR / error maps).
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    # np.rot90(..., -1) rotates the image to match the mask orientation.
    displayed = mask * np.rot90(image, -1)
    im = ax.imshow(displayed, cmap=cmap, interpolation="none", vmin=vmin, vmax=vmax)

    ax.set_aspect("equal")
    ax.axis("off")

    if colorbar:
        fig.colorbar(im, ax=ax)

    plt.savefig(filename, dpi=300, transparent=True)
    plt.close(fig)


def save_image(img, filename, cmap="gray", vmin=0, vmax=255):
    """
    Save image as a transparent PNG.

    Parameters
    ----------
    image : np.ndarray
        The 2D image to display (already normalized/scaled as needed).
    filename : Path or str
        Full output path for the saved figure.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float
        Display window limits.
    """
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.savefig(filename, bbox_inches="tight", pad_inches=0, transparent=True, dpi=200)
    plt.close(fig)



def save_snr_map(snr_map, mask_slice, filename, SNR_VMIN, SNR_VMAX):
    """Overlay the mask on an SNR map and save it as a transparent PNG.
    
    Parameters
    ----------
    snr_map : np.ndarray
        The 2D SNR map to display.
    mask_slice : np.ndarray
        The 2D binary mask applied to the (rotated) SNR map.
    filename : str
        Output filename.
    SNR_VMIN, SNR_VMAX: float
        Display window limits.
    """
    plt.imshow(mask_slice * np.rot90(snr_map, -1), vmin=SNR_VMIN, vmax=SNR_VMAX)
    plt.axis("off")
    plt.colorbar()
    plt.savefig(filename, transparent=True)
    plt.close()


def plot_metrics(
    all_metrics: dict[str, dict],
    output_dir: Path,
    figsize_per_metric: tuple = (14, 5),
    save: bool = True,
    file_format: str = "png",
    dpi: int = 150,
) -> None:
    """
    Create scatter + box plots for each metric.

    Parameters
    ----------
    all_metrics : dict[str, dict]
        ``{study_name: metrics_dict}`` as returned by process_study.
    output_dir : Path
        Destination directory for figures.
    figsize_per_metric : tuple
        Figure size (width, height) for each metric plot.
    save : bool
        Whether to save the figures to disk.
    file_format : str
        File format for saved figures.
    dpi : int
        Resolution for saved figures.
    """
    np.random.seed(1234)
    # np.random.default_rng(seed)

    # Define metric groups
    ref_based_metrics = ["MSE", "NMI", "MS-SSIM"]
    ref_free_metrics = ["BEW", "MTV", "MLC"]

    # Define methods per group
    methods_ref_based = ["input", "softgating", "discriminative network", "MRI-bridge"]
    methods_ref_based_labels = ["Input", "Soft-gating", "DiscNet", "MRI-bridge"]
    methods_ref_free = ["input", "ref_corrected", "softgating", "discriminative network", "MRI-bridge"]
    methods_ref_free_labels =  ["Input", "XD-GRASP", "Soft-gating", "DiscNet", "MRI-bridge"]

    # Study names (sorted for consistency)
    study_names = sorted(all_metrics.keys())
    n_studies = len(study_names)

    # Color map for studies
    study_colors = ["#506146",  "#346F3E", "#749E79", "#7BC288", "#9FD18E", "#ACD8B2"]
    #502419
    # --- Helper function to create one metric figure ---
    def _plot_single_metric(metric_name: str, methods: list[str], method_labels: list[str]):
        fig, axes = plt.subplots(
            1, 1,
            figsize=figsize_per_metric,
        )

        ax_scatter = axes

        method_positions = np.arange(len(methods))
        jitter_width = 0.15 # jitter for scatter within each method

        # Storage for study means (for box plot)
        # Shape: (n_methods, n_studies)
        study_means = np.full((len(methods), n_studies), np.nan)

        # --- Left panel: Scatter plot (all slice values per study) ---
        for study_idx, study_name in enumerate(study_names):
            metrics = all_metrics[study_name]
            color = study_colors[study_idx]

            for method_idx, method in enumerate(methods):
                values = metrics.get(metric_name, {}).get(method, {})
                if not values:
                    continue

                arr = np.array(list(values.values()))
                arr = arr[~np.isnan(arr)]

                if len(arr) == 0:
                    continue

                # Store the mean for box plot
                study_means[method_idx, study_idx] = np.mean(arr)

                # Jitter x-positions for this study within the method
                x_base = method_positions[method_idx]
                # Offset each study slightly
                study_offset = (study_idx - n_studies / 2 + 0.5) * (
                    jitter_width * 2 / n_studies
                ) + (study_idx - n_studies / 2 + 0.5) * 0.05
                x_jittered = x_base + study_offset + np.random.uniform(
                    -jitter_width / n_studies,
                    jitter_width / n_studies,
                    size=len(arr),
                )

                ax_scatter.scatter(
                    x_jittered,
                    arr,
                    color=color,
                    alpha=0.8,
                    s=25,
                    label=study_name if method_idx == 0 else None,
                    edgecolors="none",
                )

        ax_scatter.grid(axis="y", alpha=0.3)

        # --- Right panel: Box plot of study means ---
        # Filter out all-NaN columns
        boxplot_data = []
        valid_positions = []
        for method_idx in range(len(methods)):
            vals = study_means[method_idx]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                boxplot_data.append(vals)
                valid_positions.append(method_idx)

        if boxplot_data:
            bp = ax_scatter.boxplot(
                boxplot_data,
                positions=valid_positions,
                widths=0.5,
                patch_artist=True,
                showmeans=True,
                showfliers=False,
                meanprops=dict(
                    marker="D",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markersize=6,
                ),
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(facecolor="lightgrey", alpha=0.6),
            )

            # Overlay scatter of study means on box plot
            for bp_idx, method_idx in enumerate(valid_positions):
                vals = study_means[method_idx]
                for study_idx in range(n_studies):
                    if not np.isnan(vals[study_idx]):
                        ax_scatter.scatter(
                            method_idx + (study_idx - n_studies / 2 + 0.5) * (
                            jitter_width * 2 / n_studies) + (study_idx - n_studies / 2 + 0.5) * 0.05,
                            vals[study_idx],
                            color=study_colors[study_idx],
                            s=50,
                            # marker="H",
                            zorder=5,
                            edgecolors="black",
                            linewidths=1.5,
                        )
        ax_scatter.set_xticks(method_positions)
        ax_scatter.set_xticklabels(method_labels, rotation=0, fontsize=20) # ha="right")
        ax_scatter.tick_params(axis='y', labelsize=14)
        ax_scatter.set_ylabel(metric_name, fontsize=20)
        # ax_scatter.set_title(f"{metric_name} — Per-slice values by study")
        if metric_name == "MSE":
            ax_scatter.set_yscale("log")
            ax_scatter.set_yscale("log")

        plt.tight_layout()

        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig_path = output_dir / f"plot_{metric_name}.{file_format}"
            fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {fig_path}")

        plt.show()
        plt.close(fig)

    # --- Generate plots ---
    # Reference-based metrics
    for metric_name in ref_based_metrics:
        _plot_single_metric(metric_name, methods_ref_based, methods_ref_based_labels)

    # Reference-free metrics (includes ref_corrected)
    for metric_name in ref_free_metrics:
        _plot_single_metric(metric_name, methods_ref_free, methods_ref_free_labels)


def save_slice_plot(
    data: np.ndarray,
    save_path: str,
    *,
    vmin: float = 0,
    vmax: float = 1.4,
    cmap: str = "gray",
    dpi: int = 300,
    figsize: tuple[int, int] = (4, 4),
) -> None:
    """Save a single 2-D slice as a borderless PNG image.

    Parameters
    ----------
    data : np.ndarray (H, W)
        Image to plot.
    save_path : str
        Full path for the output ``.png`` file.
    vmin, vmax : float
        Colour-map limits passed to ``imshow``.
    cmap : str
        Matplotlib colour-map name.
    dpi : int
        Resolution of the saved figure.
    figsize : tuple[int, int]
        Figure size in inches ``(width, height)``.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.imshow(data, cmap=cmap, interpolation="none", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(save_path, dpi=dpi, transparent=True)
    plt.close(fig)


def normalize_to_max(img):
    """Normalize an array by its maximum value."""
    return img / np.percentile(img, 100)


def save_all_test_outputs(
    *,
    dataset_dir: str,
    img_name: str,
    plot_id: int,
    input_img: np.ndarray,
    ref: np.ndarray,
    slab_SR_img: np.ndarray,
    SR_img: np.ndarray,
    SR_images: np.array,
    results_folder: str,
    vmax: float = 0.6
) -> None:
    """Save all standard visualisation PNGs and the ``.mat`` file for one sample.

    Parameters
    ----------
    dataset_dir : str
        Directory where individual PNG files are written.
    img_name : str
        Base name of the current sample (without extension).
    plot_id : int
        Slice index along the first axis used for 2-D visualisation.
    input_img, ref, slab_SR_img, SR_img, SR_images: np.ndarray
        Image volumes (at least ``plot_id + 1`` slices along axis 0).
    results_folder : str
        Directory where the ``.mat`` file is written.
    vmax : float
        Upper colour-map limit for all plots.
    """
    # Define (data, filename_suffix) pairs for each plot
    
    plots = [
        (input_img[plot_id],                              f"{img_name}_input.png"),
        (ref[plot_id],                                    f"{img_name}_ref.png"),
        (SR_img[plot_id],                                 f"{img_name}_mean_BreathBRIDGE.png"),
        (slab_SR_img[plot_id],                            f"{img_name}_non_generative.png"),
        (np.abs(SR_img[plot_id] - ref[plot_id]),     f"{img_name}_diff.png")
    ]

    for data, filename in plots:
        data = normalize_to_max(data)
        vmax = 1.4
        cmap = 'gray'
        save_slice_plot(data, os.path.join(dataset_dir, filename), cmap=cmap, vmax=vmax)

    # Save .mat file
    mat_data = {
        "input":     input_img[plot_id],
        "ref":       ref[plot_id],
        "DB":        SR_img[plot_id],
        "UNet":      slab_SR_img[plot_id],
        "SR_images": SR_images[:, plot_id, ...]
    }
    os.makedirs(results_folder, exist_ok=True)
    sio.savemat(os.path.join(results_folder, f"{img_name}.mat"), mat_data)
