import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================================
# Plotting helpers
# =========================================================================
def plot_standard_comparison(
    grasp: np.ndarray,
    sharpened: np.ndarray,
    slice_file: Path,
    plot_path: Path,
    grasp_phase_index: 3, 
    crop_px: 40
) -> None:
    """Save a 3-panel comparison: raw end-ex XD-GRASP, sharpened end-ex XD-GRASP, inspiration XD-GRASP."""
    c = crop_px
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    plt.style.use("dark_background")
    plt.subplots_adjust(left=0.01, right=0.99, hspace=0.01)
    fig.suptitle(slice_file.name)

    ax1.imshow(np.rot90(np.abs(grasp[grasp_phase_index, c:-c, c:-c]), -1), cmap="gray")
    ax1.set_title("grasp end_ex", color="white")
    ax1.axis("off")

    ax2.imshow(sharpened, cmap="gray")
    ax2.set_title("sharpened grasp end_ex", color="white")
    ax2.axis("off")

    ax3.imshow(np.rot90(np.abs(grasp[0, c:-c, c:-c]), -1), cmap="gray")
    ax3.set_title("grasp in", color="white")
    ax3.axis("off")

    fig.savefig(plot_path / f"comparison_{slice_file.stem}.png", dpi=300)
    plt.close(fig)


    fig, (ax1) = plt.subplots(1, 1, figsize=(4, 4))
    plt.style.use("dark_background")
    plt.subplots_adjust(left=0.01, right=0.99, hspace=0.01)
    fig.suptitle(slice_file.name)

    ax1.imshow(sharpened, cmap="gray")
    ax1.set_title("sharpened grasp end_ex", color="white")
    ax1.axis("off")

    fig.savefig(plot_path / f"plots_{slice_file.stem}.png", dpi=300)
    plt.close(fig)


def plot_diff(
    target: np.ndarray,
    input_img: np.ndarray,
    slice_file: Path,
    plot_path: Path,
    prefix: str = "",
) -> None:
    """Save 3-panel comparison: target, input, and 1st signed and 2nd absolute difference plots."""
    for suffix, diff, cmap in [
        ("diff", target - input_img, "BrBG"),
        ("diff_abs", np.abs(target - input_img), "viridis"),
    ]:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
        plt.style.use("dark_background")
        plt.subplots_adjust(left=0.01, right=0.99, hspace=0.01)
        fig.suptitle(slice_file.name)

        ax1.imshow(target, cmap="gray")
        ax1.set_title("target", color="white")
        ax1.axis("off")

        ax2.imshow(input_img, cmap="gray")
        ax2.set_title("input", color="white")
        ax2.axis("off")

        im3 = ax3.imshow(diff, cmap=cmap)
        ax3.set_title("diff", color="white")
        fig.colorbar(im3, ax=ax3)

        fig.savefig(
            plot_path / f"{prefix}{suffix}_{slice_file.stem}.png", dpi=300
        )
        plt.close(fig)


def plot_window(
    motion_average_ave: np.ndarray,
    window_idx: int,
    slice_file: Path,
    plot_path: Path,
) -> None:
    """Save a single-panel plot for one temporal-window sub-average."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.style.use("dark_background")
    plt.subplots_adjust(left=0.01, right=0.99, hspace=0.01)
    fig.suptitle(slice_file.name)
    ax.imshow(motion_average_ave, cmap="gray")
    ax.set_title(f"motion_average win {window_idx}", color="white")
    ax.axis("off")
    fig.savefig(
        plot_path / f"motion_average_win_{window_idx}_{slice_file.stem}.png",
        dpi=300,
    )
    plt.close(fig)

