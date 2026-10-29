"""
Visualization script for PURE-corrected MRI-bridge reconstruction results.

Loads reconstruction results from .mat files, applies segmentation
masks, and generates comparison plots for the different reconstruction methods.

The script runs in one of 4 modes (set MODE below):

    "retrospective"         - retrospectively undersampled data (has a
                             reference, so a relative error map can be computed).
                             Plots: relative IQR, relative error, BreathBRIDGE mean,
                             soft-gating, input, reference, non-generative U-Net.

    "retrospective_modes_1" - retrospectively undersampled data results of 
                             MRI-bridge trained with single and full slice 
                             Plots: BreathBRIDGE mean only.

    "retrospective_modes_2" - retrospectively undersampled data of 
                             MRI-bridge trained with slab and full slice 
                             Plots: BreathBRIDGE mean only.

    "retrospective_modes_3" - retrospectively undersampled data of 
                             MRI-bridge trained with slab and patches 
                             Plots: BreathBRIDGE mean only.

Each plot is saved as a transparent PNG for later use in figures.
"""

from pathlib import Path

import os
import numpy as np
import scipy.io as sio
from eval_helpers import normalize_to_255, save_masked_image, save_image

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
RUN_NAME = os.environ["RUN_NAME"]
PROVIDED_RESULTS = os.environ["PROVIDED_RESULTS"]
DEMO_PATH = os.environ["DEMO_PATH"]
LAMBDA_CAL = os.environ["CALIBRATION_LAMBDA"]
OUT_DIR = Path(DEMO_PATH, EXPERIMENT_NAME)
BASE_DIR = Path(DEMO_PATH, PROVIDED_RESULTS)

# Mode-specific settings: input/output/mask directories and the list of images
# to process (each paired with its display window maximum, vmax).
MODE_CONFIG = {
    "retrospective": {
        "pure_path_R5": BASE_DIR / f"PURE_Results_5inferences",
        "pure_path": BASE_DIR / f"PURE_Results",
        "plots_path": OUT_DIR / f"{RUN_NAME}_Plots_PURE_Results",
        "mask_dir": BASE_DIR / "Segmentations",
        "images": [
            ("Exam8271Series12_slice_38", 210),
        ],
    },

    "retrospective_modes_1": {
        "pure_path": BASE_DIR / f"PURE_Results_single_full_slice",
        "plots_path": OUT_DIR / f"{RUN_NAME}_Plots_PURE_Results_single_full_slice",
        "mask_dir": BASE_DIR / "Segmentations",
        "images": [
            ("Exam7695-Series16_slice_38", 230),
        ],
    },

    "retrospective_modes_2": {
        "pure_path": BASE_DIR / f"PURE_Results_slab_full_slice",
        "plots_path": OUT_DIR / f"{RUN_NAME}_Plots_PURE_Results_slab_full_slice",
        "mask_dir": BASE_DIR / "Segmentations",
        "images": [
            ("Exam7695-Series16_slice_38", 230),
        ],
    },

    "retrospective_modes_3": {
        "pure_path": BASE_DIR / f"PURE_Results_slab_patch",
        "plots_path": OUT_DIR / f"{RUN_NAME}_Plots_PURE_Results_slab_patch",
        "mask_dir": BASE_DIR / "Segmentations",
        "images": [
            ("Exam7695-Series16_slice_38", 230),
        ],
    },
}

# ----------------------------------------------------------------------------
# Per-image processing
# ----------------------------------------------------------------------------
def process_image(img_name, vmax, config, MODE):
    """Load one .mat result, apply its mask, and save all plots for the mode."""
    pure_path = config["pure_path"]
    plots_path = config["plots_path"]
    mask_dir = config["mask_dir"]
    if 'pure_path_R5' in config.keys():
        pure_path_cal = pure_path
        pure_path = config["pure_path_R5"]

    # Parse exam name and slice number from the file name.
    # e.g. "Exam11358Series2500_slice_38" -> exam "Exam11358Series2500", slice 38
    exam_name = img_name.split("_")[0]
    slice_nr = int(img_name.split("_")[-1])

    # --- Load reconstruction results -------------------------------------
    mat_contents = sio.loadmat(f"{pure_path}/{img_name}.mat")
    if 'pure_path_R5' in config.keys():
        mat_contents_cal = sio.loadmat(f"{pure_path_cal}/{img_name}.mat")
        
    db = normalize_to_255(mat_contents["DB_corrected"])       # BreathBRIDGE mean
    inp = normalize_to_255(mat_contents["input_corrected"])   # undersampled input
    ref = normalize_to_255(mat_contents["ref_corrected"])     # reference CG-SENSE or XD-GRASP

    # BreathBRIDGE mean reconstruction.
    save_image(
        np.rot90(db, -1),
        plots_path / f"{img_name}_mean_BreathBRIDGE.png",
        cmap="gray", vmin=0, vmax=vmax,
    )

    # Undersampled input image.
    save_image(
        np.rot90(inp, -1),
        plots_path / f"{img_name}_input.png",
        cmap="gray", vmin=0, vmax=vmax+40,
    )

    # Reference image (saved as "_ref" in retrospective mode).
    save_image(
        np.rot90(ref, -1),
        plots_path / f"{img_name}_ref.png",
        cmap="gray", vmin=0, vmax=vmax,
    )

    if MODE == "retrospective":
        unet = normalize_to_255(mat_contents["UNet_corrected"])   # non-generative U-Net
        # Relative IQR: calibrated interquartile range divided by the DB mean,
        # halved, then scaled to percent for the 0-100 display window.
        if 'pure_path_R5' in config.keys():
            rel_iqr = float(LAMBDA_CAL) * mat_contents_cal["IQR"] / mat_contents_cal["DB_corrected"] / 2
        else:
            rel_iqr = float(LAMBDA_CAL) * mat_contents["IQR"] / mat_contents["DB_corrected"] / 2
        rel_iqr_percent = rel_iqr * 100

        # --- Load and select the segmentation mask ---------------------------
        mask_slice = np.rot90(np.load(f'{mask_dir}/Seg_{exam_name}.npz')['mask'], -1)


        # --- Plots common to both modes --------------------------------------
        # Relative IQR uncertainty map (with colorbar, percentage scale).
        save_masked_image(
            rel_iqr_percent, mask_slice,
            plots_path / f"{img_name}_rel_IQR.png",
            cmap="viridis", vmin=0, vmax=100, colorbar=True,
        )
        # Non-generative U-Net reconstruction.
        save_image(
            np.rot90(unet, -1),
            plots_path / f"{img_name}_non_generative.png",
            cmap="gray", vmin=0, vmax=vmax,
    )

        # Relative error map: |DB - reference| / DB, scaled to percent.
        # Only meaningful retrospectively, where a ground-truth reference exists.
        error = np.abs(
            mat_contents["DB_corrected"] - mat_contents["ref_corrected"]
        ) / mat_contents["DB_corrected"]
        error_percent = error * 100

        save_masked_image(
            error_percent, mask_slice,
            plots_path / f"{img_name}_rel_error.png",
            cmap="viridis", vmin=0, vmax=100, colorbar=True,
        )

        # Soft-gating reconstruction (has its own extra intensity offset).
        soft_gating = normalize_to_255(mat_contents["softgating_corrected"])
        save_image(
            np.rot90(soft_gating, -1),
            plots_path / f"{img_name}_SoftGating.png",
            cmap="gray", vmin=0, vmax=vmax+40,
        )


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    for MODE in ["retrospective", "retrospective_modes_1", "retrospective_modes_2", "retrospective_modes_3"]:
        """Process every image configured for the selected MODE."""
        config = MODE_CONFIG[MODE]

        # Make sure the output directory exists before writing any plots.
        config["plots_path"].mkdir(parents=True, exist_ok=True)

        for img_name, vmax in config["images"]:
            print(f"Processing {img_name} (vmax={vmax}) ...")
            process_image(img_name, vmax, config, MODE)

        print(f"Done. Plots saved to: {config['plots_path']}")


if __name__ == "__main__":
    main()
