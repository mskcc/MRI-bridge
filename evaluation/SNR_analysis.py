"""
SNR analysis for PURE-corrected MRI-bridge reconstruction results.

For each configured image, this script:
    1. loads the stack of images from a .mat file,
    2. loads and reorients the liver segmentation masks,
    3. for a range of "number of averages", randomly selects that many samples,
       computes an SNR map (mean / std across samples), and
    4. saves the masked SNR map and prints the mean SNR within the liver.

SNR maps are saved as transparent PNGs.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio
import os 

from evaluation.eval_helpers import save_snr_map, normalize_to_255, save_masked_image

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
PROVIDED_RESULTS = os.environ["PROVIDED_RESULTS"]
RUN_NAME = os.environ["RUN_NAME"]
DEMO_PATH = os.environ["DEMO_PATH"]
LAMBDA_CAL = os.environ["CALIBRATION_LAMBDA"]
BASE_DIR = Path(DEMO_PATH, PROVIDED_RESULTS)
OUT_DIR = Path(DEMO_PATH, EXPERIMENT_NAME)


PURE_PATH = BASE_DIR / "PURE_Results_50_windows"
OUT_PATH = OUT_DIR / f"{RUN_NAME}_Plots_PURE_Results_50_windows"
MASK_DIR = BASE_DIR / "Segmentations"

# Numbers of averages to evaluate: for each, this many samples are averaged
# before computing the SNR map.
AVERAGES_TO_EVALUATE = [1, 2, 5, 50]

# Display window for the SNR maps.
SNR_VMIN = 0
SNR_VMAX = 50

# Seed for reproducible random sample selection.
RANDOM_SEED = 1234 #42

# Images to process, paired with a display window maximum (vmax).
# NOTE: vmax is kept for consistency with the other scripts but is not used
#       for the SNR maps (which use SNR_VMIN/SNR_VMAX).
IMAGES_TO_PROCESS = [
    ("Exam8271Series12_slice_38", 170),
]

# ----------------------------------------------------------------------------
# Per-image processing
# ----------------------------------------------------------------------------
def process_image(img_name, rng):
    """Load one result, compute SNR maps for each averaging count, and save them."""
    # Parse exam name and slice number from the file name.
    # e.g. "Exam2574Series17_slice_35" -> exam "Exam2574Series17", slice 35
    exam_name = img_name.split("_")[0]
    slice_nr = int(img_name.split("_")[-1])

    # --- Load SR image stack ---------------------------------------------
    mat_contents = sio.loadmat(f"{PURE_PATH}/{img_name}.mat")
    sr_images = mat_contents["SR_images_corrected"]

    NUM_SR_SAMPLES = sr_images.shape[1]

    # --- Load and select the segmentation mask ---------------------------
    mask = np.load(f'{MASK_DIR}/Seg_{exam_name}.npz')


    mask_slice = np.rot90(mask['mask'], -1)
    liver_mask_slice = np.rot90(mask['liver_mask'], -1)

    # --- SNR analysis for each averaging count ---------------------------
    print(img_name)
    perm = rng.permutation(NUM_SR_SAMPLES)   # shuffle all indices
    for averages in AVERAGES_TO_EVALUATE:
        repetitions = int(NUM_SR_SAMPLES/averages)
        sets = perm.reshape(repetitions, averages)
        # check
        assert len(sets.reshape(-1)) == len(np.unique(sets.reshape(-1))), "Overlap detected!"

        snr_mask_all = np.zeros((repetitions, sr_images.shape[2], sr_images.shape[3]))
        liver_snr_all = []
        
        for repetition in range(repetitions):
            # Randomly pick `averages` distinct samples from the pool.
            sample_indices = sets[repetition]
            # Average the selected samples along the sample axis.
            mean = np.mean(sr_images[:, sample_indices, :, :], axis=1)
            std = np.std(mean, axis=0)
            snr_map = np.mean(mean, axis=0) / std
            snr_mask_all[repetition] = snr_map

            liver_snr = liver_mask_slice * np.rot90(snr_map, -1)
            liver_snr =  np.mean(liver_snr[liver_snr != 0])
            liver_snr_all.append(liver_snr)


            if repetition == 0:
                db = normalize_to_255(np.mean(sr_images[:, sample_indices, :, :], axis=1)[0])       # BreathBRIDGE mean
        
                # BreathBRIDGE mean reconstruction.
                save_masked_image(
                    db, mask_slice,
                    OUT_PATH / f"{img_name}_mean_BreathBRIDGE_{averages}.png",
                    cmap="gray", vmin=0, vmax=200,
                )

        save_snr_map(
            np.mean(snr_mask_all, axis=0), mask_slice,
            OUT_PATH / f"{img_name}_SNR_{averages}.png",
            SNR_VMIN, SNR_VMAX
        )

        print(f"Liver SNR is {np.mean(liver_snr_all)}")

    print("##################")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    """Process every configured image and generate SNR maps."""
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=RANDOM_SEED)

    for img_name, _vmax in IMAGES_TO_PROCESS:
        process_image(img_name, rng)


if __name__ == "__main__":
    main()