"""metrics.py — Compute image quality metrics on retrospective test cases.

This script loads the result ``.mat`` files, 
groups all slices belonging to the same study (exam), applies 
segmentation and min-max normalization to [0, 255], and computes the
following metrics **per slice**:

Reference-based (compared against ``ref``):
    - **MS-SSIM** — Multi-Scale Structural Similarity
    - **MSE** — Mean Squared Error
    - **NMI** — Normalized Mutual Information

Non-reference:
    - **BEW** — Blur Edge Widths
    - **MTV** — Mean Total Variation
    - **MLC** — Mean Line Correlation

Results are saved per study as ``.npz`` files. Boxplots are generated.

Requirements:
    pip install -e /path/to/mr-image-metrics   (or add to sys.path)

Usage::

    python metrics.py --compute_metrics (for computing metrics and plotting)
    python metrics.py                   (for plotting oly)
"""

from __future__ import annotations

import argparse
import logging
import nibabel as nib
import sys
import os
from collections import defaultdict
from pathlib import Path
from eval_helpers import plot_metrics
from preprocessing.split_config_demo import TEST_EXAMS

import sys
from unittest.mock import MagicMock

# Mock the module before importing anything from the repo
sys.modules['brisque'] = MagicMock()
sys.modules['pyrootutils'] = MagicMock()


import numpy as np
import scipy.io as sio
from scipy import ndimage

from medimetrics.metrics.ssim import MSSSIM
from medimetrics.metrics.mse import MSE
from medimetrics.metrics.nmi import NMI
from medimetrics.metrics.blur_widths import BlurWidths
from medimetrics.metrics.total_var import MeanTotalVar
from medimetrics.metrics.line_corr import MLC 

from preprocessing.split_config_demo import SLICE_RANGE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Methods stored in the .mat files
# ---------------------------------------------------------------------------
# Keys in the .mat dict → human-readable method names
METHOD_KEYS = {
    "input_corrected":      "input",
    "DB_corrected":         "MRI-bridge",
    "UNet_corrected":       "discriminative network",
    "softgating_corrected": "softgating",
}
REF_KEY = "ref_corrected"

# Metrics
REF_METRIC_NAMES = ("MS-SSIM", "MSE", "NMI")
NOREF_METRIC_NAMES = ("BEW", "MTV", "MLC")
ALL_METRIC_NAMES = REF_METRIC_NAMES + NOREF_METRIC_NAMES


# =========================================================================
# Preprocessing helpers
# =========================================================================

def minmax_bin(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Min-max normalize and quantize to [0, bins-1].

    Parameters
    ----------
    image : np.ndarray
        Input image (any shape).
    bins : int
        Number of discrete levels (default 256 → [0, 255]).

    Returns
    -------
    np.ndarray
        Normalized and binned image as float64.
    """

    mn, mx = float(image.min()), float(image.max())
    if mx - mn == 0:
        return np.zeros_like(image, dtype=np.float64)

    normed = (image.astype(np.float64) - mn) / (mx - mn)   # [0, 1]
    binned = np.round(normed * (bins - 1))                   # [0, bins-1]
    return binned  # float64 — avoids integer overflow in downstream arithmetic


# =========================================================================
# Metric computation
# =========================================================================
def compute_slice_metrics(
    img: np.ndarray,
    ref: np.ndarray | None,
    mask: np.ndarray,
    *,
    ms_ssim_obj: MSSSIM,
    mse_obj: MSE,
    nmi_obj: NMI,
    bew_obj: BlurWidths,
    mtv_obj: MeanTotalVar,
    mlc_obj: MLC,
) -> dict[str, float]:
    """Compute all metrics for a single 2-D slice.

    Parameters
    ----------
    img : np.ndarray (H, W)
        The method's output image, already normalized & binned to [0, 255].
    ref : np.ndarray (H, W) or None
        The reference image (same normalization).  If ``None``, reference-
        based metrics are skipped.
    mask : np.ndarray (H, W) of bool
        Foreground body mask.

    Returns
    -------
    dict[str, float]
        Metric name → value.
    """
    results: dict[str, float] = {}

    # Apply mask for body-region evaluation
    img_masked = img * mask
    ref_masked = ref * mask if ref is not None else None

    # --- Reference-based metrics (on masked images) -----------------------
    if ref_masked is not None:
        try:
            results["MS-SSIM"] = ms_ssim_obj.compute(ref_masked, img_masked, data_range=255.0)
        except Exception as e:
            logger.warning("MS-SSIM failed: %s", e)
            results["MS-SSIM"] = float("nan")

        results["MSE"] = mse_obj.compute(ref_masked, img_masked)
        results["NMI"] = nmi_obj.compute(ref_masked, img_masked)

    # --- Non-reference metrics (on masked images) -------------------------
    results["BEW"] = bew_obj.compute(img_masked)
    results["MTV"] = mtv_obj.compute(img_masked)
    results["MLC"] = mlc_obj.compute(img_masked)

    return results


# =========================================================================
# Study-level aggregation
# =========================================================================
def group_mat_files_by_study(results_dir: Path) -> dict[str, list[Path]]:
    """Group ``.mat`` files by exam/study name.

    Filename convention: ``<ExamName>_slice_<N>.mat``
    e.g. ``Exam2574Series17_slice_36.mat``

    Returns
    -------
    dict[str, list[Path]]
        ``{study_name: [path1, path2, …]}`` sorted by slice number.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for mat in sorted(results_dir.glob("*.mat")):
        # Split on "_slice_" to extract study name
        parts = mat.stem.rsplit("_slice_", 1)
        if len(parts) == 2 and parts[0] in TEST_EXAMS:
            study_name = parts[0]
            groups[study_name].append(mat)
        else:
            logger.warning("Skipping file with unexpected name: %s", mat.name)
    # Sort each group by slice number
    for study in groups:
        groups[study] = sorted(
            groups[study],
            key=lambda p: int(p.stem.rsplit("_slice_", 1)[1])
        )
    return dict(groups)


def process_study(
    study_name: str,
    mat_files: list[Path],
    output_dir: Path,
    mask_dir: Path,
) -> dict:
    """Load all slices for one study, compute metrics, save results.

    Parameters
    ----------
    study_name : str
        Exam identifier (e.g. ``'Exam2574Series17'``).
    mat_files : list[Path]
        Sorted list of ``.mat`` files for this study.
    output_dir : Path
        Where to write the per-study ``.npz`` result file.

    Returns
    -------
    dict
        Nested dict: ``{metric: {method: {slice_name: value}}}``.
    """
    logger.info("Processing study %s  (%d slices)", study_name, len(mat_files))

    # Instantiate metric objects
    ms_ssim_obj = MSSSIM()
    mse_obj = MSE()
    nmi_obj = NMI()
    bew_obj = BlurWidths()
    mtv_obj = MeanTotalVar()
    mlc_obj = MLC()

    # Initialize nested results dict
    # Structure: metrics[metric_name][method_name][slice_name] = value
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for m in ALL_METRIC_NAMES:
        metrics[m] = {}
        for method_label in list(METHOD_KEYS.values()) + [REF_KEY]:
            metrics[m][method_label] = {}

    for mat_file in mat_files:
        print(mat_file)
        slice_name = mat_file.stem  # e.g. "Exam2574Series17_slice_36"
        logger.info("  %s", slice_name)

        # Load the .mat file
        mat = sio.loadmat(str(mat_file))

        ref_raw = mat[REF_KEY].astype(np.float64)

        # Bin the reference
        ref_binned = minmax_bin(ref_raw)

        mask = np.load(f'{mask_dir}/Seg_{study_name}.npz')['mask']

        # --- Compute metrics for each method ------------------------------
        for mat_key, method_label in METHOD_KEYS.items():
            if mat_key not in mat:
                logger.warning("    Key '%s' not found in %s — skipping.",
                               mat_key, mat_file.name)
                continue

            img_raw = mat[mat_key].astype(np.float64)
            img_binned = minmax_bin(img_raw)

            slice_metrics = compute_slice_metrics(
                img_binned, ref_binned, mask,
                ms_ssim_obj=ms_ssim_obj, mse_obj=mse_obj, nmi_obj=nmi_obj,
                bew_obj=bew_obj, mtv_obj=mtv_obj, mlc_obj=mlc_obj,
            )

            for metric_name, value in slice_metrics.items():
                metrics[metric_name][method_label][slice_name] = value

        # --- Also compute non-reference metrics for the reference itself ---
        ref_slice_metrics = compute_slice_metrics(
            ref_binned, None, mask,
            ms_ssim_obj=ms_ssim_obj, mse_obj=mse_obj, nmi_obj=nmi_obj,
            bew_obj=bew_obj, mtv_obj=mtv_obj, mlc_obj=mlc_obj,
        )
        for metric_name, value in ref_slice_metrics.items():
            metrics[metric_name][REF_KEY][slice_name] = value

    # Save per-study results
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"metrics_{study_name}.npz",
        **{f"{m}/{method}": np.array(list(slices.values()))
           for m in metrics for method, slices in metrics[m].items()
           if slices},
    )

    return metrics


# =========================================================================
# CLI
# =========================================================================

EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
RUN_NAME = os.environ["RUN_NAME"]
DEMO_PATH = os.environ["DEMO_PATH"]
PROVIDED_RESULTS = os.environ["PROVIDED_RESULTS"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute image quality metrics on test_average_and_slab.py results.",
    )
    parser.add_argument(
        "--results_dir", type=Path,
        default=Path(DEMO_PATH, PROVIDED_RESULTS, "PURE_Results_5inferences"),
        help="Directory containing the per-slice .mat files.",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path(DEMO_PATH, EXPERIMENT_NAME, f"{RUN_NAME}_metrics"),
        help="Directory where metric results (.npz + .csv) are saved.",
    )
    parser.add_argument(
        "--mask_dir", type=Path,
        default=Path(DEMO_PATH, PROVIDED_RESULTS, "Segmentations/"),
        help="Directory where masks are saved.",
    )

    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        default=False,
        help="Compute metrics (default: off).")


    args = parser.parse_args()

    results_dir: Path = args.results_dir
    output_dir: Path = args.output_dir
    mask_dir: Path = args.mask_dir

    if not results_dir.exists():
        logger.error("Results directory does not exist: %s", results_dir)
        sys.exit(1)

    if args.compute_metrics is True:
        # Group .mat files by study
        studies = group_mat_files_by_study(results_dir)
        if not studies:
            logger.error("No .mat files found in %s", results_dir)
            sys.exit(1)

        logger.info("Found %d studies with %d total .mat files",
                    len(studies), sum(len(v) for v in studies.values()))

        # Process each study
        all_metrics: dict[str, dict] = {}
        for study_name, mat_files in sorted(studies.items()):
            all_metrics[study_name] = process_study(study_name, mat_files, output_dir, mask_dir)

        logger.info("Done. Results in %s", output_dir)

    # Plot metrics
    output_dir = Path(output_dir)
    all_metrics = {}
    # Now glob will work
    for npz_file in output_dir.glob("metrics_*.npz"):
        study_name = npz_file.stem.replace("metrics_", "")
        data = np.load(npz_file)

        metrics = {}
        for key in data.files:
            metric_name, method = key.split("/")
            if metric_name not in metrics:
                metrics[metric_name] = {}
            metrics[metric_name][method] = {
                f"slice_{i}": v for i, v in enumerate(data[key])
            }

        all_metrics[study_name] = metrics

    # Generate plots
    plot_metrics(all_metrics, output_dir=Path(DEMO_PATH, EXPERIMENT_NAME, f"{RUN_NAME}_metrics/figures"))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    main()
