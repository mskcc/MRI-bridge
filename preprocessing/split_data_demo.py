"""split_data_retro_cal.py — Preprocess reconstructed MRI slices of retrospective study and split into training/validation/calibration/test.

There are four modes:

``standard_20s``
    Input (A) = motion-averaged rnufft (20s, 100 spokes).
    Target (B) = sharpened XD-GRASP end-expiratory phase of 3 min scan.
    Target (C) = softgating reconstrucion of 100 spokes. 

``40s``
    Like standard_20s, but saves the 40s motion average (rnufft_200,
    200 spokes) as channel A.

``average``
    Like standard_20s, but saves the full motion average (rnufft_all,
    900 spokes) as channel A.

``window``
    Temporal-window sub-averages.
    Input (A) = per-window magnitude rnufft (one ``.npz`` per window index).
    No target channels B/C are written.

Output layout (example for ``standard_20s``)::

    <output_dir>/
        training/A/<exam>_slice_<n>.npz
        training/B/<exam>_slice_<n>.npz
        training/C/<exam>_slice_<n>.npz
        validation/A/…
        validation/B/…
        validation/C/…
        calibration/A/…
        calibration/B/…
        calibration/C/…
        test/A/…
        test/B/…
        test/C/…
        training.pkl          — list[str] of training slice identifiers
        validation.pkl
        test.pkl

Usage::

    python split_data.py --mode standard_20s --save
    python split_data.py --mode 40s
    python split_data.py --mode average
    python split_data.py --mode window

Split assignments and slice ranges are defined in ``split_config_retro_cal.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List

import h5py
import numpy as np
from natsort import natsorted
from skimage.filters import unsharp_mask

from preprocessing.split_config_demo import SLICE_RANGE, get_split
from preprocessing.plotting_helpers import plot_standard_comparison, plot_diff, plot_window
from preprocessing.helpers import save_list, parse_slice_number, is_representative_slice, \
    ensure_dirs, complex_to_magnitude

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported preprocessing modes
# ---------------------------------------------------------------------------
MODES = ("standard_20s", "40s", "average", "window")

# File-name glob patterns per mode
_GLOB_PATTERN = {
    "standard_20s": "*0.5.mat",
    "40s":          "*0.5.mat",
    "average":      "*0.5.mat",
    "window":       "*0.5_windows.mat",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CROP_PX = 40                 # pixels to crop from each border
NORM_PERCENTILE = 100        # percentile used for intensity normalisation
UNSHARP_RADIUS = 1.0         # unsharp-mask kernel radius
UNSHARP_AMOUNT = 1.0         # unsharp-mask strength
GRASP_PHASE_INDEX = 4        # respiratory phase index for end-expiratory XD-GRASP
BOUNDARY_MARGIN = 2          # slices within this margin of start/end are
                            # excluded from split lists (neighbour augmentation)
SOFTGATING_SUBDIR = "Results_softgating"   # subfolder for softgating slices
SOFTGATING_PREFIX = "Softgating"   # filename prefix for softgating slices

# =========================================================================
# Helper utilities
# =========================================================================

# =========================================================================
# Per-mode loading / preprocessing
# =========================================================================
def _load_mode(slice_file: Path) -> dict[str, np.ndarray]:
    """Load a single slice for **standard_20s**, **40s**, and **average** modes.

    Loads the 20s (rnufft_100), 40s (rnufft_200), and full (rnufft_all)
    temporal averages plus the XD-GRASP targets and softgating reference.
    The caller decides which average to save as channel A based on the active mode.

    Returns
    -------
    dict with keys ``'rnufft'``, ``'rnufft_200'``, ``'rnufft_all'``,
    ``'grasp'``, ``'grasp_raw'``, and ``'softgating'``.
    """

    with h5py.File(slice_file, "r") as hf:
        # Motion-averaged inputs
        motion_average = complex_to_magnitude(np.array(hf["rnufft_100"]), CROP_PX, NORM_PERCENTILE)
        motion_average_200 = complex_to_magnitude(np.array(hf["rnufft_200"]), CROP_PX, NORM_PERCENTILE)
        motion_average_all = complex_to_magnitude(np.array(hf["rnufft_all"]), CROP_PX, NORM_PERCENTILE)

        # XD-GRASP target (end-expiratory phase, 3 min)
        raw_grasp = np.array(hf["x"])
        grasp = raw_grasp["real"] + 1j * raw_grasp["imag"]
        grasp_end = np.rot90(np.abs(grasp[GRASP_PHASE_INDEX, CROP_PX:-CROP_PX, CROP_PX:-CROP_PX]), -1)
        sharpened = unsharp_mask(
            grasp_end, radius=UNSHARP_RADIUS, amount=UNSHARP_AMOUNT,
            preserve_range=False,
        )
        sharpened /= np.percentile(sharpened, NORM_PERCENTILE)

    # Soft gating results
    softgating_path = Path(
        slice_file.parents[1],
        SOFTGATING_SUBDIR,
        f"{SOFTGATING_PREFIX}_{parse_slice_number(slice_file.name)}.mat",
    )
    with h5py.File(softgating_path, "r") as hf:
       softgating = complex_to_magnitude(np.array(hf["softgating"]), CROP_PX, NORM_PERCENTILE)

    return {
        "rnufft": motion_average,
        "rnufft_200": motion_average_200,
        "rnufft_all": motion_average_all,
        "grasp": sharpened,
        "grasp_raw": grasp,
        "softgating": softgating
    }


def _load_window(slice_file: Path) -> dict[str, np.ndarray]:
    """Load a single slice in **window** mode.

    Returns
    -------
    dict with key ``'rnufft_windows'`` -> 3-D array ``(n_windows, H, W)``.
    """
    with h5py.File(slice_file, "r") as hf:
        raw = np.array(hf["rnufft_900"])
        avg_cx = raw["real"] + 1j * raw["imag"]

    n_windows = avg_cx.shape[0] - 99
    h, w = avg_cx.shape[1] - 2 * CROP_PX, avg_cx.shape[2] - 2 * CROP_PX
    windows = np.empty((n_windows, h, w), dtype=np.float64)

    for idx in range(n_windows):
        mag = np.abs(avg_cx[idx, CROP_PX:-CROP_PX, CROP_PX:-CROP_PX])
        windows[idx] = np.rot90(mag / np.percentile(mag, NORM_PERCENTILE), -1)

    return {"rnufft_windows": windows}


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------
_LOADERS = {
    "standard_20s": _load_mode,
    "40s":          _load_mode,
    "average":      _load_mode,
    "window":       _load_window,
}

# Which rnufft key to save as channel A per mode
_INPUT_KEY = {
    "standard_20s": "rnufft",
    "40s":          "rnufft_200",
    "average":      "rnufft_all",
}


# =========================================================================
# Main class
# =========================================================================
class DataSplitter:
    """Load reconstructions, preprocess slices, and write training/validation/calibration/test splits.

    Parameters
    ----------
    source_dirs : list[str | Path]
        One or more root directories containing per-exam study folders.
    output_dir : str | Path
        Destination for ``.npz`` outputs, plots, and split pickle files.
    reco_subdir : str
        Name of the reconstruction subfolder inside each study
        (e.g. ``'FISTA_results_complete_julia'``).
    preprocessing_mode : str
        One of ``'standard_20s'``, ``'40s'``, ``'average'``, or ``'window'``.
    plot_dir : str | Path | None
        Root directory for validation PNG plots.  ``None`` disables plotting.
    """

    def __init__(
        self,
        source_dirs: list[str | Path],
        output_dir: str | Path,
        reco_subdir: str,
        preprocessing_mode: str = "standard_20s",
        plot_dir: str | Path | None = None,
    ) -> None:
        if preprocessing_mode not in MODES:
            raise ValueError(
                f"Unknown preprocessing_mode {preprocessing_mode!r}. "
                f"Choose from {MODES}."
            )
        self.output_dir = Path(output_dir)
        self.reco_subdir = reco_subdir
        self.preprocessing_mode = preprocessing_mode
        self.plot_dir = Path(plot_dir) if plot_dir is not None else None

        # Discover studies across all source directories
        self.studies: list[Path] = []
        for src in source_dirs:
            src = Path(src)
            self.studies.extend(
                d for d in src.iterdir() if d.is_dir() and d.name != "Archive"
            )
        self.studies.sort()

    # ------------------------------------------------------------------ #
    # Per-study extraction
    # ------------------------------------------------------------------ #
    def extract_study(
        self,
        study: Path,
        mode: str,
        split_list: List[str],
        *,
        save: bool = True,
        plot: bool = True,
    ) -> List[str]:
        """Extract, preprocess, and (optionally) save slices for one study.

        Parameters
        ----------
        study : Path
            Path to the study directory.
        mode : str
            ``'training'``, ``'validation'``, `'calibration'``, or ``'test'``.
        split_list : list[str]
            Accumulator — new slice identifiers are appended in place.
        save : bool
            If *False*, only populate the split list (dry-run).
        plot : bool
            If *False*, skip validation plot generation.

        Returns
        -------
        list[str]
            The updated *split_list*.

        Notes
        -----
        Slices within *BOUNDARY_MARGIN* indices of the study's start / end
        boundary are excluded from the split list so that neighbouring-slice
        augmentation is always possible downstream.
        """
        logger.info(
            "Processing %-30s  [%s]  (%s)",
            study.name, mode, self.preprocessing_mode,
        )

        study_files = study / self.reco_subdir
        start, end = SLICE_RANGE[study.name]
        glob_pat = _GLOB_PATTERN[self.preprocessing_mode]
        loader = _LOADERS[self.preprocessing_mode]

        # Discover and filter slices
        all_mat = natsorted(study_files.glob(glob_pat))
        slice_files = [
            f for f in all_mat if start <= parse_slice_number(f.name) <= end
        ]

        # Prepare plot directory
        plot_path = None
        if plot and self.plot_dir is not None:
            plot_path = self.plot_dir / study.name
            plot_path.mkdir(parents=True, exist_ok=True)

        for slice_file in slice_files:
            slice_num = parse_slice_number(slice_file.name)
            logger.debug("  slice %d  (%s)", slice_num, slice_file.name)

            if save:
                t0 = time.time()
                data = loader(slice_file)
                logger.debug("    load+preprocess %.2f s", time.time() - t0)

                if self.preprocessing_mode in ("standard_20s", "40s", "average"):
                    self._save_mode(
                        data, study, mode, slice_num,
                        slice_file, plot_path, start, end, plot,
                    )
                elif self.preprocessing_mode == "window":
                    self._save_window(
                        data, study, mode, slice_num,
                        slice_file, plot_path, start, end, plot,
                    )

            # Add to split list if far enough from boundaries
            if (slice_num - BOUNDARY_MARGIN >= start
                    and slice_num + BOUNDARY_MARGIN <= end):
                split_list.append(f"{study.name}_slice_{slice_num}")

        return split_list

    # ---- mode-specific save helpers ---------------------------------- #

    def _save_mode(
        self, data: dict, study: Path, mode: str, slice_num: int,
        slice_file: Path, plot_path: Path | None,
        start: int, end: int, do_plot: bool,
    ) -> None:
        """Save A, B, and C for standard_20s / 40s / average modes.

        Channel A is selected via the ``_INPUT_KEY`` mapping.
        """
        ensure_dirs(self.output_dir, mode, ("A", "B", "C"))
        base = f"{study.name}_slice_{slice_num}"
        input_key = _INPUT_KEY[self.preprocessing_mode]

        t0 = time.time()
        np.savez(self.output_dir / mode / "A" / base, rnufft=data[input_key])
        np.savez(self.output_dir / mode / "B" / base, grasp=data["grasp"])
        np.savez(self.output_dir / mode / "C" / base, softgating=data["softgating"])
        logger.debug("    save %.2f s", time.time() - t0)

        if (do_plot and plot_path
                and is_representative_slice(slice_num, start, end)):
            plot_standard_comparison(
                data["grasp_raw"], data["grasp"], slice_file, plot_path, GRASP_PHASE_INDEX, CROP_PX
            )
            plot_diff(data["grasp"], data[input_key], slice_file, plot_path)

    def _save_window(
        self, data: dict, study: Path, mode: str, slice_num: int,
        slice_file: Path, plot_path: Path | None,
        start: int, end: int, do_plot: bool,
    ) -> None:
        """Save per-window A arrays for window mode (no B/C channels)."""
        ensure_dirs(self.output_dir, mode, ("A",))
        windows = data["rnufft_windows"]

        for w in range(windows.shape[0]):
            base = f"{study.name}_slice_{slice_num}_window_{w}"
            t0 = time.time()
            np.savez(self.output_dir / mode / "A" / base, rnufft=windows[w])
            logger.debug("    save window %d  %.2f s", w, time.time() - t0)

            if (do_plot and plot_path
                    and is_representative_slice(slice_num, start, end)):
                plot_window(windows[w], w, slice_file, plot_path)


    # ------------------------------------------------------------------ #
    # Top-level driver
    # ------------------------------------------------------------------ #
    def run(
        self, *, save: bool = True, plot: bool = True,
    ) -> tuple[List[str], List[str], List[str]]:
        """Iterate over all studies and produce training / validation / calibration / test splits.

       Exam-level assignment is governed by :func:`split_config_retro_cal.get_split`.

        Parameters
        ----------
        save : bool
            If *False*, skip writing ``.npz`` files and plots (dry-run).
        plot : bool
            If *False*, skip validation plot generation even when *save*
            is ``True``.

        Returns
        -------
        tuple[list[str], list[str], list[str]]
            ``(training_list, validation_list, calibration_list, test_list)`` of slice identifiers.
        """
        lists: dict[str, List[str]] = {
            "training": [],
            "validation": [],
            "calibration": [],
            "test": [],
        }

        for study in self.studies:
            split = get_split(study.name)
            self.extract_study(
                study, split, lists[split], save=save, plot=plot,
            )

        # Persist split lists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_list in lists.items():
            save_list(self.output_dir / f"{split_name}.pkl", split_list)
            logger.info("%s list length: %d", split_name, len(split_list))

        return lists["training"], lists["validation"], lists["calibration"], lists["test"]


# =========================================================================
# CLI entry point
# =========================================================================
def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Preprocess reconstructed MRI slices and split "
                    "into training / validation / test.",
    )
    p.add_argument(
        "--mode", choices=MODES, default="standard_20s",
        help="Preprocessing mode (default: %(default)s).",
    )
    p.add_argument(
        "--save", action="store_true", default=False,
        help="Actually write .npz files (default is dry-run).",
    )
    p.add_argument(
        "--plot", action="store_true", default=False,
        help="Write validation comparison PNGs.",
    )
    return p

GLOBAL_INPATH = os.environ["DEMO_PATH"]
RUN_NAME = os.environ["RUN_NAME"]
GLOBAL_OUTPATH = os.environ["DEMO_PATH"]
DATAFOLDER_NAME = os.environ["DATAFOLDER_NAME"]


# Default paths per mode (edit as needed)
_DEFAULT_PATHS: dict[str, dict[str, str]] = {
    "standard_20s": {
        "output":   Path(GLOBAL_OUTPATH, RUN_NAME),
        "plot_dir": Path(GLOBAL_OUTPATH, RUN_NAME + "_plots/"),
        "reco_subdir": "FISTA_results",
    },
    "40s": {
        "output":   Path(GLOBAL_OUTPATH, RUN_NAME),
        "plot_dir": Path(GLOBAL_OUTPATH, RUN_NAME + "_plots/"),
        "reco_subdir": "FISTA_results",
    },
    "average": {
        "output":   Path(GLOBAL_OUTPATH, RUN_NAME),
        "plot_dir": Path(GLOBAL_OUTPATH, RUN_NAME + "_plots/"),
        "reco_subdir": "FISTA_results",
    },
    "window": {
        "output":   Path(GLOBAL_OUTPATH, RUN_NAME + "_windows"),
        "plot_dir": Path(GLOBAL_OUTPATH, RUN_NAME + "_plots/"),
        "reco_subdir": "FISTA_results_windows",
    },
}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    args = _build_parser().parse_args()

    SOURCE_DIRS = [
        Path(GLOBAL_INPATH, DATAFOLDER_NAME)
        ]

    paths = _DEFAULT_PATHS[args.mode]

    splitter = DataSplitter(
        source_dirs=SOURCE_DIRS,
        output_dir=paths["output"],
        reco_subdir=paths["reco_subdir"],
        preprocessing_mode=args.mode,
        plot_dir=paths["plot_dir"],
    )
    splitter.run(save=args.save, plot=args.plot)
