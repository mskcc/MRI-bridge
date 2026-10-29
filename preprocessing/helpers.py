from __future__ import annotations

import numpy as np
import pickle

from pathlib import Path

# =========================================================================
# Helpers
# =========================================================================
def save_list(filepath: str | Path, data: list) -> None:
    """Serialize a Python list to a pickle file.

    Parameters
    ----------
    filepath : str or Path
        Destination file path (created or overwritten).
    data : list
        The list to persist.
    """
    with open(filepath, "wb") as fh:
        pickle.dump(data, fh)


def parse_slice_number(filename: str) -> int:
    """Extract the slice index from a reconstruction filename.

    If '4ms' is in the name, take the 3rd entry, otherwise the last. 
    Parameters
    ----------
    filename : str
        The filename (with or without extension) of a reconstruction MAT file.

    Returns
    -------
    int
        The integer slice index.
    """
    if  '4ms' in filename:
        return int(Path(filename).stem.split("_")[3])
    else:
        return int(Path(filename).stem.split("_")[-1])


def is_representative_slice(slice_number: int, start: int, end: int) -> bool:
    """Return ``True`` if *slice_number* should receive a diagnostic plot.

    A handful of evenly-spaced slices are selected so the user can visually
    verify preprocessing quality without generating a plot for every slice.
    """
    return slice_number in {
        start,
        start + 10,
        (start + end) // 2,
        end - 10,
        end,
    }


def ensure_dirs(base: Path, mode_name: str, channels: tuple[str, ...]) -> None:
    """Create ``<base>/<mode_name>/<channel>/`` for each channel."""
    for ch in channels:
        (base / mode_name / ch).mkdir(parents=True, exist_ok=True)


def complex_to_magnitude(raw: np.ndarray, crop: int, NORM_PERCENTILE: float) -> np.ndarray:
    """Convert structured complex array -> cropped, rotated magnitude image."""
    cx = raw["real"] + 1j * raw["imag"]
    mag = np.abs(cx[0, crop:-crop, crop:-crop])
    return np.rot90(mag / np.percentile(mag, NORM_PERCENTILE), -1)


def clip_and_renorm(img: np.ndarray, NORM_PERCENTILE: float) -> np.ndarray:
    """Clip to 99.99th percentile and re-normalise."""
    img = np.clip(img, 0, np.percentile(img, 99.99))
    return img / np.percentile(img, NORM_PERCENTILE)

