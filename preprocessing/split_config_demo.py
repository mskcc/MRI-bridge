"""Configuration for splitting scripts of retrospectively accelerated example data including calibration split.

Slice range convention
----------------------
``SLICE_RANGE[exam_name]`` → ``[start_slice, end_slice]`` (both inclusive).
Only slices whose index falls within this range are considered anatomically
relevant.  Values were determined by manual review of each acquisition.

Split assignment convention
---------------------------
``TEST_EXAMS``, ``CAL_EXAMS``,  ``VAL_EXAMS`` are explicit sets.  Every exam that does not
appear in either is assigned to the *training* split.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Exam-level split assignments
# ---------------------------------------------------------------------------
TEST_EXAMS: set[str] = {
    "Exam8271Series12",
}

CAL_EXAMS: set[str] = {
    "Exam884Series11",
}

VAL_EXAMS: set[str] = {
    "Exam7695-Series16",
}

# ---------------------------------------------------------------------------
# Per-exam anatomically relevant slice ranges  [start, end] (inclusive)
# ---------------------------------------------------------------------------
SLICE_RANGE: dict[str, list[int]] = {
    "Exam8271Series12":   [36, 40],
}


def get_split(exam_name: str) -> str:
    """Return the split name for a given exam.

    Parameters
    ----------
    exam_name : str
        The directory name of the exam (e.g. ``'Exam7502-Series14'``).

    Returns
    -------
    str
        One of ``'test'``, ``'validation'``, or ``'training'``.
    """
    if exam_name in TEST_EXAMS:
        return "test"
    if exam_name in CAL_EXAMS:
        return "calibration"
    if exam_name in VAL_EXAMS:
        return "validation"
    return "training"
