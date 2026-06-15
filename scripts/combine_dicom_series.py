#!/usr/bin/env python3
"""
Combine three DICOM series into a single stacked series.

Each input directory should contain one DICOM series. All slices are pooled,
sorted by z-position (ImagePositionPatient[2], falling back to SliceLocation),
and written out as a new DICOM series with fresh UIDs and renumbered instances.

Usage:
    python combine_dicom_series.py series1/ series2/ series3/ -o combined/

Requires: pydicom, numpy
    pip install pydicom numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import generate_uid


def _z_position(ds: pydicom.Dataset) -> float:
    """Return the slice's z-coordinate for sorting."""
    if "ImagePositionPatient" in ds:
        return float(ds.ImagePositionPatient[2])
    if "SliceLocation" in ds:
        return float(ds.SliceLocation)
    if "InstanceNumber" in ds:
        return float(ds.InstanceNumber)
    return 0.0


def load_series(directory: Path) -> list[pydicom.Dataset]:
    """Load all DICOM image slices from a directory, sorted by z-position."""
    datasets: list[pydicom.Dataset] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path))
        except InvalidDicomError:
            continue
        if "PixelData" in ds:
            datasets.append(ds)

    if not datasets:
        raise ValueError(f"No DICOM image slices found in {directory}")

    datasets.sort(key=_z_position)
    return datasets


def combine_series(series_list: list[list[pydicom.Dataset]]) -> list[pydicom.Dataset]:
    """Merge multiple series into one list of slices sorted by z-position."""
    combined = [ds for series in series_list for ds in series]
    combined.sort(key=_z_position)
    return combined


def stack_volume(datasets: list[pydicom.Dataset]) -> np.ndarray:
    """Return pixel data as a 3D numpy array (z, y, x), applying rescale if present."""
    slices = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(arr * slope + intercept)
    return np.stack(slices, axis=0)


def write_combined_series(
    datasets: list[pydicom.Dataset],
    output_dir: Path,
    series_description: str | None = None,
) -> None:
    """Write slices as a new DICOM series with fresh SeriesInstanceUID."""
    output_dir.mkdir(parents=True, exist_ok=True)

    new_series_uid = generate_uid()
    desc = series_description or f"Combined ({len(datasets)} slices)"

    for i, ds in enumerate(datasets, start=1):
        ds.SeriesInstanceUID = new_series_uid
        ds.SOPInstanceUID = generate_uid()
        ds.InstanceNumber = i
        ds.SeriesDescription = desc

        # Keep file_meta consistent with the new SOPInstanceUID
        if hasattr(ds, "file_meta") and ds.file_meta is not None:
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        ds.save_as(str(output_dir / f"slice_{i:04d}.dcm"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine three DICOM series into a single stacked series."
    )
    parser.add_argument(
        "series",
        nargs=3,
        type=Path,
        help="Three directories, each containing one DICOM series.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for the combined DICOM series.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional SeriesDescription for the combined series.",
    )
    args = parser.parse_args()

    print("Loading series...")
    loaded = [load_series(d) for d in args.series]
    for i, (d, series) in enumerate(zip(args.series, loaded), start=1):
        print(f"  [{i}] {d}: {len(series)} slices")

    print("Combining and sorting by z-position...")
    combined = combine_series(loaded)
    print(f"  Total slices: {len(combined)}")

    volume = stack_volume(combined)
    print(f"  Volume shape (z, y, x): {volume.shape}, dtype: {volume.dtype}")

    print(f"Writing combined series to {args.output}...")
    write_combined_series(combined, args.output, args.description)
    print("Done.")


if __name__ == "__main__":
    main()
