
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional dependency
    nib = None

def _gaussian_heatmap(shape: Tuple[int, int, int], center: Tuple[float, float, float], sigma: float) -> np.ndarray:
    """Create a single 3D Gaussian heatmap with peak 1.0."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = center
    dist2 = np.square(x - cx)+ np.square(y - cy) + np.square(z - cz)
    heatmap = np.exp(-dist2 / (2.0 * sigma ** 2))
    peak = heatmap.max()
    if peak > 0:
        heatmap = heatmap / peak
    return heatmap.astype(np.float32)


def make_heatmaps(shape: Tuple[int, int, int], center: Tuple[float, float, float], sigmas: Sequence[float]) -> np.ndarray:
    """Return stacked 3D Gaussian heatmaps for each sigma."""
    if not sigmas:
        return np.zeros((1,) + tuple(shape), dtype=np.float32)
    heatmaps = [_gaussian_heatmap(shape, center, sigma) for sigma in sigmas]
    return np.stack(heatmaps, axis=0)

HYPERPARAMS = {
    "h5": Path("h5-aneurysm.h5"),
    "localizers": Path("train_localizers.csv"),
    "series_id": "1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381",
    "heatmap_sizes": [15.0, 10.0, 5.0],
    "output_prefix": Path("outputs/gaussian_heatmaps"),
}


def _load_centers(csv_path: Path, series_id: str) -> List[Tuple[float, float, float]]:
    """Load (z, y, x) centers for a given series from a localizer CSV."""
    centers: List[Tuple[float, float, float]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        required = {"SeriesInstanceUID", "x_new", "y_new", "z_new"}
        alt = {"series_id", "new_x", "new_y", "new_z"}
        if not required.issubset(reader.fieldnames or set()):
            if alt.issubset(reader.fieldnames or set()):
                keys = ("series_id", "new_z", "new_y", "new_x")
            else:
                raise ValueError(
                    f"Localizer CSV missing required columns. Expected {required} or {alt}."
                )
        else:
            keys = ("SeriesInstanceUID", "z_new", "y_new", "x_new")

        for row in reader:
            sid = row.keys() and row.get(keys[0])
            if sid != series_id:
                continue
            try:
                z = float(row[keys[1]])
                y = float(row[keys[2]])
                x = float(row[keys[3]])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid coordinate row: {row}") from exc
            centers.append((z, y, x))
    if not centers:
        print(f"[Info] No centers found for series_id={series_id} in {csv_path}")
    return centers


def _save_nifti(array: np.ndarray, path: Path) -> None:
    """Save numpy array as NIfTI (identity affine)."""
    if nib is None:
        raise ImportError(
            "nibabel is required for NIfTI export. Install with `pip install nibabel`."
        )
    affine = np.eye(4, dtype=np.float32)
    nii = nib.Nifti1Image(array.astype(np.float32), affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(path))


def build_heatmaps(
    volume_shape: Tuple[int, int, int], centers: Sequence[Tuple[float, float, float]], sigmas: Sequence[float]
) -> np.ndarray:
    """Create stacked heatmaps, taking max over overlapping aneurysms."""
    heatmap_stack = np.zeros((len(sigmas),) + tuple(volume_shape), dtype=np.float32)
    for center in centers:
        hm = make_heatmaps(volume_shape, center, sigmas=sigmas)
        heatmap_stack = np.maximum(heatmap_stack, hm)
    return heatmap_stack



def main() -> None:
    if not HYPERPARAMS["series_id"]:
        raise ValueError("series_id is required. Pass --series-id or set it in HYPERPARAMS.")

    if not HYPERPARAMS["h5"].exists():
        raise FileNotFoundError(f"H5 file not found: {HYPERPARAMS['h5']}")
    if not HYPERPARAMS["localizers"].exists():
        raise FileNotFoundError(f"Localizers CSV not found: {HYPERPARAMS['localizers']}")

    series_id = HYPERPARAMS["series_id"]
    print(f"[1/4] Loading volume for {series_id} from {HYPERPARAMS['h5']} ...")
    with h5py.File(HYPERPARAMS["h5"], "r") as handle:
        if "series" not in handle or series_id not in handle["series"]:
            raise KeyError(f"Series {series_id} not found in {HYPERPARAMS['h5']}")
        vol_ds = handle["series"][series_id]["vol"]
        volume = vol_ds[:]

    print(f"[2/4] Loading localizer centers from {HYPERPARAMS['localizers']} ...")
    centers = _load_centers(HYPERPARAMS["localizers"], series_id)

    print(f"[3/4] Building heatmaps with sigmas {HYPERPARAMS['heatmap_sizes']} ...")
    heatmaps = build_heatmaps(volume.shape, centers, HYPERPARAMS["heatmap_sizes"])

    # Reorder to (Z, Y, X, K) for NIfTI export
    heatmaps_4d = np.moveaxis(heatmaps, 0, -1)
    merged = heatmaps.max(axis=0) if heatmaps.size else np.zeros_like(volume, dtype=np.float32)

    print("[4/4] Saving NIfTI files ...")
    _save_nifti(volume.astype(np.float32), HYPERPARAMS["output_prefix"].with_suffix(".volume.nii.gz"))
    _save_nifti(heatmaps_4d, HYPERPARAMS["output_prefix"].with_suffix(".heatmaps.nii.gz"))
    _save_nifti(merged, HYPERPARAMS["output_prefix"].with_suffix(".heatmap_max.nii.gz"))
    print("Done.")


if __name__ == "__main__":
    main()
