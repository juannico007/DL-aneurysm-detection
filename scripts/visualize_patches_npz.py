"""
Export a single series to 3D Slicer-friendly files:
  - Volume NIfTI
  - Mask NIfTI (spheres at aneurysm points)
  - Patch centers as Slicer Markups (.fcsv, IJK coordinates)
  - Aneurysm centers as Slicer Markups (.fcsv, IJK coordinates)

Configure HYPERPARAMS below, then run:
    python3 scripts/visualize_patches_npz.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
try:
    import nibabel as nib
except ImportError:  # pragma: no cover - graceful fallback message
    nib = None


HYPERPARAMS: Dict[str, object] = {
    "h5": Path("h5-aneurysm.h5"),
    "localizers": Path("train_localizers.csv"),
    "patch_csv": Path("patches/patches.csv"),
    "series_id": "1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317",
    "radius": 5.0,
    "patch_size": 64,
    "output_prefix": Path("patches_visualizer/vis_"),
}

@dataclass
class ExportMeta:
    series_id: str
    radius: float
    patch_size: int
    h5: str
    localizers: str
    patch_csv: str

def load_localizer_points(csv_path: Path) -> Dict[str, List[Tuple[float, float, float]]]:
    """Return mapping series_id -> list of (z, y, x)."""
    out: Dict[str, List[Tuple[float, float, float]]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        required = {"SeriesInstanceUID", "x_new", "y_new", "z_new"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Localizer CSV missing columns: {required}")
        for row in reader:
            sid = row["SeriesInstanceUID"]
            try:
                z = float(row["z_new"])
                y = float(row["y_new"])
                x = float(row["x_new"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Bad localizer row: {row}") from exc
            out.setdefault(sid, []).append((z, y, x))
    return out


def make_sphere_mask(shape: Sequence[int], center: Sequence[float], radius: float) -> np.ndarray:
    """Return a binary sphere mask for one center."""
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center
    dist2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return (dist2 <= radius**2).astype(np.uint8)


def load_patches_for_series(patch_csv: Path, series_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load patch centers/labels for a given series."""
    centers: List[Tuple[float, float, float]] = []
    labels: List[int] = []
    with patch_csv.open() as f:
        reader = csv.DictReader(f)
        required = {"series_id", "center_x", "center_y", "center_z", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Patches CSV missing columns: {required}")
        for row in reader:
            if row["series_id"] != series_id:
                continue
            centers.append(
                (float(row["center_z"]), float(row["center_y"]), float(row["center_x"]))
            )
            labels.append(int(row["label"]))
    if not centers:
        print(f"[Info] No patch entries found for series {series_id} in {patch_csv}")
    return np.asarray(centers, dtype=np.float32), np.asarray(labels, dtype=np.uint8)


def write_slicer_fcsv(path: Path, points: Iterable[Tuple[float, float, float]], labels: Iterable[str]) -> None:
    """Write Slicer Markups FCSV with IJK coordinates."""
    header_lines = [
        "# Markups fiducial file version = 4.11",
        "# CoordinateSystem = 2",  # 2 = IJK (voxel indices)
        "# columns = id,label,desc,associatedNodeID,x,y,z,fixed,locked,visibility,positional,display",
    ]
    rows = []
    for idx, (p, lbl) in enumerate(zip(points, labels)):
        z, y, x = p  # stored as (z, y, x)
        rows.append([
            f"F-{idx:03d}",
            lbl,
            "",
            "",
            f"{x:.3f}",
            f"{y:.3f}",
            f"{z:.3f}",
            "0",
            "0",
            "1",
            "1",
            "1",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        for line in header_lines:
            f.write(line + "\n")
        writer = csv.writer(f)
        writer.writerows(rows)


def main() -> None:
    if nib is None:
        raise ImportError("nibabel is required for NIfTI export. Install with `pip install nibabel` in your environment.")
    p = HYPERPARAMS
    h5_path = Path(p["h5"])
    loc_path = Path(p["localizers"])
    patch_csv = Path(p["patch_csv"])
    series_id = str(p["series_id"])
    radius = float(p["radius"])
    patch_size = int(p["patch_size"])
    output_prefix = Path(p["output_prefix"])

    localizers = load_localizer_points(loc_path)
    centers = localizers.get(series_id, [])

    if not centers:
        raise ValueError(f"No localizer points found for series {series_id}")

    with h5py.File(h5_path, "r") as f:
        if "series" not in f or series_id not in f["series"]:
            raise KeyError(f"Series {series_id} not found in {h5_path}")
        vol = f["series"][series_id]["vol"][:]

    # normalize dtype and clean NaNs for viewer compatibility
    vol = np.asarray(vol, dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    # build mask from all aneurysm points
    mask = np.zeros_like(vol, dtype=np.uint8)
    for c in centers:
        mask |= make_sphere_mask(vol.shape, c, radius=radius)

    patch_centers, patch_labels = load_patches_for_series(patch_csv, series_id)
    if patch_centers.size == 0:
        patch_centers = patch_centers.reshape(0, 3)
    else:
        patch_centers = patch_centers.astype(np.float32, copy=False)
    patch_labels = patch_labels.astype(np.uint8, copy=False)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # NIFTI EXPORT FOR 3D SLICER
    # -------------------------------------------------------------

    # Use identity affine; replace if real spacing/orientation is known.
    affine_matrix = np.eye(4) 

    vol_nii_path = output_prefix.with_suffix(".volume.nii.gz")
    nii_vol = nib.Nifti1Image(vol, affine_matrix)
    nib.save(nii_vol, vol_nii_path)
    print(f"[Done] Saved Volume NIfTI to {vol_nii_path}")

    mask_nii_path = output_prefix.with_suffix(".mask.nii.gz")
    nii_mask = nib.Nifti1Image(mask, affine_matrix)
    nib.save(nii_mask, mask_nii_path)
    print(f"[Done] Saved Mask NIfTI to {mask_nii_path}")

    # -------------------------------------------------------------
    # Markups: patch centers and aneurysm centers (IJK)
    # -------------------------------------------------------------
    patch_labels_text = ["Positive_Patch" if lbl == 1 else "Negative_Patch" for lbl in patch_labels]
    patch_fcsv = output_prefix.with_suffix(".patch_centers.fcsv")
    write_slicer_fcsv(patch_fcsv, patch_centers, patch_labels_text)
    print(f"[Done] Saved Patch Centers Markups file to {patch_fcsv}")

    aneurysm_fcsv = output_prefix.with_suffix(".aneurysm_centers.fcsv")
    aneurysm_labels = [f"Aneurysm_{i:03d}" for i in range(len(centers))]
    write_slicer_fcsv(aneurysm_fcsv, centers, aneurysm_labels)
    print(f"[Done] Saved Aneurysm Centers Markups file to {aneurysm_fcsv}")

    # Optional: save metadata as JSON for reference
    meta = ExportMeta(
        series_id=series_id,
        radius=radius,
        patch_size=patch_size,
        h5=str(h5_path),
        localizers=str(loc_path),
        patch_csv=str(patch_csv),
    )
    meta_path = output_prefix.with_suffix(".meta.json")
    with meta_path.open("w") as f:
        json.dump(asdict(meta), f, indent=2)
    print(f"[Done] Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
