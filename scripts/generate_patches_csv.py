"""
Generate a CSV/JSON index of 3D patches without touching the source H5.

Outputs:
  - patches/patches.csv with columns: series_id, x, y, z, label
    (x/y/z are patch centers in voxel indices; label 1=aneurysm, 0=background)
  - patches/metadata.json with patch_size, pos_neg_ratio, radius, margin, etc.

Usage:
  Adjust HYPERPARAMS at the top of this file, then run:
  python3 scripts/generate_patches_csv.py
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py


# Editable hyperparameters (similar to training_pipeline style)
HYPERPARAMS = {
    "h5": Path("h5-aneurysm.h5"),
    "localizers": Path("train_localizers.csv"),
    "output_dir": Path("patches"),
    "patch_size": 64,
    "pos_neg_ratio": "1:5",
    "radius": 5.0,
    "margin": 10.0,
    "center_gap": 4.0,
    "seed": 13,
}


def parse_ratio(ratio: str) -> Tuple[int, int, float]:
    """
    Parse a ratio string like '1:5' or a float such as '0.2' (pos/(pos+neg)).

    Returns (pos, neg, neg_per_pos_float).
    """
    ratio = ratio.strip()
    if ":" in ratio:
        parts = ratio.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid ratio format: {ratio}")
        pos = int(parts[0])
        neg = int(parts[1])
    else:
        val = float(ratio)
        if not (0 < val < 1):
            raise ValueError("If ratio is a float, it must be in (0,1) representing pos fraction.")
        # pos_fraction = val = pos / (pos+neg)
        pos = 1
        neg = max(1, int(round((1 - val) / val)))
    neg_per_pos = neg / max(pos, 1)
    return pos, neg, neg_per_pos


def load_localizer_points(csv_path: Path) -> Dict[str, List[Tuple[float, float, float]]]:
    """Load (z_new, y_new, x_new, location) points grouped by series_id."""
    points: Dict[str, List[Tuple[float, float, float, str]]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        required = {"SeriesInstanceUID", "x_new", "y_new", "z_new", "location"}
        missing = required.difference(reader.fieldnames or {})
        if missing:
            raise ValueError(f"Localizer CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            sid = row["SeriesInstanceUID"]
            try:
                x = float(row["x_new"])
                y = float(row["y_new"])
                z = float(row["z_new"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric localizer row for series {sid}: {row}") from exc
            loc = row.get("location", "").strip()
            points[sid].append((z, y, x, loc))  # store in (z, y, x, location)
    return points


def point_to_aabb_sqdist(point: Sequence[float], box_min: Sequence[float], box_max: Sequence[float]) -> float:
    """Squared distance from a point to an axis-aligned box (0 if inside)."""
    dist = 0.0
    for p, mn, mx in zip(point, box_min, box_max):
        if p < mn:
            dist += (mn - p) ** 2
        elif p > mx:
            dist += (p - mx) ** 2
    return dist


def sample_positive_center(
    center: Sequence[float],
    shape: Sequence[int],
    patch_size: int,
    margin_buffer: float,
    center_gap: float,
    rng: random.Random,
    max_attempts: int = 50,
) -> Optional[Tuple[float, float, float]]:
    """
    Sample a patch center so that the aneurysm is inside the patch with a buffer
    and not sitting at the patch center.
    """
    max_starts = [dim - patch_size for dim in shape]
    for _ in range(max_attempts):
        starts = []
        rels = []
        valid = True
        for c, dim, max_start in zip(center, shape, max_starts):
            low = max(0, math.ceil(c - (patch_size - margin_buffer)))
            high = min(max_start, math.floor(c - margin_buffer))
            if low > high:
                valid = False
                break
            s = rng.randint(int(low), int(high))
            r = c - s  # position of aneurysm within the patch
            if r < margin_buffer or r > (patch_size - margin_buffer):
                valid = False
                break
            rels.append(r)
            starts.append(s)
        if not valid:
            continue
        # Ensure aneurysm is not near the patch center
        half = patch_size / 2.0
        if any(abs(r - half) < center_gap for r in rels):
            continue
        center_out = tuple(s + half for s in starts)
        return center_out
    return None


def sample_negative_center(
    shape: Sequence[int],
    positive_centers: Iterable[Sequence[float]],
    patch_size: int,
    exclusion_radius: float,
    rng: random.Random,
    max_attempts: int = 200,
) -> Optional[Tuple[float, float, float]]:
    """Sample a negative patch center that does not intersect any positive sphere."""
    max_start = [dim - patch_size for dim in shape]
    r2 = exclusion_radius**2
    for _ in range(max_attempts):
        starts = [rng.randint(0, ms) for ms in max_start]
        half = patch_size / 2.0
        center = tuple(s + half for s in starts)
        box_min = starts
        box_max = [s + patch_size for s in starts]
        # reject if any aneurysm sphere intersects this patch
        intersects = False
        for pos in positive_centers:
            if point_to_aabb_sqdist(pos, box_min, box_max) <= r2:
                intersects = True
                break
        if intersects:
            continue
        return center
    return None


def generate_patches(
    h5_path: Path,
    localizer_csv: Path,
    output_dir: Path,
    patch_size: int = 64,
    pos_neg_ratio: str = "1:5",
    radius: float = 5.0,
    margin: float = 10.0,
    center_gap: float = 4.0,
    seed: int = 13,
) -> None:
    rng = random.Random(seed)
    pos_count, neg_count, neg_per_pos = parse_ratio(pos_neg_ratio)
    if pos_count != 1:
        print(f"[Info] Ratio {pos_neg_ratio} normalized to 1:{neg_per_pos:.2f} for sampling.")

    localizers = load_localizer_points(localizer_csv)
    if not localizers:
        raise ValueError("No localizer points found; cannot generate positive patches.")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "patches.csv"
    meta_path = output_dir / "metadata.json"

    rows: List[List[str]] = []
    total_pos = 0
    total_neg = 0

    with h5py.File(h5_path, "r") as handle:
        series_group = handle["series"]
        for sid, points in localizers.items():
            if sid not in series_group:
                print(f"[Warning] Series {sid} missing in H5, skipping.")
                continue
            vol = series_group[sid]["vol"]
            shape = vol.shape  # (D, H, W)
            if any(dim < patch_size for dim in shape):
                print(f"[Warning] Series {sid} too small for patch size {patch_size}, skipping.")
                continue

            for point in points:
                aneurysm_z, aneurysm_y, aneurysm_x, aneurysm_loc = point
                margin_buffer = radius + margin
                pos_center = sample_positive_center(
                    center=(aneurysm_z, aneurysm_y, aneurysm_x),
                    shape=shape,
                    patch_size=patch_size,
                    margin_buffer=margin_buffer,
                    center_gap=center_gap,
                    rng=rng,
                )
                if pos_center is None:
                    print(f"[Warning] Could not place positive patch for {sid} at {point}.")
                    continue

                # Compute aneurysm position in patch coordinates (relative to patch corner)
                patch_origin = (
                    pos_center[0] - patch_size / 2.0,
                    pos_center[1] - patch_size / 2.0,
                    pos_center[2] - patch_size / 2.0,
                )
                aneurysm_rel = (
                    aneurysm_x - patch_origin[2],
                    aneurysm_y - patch_origin[1],
                    aneurysm_z - patch_origin[0],
                )

                # Store as x, y, z in the CSV
                rows.append([
                    sid,
                    f"{pos_center[2]:.3f}",  # center_x
                    f"{pos_center[1]:.3f}",  # center_y
                    f"{pos_center[0]:.3f}",  # center_z
                    f"{aneurysm_x:.3f}",
                    f"{aneurysm_y:.3f}",
                    f"{aneurysm_z:.3f}",
                    f"{aneurysm_rel[0]:.3f}",  # aneurysm_x in patch coords
                    f"{aneurysm_rel[1]:.3f}",
                    f"{aneurysm_rel[2]:.3f}",
                    "1",
                    aneurysm_loc or "unknown",
                ])
                total_pos += 1

                neg_needed = int(math.ceil(neg_per_pos))
                for _ in range(neg_needed):
                    neg_center = sample_negative_center(
                        shape=shape,
                        positive_centers=[(aneurysm_z, aneurysm_y, aneurysm_x) for aneurysm_z, aneurysm_y, aneurysm_x, _ in points],
                        patch_size=patch_size,
                        exclusion_radius=radius + margin,
                        rng=rng,
                    )
                    if neg_center is None:
                        print(f"[Warning] Could not sample negative patch for {sid}; skipping one.")
                        continue
                    rows.append([
                        sid,
                        f"{neg_center[2]:.3f}",
                        f"{neg_center[1]:.3f}",
                        f"{neg_center[0]:.3f}",
                        "", "", "",
                        "", "", "",
                        "0",
                        "background",
                    ])
                    total_neg += 1

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "series_id",
            "center_x",
            "center_y",
            "center_z",
            "aneurysm_x",
            "aneurysm_y",
            "aneurysm_z",
            "aneurysm_rel_x",
            "aneurysm_rel_y",
            "aneurysm_rel_z",
            "label",
            "location",
        ])
        writer.writerows(rows)

    metadata = {
        "patch_size": patch_size,
        "pos_neg_ratio": pos_neg_ratio,
        "neg_per_pos_used": float(f"{neg_per_pos:.3f}"),
        "radius": radius,
        "margin": margin,
        "center_gap": center_gap,
        "seed": seed,
        "coords": (
            "center_x/center_y/center_z are patch centers in voxel space; "
            "aneurysm_x/y/z are global voxel coords from localizers; "
            "aneurysm_rel_* are voxel coords within the patch (origin at patch corner). "
        ),
        "counts": {"positive": total_pos, "negative": total_neg, "total": len(rows)},
    }
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Done] Wrote {len(rows)} rows to {csv_path}")
    print(f"[Done] Metadata saved to {meta_path}")


def main() -> None:
    params = HYPERPARAMS
    generate_patches(
        h5_path=Path(params["h5"]),
        localizer_csv=Path(params["localizers"]),
        output_dir=Path(params["output_dir"]),
        patch_size=int(params["patch_size"]),
        pos_neg_ratio=str(params["pos_neg_ratio"]),
        radius=float(params["radius"]),
        margin=float(params["margin"]),
        center_gap=float(params["center_gap"]),
        seed=int(params["seed"]),
    )


if __name__ == "__main__":
    main()
