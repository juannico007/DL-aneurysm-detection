from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


class ScanDataLoader:
    """Utility for enumerating series IDs and labels from the HDF5 dataset."""

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(
        self,
        h5_path: Path,
        csv_path: Path,
        max_shape: Tuple[int, int, int] = (256, 256, 256),
    ):
        train_channel = os.environ.get("SM_CHANNEL_TRAIN")
        self.train_channel = Path(train_channel) if train_channel else None

        self.h5_path = self._resolve_data_path(Path(h5_path))
        self.csv_path = self._resolve_data_path(Path(csv_path))

        self.max_shape = max_shape
        self.series_ids: List[str] = []
        self.labels: List[int] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        df = self._load_csv()
        if "Aneurysm Present" not in df.columns:
            raise ValueError("'Aneurysm Present' column missing from labels CSV.")

        with h5py.File(self.h5_path, "r") as handle:
            if "series" not in handle:
                raise KeyError("Expected group 'series' in HDF5 file.")
            series_group = handle["series"]
            for pid, row in df.iterrows():
                if pid not in series_group:
                    print(f"[Warning] Series {pid} missing in HDF5, skipping.")
                    continue

                group = series_group[pid]
                if "vol" not in group:
                    print(f"[Warning] Missing 'vol' dataset for {pid}, skipping.")
                    continue

                shape = tuple(group["vol"].shape[-3:])
                if not self._within_bounds(shape):
                    continue

                label = self._parse_label(row["Aneurysm Present"])
                if label is None:
                    print(f"[Warning] Invalid label for {pid}, skipping.")
                    continue

                self.series_ids.append(pid)
                self.labels.append(label)

        positives = int(np.sum(self.labels))
        print(
            f"Loaded {len(self.series_ids)} scans: "
            f"{positives} positive, {len(self.labels) - positives} negative"
        )

    def _load_csv(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        if "SeriesInstanceUID" not in df.columns:
            raise ValueError(f"'SeriesInstanceUID' column missing from {self.csv_path}")
        return df.set_index("SeriesInstanceUID")

    @staticmethod
    def _parse_label(raw) -> Optional[int]:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return None

    def split_data(
        self, train_ratio: float = 0.7
    ) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
        if not self.series_ids:
            raise ValueError("Dataset is empty. Ensure the HDF5 file is populated.")

        labels = np.array(self.labels, dtype=int)
        positive_idx = np.where(labels == 1)[0]
        negative_idx = np.where(labels == 0)[0]

        pos_train, pos_val = self._split_indices(positive_idx, train_ratio)
        neg_train, neg_val = self._split_indices(negative_idx, train_ratio)

        train_idx = self._concat_indices([pos_train, neg_train])
        val_idx = self._concat_indices([pos_val, neg_val])

        rng = np.random.default_rng()
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        x_train = [self.series_ids[i] for i in train_idx]
        y_train = labels[train_idx]
        x_val = [self.series_ids[i] for i in val_idx]
        y_val = labels[val_idx]

        print(
            f"\nDataset split:\n"
            f"  Training: {len(x_train)} samples "
            f"(pos: {int(np.sum(y_train))}, neg: {len(y_train) - int(np.sum(y_train))})\n"
            f"  Validation: {len(x_val)} samples "
            f"(pos: {int(np.sum(y_val))}, neg: {len(y_val) - int(np.sum(y_val))})"
        )

        return x_train, y_train, x_val, y_val

    def _split_indices(
        self, indices: np.ndarray, train_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if indices.size == 0:
            empty = np.empty(0, dtype=int)
            return empty, empty
        cutoff = int(train_ratio * len(indices))
        return indices[:cutoff], indices[cutoff:]

    @staticmethod
    def _concat_indices(parts: List[np.ndarray]) -> np.ndarray:
        filtered = [p for p in parts if p.size > 0]
        if not filtered:
            return np.empty(0, dtype=int)
        if len(filtered) == 1:
            return filtered[0].copy()
        return np.concatenate(filtered)

    def _within_bounds(self, shape: Tuple[int, int, int]) -> bool:
        return all(dim <= limit for dim, limit in zip(shape, self.max_shape))

    def _resolve_data_path(self, candidate: Path) -> Path:
        """Locate `candidate` relative to the SageMaker channel, repo root, or CWD."""
        if candidate.is_absolute():
            return candidate

        relative_candidates = [candidate]
        if candidate.suffix == "":
            relative_candidates.append(candidate.with_name(candidate.name + ".h5"))

        search_roots = []
        if self.train_channel is not None:
            search_roots.append(self.train_channel)
        search_roots.append(self.PROJECT_ROOT)
        search_roots.append(Path("."))

        for root in search_roots:
            for option in relative_candidates:
                path = root / option
                if path.exists():
                    return path

        # If nothing exists, prefer the channel path when available for clearer errors.
        if self.train_channel is not None:
            return self.train_channel / candidate
        return self.PROJECT_ROOT / candidate
