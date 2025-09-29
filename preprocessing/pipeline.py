# preprocess/pipeline.py
from pathlib import Path
from typing import Optional, Literal
import multiprocessing
import pandas as pd
import itk
from .resample import resample_to_spacing
from .registry import REGISTRY
from .policy import ModalityPolicy

class Preprocess:
    """Coordinate the high-level preprocessing workflow for a dataset.

    Parameters
    ----------
    input_root : Path
        Directory that contains the raw data (expects a `series/` subfolder and
        metadata CSV files such as `train.csv`).
    output_root : Path
        Destination directory where processed volumes and metadata are written.
    num_workers : int, optional
        Maximum number of worker processes to spawn. If omitted the value is
        inferred from the machine's CPU count.
    itk_threads : int, optional
        Per-process thread budget for ITK operations. Auto-derived when not
        supplied.
    pipeline_version : str
        Identifier stored in the generated metadata for reproducibility.
    output_format : {"nii", "nii.gz"}
        File format extension used when saving preprocessed volumes.
    voxel_size : tuple[float, float, float]
        Target spacing (in millimetres) for resampled images.
    oversub_factor : float
        Multiplier applied to the CPU count to determine the shared thread pool
        budget used to balance processes versus ITK threads.
    """
    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        num_workers: Optional[int] = None,
        itk_threads: Optional[int] = None,
        pipeline_version: str = "0.0.1",
        output_format: Literal["nii", "nii.gz"] = "nii.gz",
        voxel_size: tuple[float, float, float] = (1, 1, 1),
        oversub_factor: float = 1.5,
    ):
        """Store configuration and derive worker/thread allocation."""
        self.input_root = input_root
        self.output_root = output_root
        self.pipeline_version = pipeline_version
        self.output_format = output_format
        self.voxel_size = voxel_size
        self.oversub_factor = oversub_factor

        total_cores = multiprocessing.cpu_count()
        target_threads = int(total_cores * oversub_factor)
        if num_workers is None and itk_threads is None:
            num_workers = max(1, total_cores // 4)
            itk_threads = max(1, target_threads // num_workers)
        elif num_workers is not None and itk_threads is None:
            itk_threads = max(1, target_threads // num_workers)
        elif num_workers is None and itk_threads is not None:
            num_workers = max(1, target_threads // itk_threads)
        if num_workers * itk_threads > target_threads:
            itk_threads = max(1, target_threads // num_workers)

        self.num_workers = num_workers
        self.itk_threads = itk_threads

    def _process_one_series(self, series_id: str, modality: Optional[str]) -> str:
        """Process a single imaging series and persist the result.

        Parameters
        ----------
        series_id : str
            Identifier of the series (typically a folder name inside
            `input_root/series`).
        modality : str, optional
            Imaging modality label used to resolve the appropriate policy. May
            be ``None`` when not available.

        Returns
        -------
        str
            Absolute path to the written preprocessed volume file.
        """
        # 1) read + resample
        image = resample_to_spacing(self.input_root, series_id, self.voxel_size)

        # 2) pick policy
        policy_cls = REGISTRY.get((modality or "").upper(), ModalityPolicy)
        policy: ModalityPolicy = policy_cls()

        ctx = {
            "series_id": series_id,
            "modality": modality,
            "voxel_size": self.voxel_size,
            "pipeline_version": self.pipeline_version,
        }

        image = policy.pre_hooks(image, ctx)
        image = policy.crop(image, ctx)
        image = policy.normalize(image, ctx)
        image = policy.post_hooks(image, ctx)

        # 3) write
        out_dir = self.output_root / "series"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{series_id}.{self.output_format}"
        itk.imwrite(image, str(out_path), compression=(self.output_format == "nii.gz"))
        return str(out_path)

    def preprocess_generate_metadata(self):
        """Write a JSON metadata summary for the current preprocessing run."""
        import json
        df = pd.read_csv(self.input_root / "train.csv")
        meta = {
            "num_series": int(df["SeriesInstanceUID"].nunique()),
            "modality_counts": df["Modality"].value_counts().to_dict(),
            "pipeline_version": self.pipeline_version,
            "voxel_size": self.voxel_size,
            "output_format": self.output_format,
            "num_workers": self.num_workers,
            "itk_threads": self.itk_threads,
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        with open(self.output_root / "metadata.json", "w") as f:
            json.dump(meta, f, indent=4)

    def preprocess_generate_overview(self, series_id: Optional[str] = None) -> Optional[Path]:
        """Create a quick-look figure comparing raw and processed volumes.

        Parameters
        ----------
        series_id : str, optional
            Series identifier to visualise. When omitted the first entry in
            ``train.csv`` is used.

        Returns
        -------
        Path | None
            Location of the written figure, or ``None`` when generation was
            skipped (for example because dependencies were missing).
        """
        try:
            import numpy as np  # type: ignore
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            print("matplotlib not available - skipping preprocessing overview figure")
            return None

        if series_id is None:
            try:
                df = pd.read_csv(self.input_root / "train.csv")
            except FileNotFoundError:
                print("train.csv missing - unable to locate series for overview figure")
                return None
            if df.empty:
                print("train.csv empty - skipping preprocessing overview figure")
                return None
            series_id = str(df.iloc[0]["SeriesInstanceUID"])

        raw_path = self.input_root / "series" / series_id
        processed_path = self.output_root / "series" / f"{series_id}.{self.output_format}"

        if not processed_path.exists():
            print(f"processed series not found ({processed_path}) – skipping overview figure")
            return None
        if not raw_path.exists():
            print(f"raw series not found ({raw_path}) – skipping overview figure")
            return None

        try:
            raw_image = itk.imread(str(raw_path), itk.F)
        except Exception as exc:
            print(f"failed to read raw series {series_id}: {exc}")
            return None

        try:
            processed_image = itk.imread(str(processed_path), itk.F)
        except Exception as exc:
            print(f"failed to read processed series {series_id}: {exc}")
            return None

        raw_arr = itk.GetArrayFromImage(raw_image).astype("float32")
        processed_arr = itk.GetArrayFromImage(processed_image).astype("float32")

        def _finite_values(data: "np.ndarray") -> "np.ndarray":
            flat = data.reshape(-1)
            finite = flat[np.isfinite(flat)]
            return finite if finite.size else flat

        def _extract_axial_slice(
            data: "np.ndarray",
            *,
            z_idx: Optional[int] = None,
        ) -> "np.ndarray":
            z = data.shape[0]
            if z_idx is None:
                z_idx = z // 2
            z_idx = int(np.clip(z_idx, 0, z - 1))
            return data[z_idx]

        def _window(data: "np.ndarray") -> tuple[float, float]:
            finite = _finite_values(data)
            if finite.size == 0:
                return 0.0, 1.0
            lo, hi = np.percentile(finite, [1.0, 99.0])
            if hi <= lo:
                hi = lo + 1.0
            return float(lo), float(hi)

        def _stats(image, data: "np.ndarray") -> dict:
            finite = _finite_values(data)
            spacing = image.GetSpacing()
            size = image.GetLargestPossibleRegion().GetSize()
            return {
                "shape_zyx": tuple(int(v) for v in data.shape),
                "size_xyz": tuple(int(v) for v in size),
                "spacing_xyz": tuple(round(float(v), 3) for v in spacing),
                "intensity_min": round(float(finite.min()) if finite.size else float(data.min()), 2),
                "intensity_max": round(float(finite.max()) if finite.size else float(data.max()), 2),
                "intensity_mean": round(float(finite.mean()) if finite.size else float(data.mean()), 2),
                "intensity_std": round(float(finite.std()) if finite.size else float(data.std()), 2),
            }

        z_mid = processed_arr.shape[0] // 2
        y_mid = processed_arr.shape[1] // 2
        x_mid = processed_arr.shape[2] // 2

        def _processed_to_raw_index(z_idx: int, y_idx: int, x_idx: int) -> tuple[int, int, int]:
            itk_index = (int(x_idx), int(y_idx), int(z_idx))
            try:
                physical_point = processed_image.TransformIndexToPhysicalPoint(itk_index)
                raw_cont = raw_image.TransformPhysicalPointToContinuousIndex(physical_point)
            except RuntimeError:
                return (
                    int(np.clip(z_idx, 0, raw_arr.shape[0] - 1)),
                    int(np.clip(y_idx, 0, raw_arr.shape[1] - 1)),
                    int(np.clip(x_idx, 0, raw_arr.shape[2] - 1)),
                )
            raw_z = int(np.clip(int(round(raw_cont[2])), 0, raw_arr.shape[0] - 1))
            raw_y = int(np.clip(int(round(raw_cont[1])), 0, raw_arr.shape[1] - 1))
            raw_x = int(np.clip(int(round(raw_cont[0])), 0, raw_arr.shape[2] - 1))
            return raw_z, raw_y, raw_x

        raw_z_idx, _, _ = _processed_to_raw_index(z_mid, y_mid, x_mid)

        raw_axial = _extract_axial_slice(raw_arr, z_idx=raw_z_idx)
        proc_axial = _extract_axial_slice(processed_arr, z_idx=z_mid)
        raw_window  = _window(raw_arr)
        proc_window = _window(processed_arr)
        raw_stats   = _stats(raw_image, raw_arr)
        proc_stats  = _stats(processed_image, processed_arr)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8), gridspec_kw={"wspace": 0.25})
        fig.suptitle(f"Preprocessing overview – series {series_id}")

        hist_bins = 80

        def _plot_row(ax_row, axial, data, window, stats, title_prefix):
            ax_row[0].imshow(axial, cmap="gray", vmin=window[0], vmax=window[1])
            ax_row[0].set_title(f"{title_prefix} axial")
            ax_row[0].axis("off")

            finite = _finite_values(data)
            if finite.size > 200_000:
                rng = np.random.default_rng(seed=0)
                finite = rng.choice(finite, size=200_000, replace=False)
            ax_row[1].hist(finite, bins=hist_bins, color="#4a90e2", alpha=0.85)
            ax_row[1].set_title(f"{title_prefix} intensity histogram")
            ax_row[1].set_ylabel("voxel count")
            ax_row[1].set_xlabel("intensity")
            ax_row[1].tick_params(axis="x", rotation=45)

            ax_row[2].axis("off")
            ax_row[2].set_xlim(0, 1)
            ax_row[2].set_ylim(0, 1)
            text_lines = [
                f"Shape (z,y,x): {stats['shape_zyx']}",
                f"Size (x,y,z): {stats['size_xyz']}",
                f"Spacing (x,y,z): {stats['spacing_xyz']}",
                f"Min / Max: {stats['intensity_min']} / {stats['intensity_max']}",
                f"Mean ± Std: {stats['intensity_mean']} ± {stats['intensity_std']}",
            ]
            ax_row[2].text(
                0.02,
                0.98,
                "\n".join(text_lines),
                va="top",
                ha="left",
                fontsize=10,
                transform=ax_row[2].transAxes,
            )

        _plot_row(axes[0], raw_axial, raw_arr, raw_window, raw_stats, "Raw")
        _plot_row(axes[1], proc_axial, processed_arr, proc_window, proc_stats, "Processed")

        fig.tight_layout(pad=1.2, rect=(0, 0, 1, 0.92), w_pad=0.2, h_pad=0.6)

        output_path = self.output_root / "preprocessing_overview.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def run(self, batch_size: int = 1):
        """Execute the preprocessing pipeline over the dataset.

        Parameters
        ----------
        batch_size : int, default=1
            Number of series assigned to a worker process at a time. Increasing
            the batch size can improve throughput on large machines.
        """
        from .workers import run_in_process_batches
        self.preprocess_generate_metadata()

        df = pd.read_csv(self.input_root / "train.csv")
        series_items = [(row["SeriesInstanceUID"], row["Modality"]) for _, row in df.iterrows()]
        run_in_process_batches(series_items, batch_size, self)

        sample_id = next((sid for sid, _ in series_items if isinstance(sid, str) and sid), None)
        if sample_id:
            try:
                self.preprocess_generate_overview(sample_id)
            except Exception as exc:  # pragma: no cover - safeguards against optional dependency issues
                print(f"Failed to create preprocessing overview figure: {exc}")
