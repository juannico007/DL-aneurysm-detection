# preprocess/workers.py
import os
from concurrent import futures
from pathlib import Path
from typing import Sequence

import itk
from tqdm import tqdm

# We will have for each process a batch of series
# We will take into account the number of threads available and allocate them to each series in the batch
def _worker_batch(batch_items, config: dict):
    """Process a batch of series inside a worker process.

    Parameters
    ----------
    batch_items : Sequence[tuple[str, str | None]]
        Series identifiers paired with modalities assigned to this process.
    config : dict
        Serialized configuration dictionary produced by the parent process.

    Returns
    -------
    None
        Processed volumes are written to disk; no value is returned.
    """
    from .pipeline import Preprocess
    num_series = len(batch_items)

    thread_allocs = _allocate_threads(
        batch_items,
        total_threads=int(config["itk_threads"]),
        series_root=Path(config["input_root"]) / "series",
    )

    self = Preprocess(
        input_root=Path(config["input_root"]),
        output_root=Path(config["output_root"]),
        num_workers=1,
        pipeline_version=config["pipeline_version"],
        output_format=config["output_format"],
        voxel_size=tuple(config["voxel_size"]),
        itk_threads=int(config["itk_threads"]),
    )

    def process_one(series_modality, n_threads):
        """Resample and post-process a single series using ``n_threads``."""
        series_id, modality = series_modality
        itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(n_threads)
        return self._process_one_series(series_id, modality=modality)

    # Use ThreadPoolExecutor to manage threads for each series in the batch
    with futures.ThreadPoolExecutor(max_workers=num_series) as pool:
        futs = [pool.submit(process_one, sm, number_threads) for sm, number_threads in zip(batch_items, thread_allocs)]
        for f in futs:
            _ = f.result()
    return None


# Main function to run preprocessing in parallel batches
def run_in_process_batches(series_items, batch_size: int, preprocess_obj):
    """Dispatch preprocessing work across multiple processes.

    Parameters
    ----------
    series_items : Sequence[tuple[str, str | None]]
        Collection of series identifiers and modalities to process.
    batch_size : int
        Number of series passed to each worker invocation.
    preprocess_obj : Preprocess
        Configured pipeline instance supplying shared configuration.

    Returns
    -------
    None
        Side effects only; output files are written by worker processes.
    """
    cfg = {
        "input_root": str(preprocess_obj.input_root),
        "output_root": str(preprocess_obj.output_root),
        "pipeline_version": preprocess_obj.pipeline_version,
        "output_format": preprocess_obj.output_format,
        "voxel_size": preprocess_obj.voxel_size,
        "itk_threads": int(preprocess_obj.itk_threads),
    }
    batches = [series_items[i:i + batch_size] for i in range(0, len(series_items), batch_size)]
    ex = futures.ProcessPoolExecutor(max_workers=preprocess_obj.num_workers)
    try:
        # Submit batches to the executor
        futs = [ex.submit(_worker_batch, item, cfg) for item in batches]
        for _ in tqdm(futures.as_completed(futs), total=len(futs), desc="Processing series (parallel)"):
            pass
    finally:
        # Ensure the executor is properly shut down
        ex.shutdown(wait=False, cancel_futures=False)


def _allocate_threads(
    batch_items: Sequence[tuple[str, str | None]],
    total_threads: int,
    series_root: Path,
    min_threads: int = 1,
) -> list[int]:
    """Distribute threads across series according to a cheap weight metric."""
    if total_threads < len(batch_items):
        total_threads = len(batch_items)

    weights = [
        _estimate_series_weight(series_root, series_id, modality)
        for series_id, modality in batch_items
    ]
    weight_sum = sum(weights) or len(batch_items)

    raw_allocs = [(weight / weight_sum) * total_threads for weight in weights]
    thread_allocs = [max(min_threads, int(value)) for value in raw_allocs]

    used = sum(thread_allocs)
    remainder = total_threads - used
    if remainder != 0:
        fractional = [(value - int(value), idx) for idx, value in enumerate(raw_allocs)]
        if remainder > 0:
            for _, idx in sorted(fractional, reverse=True)[:remainder]:
                thread_allocs[idx] += 1
        else:
            for _, idx in sorted(fractional)[: abs(remainder)]:
                if thread_allocs[idx] > min_threads:
                    thread_allocs[idx] -= 1

    return thread_allocs


def _estimate_series_weight(series_root: Path, series_id: str, modality: str | None) -> int:
    """Return a small integer weight that approximates series complexity."""
    path = series_root / series_id
    count = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file():
                    count += 1
                if count >= 512:  # Cap to keep directory scans fast.
                    break
    except FileNotFoundError:
        return 1

    # Placeholder for modality-based adjustments if needed later.
    return max(count, 1)
