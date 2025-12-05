# preprocess/workers.py
import os
from concurrent import futures
from pathlib import Path
from typing import Sequence
import csv
import itk
from tqdm import tqdm
from typing import Dict, List
import multiprocessing
import signal 

def load_csv(path: Path):
    """Load a CSV file as a list of dictionaries."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def load_done_ids(preproc_csv: Path):
    """Load processed series IDs from preprocessed/train.csv."""
    if not preproc_csv.exists():
        return set()
    with open(preproc_csv, newline="") as f:
        reader = csv.DictReader(f)
        return {row["SeriesInstanceUID"] for row in reader if row.get("SeriesInstanceUID")}
    
def append_to_csv(path: Path, header: List[str], row: Dict[str, str], lock: multiprocessing.Lock):
    """Thread-safe append of one row to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    
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
    try:
        from .pipeline import Preprocess
    except Exception as exc:
        print(f"[worker] failed importing pipeline in child process: {exc}")

    cancel_event = config.get("cancel_event")
    if cancel_event is not None and cancel_event.is_set():
        return None
    
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

    #checkpoint code
    preproc_csv = Path(config["preproc_csv"])
    csv_header  = config["csv_header"]
    csv_lock    = config["csv_lock"]    
    rows  = config["rows"] 
        
    def process_one(series_modality, n_threads):
        """Resample and post-process a single series using ``n_threads``."""
        series_id, modality = series_modality
        itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(n_threads)
        try:
            self._process_one_series(series_id, modality=modality)
            series_dir = Path(config["input_root"]) / "series" / series_id
            if not series_dir.exists():
                raise FileNotFoundError(f"Series folder missing: {series_dir}")
            
            #checkpoint code
            row = rows.get(series_id)
            if row is not None:
                append_to_csv(preproc_csv, csv_header, row, csv_lock)
            return (series_id, True)
        except Exception as e:
            print(f"[!] {series_id} failed: {e}")
            return (series_id, False)

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
    preproc_csv = preprocess_obj.output_root / "train.csv"

    # Load CSVs
    all_rows = load_csv(preprocess_obj.input_root / "train.csv")
    if not all_rows:
        print("❌ No entries found in input train.csv")
        return

    header = list(all_rows[0].keys())
    done_ids = load_done_ids(preproc_csv)
    remaining_rows = [r for r in all_rows if r["SeriesInstanceUID"] not in done_ids]
    rows_by_id = {r["SeriesInstanceUID"]: r for r in remaining_rows}
    
    manager = multiprocessing.Manager()
    csv_lock = manager.Lock()
    cancel_event = manager.Event()

    def _handle_sigint(sig, frame):
        print("\n🛑 Ctrl+C received — finishing current series and stopping new ones...")
        cancel_event.set()
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        pass

    cfg = {
        "input_root": str(preprocess_obj.input_root),
        "output_root": str(preprocess_obj.output_root),
        "pipeline_version": preprocess_obj.pipeline_version,
        "output_format": preprocess_obj.output_format,
        "voxel_size": preprocess_obj.voxel_size,
        "itk_threads": int(preprocess_obj.itk_threads),

        "preproc_csv": str(preproc_csv),
        "csv_header": header,
        "csv_lock": csv_lock,
        "rows": rows_by_id,

        "cancel_event": cancel_event,
    }

    batches = [series_items[i:i + batch_size] for i in range(0, len(series_items), batch_size)]
    ex = futures.ProcessPoolExecutor(max_workers=preprocess_obj.num_workers)
    try:
        # Submit batches to the executor
        futs = [ex.submit(_worker_batch, item, cfg) for item in batches]
        for _ in tqdm(futures.as_completed(futs), total=len(futs), desc="Processing series (parallel)"):
            pass
    except KeyboardInterrupt:
        cancel_event.set()
        print("\n🛑 Stopping")
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
