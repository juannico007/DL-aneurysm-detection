from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def get_folder_size(path):
    """
    Get the total size of all files in the specified folder.
    """
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total

def _copy_one_series(source, destination):
    """
    Copy a single series folder from source to destination if source exists and destination does not exist.
    """
    if os.path.exists(source) and not os.path.exists(destination):
        shutil.copytree(source, destination)

def copy_subset(folder_path, out_path, modality="CTA", n_series=20, workers=8, see_size=True):
    """
    Copy a subset of series folders based on the specified modality and number of series.
    """

    train_csv = folder_path + "/train.csv"
    images = folder_path + "/series"
    OUT = Path(out_path)
    df = pd.read_csv(train_csv)

    # get all series with the specified modality
    scans = df[df["Modality"] == modality]

    subset_series = scans["SeriesInstanceUID"].unique()[:n_series]

    if see_size:
        # estimate folder size by getting sizes of all folders in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(get_folder_size, os.path.join(images, str(s)))
                    for s in subset_series]
            total_bytes = sum(f.result() for f in tqdm(as_completed(futures), total=len(futures)))
        folder_size_gb = total_bytes / (1024 ** 3)

        print("Approximate folder size (GB):", folder_size_gb)
        input("Press Enter to continue...")

    os.makedirs(OUT / "series", exist_ok=True)

    # copy all series in parallel
    with ThreadPoolExecutor(max_workers=workers) as ex:
        copied_data = []
        for sid in subset_series:
            src = images + f"/{str(sid)}"
            dst = OUT / "series" / str(sid)
            copied_data.append(ex.submit(_copy_one_series, src, dst))

        for f in tqdm(as_completed(copied_data), total=len(copied_data), desc="Copying"):
            f.result()  # raise errors if any

    subset_df = scans[scans["SeriesInstanceUID"].isin(subset_series)]
    subset_df.to_csv(OUT / "train.csv", index=False)

copy_subset(
    folder_path="../data/rsna-intracranial-aneurysm-detection",
    out_path="mini-rsna-intracranial-aneurysm-detection",
    modality="CTA",
    n_series=20,
    workers=8,
    see_size=True
)