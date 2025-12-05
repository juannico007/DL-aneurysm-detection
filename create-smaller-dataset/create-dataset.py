from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from subprocess import call

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
    Efficiently copy a folder using robocopy on Windows or rsync on Linux/macOS.
    """
    if not os.path.exists(source) or os.path.exists(destination):
        return

    if os.name == "nt":
        # /E copies subdirectories including empty ones
        # /COPYALL preserves timestamps, ownership, permissions, etc.
        # /MT:8 uses 8 threads (adjust as needed)
        call(f'robocopy "{source}" "{destination}" /E /COPY:DAT /MT:8 /R:1 /W:1 /NFL /NDL /NJH /NJS /NP /NS /NC', shell=True)
    else:
        # fallback for non-Windows systems
        call(["rsync", "-a", "--info=progress2", source + "/", destination])

def copy_subset(folder_path, out_path, modality="CTA", n_series=20, workers=8, see_size=True, balanced = True):
    """
    Copy a subset of series folders based on the specified modality and number of series.
    """
   
    train_csv = folder_path + "/train.csv"
    images = folder_path + "/series"
    OUT = Path(out_path)
    df = pd.read_csv(train_csv)

    # get all series with the specified modality
    scans = df[df["Modality"] == modality]
    print("Taking",len(scans), "scans to make the subset")

    if n_series == -1:
        n_series = len(scans)

    if balanced: 
        #take equal amount of positive and negative samples
        positive_scans = scans[scans["Aneurysm Present"] == 1]
        negative_scans = scans[scans["Aneurysm Present"] == 0]

        subset_series = []
        print("There are", len(positive_scans), "positive scans and", len(negative_scans), "negative scans")
        if min(len(positive_scans), len(negative_scans)) < n_series // 2:
            print("Not enough positive scans, the result will be unbalanced")
            if len(positive_scans) < len(negative_scans):
                add_series = [positive_scans, negative_scans]
            else:
                add_series = [negative_scans, positive_scans]
            subset_series += list(add_series[0]["SeriesInstanceUID"].unique())
            subset_series += list(add_series[1]["SeriesInstanceUID"].unique()[:n_series - len(subset_series)])
        else:
            print("Creating balanced subset")
            subset_series += list(positive_scans["SeriesInstanceUID"].unique()[:n_series // 2])
            subset_series += list(negative_scans["SeriesInstanceUID"].unique()[:n_series - len(subset_series)])
    else:
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
    out_path="../ct_subset",
    modality="CTA",
    n_series=-1,
    workers=8,
    see_size=True,
    balanced=True
)