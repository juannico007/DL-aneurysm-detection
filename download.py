import kagglehub
from pathlib import Path
import shutil
from tqdm import tqdm

# Where to save locally
local_output = Path("local_rsna")
local_output.mkdir(parents=True, exist_ok=True)

# 1. Download dataset to KaggleHub cache
root = Path(kagglehub.dataset_download(
    "juannicolasquintero/rsna-intracranial-aneurysm-detection-cta"
))

print("Downloaded dataset to:", root)

# 2. Locate the `series` folder
series_dir = root / "series"
if not series_dir.exists():
    raise FileNotFoundError(f"'series' directory not found under {root}")

# 3. Collect all series (.npz)
series_paths = sorted(series_dir.glob("*.npz"))
print(f"Found {len(series_paths)} series")

# === NEW PART: COPY SERIES LOCALLY (with tqdm) ===
local_series_dir = local_output / "series"
local_series_dir.mkdir(exist_ok=True)

print("\nCopying series files:")
for src in tqdm(series_paths, desc="Copying .npz", unit="file"):
    dst = local_series_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)

print("✔ Series copied to:", local_series_dir)

# 4. Copy train.csv (also show tqdm for consistency)
train_csv = root / "train.csv"
if not train_csv.exists():
    raise FileNotFoundError(f"'train.csv' not found in {root}")

print("\nCopying train.csv:")
for _ in tqdm(range(1), desc="Copying train.csv"):
    shutil.copy2(train_csv, local_output / "train.csv")

print("✔ train.csv copied to:", local_output / "train.csv")
