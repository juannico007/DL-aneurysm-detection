from os import listdir
from os.path import isfile, join
from pathlib import Path
import csv
import pandas as pd

def main():
    source_dir = Path("./segmentations/series/")
    target_dir = Path("./masks")
    train_df = pd.read_csv(Path("./ct_subset/train.csv"))
    localizers_df = pd.read_csv("./segmentations/train_localizers.csv")
    
    filtered_localizers_df = localizers_df[localizers_df["SeriesInstanceUID"].isin(train_df["SeriesInstanceUID"])]
    filtered_localizers_df.to_csv("filtered_train_localizers.csv", index=False)
        
    sop_ids = set(filtered_localizers_df["SOPInstanceUID"].astype(str))
    copied, missing = 0, 0
    for sop in sop_ids:
        print(sop)
        src_file = source_dir / f"{sop}.nii"
        dst_file = target_dir / src_file.name
        
        print(src_file)
        if src_file.exists():
            shutil.copy(src_file, dst_file)
            copied += 1
        else:
            missing += 1

    print(f"✅ Copied {copied} masks to '{target_dir}'.")
    # if missing > 0:
    #     print(f"⚠️ {missing} masks not found in source folder.")

if __name__ == "__main__":
    main()