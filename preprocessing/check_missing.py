import os
import pandas as pd
import numpy as np
import itk

train_df = pd.read_csv("../ct_subset/train.csv")
ids = train_df["SeriesInstanceUID"].unique()

ids = [i+".nii.gz" for i in ids]

files = os.listdir("../ct_preprocessed/series")
missing = np.array([i for i in ids if i not in files])

missing = [i[:-7] for i in missing]
missing_train = train_df[train_df["SeriesInstanceUID"].isin(missing)]
missing_train.to_csv("../preprocessing_failing/train.csv", index = False)

for file in missing:
    dicom_dir = "../preprocessing_failing/series/"+file

    names_generator = itk.GDCMSeriesFileNames.New()
    names_generator.SetUseSeriesDetails(True)
    names_generator.SetDirectory(dicom_dir)

    series_uids = names_generator.GetSeriesUIDs()

    for uid in series_uids:
        files = names_generator.GetFileNames(uid)
        print(f"Series {uid} has {len(files)} file(s)")
    print()