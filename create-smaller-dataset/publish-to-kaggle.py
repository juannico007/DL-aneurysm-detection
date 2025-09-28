import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def _write_metadata(dataset_path: str | Path,
                    dataset_id: str,       # "owner/slug" (owner must be the authenticated user)
                    title: str,
                    description: str,
                    license_name: str = "CC0-1.0",
                    private: bool = True):
    p = Path(dataset_path)
    p.mkdir(parents=True, exist_ok=True)
    owner, slug = dataset_id.split("/", 1)

    # NOTE: Kaggle ignores "subtitle" in metadata; fold it into description if you want it shown.
    metadata = {
        "title": title,
        "id": dataset_id,               # REQUIRED: "username/slug"
        "licenses": [{"name": license_name}],
        "description": description,
        "private": private              # for UI; API call also takes public=...
    }
    (p / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return slug

def publish_to_kaggle(dataset_path: str,
                      dataset_id: str,     # "owner/slug"
                      title: str,
                      description: str,
                      private: bool = True,
                      convert_to_csv: bool = False,
                      dir_mode: str = "zip",
                      version_notes: str = "Update"):
    
    _write_metadata(dataset_path, dataset_id, title, description, private=private)

    api = KaggleApi()
    api.authenticate()

    # If it exists -> version; else -> create new
    try:
        api.dataset_view(dataset_id)  # raises if not found
        api.dataset_create_version(
            folder=dataset_path,
            version_notes=version_notes,
            convert_to_csv=convert_to_csv,
            dir_mode=dir_mode
        )
        print(f"New version pushed for '{dataset_id}'.")
    except Exception:
        api.dataset_create_new(
            folder=dataset_path,
            public=not private,
            quiet=False,
            convert_to_csv=convert_to_csv,
            dir_mode=dir_mode
        )
        print(f"Dataset '{dataset_id}' created.")

# --- Example usage ---
n_series = 200
modality = "CTA"

publish_to_kaggle(
    dataset_path="mini-rsna-intracranial-aneurysm-detection",
    dataset_id="mihaibivol25/mini-rsna-intracranial-aneurysm-detection",
    title="Mini RSNA Intracranial Aneurysm Detection Dataset",
    description=f"A subset of {n_series} {modality} series from the RSNA Intracranial Aneurysm Detection dataset.",
    private=True,
    convert_to_csv=False,
    dir_mode="zip",
    version_notes=f"Add {n_series} {modality} series"
)