#!/usr/bin/env python3
"""
Download SageMaker training artifacts for a specific GitHub user/model/version
triplet and normalize them into cloud_models/<user>/<model>_<version>/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUD_ROOT = REPO_ROOT / "cloud_models"
NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}
DEFAULT_ENV_FILE = REPO_ROOT / "src/.env"


class DownloadError(Exception):
    """Raised when an artifact cannot be pulled from S3."""

    def __init__(self, message: str, *, code: Optional[str] = None):
        super().__init__(message)
        self.code = code


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def str_to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def require_env(*names: str, default: Optional[str] = None) -> str:
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    if default is not None:
        return default
    raise SystemExit(f"Missing required environment variable(s): {', '.join(names)}")


def get_env_list(name: str) -> Optional[List[str]]:
    value = os.environ.get(name)
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull SageMaker artifacts into cloud_models/.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to the .env file containing RUN_MODEL_* variables (default: src/.env).",
    )
    parser.add_argument("--bucket", help="Override the target S3 bucket (PULL_BUCKET/S3_BUCKET).")
    parser.add_argument("--prefix", help="Override the S3 prefix/job namespace (PULL_PREFIX).")
    parser.add_argument("--jobs", nargs="+", help="Explicit job names to inspect (PULL_JOBS).")
    parser.add_argument("--github-user", help="GitHub handle to filter metadata (defaults to GITHUB_USER).")
    parser.add_argument("--model-name", help="Model name to filter metadata (defaults to RUN_MODEL_NAME).")
    parser.add_argument(
        "--model-version",
        help="Specific model version to keep (defaults to RUN_MODEL_VERSION or 'latest').",
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Download every version discovered, ignoring RUN_MODEL_VERSION.",
    )
    parser.add_argument("--output-root", type=Path, help="Directory to store downloads (PULL_OUTPUT_ROOT).")
    parser.add_argument("--region", help="AWS region.")
    parser.add_argument("--aws-access-key-id", help="AWS access key ID override.")
    parser.add_argument("--aws-secret-access-key", help="AWS secret access key override.")
    parser.add_argument("--iam-role-name", help="IAM role to assume before contacting S3.")
    parser.add_argument(
        "--keep-tarballs",
        action="store_true",
        help="Keep downloaded model/output tarballs instead of deleting temporary copies.",
    )
    return parser.parse_args()


def apply_cli_overrides(cli: argparse.Namespace) -> None:
    os.environ["PULL_ENV_FILE"] = str(cli.env_file)
    mapping = [
        ("bucket", "S3_BUCKET"),
        ("prefix", "PULL_PREFIX"),
        ("github_user", "GITHUB_USER"),
        ("model_name", "RUN_MODEL_NAME"),
        ("model_version", "RUN_MODEL_VERSION"),
        ("output_root", "PULL_OUTPUT_ROOT"),
        ("region", "AWS_REGION"),
        ("aws_access_key_id", "AWS_ACCESS_KEY_ID"),
        ("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
        ("iam_role_name", "IAM_ROLE_NAME"),
    ]
    for attr, env_name in mapping:
        value = getattr(cli, attr)
        if value is None:
            continue
        os.environ[env_name] = str(value)

    if cli.jobs:
        os.environ["PULL_JOBS"] = ",".join(cli.jobs)
    if cli.all_versions:
        os.environ["PULL_ALL_VERSIONS"] = "1"
    if cli.keep_tarballs:
        os.environ["PULL_KEEP_TARBALLS"] = "1"


def bootstrap_run_identifiers(env_file: Path) -> None:
    load_env_file(env_file)
    github_user = os.environ.get("PULL_GITHUB_USER") or os.environ.get("GITHUB_USER")
    model_name = os.environ.get("PULL_MODEL_NAME") or os.environ.get("RUN_MODEL_NAME")
    model_version = os.environ.get("PULL_VERSION") or os.environ.get("RUN_MODEL_VERSION") or "latest"
    missing = []
    if not github_user:
        missing.append("GITHUB_USER")
    if not model_name:
        missing.append("RUN_MODEL_NAME")
    if missing:
        names = ", ".join(missing)
        raise SystemExit(
            f"Missing required identifiers ({names}). "
            f"Populate {env_file} or pass CLI overrides before running the pull."
        )
    os.environ.setdefault("PULL_GITHUB_USER", github_user)
    os.environ.setdefault("PULL_MODEL_NAME", model_name)
    os.environ.setdefault("PULL_VERSION", model_version)


def build_settings() -> SimpleNamespace:
    env_file = Path(os.environ.get("PULL_ENV_FILE", REPO_ROOT / "src/.env"))
    load_env_file(env_file)

    bucket = require_env("PULL_BUCKET", "S3_BUCKET")
    prefix = os.environ.get("PULL_PREFIX", "")
    jobs = get_env_list("PULL_JOBS")
    region = os.environ.get("AWS_REGION")
    aws_access = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    iam_role = os.environ.get("IAM_ROLE_NAME")
    github_user = require_env("PULL_GITHUB_USER", "GITHUB_USER")
    model_name = os.environ.get("PULL_MODEL_NAME", os.environ.get("RUN_MODEL_NAME"))
    version = os.environ.get("PULL_VERSION", os.environ.get("RUN_MODEL_VERSION", "latest"))
    all_versions = str_to_bool(os.environ.get("PULL_ALL_VERSIONS"), default=False)
    output_root = Path(os.environ.get("PULL_OUTPUT_ROOT", str(CLOUD_ROOT)))
    keep_tarballs = str_to_bool(os.environ.get("PULL_KEEP_TARBALLS"), default=False)

    return SimpleNamespace(
        env_file=env_file,
        bucket=bucket,
        prefix=prefix,
        jobs=jobs,
        region=region,
        aws_access_key_id=aws_access,
        aws_secret_access_key=aws_secret,
        iam_role_name=iam_role,
        github_user=github_user,
        model_name=model_name,
        version=version,
        all_versions=all_versions,
        output_root=output_root,
        keep_tarballs=keep_tarballs,
    )


def create_session(args) -> boto3.Session:
    session_kwargs: Dict[str, Optional[str]] = {}
    if args.region:
        session_kwargs["region_name"] = args.region

    if args.aws_access_key_id or args.aws_secret_access_key:
        if not (args.aws_access_key_id and args.aws_secret_access_key):
            raise SystemExit("Provide both --aws-access-key-id and --aws-secret-access-key.")
        session_kwargs["aws_access_key_id"] = args.aws_access_key_id
        session_kwargs["aws_secret_access_key"] = args.aws_secret_access_key

    base_session = boto3.Session(**session_kwargs)

    if not args.iam_role_name:
        return base_session

    sts_client = base_session.client("sts")
    identity = sts_client.get_caller_identity()
    account_id = identity["Account"]
    role_arn = args.iam_role_name if args.iam_role_name.startswith("arn:") else f"arn:aws:iam::{account_id}:role/{args.iam_role_name}"

    try:
        assumed = sts_client.assume_role(RoleArn=role_arn, RoleSessionName="pull-models")
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "AccessDenied":
            print(
                f"[Warning] AssumeRole denied for {role_arn}. "
                "Proceeding with base credentials. Ensure the role trust policy allows this user if role access is required."
            )
            return base_session
        raise

    creds = assumed["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=args.region,
    )


def normalize_prefix(prefix: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/" if prefix else ""


def discover_jobs(s3_client, bucket: str, prefix: str) -> Dict[str, float]:
    jobs: Dict[str, float] = {}
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for content in page.get("Contents", []):
            key = content["Key"]
            if not key or key.endswith("/"):
                continue
            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue
            job = rel.split("/", 1)[0]
            last_modified = content["LastModified"].timestamp()
            if job not in jobs or last_modified > jobs[job]:
                jobs[job] = last_modified
    return jobs


def download_file(client, bucket: str, key: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, str(destination))
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        raise DownloadError(
            f"Failed to download s3://{bucket}/{key}: {exc}",
            code=code,
        ) from exc
    return destination


def safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = dest / member.name
        if not str(member_path.resolve()).startswith(str(dest)):
            raise DownloadError(f"Unsafe path detected in tarball: {member.name}")
    tar.extractall(dest)


def extract_tarball(tar_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        safe_extract(tar, dest)
    return dest


def collect_artifacts(model_root: Path, output_root: Path, job_name: str) -> List[Dict]:
    """
    Discover metadata.json files inside the extracted SageMaker output tarball and infer where the
    model/checkpoint files live. Earlier versions of the training metadata did not store explicit
    "storage" information, so we derive the paths from the metadata location and artifact_id.
    """

    artifacts: List[Dict] = []
    for meta_path in output_root.rglob("metadata.json"):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue

        files = metadata.get("files", {})
        checkpoint_name = files.get("checkpoint")
        if not checkpoint_name:
            continue

        artifact_id = metadata.get("artifact_id")
        output_base = meta_path.parent

        candidate_model_dirs: List[Path] = []
        if artifact_id:
            candidate_model_dirs.append(model_root / artifact_id)
        try:
            rel_to_output = output_base.relative_to(output_root)
        except ValueError:
            rel_to_output = None
        if rel_to_output:
            candidate_model_dirs.append(model_root / rel_to_output)
        candidate_model_dirs.append(model_root)

        model_base = None
        for candidate in candidate_model_dirs:
            if (candidate / checkpoint_name).exists():
                model_base = candidate
                break

        if model_base is None:
            # Fall back to any directory that contains the checkpoint file
            found = list(model_root.rglob(checkpoint_name))
            if found:
                model_base = found[0].parent

        if model_base is None:
            print(f"[Warning] Could not locate checkpoint '{checkpoint_name}' for metadata at {meta_path}")
            continue

        artifacts.append(
            {
                "metadata": metadata,
                "model_base": model_base.resolve(),
                "output_base": output_base.resolve(),
                "job_name": job_name,
            }
        )
    return artifacts


def should_download(version: str, downloaded: set[str], args: argparse.Namespace) -> bool:
    if args.all_versions:
        return version not in downloaded
    if args.version != "latest":
        return version == args.version and version not in downloaded
    return len(downloaded) == 0


def copy_artifact(
    artifact: Dict,
    args: argparse.Namespace,
    target_root: Path,
) -> Optional[Path]:
    meta = artifact["metadata"]
    version = meta["model_version"]
    model_name = meta["model_name"]
    github_user = meta["github_user"]

    model_base = artifact["model_base"]
    output_base = artifact["output_base"]

    checkpoint_name = meta["files"]["checkpoint"]
    history_name = meta["files"]["history"]
    hyper_name = meta["files"]["hyperparameters"]

    checkpoint_src = model_base / checkpoint_name
    history_src = output_base / history_name
    hyper_src = output_base / hyper_name
    metadata_src = output_base / "metadata.json"

    if not checkpoint_src.exists():
        print(f"[Warning] Missing checkpoint at {checkpoint_src}")
        return None

    version_folder = f"{model_name}_{version}"
    dest_dir = target_root / github_user / version_folder
    dest_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_src, dest_dir / checkpoint_name)
    if history_src.exists():
        shutil.copy2(history_src, dest_dir / history_name)
    if hyper_src.exists():
        shutil.copy2(hyper_src, dest_dir / hyper_name)
    if metadata_src.exists():
        enriched = meta.copy()
        enriched["downloaded_at"] = datetime.utcnow().isoformat() + "Z"
        enriched["s3_job_name"] = artifact["job_name"]
        enriched["bucket"] = args.bucket
        enriched["prefix"] = args.prefix.strip("/")
        with open(dest_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(enriched, handle, indent=2, sort_keys=True)

    print(f"Saved version '{version}' to {dest_dir}")
    return dest_dir


def main() -> None:
    cli_args = parse_cli_args()
    bootstrap_run_identifiers(cli_args.env_file)
    apply_cli_overrides(cli_args)
    args = build_settings()
    session = create_session(args)
    s3_client = session.client("s3")
    prefix = normalize_prefix(args.prefix)
    jobs = discover_jobs(s3_client, args.bucket, prefix)

    if not jobs:
        raise SystemExit(f"No jobs found under s3://{args.bucket}/{prefix}")

    if args.jobs:
        job_order = args.jobs
    else:
        job_order = [job for job, _ in sorted(jobs.items(), key=lambda item: item[1], reverse=True)]
        if job_order:
            print(f"No job list provided in env. Trying newest job first: {job_order[0]}")

    missing = [job for job in job_order if job not in jobs]
    if missing:
        raise SystemExit(f"Job(s) not found under prefix: {', '.join(missing)}")

    downloaded_versions: set[str] = set()
    downloads_done = 0

    for job in job_order:
        model_key = f"{prefix}{job}/output/model.tar.gz"
        output_key = f"{prefix}{job}/output/output.tar.gz"

        with tempfile.TemporaryDirectory(prefix="pull_models_") as tmpdir:
            tmpdir = Path(tmpdir)
            model_tar = tmpdir / "model.tar.gz"
            output_tar = tmpdir / "output.tar.gz"

            def skip_download(exc: DownloadError, key_name: str) -> bool:
                if exc.code in NOT_FOUND_CODES and args.jobs is None:
                    print(f"[Warning] Job '{job}' missing {key_name}. Trying next job.")
                    return True
                raise exc

            try:
                download_file(s3_client, args.bucket, model_key, model_tar)
            except DownloadError as exc:
                if skip_download(exc, model_key):
                    continue
            try:
                download_file(s3_client, args.bucket, output_key, output_tar)
            except DownloadError as exc:
                if skip_download(exc, output_key):
                    continue

            model_extract = extract_tarball(model_tar, tmpdir / "model")
            output_extract = extract_tarball(output_tar, tmpdir / "output")

            artifacts = collect_artifacts(model_extract, output_extract, job)
            if not artifacts:
                print(f"[Warning] No metadata found in job '{job}'. Skipping.")
                continue

            for artifact in artifacts:
                meta = artifact["metadata"]
                if meta.get("github_user") != args.github_user:
                    continue
                if args.model_name and meta.get("model_name") != args.model_name:
                    continue

                version = meta.get("model_version")
                if not version or not should_download(version, downloaded_versions, args):
                    continue

                dest_root = args.output_root
                saved = copy_artifact(artifact, args, dest_root)
                if saved:
                    downloaded_versions.add(version)
                    downloads_done += 1
                    if not args.all_versions:
                        if args.version == "latest":
                            break
                        if args.version == version:
                            break
            else:
                model_extract = None  # keep scope

            if not args.all_versions:
                if args.version == "latest" and downloads_done:
                    break
                if args.version != "latest" and args.version in downloaded_versions:
                    break

    if not downloads_done:
        raise SystemExit("No matching artifacts were downloaded. Check filters and available jobs.")


if __name__ == "__main__":
    main()
