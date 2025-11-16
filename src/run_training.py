"""
Wrapper around `accelerate launch` that selects the correct config/env per mode.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENTRY = REPO_ROOT / "src" / "train_entry.py"
MODE_ENV_VAR = "TRAINING_MODE"
VALID_MODES = {"local", "cloud"}


def load_env(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Cloud mode requested but env file missing: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def extract_mode(argv: List[str]) -> tuple[str, List[str]]:
    mode = os.environ.get(MODE_ENV_VAR, "local")
    cleaned: List[str] = []
    idx = 0

    while idx < len(argv):
        arg = argv[idx]
        if arg == "--mode":
            if idx + 1 >= len(argv):
                raise SystemExit("--mode flag requires a value (local/cloud).")
            mode = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith("--mode="):
            value = arg.split("=", 1)[1]
            if not value:
                raise SystemExit("--mode flag requires a value (local/cloud).")
            mode = value
            idx += 1
            continue
        cleaned.append(arg)
        idx += 1

    if mode not in VALID_MODES:
        raise SystemExit(f"Unsupported mode '{mode}'. Expected one of: {', '.join(sorted(VALID_MODES))}.")

    return mode, cleaned


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training via accelerate with mode-aware defaults.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / "src/.env",
        help="Env file to load when --mode cloud.",
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        default=REPO_ROOT / "configs/accelerate_local.yaml",
        help="Accelerate config file for local runs.",
    )
    parser.add_argument(
        "--cloud-config",
        type=Path,
        default=REPO_ROOT / "configs/accelerate_cloud.yaml",
        help="Accelerate config file for cloud runs.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    mode, remaining = extract_mode(argv)
    args = parse_args(remaining)
    os.environ[MODE_ENV_VAR] = mode
    env = os.environ.copy()
    config_path = args.local_config

    if not TRAIN_ENTRY.exists():
        raise FileNotFoundError(f"Training entry point missing: {TRAIN_ENTRY}")

    if mode == "cloud":
        load_env(args.env_file)
        os.environ[MODE_ENV_VAR] = mode
        env = os.environ.copy()
        if not env.get("AWS_ACCESS_KEY_ID") or not env.get("AWS_SECRET_ACCESS_KEY"):
            raise SystemExit("Cloud mode requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the .env file.")
        config_path = args.cloud_config

        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            str(config_path),
            "--aws_access_key_id",
            env.get("AWS_ACCESS_KEY_ID"),
            "--aws_secret_access_key",
            env.get("AWS_SECRET_ACCESS_KEY"),
            str(TRAIN_ENTRY),
        ]
    else:
        env[MODE_ENV_VAR] = mode
        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            str(config_path),
            str(TRAIN_ENTRY),
        ]

    print("Running:", " ".join(str(part) for part in cmd))
    result = subprocess.run(cmd, check=False, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
