#!/usr/bin/env python3
"""
Stream CloudWatch logs for a SageMaker training job created through accelerate.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

LOG_GROUP_DEFAULT = "/aws/sagemaker/TrainingJobs"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / "src" / ".env"
DEFAULT_REGION = "eu-north-1"


def load_env_file(path: Path) -> None:
    """Populate os.environ with key/value pairs from a .env style file."""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tail CloudWatch logs for an existing SageMaker training job."
    )
    parser.add_argument(
        "job_name",
        help="Full SageMaker training job name (e.g. accelerate-sagemaker-1-2025-11-15-22-44-56-480)",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help=f"Optional env file with AWS credentials (default: {DEFAULT_ENV_FILE})",
    )
    parser.add_argument(
        "--region",
        help="AWS region that hosts the training job (falls back to AWS_REGION/AWS_DEFAULT_REGION).",
    )
    parser.add_argument(
        "--log-group",
        default=LOG_GROUP_DEFAULT,
        help=f"CloudWatch Logs group to inspect (default: {LOG_GROUP_DEFAULT})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between log fetches when following (default: 5).",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Only stream new events instead of replaying the entire log history.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch available logs once and exit instead of following.",
    )
    return parser.parse_args()


def resolve_region(name: Optional[str]) -> Optional[str]:
    if name:
        return name
    env_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    return env_region or DEFAULT_REGION


def create_logs_client(region: Optional[str]):
    session = boto3.Session(region_name=region) if region else boto3.Session()
    return session.client("logs")


def list_streams(logs_client, job_name: str, log_group: str) -> Iterable[str]:
    paginator = logs_client.get_paginator("describe_log_streams")
    for page in paginator.paginate(logGroupName=log_group, logStreamNamePrefix=job_name):
        for stream in page.get("logStreams", []):
            name = stream.get("logStreamName")
            if not name or not name.startswith(job_name):
                continue
            yield name


def fetch_events(
    logs_client,
    *,
    log_group: str,
    stream: str,
    token: Optional[str],
    start_from_head: bool,
) -> Tuple[Iterable[dict], str]:
    kwargs = {
        "logGroupName": log_group,
        "logStreamName": stream,
        "startFromHead": start_from_head,
    }
    if token:
        kwargs["nextToken"] = token
    response = logs_client.get_log_events(**kwargs)
    return response.get("events", []), response.get("nextForwardToken")


def render_timestamp(ts_ms: Optional[int]) -> str:
    if ts_ms is None:
        return "unknown"
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def stream_log_events(
    logs_client,
    *,
    job_name: str,
    log_group: str,
    poll_interval: float,
    follow: bool,
    skip_history: bool,
) -> None:
    cursors: Dict[str, Dict[str, Optional[str]]] = {}
    warned_missing_streams = False

    while True:
        try:
            streams = sorted(set(list_streams(logs_client, job_name, log_group)))
        except logs_client.exceptions.ResourceNotFoundException:
            if warned_missing_streams is False:
                print(
                    f"[Info] Log group {log_group} not found yet; waiting for SageMaker to create it...",
                    file=sys.stderr,
                )
                warned_missing_streams = True
            if not follow:
                raise SystemExit(f"Log group {log_group} does not exist.")
            time.sleep(poll_interval)
            continue
        except ClientError as exc:
            raise SystemExit(f"Failed to list log streams: {exc}") from exc

        if not streams:
            if warned_missing_streams is False:
                print(
                    f"[Info] Waiting for log streams matching {job_name} to appear...",
                    file=sys.stderr,
                )
                warned_missing_streams = True
            if not follow:
                print(f"[Info] No log streams found for {job_name}.", file=sys.stderr)
                return
            time.sleep(poll_interval)
            continue

        warned_missing_streams = False
        new_events_found = False

        for stream in streams:
            cursor = cursors.setdefault(
                stream,
                {"token": None, "start_from_head": not skip_history},
            )
            try:
                events, token = fetch_events(
                    logs_client,
                    log_group=log_group,
                    stream=stream,
                    token=cursor["token"],
                    start_from_head=cursor["start_from_head"],
                )
            except logs_client.exceptions.ResourceNotFoundException:
                cursors.pop(stream, None)
                continue
            except ClientError as exc:
                print(f"[Warning] Failed to read {stream}: {exc}", file=sys.stderr)
                continue

            cursor["token"] = token
            cursor["start_from_head"] = False

            for event in events:
                timestamp = render_timestamp(event.get("timestamp"))
                message = event.get("message", "").rstrip("\n")
                print(f"[{timestamp}] {stream}: {message}")
                new_events_found = True

        if not follow:
            if not new_events_found:
                return
            continue

        time.sleep(poll_interval)


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file).expanduser())
    region = resolve_region(args.region)

    try:
        logs_client = create_logs_client(region)
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(f"Unable to create CloudWatch Logs client: {exc}") from exc

    follow = not args.once
    try:
        stream_log_events(
            logs_client,
            job_name=args.job_name,
            log_group=args.log_group,
            poll_interval=args.poll_interval,
            follow=follow,
            skip_history=args.skip_history,
        )
    except KeyboardInterrupt:
        print("\n[Info] Stopped log streaming.", file=sys.stderr)


if __name__ == "__main__":
    main()
