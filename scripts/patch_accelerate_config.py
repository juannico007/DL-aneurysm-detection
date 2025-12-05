#!/usr/bin/env python3
"""
Utility script to patch Hugging Face Accelerate's SageMakerConfig so it includes
`num_processes` and `parallelism_config` defaults. This mirrors the manual edit
we tested locally and lets the rest of the team apply it reliably.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Iterable


def _default_config_path() -> Path:
    """Return the path to accelerate/commands/config/config_args.py inside the active env."""
    try:
        import accelerate  # type: ignore
    except ImportError as exc:  # pragma: no cover - script is for manual use
        raise SystemExit(
            "Unable to import accelerate. Please activate the same virtualenv you use for training."
        ) from exc

    module_path = Path(inspect.getfile(accelerate))
    return module_path.parent / "commands" / "config" / "config_args.py"


def _find_block(lines: list[str], header: str) -> tuple[int, int]:
    """Locate the block that starts with `header` (e.g. 'class SageMakerConfig')."""
    start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith(header):
            start = idx
            break
    if start is None:
        raise SystemExit(f"Unable to find `{header}` in config file.")

    indent_prefix = " " * 4
    end = start + 1
    while end < len(lines) and lines[end].startswith(indent_prefix):
        end += 1
    return start, end


def _ensure_line_after(
    lines: list[str], search_text: str, new_line: str, block_slice: slice
) -> bool:
    """Insert `new_line` after the line containing `search_text` within `block_slice`."""
    block = lines[block_slice]
    if any(new_line.strip().startswith(part.strip()) for part in block):
        return False

    for local_idx, line in enumerate(block):
        if search_text in line:
            lines.insert(block_slice.start + local_idx + 1, new_line)
            return True

    raise SystemExit(f"Unable to find `{search_text}` inside SageMakerConfig block.")


def patch_config(config_path: Path) -> bool:
    """Apply the required edits in-place, returning True if anything changed."""
    contents = config_path.read_text().splitlines()
    block_start, block_end = _find_block(contents, "class SageMakerConfig")

    changed = False
    indent = " " * 4
    if _ensure_line_after(
        contents,
        "num_machines: int = 1",
        f"{indent}num_processes: int = 1",
        slice(block_start, block_end),
    ):
        block_end += 1
        changed = True

    if _ensure_line_after(
        contents,
        "gpu_ids: str = \"all\"",
        f"{indent}parallelism_config: Optional[dict] = None",
        slice(block_start, block_end),
    ):
        block_end += 1
        changed = True

    if changed:
        config_path.write_text("\n".join(contents) + "\n")
    return changed


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to accelerate/commands/config/config_args.py (auto-detected if omitted).",
    )
    args = parser.parse_args(argv)

    config_path = args.config_path or _default_config_path()
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    changed = patch_config(config_path)
    if changed:
        print(f"Updated {config_path}")
    else:
        print(f"No changes needed in {config_path}")


if __name__ == "__main__":
    main()
