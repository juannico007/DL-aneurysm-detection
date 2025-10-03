"""Auto-import policy modules so they register themselves."""

from importlib import import_module
from pathlib import Path


def _auto_import_policies():
    """Import all policy modules so registration side effects run."""
    pkg_path = Path(__file__).resolve().parent
    for path in pkg_path.glob("*.py"):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        import_module(f"{__name__}.{path.stem}")


_auto_import_policies()


__all__ = []
