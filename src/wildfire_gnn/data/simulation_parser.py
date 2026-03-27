from __future__ import annotations

from pathlib import Path


def discover_metadata_files(metadata_dir: str | Path) -> list[Path]:
    """Return sorted metadata files from the metadata directory."""
    path = Path(metadata_dir)
    if not path.exists():
        raise FileNotFoundError(f"Metadata directory not found: {path}")

    files = sorted([p for p in path.iterdir() if p.is_file()])
    return files