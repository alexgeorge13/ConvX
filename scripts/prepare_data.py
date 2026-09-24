"""Download and validate the datasets used by the ConvX workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_root(root: Path) -> Path:
    """Resolve relative dataset paths from the repository root."""
    if root.is_absolute():
        return root
    return (PROJECT_ROOT / root).resolve()


def _load_dataset(name: str, root: Path, create_dataset: Callable[[], Any]) -> int:
    """Create a torchvision dataset, downloading it when needed, and check one item."""
    root.mkdir(parents=True, exist_ok=True)
    print(f"\nPreparing {name} in: {root}")
    try:
        dataset = create_dataset()
        count = len(dataset)
        if count == 0:
            raise RuntimeError("The dataset split contains no samples.")
        # Read one sample so missing or unreadable files are reported here.
        dataset[0]
    except Exception as exc:
        raise RuntimeError(
            f"Could not download or validate {name} at {root}. "
            "Check your internet connection and that the folder is writable, "
            "then run this command again."
        ) from exc

    print(f"Ready: {name} ({count:,} samples)")
    return count


def prepare_sbd(root: Path) -> int:
    """Download/validate the SBD training split used by ConvX and scribble tools."""
    root = _resolve_root(root)
    return _load_dataset(
        "SBD train_noval segmentation",
        root,
        lambda: datasets.SBDataset(
            root=str(root),
            image_set="train_noval",
            mode="segmentation",
            download=True,
        ),
    )


def prepare_voc(root: Path) -> int:
    """Download/validate the PASCAL VOC 2012 validation split."""
    root = _resolve_root(root)
    return _load_dataset(
        "PASCAL VOC 2012 validation segmentation",
        root,
        lambda: datasets.VOCSegmentation(
            root=str(root),
            year="2012",
            image_set="val",
            download=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify the SBD and PASCAL VOC datasets used by ConvX. "
            "Datasets are stored in data_sbd/ and data_voc/ by default."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--sbd-only",
        action="store_true",
        help="Prepare only SBD (enough for the scribble annotator).",
    )
    group.add_argument(
        "--voc-only",
        action="store_true",
        help="Prepare only the PASCAL VOC 2012 validation split.",
    )
    parser.add_argument(
        "--sbd-root",
        type=Path,
        default=PROJECT_ROOT / "data_sbd",
        help="SBD dataset directory (default: <project>/data_sbd).",
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=PROJECT_ROOT / "data_voc",
        help="VOC dataset directory (default: <project>/data_voc).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.voc_only:
        prepare_sbd(args.sbd_root)
    if not args.sbd_only:
        prepare_voc(args.voc_root)
    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
