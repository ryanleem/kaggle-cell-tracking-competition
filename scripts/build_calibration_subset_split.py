#!/usr/bin/env python
"""Derive a small validation-subset split file from an existing fold split.

Used to build a representative screening subset for threshold-calibration
Stages A and B, so those stages run prediction+evaluation on a handful of
known validation datasets instead of the full fold. The subset file follows
exactly the same schema as the source split file (a JSON list of
``{"split": int, "train": [...], "test": [...]}`` fold objects): ``train`` is
copied unchanged from the source fold and ``test`` is replaced by the
requested subset of dataset names, which must all already be members of the
source fold's ``test`` list.

This script only reads JSON and writes JSON — it never touches training data,
model weights, or runs inference.

Usage:
    python scripts/build_calibration_subset_split.py \
        --source-split experiments/splits/baseline_fold0_seed0.json \
        --source-split-index 0 \
        --dataset-names 44b6_d5e7d891 44b6_d754aa59 44b6_e57ff5c6 \
                        6bba_268e1230 6bba_3a1849c2 6bba_afb141ff \
        --output-split experiments/splits/calibration_subset_fold0.json \
        --output-split-index 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class SubsetSplitError(RuntimeError):
    """Raised when the source split or requested subset is invalid."""


def _load_fold(source_split_path: Path, source_split_index: int) -> dict:
    try:
        raw = json.loads(source_split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubsetSplitError(f"cannot read source split {source_split_path}: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise SubsetSplitError("source split JSON must be a non-empty list of fold objects")
    matches = [item for item in raw if isinstance(item, dict) and item.get("split") == source_split_index]
    if len(matches) != 1:
        raise SubsetSplitError(
            f"source split JSON must contain exactly one fold with split={source_split_index}",
        )
    fold = matches[0]
    if set(fold) != {"split", "train", "test"}:
        raise SubsetSplitError("source split fold schema must contain exactly split, train, and test")
    if isinstance(fold["split"], bool) or not isinstance(fold["split"], int):
        raise SubsetSplitError("source split value must be an integer")
    if not all(isinstance(fold[key], list) for key in ("train", "test")):
        raise SubsetSplitError("source split train and test values must be lists")
    if not all(isinstance(name, str) and name for name in fold["train"] + fold["test"]):
        raise SubsetSplitError("source split dataset names must be non-empty strings")
    return fold


def build_subset_fold(
    source_fold: dict,
    dataset_names: list[str],
    output_split_index: int,
) -> dict:
    """Build a subset fold object reusing the source schema and ``train`` list.

    ``dataset_names`` must be a duplicate-free, non-empty list, and every name
    must already be a member of ``source_fold["test"]`` — the subset is a
    screening slice of the *validation* set, never a mix of train and test.
    """
    if not dataset_names:
        raise SubsetSplitError("subset dataset name list must not be empty")
    if len(dataset_names) != len(set(dataset_names)):
        raise SubsetSplitError("subset dataset name list contains duplicates")

    source_test = set(source_fold["test"])
    missing = sorted(set(dataset_names) - source_test)
    if missing:
        raise SubsetSplitError(
            f"subset dataset name(s) not found in source split's test list: {missing}",
        )

    return {
        "split": output_split_index,
        "train": list(source_fold["train"]),
        "test": list(dataset_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--source-split-index", type=int, default=0)
    parser.add_argument("--dataset-names", type=str, nargs="+", required=True)
    parser.add_argument("--output-split", type=Path, required=True)
    parser.add_argument("--output-split-index", type=int, default=0)
    args = parser.parse_args()

    try:
        source_fold = _load_fold(args.source_split, args.source_split_index)
        subset_fold = build_subset_fold(source_fold, args.dataset_names, args.output_split_index)
    except SubsetSplitError as exc:
        parser.error(str(exc))
        return

    args.output_split.parent.mkdir(parents=True, exist_ok=True)
    args.output_split.write_text(json.dumps([subset_fold], indent=2) + "\n", encoding="utf-8")
    print(f"Wrote subset split ({len(subset_fold['test'])} test datasets) to {args.output_split}")


if __name__ == "__main__":
    main()
