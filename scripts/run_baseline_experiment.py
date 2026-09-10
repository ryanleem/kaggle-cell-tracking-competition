#!/usr/bin/env python
"""Run and provenance a source-verified baseline experiment.

The runner deliberately shells out to the repository's existing CLIs.  It does
not import model code or reproduce training, prediction, metric, or conversion
logic, so the commands recorded in each run directory remain the authoritative
execution path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "src" / "tracking_cellmot" / "metrics.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_unet_transformer.py"
PREDICT_SCRIPT = REPO_ROOT / "scripts" / "predict_unet_transformer.py"
EVALUATE_SCRIPT = REPO_ROOT / "scripts" / "evaluate.py"
GEFFS_TO_CSV_SCRIPT = REPO_ROOT / "scripts" / "geffs_to_csv.py"
CSV_TO_GEFFS_SCRIPT = REPO_ROOT / "scripts" / "csv_to_geffs.py"
SOURCE_PATHS = {
    "train_script": TRAIN_SCRIPT,
    "predict_script": PREDICT_SCRIPT,
    "evaluate_script": EVALUATE_SCRIPT,
    "geffs_to_csv_script": GEFFS_TO_CSV_SCRIPT,
    "csv_to_geffs_script": CSV_TO_GEFFS_SCRIPT,
    "metrics.py": METRICS_PATH,
}
EXPECTED_CSV_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
]


class ExperimentError(RuntimeError):
    """Raised when a preflight or experiment invariant fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _paired_inventory(data_dir: Path) -> list[str]:
    """Match the training CLI's exact paired-inventory expression."""
    return sorted(
        p.name[:-5]
        for p in data_dir.glob("*.zarr")
        if (data_dir / f"{p.name[:-5]}.geff").exists()
    )


def _load_and_validate_split(
    split_path: Path,
    data_dir: Path,
    split: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the selected fold and its exact coverage of paired datasets."""
    try:
        raw = json.loads(split_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"cannot read split file {split_path}: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise ExperimentError("split JSON must be a non-empty list of fold objects")
    matches = [item for item in raw if isinstance(item, dict) and item.get("split") == split]
    if len(matches) != 1:
        raise ExperimentError(f"split JSON must contain exactly one fold with split={split}")
    fold = matches[0]
    if set(fold) != {"split", "train", "test"}:
        raise ExperimentError("split fold schema must contain exactly split, train, and test")
    if isinstance(fold["split"], bool) or not isinstance(fold["split"], int):
        raise ExperimentError("split value must be an integer")
    if not all(isinstance(fold[key], list) for key in ("train", "test")):
        raise ExperimentError("split train and test values must be lists")
    if not all(isinstance(name, str) and name for name in fold["train"] + fold["test"]):
        raise ExperimentError("split dataset names must be non-empty strings")

    train = fold["train"]
    test = fold["test"]
    if len(train) != len(set(train)) or len(test) != len(set(test)):
        raise ExperimentError("split contains duplicate dataset names")
    overlap = sorted(set(train) & set(test))
    if overlap:
        raise ExperimentError(f"split train/test overlap: {overlap}")

    paired = _paired_inventory(data_dir)
    selected = set(train) | set(test)
    missing = sorted(set(paired) - selected)
    extra = sorted(selected - set(paired))
    if missing or extra:
        raise ExperimentError(f"split coverage mismatch: missing={missing}, extra={extra}")

    shuffled = list(paired)
    random.Random(0).shuffle(shuffled)
    n_val = max(1, len(shuffled) // 10)
    algorithm_reproduction = train == shuffled[n_val:] and test == shuffled[:n_val]
    validation = {
        "paired_count": len(paired),
        "train_count": len(train),
        "test_count": len(test),
        "overlap": overlap,
        "train_duplicates": sorted(name for name in set(train) if train.count(name) > 1),
        "test_duplicates": sorted(name for name in set(test) if test.count(name) > 1),
        "missing": missing,
        "extra": extra,
        "algorithm_reproduction": algorithm_reproduction,
    }
    if not algorithm_reproduction:
        raise ExperimentError("split ordering does not reproduce the seed-0 fallback algorithm")
    return fold, validation


def _git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=all")
    dirty_entries = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(dirty_entries),
        "status_porcelain": dirty_entries,
    }


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _nonempty_path(path: Path) -> bool:
    if path.is_dir():
        return any(path.iterdir())
    return path.is_file() and path.stat().st_size > 0


def _prediction_stems(prediction_dir: Path, expected: list[str]) -> dict[str, list[str]]:
    paths = sorted(prediction_dir.glob("*.geff")) if prediction_dir.exists() else []
    actual = {path.stem for path in paths}
    expected_set = set(expected)
    missing = sorted(expected_set - actual)
    extra = sorted(actual - expected_set)
    empty = sorted(path.stem for path in paths if not _nonempty_path(path))
    if not expected:
        raise ExperimentError("validation split is empty")
    if not paths:
        raise ExperimentError(f"validation predictions are empty: {prediction_dir}")
    if missing or extra or empty:
        raise ExperimentError(
            f"validation prediction coverage failure: missing={missing}, extra={extra}, empty={empty}",
        )
    return {"actual": sorted(actual), "missing": missing, "extra": extra, "empty": empty}


def _integer(value: str, field: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentError(f"CSV {field} is not numeric: {value!r}") from exc
    if not number.is_integer():
        raise ExperimentError(f"CSV {field} is not an integer: {value!r}")
    return int(number)


def validate_csv(csv_path: Path, expected_datasets: list[str]) -> dict[str, Any]:
    """Validate the repository converter's submission CSV contract."""
    if len(expected_datasets) != len(set(expected_datasets)):
        raise ExperimentError("validation dataset list contains duplicates")
    if not expected_datasets:
        raise ExperimentError("validation dataset list is empty")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_CSV_COLUMNS:
            raise ExperimentError(
                f"CSV columns differ from {EXPECTED_CSV_COLUMNS}: {reader.fieldnames}",
            )
        rows = list(reader)

    ids = [_integer(row["id"], "id") for row in rows]
    if ids != list(range(len(rows))):
        raise ExperimentError("CSV id column must be consecutive and unique from zero")

    expected = set(expected_datasets)
    actual = {row["dataset"] for row in rows}
    if actual != expected:
        raise ExperimentError(
            f"CSV dataset coverage differs from validation split: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}",
        )

    node_ids: dict[str, set[int]] = {name: set() for name in expected_datasets}
    node_times: dict[str, dict[int, int]] = {name: {} for name in expected_datasets}
    edges: list[tuple[str, int, int]] = []
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ExperimentError("CSV contains a row with missing or extra fields")
        dataset = row["dataset"]
        if row["row_type"] == "node":
            node_id = _integer(row["node_id"], "node_id")
            if node_id < 0 or node_id in node_ids[dataset]:
                raise ExperimentError(f"duplicate or invalid node ID in dataset {dataset}: {node_id}")
            if _integer(row["source_id"], "source_id") != -1 or _integer(row["target_id"], "target_id") != -1:
                raise ExperimentError(f"node row has non-sentinel edge endpoints in {dataset}")
            node_times[dataset][node_id] = _integer(row["t"], "t")
            for field in ("z", "y", "x"):
                _integer(row[field], field)
            node_ids[dataset].add(node_id)
        elif row["row_type"] == "edge":
            sentinel_fields = ("node_id", "t", "z", "y", "x")
            if any(_integer(row[field], field) != -1 for field in sentinel_fields):
                raise ExperimentError(f"edge row has invalid node sentinels in {dataset}")
            source = _integer(row["source_id"], "source_id")
            target = _integer(row["target_id"], "target_id")
            if source < 0 or target < 0:
                raise ExperimentError(f"edge row has negative endpoint in {dataset}")
            edges.append((dataset, source, target))
        else:
            raise ExperimentError(f"invalid CSV row_type: {row['row_type']!r}")

    duplicate_edges = sorted(
        f"{dataset}:{source}->{target}"
        for dataset, source, target in set(edges)
        if edges.count((dataset, source, target)) > 1
    )
    if duplicate_edges:
        raise ExperimentError(f"CSV contains duplicate edges: {duplicate_edges}")
    for dataset, source, target in edges:
        if source not in node_ids[dataset] or target not in node_ids[dataset]:
            raise ExperimentError(f"edge endpoint is not a node in dataset {dataset}: {source}->{target}")
        if node_times[dataset][source] >= node_times[dataset][target]:
            raise ExperimentError(f"edge is not forward in time in dataset {dataset}: {source}->{target}")

    return {
        "columns": EXPECTED_CSV_COLUMNS,
        "row_count": len(rows),
        "dataset_count": len(actual),
        "node_count": sum(len(ids_for_dataset) for ids_for_dataset in node_ids.values()),
        "edge_count": len(edges),
        "datasets": sorted(actual),
    }


def _command_strings(command_map: dict[str, list[str]]) -> dict[str, list[str]]:
    return {name: [str(part) for part in command] for name, command in command_map.items()}


def _validate_checkpoint_iters(config: dict[str, Any]) -> None:
    """Validate the optional periodic checkpoint schedule in an experiment config."""
    if "checkpoint_iters" not in config or config["checkpoint_iters"] is None:
        return

    values = config["checkpoint_iters"]
    if not isinstance(values, list):
        raise ExperimentError("config checkpoint_iters must be a list of integers")
    if not values:
        raise ExperimentError("config checkpoint_iters must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ExperimentError("config checkpoint_iters must contain only integers")
    if any(value <= 0 for value in values):
        raise ExperimentError("config checkpoint_iters must contain only positive update numbers")
    if len(values) != len(set(values)):
        raise ExperimentError("config checkpoint_iters must not contain duplicates")
    if values != sorted(values):
        raise ExperimentError("config checkpoint_iters must be in strictly increasing order")

    max_iters = config["max_iters"]
    if isinstance(max_iters, bool) or not isinstance(max_iters, int):
        raise ExperimentError("config max_iters must be an integer when checkpoint_iters is set")
    if any(value > max_iters for value in values):
        raise ExperimentError("config checkpoint_iters cannot exceed max_iters")
    n_epochs = config["epochs"]
    if isinstance(n_epochs, bool) or not isinstance(n_epochs, int):
        raise ExperimentError("config epochs must be an integer when checkpoint_iters is set")
    if n_epochs != 1:
        raise ExperimentError("config checkpoint_iters currently requires epochs == 1")


def _periodic_checkpoint_filename(iteration: int) -> str:
    return f"edge_predictor_iter_{iteration:06d}.pth"


def _validate_periodic_outputs(
    checkpoint_dir: Path,
    history_path: Path,
    checkpoint_iters: list[int],
) -> list[str]:
    """Require and validate all periodic checkpoints emitted by training."""
    expected_names = [_periodic_checkpoint_filename(iteration) for iteration in checkpoint_iters]
    missing = [name for name in expected_names if not (checkpoint_dir / name).is_file()]
    if missing:
        raise ExperimentError(f"periodic checkpoints missing after training: {missing}")
    if not history_path.is_file():
        raise ExperimentError(f"training history not found after training: {history_path}")

    try:
        records = [json.loads(line) for line in history_path.read_text().splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"cannot read training history {history_path}: {exc}") from exc
    if len(records) != len(expected_names):
        raise ExperimentError(
            f"training history has {len(records)} records; expected {len(expected_names)}",
        )

    required_fields = {
        "iteration", "running_avg_edge_loss", "running_avg_detection_loss",
        "checkpoint_filename", "elapsed_training_seconds",
    }
    for record, iteration, expected_name in zip(records, checkpoint_iters, expected_names):
        if not isinstance(record, dict) or not required_fields <= set(record):
            raise ExperimentError("training history contains a record with missing required fields")
        if record["iteration"] != iteration or record["checkpoint_filename"] != expected_name:
            raise ExperimentError("training history records are not ordered as requested checkpoints")
    return expected_names


def build_commands(
    config: dict[str, Any],
    run_method: str,
    data_dir: Path,
    split_path: Path,
    checkpoint_path: Path,
    prediction_dir: Path,
    run_dir: Path,
) -> dict[str, list[str]]:
    """Build only flags verified against the current repository CLIs."""
    common = [
        "--method", run_method,
        "--data-dir", str(data_dir),
        "--splits", str(split_path),
        "--split", str(config["split"]),
    ]
    train = [
        str(sys.executable), str(TRAIN_SCRIPT), *common,
        "--seed", str(config["seed"]),
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
        "--batch-size", str(config["batch_size"]),
        "--num-workers", str(config["num_workers"]),
        "--unet-out-channels", str(config["unet_out_channels"]),
        "--unet-layers", str(config["unet_layers"]),
        "--downsample", str(config["downsample"]),
        "--det-loss-weight", str(config["det_loss_weight"]),
        "--det-neg-weight", str(config["det_neg_weight"]),
        "--max-iters", str(config["max_iters"]),
        "--window-size", str(config["window_size"]),
        "--pool-kernel-um", str(config["pool_kernel_um"]),
    ]
    if config.get("checkpoint_iters"):
        train.extend([
            "--checkpoint-iters",
            ",".join(str(value) for value in config["checkpoint_iters"]),
        ])
    predict = [
        str(sys.executable), str(PREDICT_SCRIPT), *common,
        "--weights", str(checkpoint_path),
        "--det-threshold", str(config["det_threshold"]),
        "--pool-kernel-um", str(config["pool_kernel_um"]),
    ]
    evaluate = [
        str(sys.executable), str(EVALUATE_SCRIPT),
        "--pred-dir", str(prediction_dir),
        "--gt-dir", str(data_dir),
        "--strict", "--json-out", str(run_dir / "metrics.json"),
    ]
    roundtrip_evaluate = [
        str(sys.executable), str(EVALUATE_SCRIPT),
        "--pred-dir", str(run_dir / "reconstructed_predictions"),
        "--gt-dir", str(data_dir),
        "--strict", "--json-out", str(run_dir / "metrics_roundtrip.json"),
    ]
    return {
        "training": train,
        "prediction": predict,
        "evaluation": evaluate,
        "geffs_to_csv": [
            str(sys.executable), str(GEFFS_TO_CSV_SCRIPT),
            "--in-dir", str(prediction_dir), "--csv", str(run_dir / "predictions.csv"),
        ],
        "csv_to_geffs": [
            str(sys.executable), str(CSV_TO_GEFFS_SCRIPT),
            "--csv", str(run_dir / "predictions.csv"),
            "--out-dir", str(run_dir / "reconstructed_predictions"),
        ],
        "roundtrip_evaluation": roundtrip_evaluate,
    }


def _build_prediction_command(
    config: dict[str, Any],
    run_method: str,
    data_dir: Path,
    split_path: Path,
    checkpoint_path: Path,
) -> list[str]:
    """Build the repository prediction command for one fixed checkpoint."""
    return [
        str(sys.executable), str(PREDICT_SCRIPT),
        "--method", run_method,
        "--data-dir", str(data_dir),
        "--splits", str(split_path),
        "--split", str(config["split"]),
        "--weights", str(checkpoint_path),
        "--det-threshold", str(config["det_threshold"]),
        "--pool-kernel-um", str(config["pool_kernel_um"]),
        "--tracking", str(config["tracking"]),
    ]


def _build_checkpoint_evaluation_commands(
    config: dict[str, Any],
    method: str,
    data_dir: Path,
    split_path: Path,
    checkpoint_path: Path,
    prediction_dir: Path,
    metrics_path: Path,
) -> dict[str, list[str]]:
    """Build prediction/evaluation commands for one periodic checkpoint."""
    return {
        "prediction": _build_prediction_command(
            config, method, data_dir, split_path, checkpoint_path,
        ),
        "evaluation": [
            str(sys.executable), str(EVALUATE_SCRIPT),
            "--pred-dir", str(prediction_dir),
            "--gt-dir", str(data_dir),
            "--strict", "--json-out", str(metrics_path),
        ],
    }


_REQUIRED_PER_DATASET_METRICS = {
    "edge_tp", "edge_fp", "edge_fn", "num_pred_nodes", "total_node_ratio",
    "node_recall", "edge_jaccard", "adj_edge_jaccard",
}


def _validate_checkpoint_metrics(
    metrics_path: Path,
    expected_datasets: list[str],
) -> dict[str, Any]:
    """Validate one strict official-evaluation JSON report and return it."""
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(
            f"cannot read checkpoint evaluation metrics {metrics_path}: {exc}",
        ) from exc
    if not isinstance(metrics, dict):
        raise ExperimentError(f"checkpoint evaluation metrics are not an object: {metrics_path}")

    evaluated = metrics.get("evaluated_datasets")
    if (
        not isinstance(evaluated, list)
        or not all(isinstance(name, str) for name in evaluated)
        or len(evaluated) != len(expected_datasets)
        or set(evaluated) != set(expected_datasets)
    ):
        raise ExperimentError(
            f"checkpoint evaluation datasets differ from validation split: "
            f"{evaluated} != {expected_datasets}",
        )
    skipped = metrics.get("skipped_datasets")
    if not isinstance(skipped, list) or skipped != []:
        raise ExperimentError(
            f"checkpoint evaluation skipped validation datasets: {skipped}",
        )

    summary = metrics.get("summary_metrics")
    if not isinstance(summary, dict):
        raise ExperimentError(f"checkpoint evaluation summary_metrics is missing or malformed: {metrics_path}")
    score = summary.get("score")
    if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)):
        raise ExperimentError(f"checkpoint evaluation score is not finite: {score!r}")

    per_dataset = metrics.get("per_dataset_metrics")
    if not isinstance(per_dataset, list) or len(per_dataset) != len(expected_datasets):
        raise ExperimentError(
            f"checkpoint evaluation per_dataset_metrics is missing or incomplete: {metrics_path}",
        )
    per_dataset_names: list[str] = []
    for row in per_dataset:
        if not isinstance(row, dict) or not isinstance(row.get("dataset"), str):
            raise ExperimentError(f"checkpoint evaluation has malformed per-dataset metrics: {metrics_path}")
        row_metrics = row.get("metrics")
        if not isinstance(row_metrics, dict) or not _REQUIRED_PER_DATASET_METRICS <= set(row_metrics):
            raise ExperimentError(
                f"checkpoint evaluation per-dataset metrics are incomplete for {row.get('dataset')!r}",
            )
        per_dataset_names.append(row["dataset"])
    if len(set(per_dataset_names)) != len(expected_datasets) or set(per_dataset_names) != set(expected_datasets):
        raise ExperimentError(
            f"checkpoint per-dataset metrics differ from validation split: "
            f"{per_dataset_names} != {expected_datasets}",
        )
    return metrics


def _checkpoint_evaluation_method(run_method: str, iteration: int) -> str:
    return f"{run_method}_checkpoint_iter_{iteration:06d}"


def _checkpoint_evaluation_prediction_dir(
    repo_root: Path, username: str, method: str, split: int,
) -> Path:
    return repo_root / "predictions" / username / method / f"split_{split}"


def _remove_checkpoint_evaluation_predictions(
    prediction_dir: Path, repo_root: Path, username: str, method: str, split: int,
) -> None:
    """Remove exactly one generated periodic-evaluation prediction directory."""
    expected = _checkpoint_evaluation_prediction_dir(repo_root, username, method, split)
    if prediction_dir.resolve() != expected.resolve():
        raise ExperimentError(f"refusing to remove unexpected checkpoint prediction directory: {prediction_dir}")
    if prediction_dir.exists():
        if not prediction_dir.is_dir():
            raise ExperimentError(f"checkpoint prediction path is not a directory: {prediction_dir}")
        shutil.rmtree(prediction_dir)


def _evaluate_periodic_checkpoints(
    config: dict[str, Any],
    data_dir: Path,
    split_path: Path,
    fold: dict[str, Any],
    run_method: str,
    username: str,
    run_dir: Path,
    periodic_checkpoint_paths: list[Path],
    periodic_checkpoint_config_path: Path,
    provenance: dict[str, Any],
    provenance_path: Path,
) -> dict[str, Any]:
    """Evaluate, validate, preserve, and select every periodic checkpoint."""
    if not periodic_checkpoint_config_path.is_file() or not _nonempty_path(periodic_checkpoint_config_path):
        raise ExperimentError(
            "periodic checkpoint config is missing or empty: "
            f"{periodic_checkpoint_config_path}",
        )
    evaluation_root = run_dir / "checkpoint_evaluations"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    checkpoint_iters = config["checkpoint_iters"]

    for iteration, checkpoint_path in zip(checkpoint_iters, periodic_checkpoint_paths, strict=True):
        method = _checkpoint_evaluation_method(run_method, iteration)
        stage_dir = evaluation_root / f"iter_{iteration:06d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir = _checkpoint_evaluation_prediction_dir(
            REPO_ROOT, username, method, int(config["split"]),
        )
        metrics_path = stage_dir / "metrics.json"
        commands = _build_checkpoint_evaluation_commands(
            config, method, data_dir, split_path, checkpoint_path,
            prediction_dir, metrics_path,
        )
        prediction_stage = f"checkpoint_iter_{iteration:06d}_prediction"
        evaluation_stage = f"checkpoint_iter_{iteration:06d}_evaluation"

        _run_stage(prediction_stage, commands["prediction"], stage_dir, provenance, provenance_path)
        prediction_validation = _prediction_stems(prediction_dir, fold["test"])
        _run_stage(evaluation_stage, commands["evaluation"], stage_dir, provenance, provenance_path)
        metrics = _validate_checkpoint_metrics(metrics_path, fold["test"])

        record = {
            "iteration": iteration,
            "checkpoint_filename": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path),
            "periodic_checkpoint_config_path": str(periodic_checkpoint_config_path),
            "method": method,
            "prediction_dir": str(prediction_dir),
            "metrics_path": str(metrics_path),
            "prediction_stage": prediction_stage,
            "evaluation_stage": evaluation_stage,
            "prediction_log_dir": str(stage_dir),
            "prediction_validation": prediction_validation,
            "complete_metrics": metrics,
            "summary_metrics": metrics["summary_metrics"],
            "per_dataset_metrics": metrics["per_dataset_metrics"],
            "prediction_wall_seconds": provenance["stages"][prediction_stage]["wall_seconds"],
            "evaluation_wall_seconds": provenance["stages"][evaluation_stage]["wall_seconds"],
        }
        records.append(record)

        # Metrics and logs have been written under run_dir and the record is
        # durable before removing this iteration's uniquely named predictions.
        _write_json(
            run_dir / "checkpoint_evaluations.json",
            {
                "schema_version": 1,
                "selection_metric": "summary_metrics.score",
                "tie_rule": "later_iteration_wins_equal_scores",
                "periodic_checkpoint_config_path": str(periodic_checkpoint_config_path),
                "evaluations": records,
            },
        )
        _remove_checkpoint_evaluation_predictions(
            prediction_dir, REPO_ROOT, username, method, int(config["split"]),
        )

    selected = max(
        records,
        key=lambda record: (float(record["summary_metrics"]["score"]), record["iteration"]),
    )
    evaluation_report = {
        "schema_version": 1,
        "selection_metric": "summary_metrics.score",
        "tie_rule": "later_iteration_wins_equal_scores",
        "periodic_checkpoint_config_path": str(periodic_checkpoint_config_path),
        "evaluations": records,
        "selected_iteration": selected["iteration"],
        "selected_checkpoint": selected["checkpoint_filename"],
        "selected_checkpoint_path": selected["checkpoint_path"],
        "selected_score": selected["summary_metrics"]["score"],
    }
    _write_json(run_dir / "checkpoint_evaluations.json", evaluation_report)
    return evaluation_report


def _run_logged_command(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    elapsed = time.monotonic() - started
    stdout_path.write_text(result.stdout or "")
    stderr_path.write_text(result.stderr or "")
    return {"returncode": result.returncode, "wall_seconds": elapsed}


def _run_stage(
    stage: str,
    command: list[str],
    run_dir: Path,
    provenance: dict[str, Any],
    provenance_path: Path,
) -> None:
    result = _run_logged_command(
        command,
        run_dir / f"{stage}.stdout.log",
        run_dir / f"{stage}.stderr.log",
    )
    provenance.setdefault("stages", {})[stage] = result
    _write_json(provenance_path, provenance)
    if result["returncode"] != 0:
        raise ExperimentError(
            f"{stage} failed with return code {result['returncode']}; see {run_dir / f'{stage}.stderr.log'}",
        )


def run_experiment(
    config_path: Path,
    data_dir: Path,
    output_root: Path,
    allow_dirty: bool = False,
) -> Path:
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ExperimentError("experiment config must be a JSON object")
    required = {
        "split", "seed", "epochs", "max_iters", "batch_size", "num_workers", "lr",
        "det_loss_weight", "det_neg_weight", "unet_out_channels", "unet_layers",
        "downsample", "window_size", "pool_kernel_um", "det_threshold", "tracking",
        "split_file",
    }
    missing_config = sorted(required - set(config))
    if missing_config:
        raise ExperimentError(f"config missing required keys: {missing_config}")
    _validate_checkpoint_iters(config)
    if config["tracking"] != "greedy":
        raise ExperimentError("this runner supports only tracking=greedy")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise ExperimentError("config seed must be an integer")

    config_path = config_path.resolve()
    data_dir = data_dir.resolve()
    output_root = output_root.resolve()
    split_path = Path(config["split_file"])
    if not split_path.is_absolute():
        split_path = REPO_ROOT / split_path
    split_path = split_path.resolve()
    if not split_path.exists():
        raise ExperimentError(f"split file does not exist: {split_path}")
    if not data_dir.is_dir():
        raise ExperimentError(f"data directory does not exist: {data_dir}")

    git = _git_provenance()
    if git["dirty"] and not allow_dirty:
        raise ExperimentError(
            "refusing to run from a dirty Git worktree; use --allow-dirty as an explicit development override",
        )

    fold, split_validation = _load_and_validate_split(split_path, data_dir, int(config["split"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{config_path.stem}_{stamp}_{uuid.uuid4().hex[:10]}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    run_method = f"{config.get('method', 'baseline')}_{run_name}"

    # dataspec.py uses USER first and USERNAME second, matching the prediction CLI.
    username = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
    prediction_dir = REPO_ROOT / "predictions" / username / run_method / f"split_{config['split']}"
    checkpoint_path = REPO_ROOT / "weights" / run_method / f"split_{config['split']}" / "edge_predictor_best.pth"
    commands = build_commands(
        config, run_method, data_dir, split_path, checkpoint_path, prediction_dir, run_dir,
    )
    resolved_config = {
        **config,
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "split_file": str(split_path),
        "output_root": str(output_root),
        "run_name": run_name,
        "run_method": run_method,
        "seed_passed_to_training": True,
        "development_override": allow_dirty,
    }
    resolved_config_path = run_dir / "resolved_config.json"
    provenance_path = run_dir / "provenance.json"
    _write_json(resolved_config_path, resolved_config)
    provenance = {
        "schema_version": 1,
        "run_name": run_name,
        "seed": config["seed"],
        "git": git,
        "python_executable": str(Path(sys.executable).resolve()),
        "hashes": {
            **{
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in SOURCE_PATHS.items()
            },
            "config_json": {"path": str(config_path), "sha256": _sha256(config_path)},
            "split_json": {"path": str(split_path), "sha256": _sha256(split_path)},
        },
        "resolved_config": str(resolved_config_path),
        "split_validation": split_validation,
        "fold": fold,
        "commands": _command_strings(commands),
        "stages": {},
        "status": "prepared",
    }
    _write_json(provenance_path, provenance)

    started_all = time.monotonic()
    try:
        _run_stage("training", commands["training"], run_dir, provenance, provenance_path)
        if not checkpoint_path.is_file():
            raise ExperimentError(f"training checkpoint not found: {checkpoint_path}")
        checkpoint_config = checkpoint_path.parent / "config.json"
        if not checkpoint_config.is_file():
            raise ExperimentError(f"checkpoint config not found adjacent to checkpoint: {checkpoint_config}")
        _copy_path(checkpoint_path, run_dir / "checkpoint" / checkpoint_path.name)
        _copy_path(checkpoint_config, run_dir / "checkpoint" / "config.json")

        periodic_checkpoint_paths: list[Path] | None = None
        training_history_path: str | None = None
        official_checkpoint_path: Path | None = None
        checkpoint_evaluations: dict[str, Any] | None = None
        if config.get("checkpoint_iters"):
            training_output_dir = checkpoint_path.parent
            periodic_checkpoint_config_path = run_dir / "checkpoints" / "config.json"
            _copy_path(checkpoint_config, periodic_checkpoint_config_path)
            if not periodic_checkpoint_config_path.is_file() or not _nonempty_path(periodic_checkpoint_config_path):
                raise ExperimentError(
                    "periodic checkpoint config was not copied or is empty: "
                    f"{periodic_checkpoint_config_path}",
                )
            provenance["periodic_checkpoint_config_path"] = str(periodic_checkpoint_config_path)
            _write_json(provenance_path, provenance)
            history_source = training_output_dir / "training_history.jsonl"
            periodic_names = _validate_periodic_outputs(
                training_output_dir,
                history_source,
                config["checkpoint_iters"],
            )
            periodic_checkpoint_paths = []
            for name in periodic_names:
                destination = run_dir / "checkpoints" / name
                _copy_path(training_output_dir / name, destination)
                periodic_checkpoint_paths.append(destination)
            history_destination = run_dir / "training_history.jsonl"
            _copy_path(history_source, history_destination)
            training_history_path = str(history_destination)

            checkpoint_evaluations = _evaluate_periodic_checkpoints(
                config,
                data_dir,
                split_path,
                fold,
                run_method,
                username,
                run_dir,
                periodic_checkpoint_paths,
                periodic_checkpoint_config_path,
                provenance,
                provenance_path,
            )
            selected_iteration = checkpoint_evaluations["selected_iteration"]
            selected_filename = checkpoint_evaluations["selected_checkpoint"]
            selected_source = run_dir / "checkpoints" / selected_filename
            if selected_source != Path(checkpoint_evaluations["selected_checkpoint_path"]):
                raise ExperimentError(
                    "checkpoint selection path does not refer to the preserved run checkpoint",
                )
            official_checkpoint_path = run_dir / "checkpoint" / "edge_predictor_official_best.pth"
            _copy_path(selected_source, official_checkpoint_path)
            if not official_checkpoint_path.is_file() or not _nonempty_path(official_checkpoint_path):
                raise ExperimentError(f"official selected checkpoint was not copied: {official_checkpoint_path}")

            # The original acc*recall-selected artifact remains at
            # run_dir/checkpoint/edge_predictor_best.pth. Only the final
            # prediction command switches to the official scorer-selected file.
            commands["prediction"] = _build_prediction_command(
                config, run_method, data_dir, split_path, official_checkpoint_path,
            )
            provenance["commands"] = _command_strings(commands)
            provenance.update({
                "checkpoint_selection_metric": checkpoint_evaluations["selection_metric"],
                "selected_checkpoint_iteration": selected_iteration,
                "selected_checkpoint_score": checkpoint_evaluations["selected_score"],
                "selected_checkpoint_path": str(selected_source),
                "checkpoint_evaluations_artifact_path": str(run_dir / "checkpoint_evaluations.json"),
                "periodic_checkpoint_config_path": str(periodic_checkpoint_config_path),
                "training_selected_checkpoint_path": str(run_dir / "checkpoint" / checkpoint_path.name),
                "official_selected_checkpoint_path": str(official_checkpoint_path),
            })
            _write_json(provenance_path, provenance)

        _run_stage("prediction", commands["prediction"], run_dir, provenance, provenance_path)
        prediction_validation = _prediction_stems(prediction_dir, fold["test"])
        provenance["prediction_validation"] = prediction_validation
        _copy_path(prediction_dir, run_dir / "predictions")
        _write_json(provenance_path, provenance)

        _run_stage("evaluation", commands["evaluation"], run_dir, provenance, provenance_path)
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        evaluated = metrics.get("evaluated_datasets")
        if set(evaluated or []) != set(fold["test"]):
            raise ExperimentError(
                f"evaluation dataset names differ from validation split: {evaluated} != {fold['test']}",
            )
        if metrics.get("skipped_datasets"):
            raise ExperimentError(f"evaluation skipped validation datasets: {metrics['skipped_datasets']}")
        if set(metrics.get("evaluated_datasets", [])) != set(fold["test"]):
            raise ExperimentError("evaluation coverage does not equal validation coverage")

        _run_stage("geffs_to_csv", commands["geffs_to_csv"], run_dir, provenance, provenance_path)
        csv_validation = validate_csv(run_dir / "predictions.csv", fold["test"])
        _write_json(run_dir / "csv_validation.json", csv_validation)

        _run_stage("csv_to_geffs", commands["csv_to_geffs"], run_dir, provenance, provenance_path)
        reconstructed_validation = _prediction_stems(run_dir / "reconstructed_predictions", fold["test"])
        provenance["reconstructed_prediction_validation"] = reconstructed_validation
        _write_json(provenance_path, provenance)

        _run_stage("roundtrip_evaluation", commands["roundtrip_evaluation"], run_dir, provenance, provenance_path)
        roundtrip_metrics = json.loads((run_dir / "metrics_roundtrip.json").read_text())
        if set(roundtrip_metrics.get("evaluated_datasets", [])) != set(fold["test"]):
            raise ExperimentError("round-trip evaluation coverage differs from validation split")
        if roundtrip_metrics.get("skipped_datasets"):
            raise ExperimentError(
                f"round-trip evaluation skipped validation datasets: {roundtrip_metrics['skipped_datasets']}",
            )
        if roundtrip_metrics.get("summary_metrics") != metrics.get("summary_metrics"):
            raise ExperimentError("round-trip summary metrics differ from original evaluation")

        elapsed = time.monotonic() - started_all
        provenance["wall_seconds_total"] = elapsed
        provenance["status"] = "success"
        _write_json(provenance_path, provenance)
        artifacts = {
            "checkpoint": str(run_dir / "checkpoint" / checkpoint_path.name),
            "checkpoint_config": str(run_dir / "checkpoint" / "config.json"),
            "predictions": str(run_dir / "predictions"),
            "csv": str(run_dir / "predictions.csv"),
            "reconstructed_predictions": str(run_dir / "reconstructed_predictions"),
            "metrics": str(metrics_path),
            "metrics_roundtrip": str(run_dir / "metrics_roundtrip.json"),
            "provenance": str(provenance_path),
            "resolved_config": str(resolved_config_path),
        }
        if periodic_checkpoint_paths is not None and training_history_path is not None:
            artifacts["periodic_checkpoints"] = [str(path) for path in periodic_checkpoint_paths]
            artifacts["training_history"] = training_history_path
            if checkpoint_evaluations is None or official_checkpoint_path is None:
                raise ExperimentError("periodic checkpoint selection did not produce an official checkpoint")
            artifacts["checkpoint"] = str(official_checkpoint_path)
            artifacts["training_selected_checkpoint"] = str(run_dir / "checkpoint" / checkpoint_path.name)
            artifacts["official_selected_checkpoint"] = str(official_checkpoint_path)
            artifacts["checkpoint_evaluations"] = str(run_dir / "checkpoint_evaluations.json")
            artifacts["periodic_checkpoint_config"] = str(periodic_checkpoint_config_path)
        final_result = {
            "status": "success",
            "run_name": run_name,
            "seed": config["seed"],
            "git_commit": git["commit"],
            "summary_metrics": metrics["summary_metrics"],
            "roundtrip_summary_metrics": roundtrip_metrics["summary_metrics"],
            "wall_seconds_total": elapsed,
            "artifacts": artifacts,
        }
        if checkpoint_evaluations is not None and official_checkpoint_path is not None:
            final_result.update({
                "checkpoint_selection_metric": checkpoint_evaluations["selection_metric"],
                "selected_checkpoint_iteration": checkpoint_evaluations["selected_iteration"],
                "selected_checkpoint_score": checkpoint_evaluations["selected_score"],
                "selected_checkpoint_path": checkpoint_evaluations["selected_checkpoint_path"],
                "checkpoint_evaluations_artifact_path": str(run_dir / "checkpoint_evaluations.json"),
                "periodic_checkpoint_config_path": str(periodic_checkpoint_config_path),
                "training_selected_checkpoint_path": str(run_dir / "checkpoint" / checkpoint_path.name),
                "official_selected_checkpoint_path": str(official_checkpoint_path),
            })
        _write_json(
            run_dir / "final_result.json",
            final_result,
        )
    except Exception as exc:
        provenance["wall_seconds_total"] = time.monotonic() - started_all
        provenance["status"] = "failed"
        provenance["failure"] = f"{type(exc).__name__}: {exc}"
        _write_json(provenance_path, provenance)
        raise
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the source-verified baseline experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--allow-dirty", "--allow-dirty-worktree", dest="allow_dirty", action="store_true",
        help="Explicit development override allowing a dirty Git worktree.",
    )
    args = parser.parse_args()
    try:
        run_dir = run_experiment(args.config, args.data_dir, args.output_root, args.allow_dirty)
    except (ExperimentError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(run_dir)


if __name__ == "__main__":
    main()
