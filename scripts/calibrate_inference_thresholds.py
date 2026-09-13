#!/usr/bin/env python
"""Generic, resumable, compute-conscious inference-threshold calibration runner.

The runner shells out to the repository's existing ``predict_unet_transformer.py``
and ``evaluate.py`` CLIs — it never imports model or training code and never
invokes ``train_unet_transformer.py``. Each *trial* is one (checkpoint,
det_threshold, edge_threshold) combination: prediction runs against a unique
scratch directory, is validated and copied into the trial's own isolated
output directory, then scored with ``evaluate.py --strict --json-out``.

Results are written incrementally to a single JSON report
(``<output-dir>/calibration_results.json``) after every trial, using an
atomic write-temp-then-replace so a crash mid-run never corrupts previously
completed trials. Re-running with the same ``--output-dir`` resumes: a prior
trial is only reused if its calibration config, checkpoint (and checkpoint
config.json), split file, data directory, parameters, predictions, and
on-disk metrics all still validate exactly against what's on disk right now
— anything else is re-executed from scratch, and the trial's own output
directory plus its deterministic scratch prediction directory are wiped
first so stale files can never leak into a fresh run.

Every stage persists a deterministic, tamper-evident selection alongside its
trial records, and every stage requires its *entire* planned grid to have
succeeded with a finite score before it will select anything — a partial
grid (any failed or missing trial) is never silently selected from. Stage A
writes ``selected_det_thresholds``, Stage B writes ``selected_pairs``, and
Stage C writes ``selected_final_trial`` (the single official winner, with
its checkpoint, thresholds, score, and full metrics). Stage B and Stage C
each strictly validate the previous stage's results file before trusting
it: schema version, stage identity, config hash, data directory, the exact
expected trial grid (IDs and every parameter, generated fresh from the
current config and checkpoints — not merely a matching count), full
success with finite scores, screening status, an on-disk revalidation of
every successful trial's checkpoint/config hashes, split hash, prediction
contents (via a deterministic per-GEFF manifest), and metrics, and the
recorded selection must all check out, or the run refuses to start.

Staged usage — see ``experiments/configs/inference_threshold_calibration.json``
for the reviewed grid parameters:

    # Stage A: screen det_threshold on the 10000 checkpoint, subset data.
    python scripts/calibrate_inference_thresholds.py \\
        --config experiments/configs/inference_threshold_calibration.json \\
        --stage A --checkpoint 10000=weights/.../edge_predictor_iter_010000.pth \\
        --data-dir data/train --output-dir experiments/calibration_runs/stage_a

    # Stage B: screen edge_threshold on the best two Stage-A thresholds.
    python scripts/calibrate_inference_thresholds.py \\
        --config experiments/configs/inference_threshold_calibration.json \\
        --stage B --checkpoint 10000=weights/.../edge_predictor_iter_010000.pth \\
        --data-dir data/train --output-dir experiments/calibration_runs/stage_b \\
        --previous-results experiments/calibration_runs/stage_a/calibration_results.json

    # Stage C: confirm the best two pairs on both checkpoints, full fold.
    python scripts/calibrate_inference_thresholds.py \\
        --config experiments/configs/inference_threshold_calibration.json \\
        --stage C --checkpoint 7500=... --checkpoint 10000=... \\
        --data-dir data/train --output-dir experiments/calibration_runs/stage_c \\
        --previous-results experiments/calibration_runs/stage_b/calibration_results.json

Stage A and B trials run on a small representative subset of the validation
fold (see ``build_calibration_subset_split.py``) purely to limit GPU use;
their scores are SCREENING scores, not final validation scores. Only Stage C
runs on the complete validation fold and produces final validation scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT = REPO_ROOT / "scripts" / "predict_unet_transformer.py"
EVALUATE_SCRIPT = REPO_ROOT / "scripts" / "evaluate.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_unet_transformer.py"

SCHEMA_VERSION = 1
HEARTBEAT_INTERVAL_SECONDS = 60.0

# The reviewed calibration plan pins exactly which checkpoints each stage may
# touch. Config validation rejects any drift from this list so an edited
# config can never silently expand the reviewed scope.
_REVIEWED_STAGE_AB_CHECKPOINT_LABELS = ["10000"]
_REVIEWED_STAGE_C_CHECKPOINT_LABELS = ["7500", "10000"]

_REQUIRED_CONFIG_KEYS = {
    "pool_kernel_um", "tracking", "current_det_threshold", "edge_threshold_default",
    "subset_split_file", "subset_split", "subset_dataset_names",
    "full_split_file", "full_split",
    "stage_a", "stage_b", "stage_c",
}
_REQUIRED_STAGE_KEYS = {
    "stage_a": {"checkpoint_labels", "det_thresholds", "edge_thresholds", "n_best_det_thresholds"},
    "stage_b": {"checkpoint_labels", "edge_thresholds", "n_best_pairs"},
    "stage_c": {"checkpoint_labels"},
}

_STAGE_KEY_BY_ARG = {"A": "stage_a", "B": "stage_b", "C": "stage_c"}


class CalibrationError(RuntimeError):
    """Raised when a calibration precondition, config, or trial invariant fails."""


# =============================================================================
# Path resolution — deterministic regardless of the caller's working directory
# =============================================================================

def _resolve_repo_path(value: str | Path) -> Path:
    """Resolve a config-declared repository file against REPO_ROOT.

    Config files reference repository paths (e.g. split files) as strings
    relative to the repository root, not to whatever directory the runner
    happens to be invoked from. Absolute paths pass through unchanged.
    """
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _resolve_cli_path(value: str | Path) -> Path:
    """Resolve a CLI-supplied path to an absolute path once, up front.

    Every path recorded in a trial record or used to build a command is
    resolved through here so resume validation and provenance stay stable
    even if the runner is later re-invoked from a different directory.
    """
    return Path(value).resolve()


# =============================================================================
# Small filesystem / provenance helpers
# =============================================================================

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_sha256(path: str | None) -> str | None:
    if not path:
        return None
    try:
        candidate = Path(path)
        if candidate.is_file():
            return _sha256(candidate)
    except OSError:
        pass
    return None


def _require_sha256(path: Path) -> str:
    value = _safe_sha256(str(path))
    if value is None:
        raise CalibrationError(f"cannot hash required file (missing or unreadable): {path}")
    return value


def _nonempty_path(path: Path) -> bool:
    if path.is_dir():
        return any(path.iterdir())
    return path.is_file() and path.stat().st_size > 0


def _write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON to ``path`` via write-temp-then-replace so it is never partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:10]}")
    tmp_path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()

    try:
        status = git("status", "--porcelain", "--untracked-files=all")
        dirty_entries = [line for line in status.splitlines() if line.strip()]
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(dirty_entries),
            "status_porcelain": dirty_entries,
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": f"{type(exc).__name__}: {exc}"}


def _environment_provenance() -> dict[str, Any]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "git": _git_provenance(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Split inspection
# =============================================================================

def _expected_datasets(split_file: Path, split: int) -> list[str]:
    try:
        raw = json.loads(Path(split_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read split file {split_file}: {exc}") from exc
    if not isinstance(raw, list):
        raise CalibrationError(f"split file {split_file} must be a JSON list of fold objects")
    matches = [item for item in raw if isinstance(item, dict) and item.get("split") == split]
    if len(matches) != 1:
        raise CalibrationError(f"split file {split_file} must contain exactly one fold with split={split}")
    test = matches[0].get("test")
    if not isinstance(test, list) or not test or not all(isinstance(name, str) and name for name in test):
        raise CalibrationError(f"split file {split_file} fold {split} has an invalid or empty test list")
    if len(test) != len(set(test)):
        raise CalibrationError(f"split file {split_file} fold {split} test list has duplicates")
    return sorted(test)


# =============================================================================
# Config validation helpers
# =============================================================================

def _validate_finite_number(name: str, value: Any, *, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CalibrationError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise CalibrationError(f"{name} must be finite, got {value!r}")
    if lo is not None and value < lo:
        raise CalibrationError(f"{name} must be >= {lo}, got {value!r}")
    if hi is not None and value > hi:
        raise CalibrationError(f"{name} must be <= {hi}, got {value!r}")
    return value


def _validate_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise CalibrationError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_subset_matches_split(config: dict[str, Any]) -> None:
    subset_split_file = _resolve_repo_path(config["subset_split_file"])
    expected = _expected_datasets(subset_split_file, config["subset_split"])
    configured = config["subset_dataset_names"]
    if sorted(configured) != expected:
        raise CalibrationError(
            "config subset_dataset_names does not exactly match the committed subset "
            f"split file's test list: {sorted(configured)} != {expected}",
        )


# =============================================================================
# Config loading
# =============================================================================

def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the calibration config (thresholds live here, not in code)."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read calibration config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise CalibrationError("calibration config must be a JSON object")

    missing = sorted(_REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise CalibrationError(f"calibration config missing required keys: {missing}")

    for stage_key, required in _REQUIRED_STAGE_KEYS.items():
        stage_cfg = config[stage_key]
        if not isinstance(stage_cfg, dict):
            raise CalibrationError(f"calibration config {stage_key} must be an object")
        stage_missing = sorted(required - set(stage_cfg))
        if stage_missing:
            raise CalibrationError(f"calibration config {stage_key} missing required keys: {stage_missing}")

    _validate_finite_number("pool_kernel_um", config["pool_kernel_um"], lo=1e-9)
    _validate_finite_number("current_det_threshold", config["current_det_threshold"], lo=0.0, hi=1.0)
    _validate_finite_number("edge_threshold_default", config["edge_threshold_default"], lo=0.0, hi=1.0)

    if config["tracking"] != "greedy":
        raise CalibrationError(f"tracking must be 'greedy', got {config['tracking']!r}")

    for key in ("subset_split", "full_split"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CalibrationError(f"{key} must be a non-negative integer, got {value!r}")

    if not isinstance(config["subset_dataset_names"], list) or not config["subset_dataset_names"]:
        raise CalibrationError("subset_dataset_names must be a non-empty list")
    if len(config["subset_dataset_names"]) != len(set(config["subset_dataset_names"])):
        raise CalibrationError("subset_dataset_names contains duplicates")

    stage_a = config["stage_a"]
    if not isinstance(stage_a["det_thresholds"], list) or not stage_a["det_thresholds"]:
        raise CalibrationError("stage_a.det_thresholds must be a non-empty list")
    for value in stage_a["det_thresholds"]:
        _validate_finite_number("stage_a.det_thresholds[*]", value, lo=0.0, hi=1.0)
    if len(stage_a["det_thresholds"]) != len(set(stage_a["det_thresholds"])):
        raise CalibrationError(
            "stage_a.det_thresholds contains duplicates (duplicate thresholds would collide "
            "on the same trial_id and trial directory)",
        )
    if not isinstance(stage_a["edge_thresholds"], list) or not stage_a["edge_thresholds"]:
        raise CalibrationError("stage_a.edge_thresholds must be a non-empty list")
    for value in stage_a["edge_thresholds"]:
        _validate_finite_number("stage_a.edge_thresholds[*]", value, lo=0.0, hi=1.0)
    if len(stage_a["edge_thresholds"]) != len(set(stage_a["edge_thresholds"])):
        raise CalibrationError(
            "stage_a.edge_thresholds contains duplicates (duplicate thresholds would collide "
            "on the same trial_id and trial directory)",
        )
    n_best_det = _validate_positive_int("stage_a.n_best_det_thresholds", stage_a["n_best_det_thresholds"])
    unique_dets = len(set(stage_a["det_thresholds"]))
    if n_best_det > unique_dets:
        raise CalibrationError(
            f"stage_a.n_best_det_thresholds ({n_best_det}) exceeds the number of unique "
            f"det_thresholds available ({unique_dets})",
        )
    if not isinstance(stage_a["checkpoint_labels"], list) or not stage_a["checkpoint_labels"]:
        raise CalibrationError("stage_a.checkpoint_labels must be a non-empty list")
    if len(stage_a["checkpoint_labels"]) != len(set(stage_a["checkpoint_labels"])):
        raise CalibrationError("stage_a.checkpoint_labels contains duplicates")

    stage_b = config["stage_b"]
    if not isinstance(stage_b["edge_thresholds"], list) or not stage_b["edge_thresholds"]:
        raise CalibrationError("stage_b.edge_thresholds must be a non-empty list")
    for value in stage_b["edge_thresholds"]:
        _validate_finite_number("stage_b.edge_thresholds[*]", value, lo=0.0, hi=1.0)
    if len(stage_b["edge_thresholds"]) != len(set(stage_b["edge_thresholds"])):
        raise CalibrationError(
            "stage_b.edge_thresholds contains duplicates (duplicate thresholds would collide "
            "on the same trial_id and trial directory)",
        )
    n_best_pairs = _validate_positive_int("stage_b.n_best_pairs", stage_b["n_best_pairs"])
    max_pairs = n_best_det * len(set(stage_b["edge_thresholds"]))
    if n_best_pairs > max_pairs:
        raise CalibrationError(
            f"stage_b.n_best_pairs ({n_best_pairs}) exceeds the number of unique candidate "
            f"pairs available from the config ({max_pairs})",
        )
    if not isinstance(stage_b["checkpoint_labels"], list) or not stage_b["checkpoint_labels"]:
        raise CalibrationError("stage_b.checkpoint_labels must be a non-empty list")
    if len(stage_b["checkpoint_labels"]) != len(set(stage_b["checkpoint_labels"])):
        raise CalibrationError("stage_b.checkpoint_labels contains duplicates")

    stage_c = config["stage_c"]
    if not isinstance(stage_c["checkpoint_labels"], list) or not stage_c["checkpoint_labels"]:
        raise CalibrationError("stage_c.checkpoint_labels must be a non-empty list")
    if len(stage_c["checkpoint_labels"]) != len(set(stage_c["checkpoint_labels"])):
        raise CalibrationError("stage_c.checkpoint_labels contains duplicates")

    if stage_a["checkpoint_labels"] != _REVIEWED_STAGE_AB_CHECKPOINT_LABELS:
        raise CalibrationError(
            f"stage_a.checkpoint_labels {stage_a['checkpoint_labels']} does not match the "
            f"reviewed plan {_REVIEWED_STAGE_AB_CHECKPOINT_LABELS}",
        )
    if stage_b["checkpoint_labels"] != _REVIEWED_STAGE_AB_CHECKPOINT_LABELS:
        raise CalibrationError(
            f"stage_b.checkpoint_labels {stage_b['checkpoint_labels']} does not match the "
            f"reviewed plan {_REVIEWED_STAGE_AB_CHECKPOINT_LABELS}",
        )
    if stage_c["checkpoint_labels"] != _REVIEWED_STAGE_C_CHECKPOINT_LABELS:
        raise CalibrationError(
            f"stage_c.checkpoint_labels {stage_c['checkpoint_labels']} does not match the "
            f"reviewed plan {_REVIEWED_STAGE_C_CHECKPOINT_LABELS}",
        )

    _validate_subset_matches_split(config)

    return config


# =============================================================================
# Checkpoint labels
# =============================================================================

def parse_checkpoint_args(values: list[str] | None) -> dict[str, str]:
    """Parse repeated ``--checkpoint LABEL=PATH`` args into a label -> path map."""
    checkpoints: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise CalibrationError(f"--checkpoint must be LABEL=PATH, got {item!r}")
        label, _, path = item.partition("=")
        label = label.strip()
        if not label:
            raise CalibrationError(f"--checkpoint has an empty label: {item!r}")
        if not path:
            raise CalibrationError(f"--checkpoint has an empty path: {item!r}")
        if label in checkpoints:
            raise CalibrationError(f"duplicate --checkpoint label {label!r} (was {checkpoints[label]!r})")
        checkpoints[label] = path
    return checkpoints


def _resolve_checkpoint_path(checkpoints: dict[str, str], label: str) -> str:
    if label not in checkpoints:
        raise CalibrationError(
            f"no --checkpoint provided for label {label!r}; available labels: {sorted(checkpoints)}",
        )
    return str(checkpoints[label])


def _checkpoint_config_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / "config.json"


def _verify_checkpoint_files(checkpoint_path: Path) -> tuple[str, str]:
    """Verify the checkpoint and its adjacent config.json exist; return their sha256 hashes.

    Prediction reads ``config.json`` from the checkpoint's directory to
    reconstruct the model and resolve ``pool_kernel_um`` — inference cannot
    proceed correctly without it, so its presence and hash are load-bearing.
    """
    if not checkpoint_path.is_file():
        raise CalibrationError(f"checkpoint not found: {checkpoint_path}")
    config_path = _checkpoint_config_path(checkpoint_path)
    if not config_path.is_file():
        raise CalibrationError(
            f"checkpoint config.json not found next to checkpoint (required for inference): {config_path}",
        )
    return _sha256(checkpoint_path), _sha256(config_path)


# =============================================================================
# Trial construction (pure — no filesystem reads beyond string path resolution)
# =============================================================================

def _trial_id(stage: str, checkpoint_label: str, det_threshold: float, edge_threshold: float) -> str:
    return f"{stage}__ckpt-{checkpoint_label}__det-{det_threshold:.4f}__edge-{edge_threshold:.4f}"


def _make_trial(
    *,
    stage: str,
    checkpoint_label: str,
    checkpoint_path: str,
    det_threshold: float,
    edge_threshold: float,
    pool_kernel_um: float,
    tracking: str,
    split_file: str,
    split: int,
    is_screening: bool,
) -> dict[str, Any]:
    det_threshold = float(det_threshold)
    edge_threshold = float(edge_threshold)
    return {
        "trial_id": _trial_id(stage, checkpoint_label, det_threshold, edge_threshold),
        "stage": stage,
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(checkpoint_path),
        "det_threshold": det_threshold,
        "edge_threshold": edge_threshold,
        "pool_kernel_um": float(pool_kernel_um),
        "tracking": tracking,
        "split_file": str(_resolve_repo_path(split_file)),
        "split": int(split),
        "is_screening": bool(is_screening),
    }


def build_stage_a_trials(config: dict[str, Any], checkpoints: dict[str, str]) -> list[dict[str, Any]]:
    """Stage A: sweep det_threshold at the default edge_threshold, subset data, one checkpoint."""
    stage_cfg = config["stage_a"]
    trials = []
    for label in stage_cfg["checkpoint_labels"]:
        checkpoint_path = _resolve_checkpoint_path(checkpoints, label)
        for det_threshold in stage_cfg["det_thresholds"]:
            for edge_threshold in stage_cfg["edge_thresholds"]:
                trials.append(_make_trial(
                    stage="stage_a",
                    checkpoint_label=label,
                    checkpoint_path=checkpoint_path,
                    det_threshold=det_threshold,
                    edge_threshold=edge_threshold,
                    pool_kernel_um=config["pool_kernel_um"],
                    tracking=config["tracking"],
                    split_file=config["subset_split_file"],
                    split=config["subset_split"],
                    is_screening=True,
                ))
    return trials


def build_stage_b_trials(
    config: dict[str, Any],
    checkpoints: dict[str, str],
    best_det_thresholds: list[float],
) -> list[dict[str, Any]]:
    """Stage B: sweep edge_threshold at the best Stage-A det_thresholds, subset data."""
    stage_cfg = config["stage_b"]
    trials = []
    for label in stage_cfg["checkpoint_labels"]:
        checkpoint_path = _resolve_checkpoint_path(checkpoints, label)
        for det_threshold in best_det_thresholds:
            for edge_threshold in stage_cfg["edge_thresholds"]:
                trials.append(_make_trial(
                    stage="stage_b",
                    checkpoint_label=label,
                    checkpoint_path=checkpoint_path,
                    det_threshold=det_threshold,
                    edge_threshold=edge_threshold,
                    pool_kernel_um=config["pool_kernel_um"],
                    tracking=config["tracking"],
                    split_file=config["subset_split_file"],
                    split=config["subset_split"],
                    is_screening=True,
                ))
    return trials


def build_stage_c_trials(
    config: dict[str, Any],
    checkpoints: dict[str, str],
    best_pairs: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Stage C: confirm the best Stage-B (det, edge) pairs on every checkpoint, full fold."""
    stage_cfg = config["stage_c"]
    trials = []
    for label in stage_cfg["checkpoint_labels"]:
        checkpoint_path = _resolve_checkpoint_path(checkpoints, label)
        for det_threshold, edge_threshold in best_pairs:
            trials.append(_make_trial(
                stage="stage_c",
                checkpoint_label=label,
                checkpoint_path=checkpoint_path,
                det_threshold=det_threshold,
                edge_threshold=edge_threshold,
                pool_kernel_um=config["pool_kernel_um"],
                tracking=config["tracking"],
                split_file=config["full_split_file"],
                split=config["full_split"],
                is_screening=False,
            ))
    return trials


# =============================================================================
# Official-score selection with deterministic, documented tie-breaking
# =============================================================================

def _finite_score(record: dict[str, Any]) -> float | None:
    summary = record.get("summary_metrics")
    if not isinstance(summary, dict):
        return None
    score = summary.get("score")
    if isinstance(score, bool) or not isinstance(score, Real):
        return None
    score = float(score)
    return score if math.isfinite(score) else None


def _require_complete_successful_grid(trial_records: list[dict[str, Any]], stage_label: str) -> None:
    """Require every planned trial in ``trial_records`` to have succeeded with a
    finite official score. A stage never selects from a partial grid: failed
    trials stay recorded (so rerunning the stage resumes the successes and
    retries only the failures), but selection itself refuses to run until the
    full planned grid is successful.
    """
    if not trial_records:
        raise CalibrationError(f"{stage_label} selection requires a non-empty planned trial grid, found 0")
    incomplete = sorted(
        str(r.get("trial_id", "<unknown>")) for r in trial_records
        if r.get("status") != "success" or _finite_score(r) is None
    )
    if incomplete:
        raise CalibrationError(
            f"{stage_label} selection requires all {len(trial_records)} planned trial(s) to have "
            f"succeeded with a finite official score; {len(incomplete)} did not (rerun the stage to "
            f"resume successes and retry failures): {incomplete}",
        )


def select_best_det_thresholds(trial_records: list[dict[str, Any]], n: int) -> list[float]:
    """Rank Stage-A trials by score desc; ties broken by ascending det_threshold.

    Raises unless every planned trial succeeded with a finite score (a
    partial grid is never selected from), and unless at least ``n`` unique
    det_threshold candidates result.
    """
    n = _validate_positive_int("n", n)
    _require_complete_successful_grid(trial_records, "stage A")
    best_by_threshold: dict[float, tuple[float, dict[str, Any]]] = {}
    for record in trial_records:
        score = _finite_score(record)
        key = record["det_threshold"]
        if key not in best_by_threshold or score > best_by_threshold[key][0]:
            best_by_threshold[key] = (score, record)

    ranked = sorted(best_by_threshold.values(), key=lambda pair: (-pair[0], pair[1]["det_threshold"]))
    if len(ranked) < n:
        raise CalibrationError(
            f"stage A selection requires {n} successful unique det_threshold candidate(s) from a "
            f"complete grid, found {len(ranked)}",
        )
    return [record["det_threshold"] for _, record in ranked[:n]]


def select_best_pairs(trial_records: list[dict[str, Any]], n: int) -> list[tuple[float, float]]:
    """Rank Stage-B trials by score desc; ties broken by ascending edge_threshold, then det_threshold.

    Raises unless every planned trial succeeded with a finite score (a
    partial grid is never selected from), and unless at least ``n`` unique
    (det, edge) candidates result.
    """
    n = _validate_positive_int("n", n)
    _require_complete_successful_grid(trial_records, "stage B")
    best_by_pair: dict[tuple[float, float], tuple[float, dict[str, Any]]] = {}
    for record in trial_records:
        score = _finite_score(record)
        key = (record["det_threshold"], record["edge_threshold"])
        if key not in best_by_pair or score > best_by_pair[key][0]:
            best_by_pair[key] = (score, record)

    ranked = sorted(
        best_by_pair.values(),
        key=lambda pair: (-pair[0], pair[1]["edge_threshold"], pair[1]["det_threshold"]),
    )
    if len(ranked) < n:
        raise CalibrationError(
            f"stage B selection requires {n} successful unique (det,edge) candidate(s) from a "
            f"complete grid, found {len(ranked)}",
        )
    return [(record["det_threshold"], record["edge_threshold"]) for _, record in ranked[:n]]


def select_final_best(
    trial_records: list[dict[str, Any]],
    checkpoint_label_order: list[str],
) -> dict[str, Any]:
    """Rank Stage-C trials by score desc; ties broken by ascending edge_threshold, then
    det_threshold, then by position of checkpoint_label in ``checkpoint_label_order``
    (earlier-listed checkpoints win ties).

    Raises unless every planned Stage-C trial succeeded with a finite score —
    Stage C never selects a winner from a partial grid.
    """
    _require_complete_successful_grid(trial_records, "stage C")
    rank_of = {label: index for index, label in enumerate(checkpoint_label_order)}
    ranked = sorted(
        trial_records,
        key=lambda record: (
            -_finite_score(record),
            record["edge_threshold"],
            record["det_threshold"],
            rank_of.get(record["checkpoint_label"], len(rank_of)),
        ),
    )
    return ranked[0]


def _selected_final_trial_summary(trial_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": trial_record["trial_id"],
        "checkpoint_label": trial_record["checkpoint_label"],
        "checkpoint_path": trial_record["checkpoint_path"],
        "checkpoint_sha256": trial_record["checkpoint_sha256"],
        "det_threshold": trial_record["det_threshold"],
        "edge_threshold": trial_record["edge_threshold"],
        "official_score": trial_record["summary_metrics"]["score"],
        "summary_metrics": trial_record["summary_metrics"],
    }


# =============================================================================
# Previous-stage result validation (strict — rejects anything but a complete,
# successful, same-config result file from the exact expected stage)
# =============================================================================

def _normalize_selection(selection_key: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{selection_key} must be a list")
    if selection_key == "selected_pairs":
        normalized = []
        for pair in value:
            if not (isinstance(pair, list) and len(pair) == 2):
                raise ValueError("malformed pair entry")
            normalized.append((float(pair[0]), float(pair[1])))
        return normalized
    return [float(x) for x in value]


def _validate_trial_records_shape(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise CalibrationError("previous-results 'trials' must be a non-empty list")
    ids: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise CalibrationError("previous-results contains a malformed trial record")
        trial_id = record.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise CalibrationError("previous-results trial record is missing a valid trial_id")
        if record.get("status") not in ("success", "failed"):
            raise CalibrationError(f"previous-results trial {trial_id} has an invalid status: {record.get('status')!r}")
        ids.append(trial_id)
    if len(ids) != len(set(ids)):
        raise CalibrationError("previous-results has duplicate trial_id values")
    return records


def _load_and_check_report_envelope(
    path: Path, *, expected_stage_key: str, config: dict[str, Any], config_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Read, parse, and check the report-level invariants (schema, stage, config hash).

    Returns the parsed report, its (shape-validated but not yet grid-checked)
    trial records, and the config's current sha256 for reuse by the caller.
    """
    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationError(f"cannot read previous-results file {path}: {exc}") from exc
    try:
        report = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"previous-results file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise CalibrationError(f"previous-results file must be a JSON object: {path}")

    if report.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationError(
            f"previous-results schema_version {report.get('schema_version')!r} != {SCHEMA_VERSION}",
        )
    if report.get("stage") != expected_stage_key:
        raise CalibrationError(
            f"previous-results is from stage {report.get('stage')!r}, expected {expected_stage_key!r}",
        )

    expected_config_sha256 = _require_sha256(config_path)
    if report.get("config_sha256") != expected_config_sha256:
        raise CalibrationError(
            "previous-results was produced from a different calibration config than the one in "
            f"use now (sha256 {report.get('config_sha256')!r} != {expected_config_sha256!r}); "
            "rerun the prior stage with the current config",
        )

    records = _validate_trial_records_shape(report.get("trials"))
    return report, records, expected_config_sha256


_GRID_MATCH_FIELDS = (
    "checkpoint_label", "checkpoint_path", "det_threshold", "edge_threshold",
    "pool_kernel_um", "tracking", "split_file", "split", "is_screening",
)


def _validate_report_matches_expected_grid(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    expected_trials: list[dict[str, Any]],
    *,
    expected_stage_key: str,
    require_screening: bool,
    config_sha256: str,
    data_dir: Path,
) -> None:
    """Require ``records`` to be exactly ``expected_trials``: the same trial IDs,
    the same parameters per trial, every trial successful with a finite score,
    and every successful trial's on-disk artifacts (checkpoint, checkpoint
    config, split file, predictions, and metrics) independently revalidated —
    never merely trusted from the report file. Rejects missing, extra,
    duplicated, or substituted trials, and any altered checkpoint label,
    threshold, split identity, screening status, pool kernel, or tracking
    value.
    """
    resolved_data_dir = str(Path(data_dir).resolve())
    if report.get("data_dir") != resolved_data_dir:
        raise CalibrationError(
            f"previous-results data_dir {report.get('data_dir')!r} does not match the current "
            f"resolved data_dir {resolved_data_dir!r}",
        )

    records_by_id = {r["trial_id"]: r for r in records}
    expected_ids = {t["trial_id"] for t in expected_trials}
    actual_ids = set(records_by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise CalibrationError(
            f"previous-results trial set does not exactly match the expected {expected_stage_key} "
            f"grid generated from the current config and checkpoints: missing={missing}, extra={extra}",
        )

    for expected_trial in expected_trials:
        record = records_by_id[expected_trial["trial_id"]]
        if record.get("stage") != expected_stage_key:
            raise CalibrationError(
                f"previous-results trial {expected_trial['trial_id']} has stage {record.get('stage')!r}, "
                f"expected {expected_stage_key!r}",
            )
        for field in _GRID_MATCH_FIELDS:
            if record.get(field) != expected_trial.get(field):
                raise CalibrationError(
                    f"previous-results trial {expected_trial['trial_id']} field {field!r} does not "
                    f"match the expected grid: {record.get(field)!r} != {expected_trial.get(field)!r}",
                )
        if record.get("status") != "success":
            raise CalibrationError(
                f"previous-results trial {expected_trial['trial_id']} did not succeed "
                f"(status={record.get('status')!r}); every trial in the previous stage must succeed "
                "before the next stage may begin",
            )
        if _finite_score(record) is None:
            raise CalibrationError(
                f"previous-results trial {expected_trial['trial_id']} has a non-finite official score",
            )
        if require_screening and record.get("is_screening") is not True:
            raise CalibrationError(
                f"previous-results trial {expected_trial['trial_id']} is not marked as a screening run "
                "(is_screening must be True for Stage A/B)",
            )
        if not _resume_is_valid(record, expected_trial, config_sha256=config_sha256, data_dir=data_dir):
            raise CalibrationError(
                f"previous-results trial {expected_trial['trial_id']} failed on-disk revalidation "
                "(checkpoint, checkpoint config, split file, predictions, or metrics on disk no "
                "longer match what is recorded — rerun the prior stage)",
            )


def _validate_and_return_selection(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    selection_key: str,
    expected_selection_count: int,
    selector: Callable[[list[dict[str, Any]], int], list[Any]],
) -> list[Any]:
    recomputed_selection = selector(records, expected_selection_count)
    try:
        normalized_stored = _normalize_selection(selection_key, report.get(selection_key))
    except (ValueError, TypeError) as exc:
        raise CalibrationError(f"previous-results {selection_key} is missing or malformed: {exc}") from exc
    if len(normalized_stored) != expected_selection_count:
        raise CalibrationError(
            f"previous-results {selection_key} must contain exactly {expected_selection_count} entries, "
            f"found {len(normalized_stored)}",
        )
    normalized_recomputed = (
        [tuple(pair) for pair in recomputed_selection] if selection_key == "selected_pairs"
        else [float(x) for x in recomputed_selection]
    )
    if normalized_stored != normalized_recomputed:
        raise CalibrationError(
            f"previous-results {selection_key} does not match a deterministic recomputation from "
            "its own trial records (possible tampering or corruption)",
        )
    return recomputed_selection


def _extract_expected_det_thresholds_from_stage_b_records(
    records: list[dict[str, Any]], config: dict[str, Any],
) -> list[float]:
    """Recover the det_threshold values Stage B's grid was built from.

    Stage C only receives Stage B's results file, not Stage A's, so the exact
    Stage-B grid is reconstructed from the det_threshold values actually
    present in Stage B's own trial records — constrained to be exactly
    ``stage_a.n_best_det_thresholds`` distinct values, each one drawn from
    the reviewed ``stage_a.det_thresholds`` grid, so a fabricated or
    out-of-grid value can never slip through.
    """
    n_best_det = config["stage_a"]["n_best_det_thresholds"]
    valid_det_thresholds = set(config["stage_a"]["det_thresholds"])
    used = sorted({
        float(r["det_threshold"]) for r in records
        if isinstance(r.get("det_threshold"), Real) and not isinstance(r.get("det_threshold"), bool)
    })
    if len(used) != n_best_det:
        raise CalibrationError(
            f"previous-results stage_b trials use {len(used)} distinct det_threshold value(s), "
            f"expected exactly {n_best_det} (stage_a.n_best_det_thresholds)",
        )
    invalid = sorted(v for v in used if v not in valid_det_thresholds)
    if invalid:
        raise CalibrationError(
            "previous-results stage_b trials use det_threshold value(s) outside the reviewed "
            f"stage_a.det_thresholds grid: {invalid}",
        )
    return used


def load_stage_a_report_for_stage_b(
    path: Path,
    config: dict[str, Any],
    config_path: Path,
    checkpoints: dict[str, str],
    data_dir: Path,
) -> tuple[dict[str, Any], list[float]]:
    """Strictly validate a Stage-A results file before Stage B may consume it.

    Generates the exact expected Stage-A trial grid from the current config
    and checkpoints, requires the report's trials to match it exactly, and
    revalidates every successful trial's on-disk artifacts.
    """
    report, records, config_sha256 = _load_and_check_report_envelope(
        path, expected_stage_key="stage_a", config=config, config_path=config_path,
    )
    expected_trials = build_stage_a_trials(config, checkpoints)
    _validate_report_matches_expected_grid(
        report, records, expected_trials,
        expected_stage_key="stage_a", require_screening=True,
        config_sha256=config_sha256, data_dir=data_dir,
    )
    n = config["stage_a"]["n_best_det_thresholds"]
    selection = _validate_and_return_selection(
        report, records,
        selection_key="selected_det_thresholds", expected_selection_count=n,
        selector=select_best_det_thresholds,
    )
    return report, selection


def load_stage_b_report_for_stage_c(
    path: Path,
    config: dict[str, Any],
    config_path: Path,
    checkpoints: dict[str, str],
    data_dir: Path,
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    """Strictly validate a Stage-B results file before Stage C may consume it.

    Generates the exact expected Stage-B trial grid from the current config,
    checkpoints, and the det_threshold values actually used by the report
    (constrained to the reviewed Stage-A grid), requires the report's trials
    to match it exactly, and revalidates every successful trial's on-disk
    artifacts.
    """
    report, records, config_sha256 = _load_and_check_report_envelope(
        path, expected_stage_key="stage_b", config=config, config_path=config_path,
    )
    best_det_thresholds = _extract_expected_det_thresholds_from_stage_b_records(records, config)
    expected_trials = build_stage_b_trials(config, checkpoints, best_det_thresholds)
    _validate_report_matches_expected_grid(
        report, records, expected_trials,
        expected_stage_key="stage_b", require_screening=True,
        config_sha256=config_sha256, data_dir=data_dir,
    )
    n = config["stage_b"]["n_best_pairs"]
    selection = _validate_and_return_selection(
        report, records,
        selection_key="selected_pairs", expected_selection_count=n,
        selector=select_best_pairs,
    )
    return report, selection


# =============================================================================
# Prediction scratch directory (unique per trial, cleaned up after copying)
# =============================================================================

def _prediction_scratch_dir(repo_root: Path, username: str, method: str, split: int) -> Path:
    return repo_root / "predictions" / username / method / f"split_{split}"


def _remove_prediction_scratch_dir(
    path: Path, repo_root: Path, username: str, method: str, split: int,
) -> None:
    """Remove exactly the one scratch prediction directory this trial created.

    Guarded by an exact-path check so a bug elsewhere can never turn this into
    a broader delete — this is the only deletion the runner ever performs
    outside a trial's own directory, and it only ever targets a directory
    whose name is unique to this trial (via the trial-scoped ``method``).
    """
    expected = _prediction_scratch_dir(repo_root, username, method, split)
    if path.resolve() != expected.resolve():
        raise CalibrationError(f"refusing to remove unexpected prediction scratch directory: {path}")
    if path.exists():
        if not path.is_dir():
            raise CalibrationError(f"prediction scratch path is not a directory: {path}")
        shutil.rmtree(path)


def _remove_trial_dir(trial_dir: Path, output_dir: Path, trial_id: str) -> None:
    """Remove exactly one trial's own output directory before it is (re-)executed.

    Guarded the same way as the scratch-directory removal: the path must
    match the expected ``<output_dir>/trials/<trial_id>`` exactly, so a
    rerun of one trial can never touch another trial's completed results.
    """
    trials_root = (output_dir / "trials").resolve()
    expected = trials_root / trial_id
    resolved = trial_dir.resolve()
    if resolved != expected or resolved.parent != trials_root:
        raise CalibrationError(f"refusing to remove unexpected trial directory: {trial_dir}")
    if resolved.exists():
        if not resolved.is_dir():
            raise CalibrationError(f"trial path is not a directory: {resolved}")
        shutil.rmtree(resolved)


# =============================================================================
# Validation of prediction coverage and strict evaluation reports
# =============================================================================

def _validate_predictions(prediction_dir: Path, expected: list[str]) -> dict[str, Any]:
    """Require exactly one nonempty GEFF per expected dataset — no missing, extra, or empty."""
    if not prediction_dir.exists():
        raise CalibrationError(f"prediction output directory missing: {prediction_dir}")
    paths = sorted(prediction_dir.glob("*.geff"))
    actual = {p.stem for p in paths}
    expected_set = set(expected)
    missing = sorted(expected_set - actual)
    extra = sorted(actual - expected_set)
    empty = sorted(p.stem for p in paths if not _nonempty_path(p))
    if missing or extra or empty:
        raise CalibrationError(
            f"prediction coverage failure for {prediction_dir}: missing={missing}, extra={extra}, empty={empty}",
        )
    return {"datasets": sorted(actual), "missing": missing, "extra": extra, "empty": empty}


def _hash_geff_entry(path: Path) -> dict[str, Any]:
    """Deterministic content manifest entry for one GEFF, whether a file or a directory.

    A directory GEFF is hashed by walking every member file in sorted
    relative-path order and folding each member's relative path and content
    hash into one digest, so a change to any member's bytes — even with the
    same file count and total size — changes the result.
    """
    if path.is_dir():
        members = sorted(p for p in path.rglob("*") if p.is_file())
        digest = hashlib.sha256()
        total_bytes = 0
        for member in members:
            rel = member.relative_to(path).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(_sha256(member).encode("ascii"))
            digest.update(b"\n")
            total_bytes += member.stat().st_size
        return {
            "kind": "dir", "member_count": len(members),
            "total_bytes": total_bytes, "sha256": digest.hexdigest(),
        }
    return {
        "kind": "file", "member_count": 1,
        "total_bytes": path.stat().st_size, "sha256": _sha256(path),
    }


def _prediction_manifest(prediction_dir: Path, expected: list[str]) -> dict[str, dict[str, Any]]:
    """Deterministic per-dataset content manifest for a prediction directory.

    Covers both file GEFFs and directory GEFFs, recording member counts,
    total bytes, and sha256 per dataset. Resume recomputes this manifest from
    what's on disk and requires it to equal what was recorded, so a GEFF that
    stays present and nonempty but whose bytes have changed is still rejected.
    """
    return {name: _hash_geff_entry(prediction_dir / f"{name}.geff") for name in sorted(expected)}


_REQUIRED_PER_DATASET_METRICS = {
    "edge_tp", "edge_fp", "edge_fn", "num_pred_nodes", "total_node_ratio",
    "node_recall", "edge_jaccard", "adj_edge_jaccard",
}


def _validate_strict_metrics(metrics_path: Path, expected: list[str]) -> dict[str, Any]:
    """Validate one strict ``evaluate.py --json-out`` report: full, unskipped coverage."""
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read evaluation metrics {metrics_path}: {exc}") from exc
    if not isinstance(metrics, dict):
        raise CalibrationError(f"evaluation metrics are not an object: {metrics_path}")

    evaluated = metrics.get("evaluated_datasets")
    if (
        not isinstance(evaluated, list)
        or len(evaluated) != len(expected)
        or set(evaluated) != set(expected)
    ):
        raise CalibrationError(f"evaluation datasets differ from expected: {evaluated} != {expected}")
    skipped = metrics.get("skipped_datasets")
    if not isinstance(skipped, list) or skipped != []:
        raise CalibrationError(f"evaluation skipped dataset(s): {skipped}")

    summary = metrics.get("summary_metrics")
    if not isinstance(summary, dict):
        raise CalibrationError(f"evaluation summary_metrics missing or malformed: {metrics_path}")
    score = summary.get("score")
    if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)):
        raise CalibrationError(f"evaluation score is not finite: {score!r}")

    per_dataset = metrics.get("per_dataset_metrics")
    if not isinstance(per_dataset, list) or len(per_dataset) != len(expected):
        raise CalibrationError(f"evaluation per_dataset_metrics missing or incomplete: {metrics_path}")
    names = []
    for row in per_dataset:
        if not isinstance(row, dict) or not isinstance(row.get("dataset"), str):
            raise CalibrationError(f"evaluation has a malformed per-dataset metrics row: {metrics_path}")
        row_metrics = row.get("metrics")
        if not isinstance(row_metrics, dict) or not _REQUIRED_PER_DATASET_METRICS <= set(row_metrics):
            raise CalibrationError(f"evaluation per-dataset metrics incomplete for {row.get('dataset')!r}")
        names.append(row["dataset"])
    if set(names) != set(expected):
        raise CalibrationError(f"evaluation per-dataset metrics differ from expected: {names} != {expected}")

    return metrics


def _metrics_match_record(metrics_path: Path, prior: dict[str, Any], expected: list[str]) -> bool:
    """Re-read metrics currently on disk and require they equal the values in the trial record.

    A resumed trial must not silently accept a results file whose metrics
    were edited (or regenerated differently) since the record was written.
    """
    try:
        metrics = _validate_strict_metrics(metrics_path, expected)
    except CalibrationError:
        return False
    return (
        metrics["summary_metrics"] == prior.get("summary_metrics")
        and metrics["per_dataset_metrics"] == prior.get("per_dataset_metrics")
    )


# =============================================================================
# Command construction (prediction + strict evaluation only — never training)
# =============================================================================

def build_prediction_command(trial: dict[str, Any], data_dir: Path, method: str) -> list[str]:
    return [
        str(sys.executable), str(PREDICT_SCRIPT),
        "--method", method,
        "--data-dir", str(data_dir),
        "--splits", str(trial["split_file"]),
        "--split", str(trial["split"]),
        "--weights", str(trial["checkpoint_path"]),
        "--det-threshold", str(trial["det_threshold"]),
        "--edge-threshold", str(trial["edge_threshold"]),
        "--pool-kernel-um", str(trial["pool_kernel_um"]),
        "--tracking", str(trial["tracking"]),
    ]


def build_evaluation_command(prediction_dir: Path, data_dir: Path, metrics_path: Path) -> list[str]:
    return [
        str(sys.executable), str(EVALUATE_SCRIPT),
        "--pred-dir", str(prediction_dir),
        "--gt-dir", str(data_dir),
        "--strict", "--json-out", str(metrics_path),
    ]


# =============================================================================
# Progress visibility — immediate start message + periodic heartbeat
# =============================================================================

def _format_progress_prefix(trial: dict[str, Any], trial_index: int, total_trials: int) -> str:
    return (
        f"[stage={trial['stage']} trial {trial_index}/{total_trials} "
        f"checkpoint={trial['checkpoint_label']} det={trial['det_threshold']:.4f} "
        f"edge={trial['edge_threshold']:.4f}]"
    )


def _print_trial_start(trial: dict[str, Any], trial_index: int, total_trials: int) -> None:
    print(f"{_format_progress_prefix(trial, trial_index, total_trials)} starting", flush=True)


def _print_heartbeat(
    trial: dict[str, Any], trial_index: int, total_trials: int, phase: str, elapsed_seconds: float,
) -> None:
    print(
        f"{_format_progress_prefix(trial, trial_index, total_trials)} "
        f"phase={phase} elapsed={elapsed_seconds:.0f}s",
        flush=True,
    )


def _command_references_training_script(command: list[str]) -> bool:
    """True if any argument in ``command`` resolves to the training script.

    Catches every equivalent way the training script could be spelled: an
    absolute path, a path relative to the repository root, Windows or POSIX
    separators, redundant ``.``/``..`` segments, or any other path string
    that resolves to the same file on disk — not just a byte-for-byte match
    against ``str(TRAIN_SCRIPT)``.
    """
    train_script_resolved = TRAIN_SCRIPT.resolve()
    for arg in command:
        if not isinstance(arg, str) or not arg:
            continue
        candidate = Path(arg)
        candidate = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
        try:
            resolved = candidate.resolve()
        except (OSError, ValueError):
            continue
        if resolved == train_script_resolved:
            return True
    return False


def _stream_subprocess(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    on_heartbeat: Callable[[float], None] = lambda elapsed: None,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Run ``command``, streaming stdout/stderr to separate log files live.

    Output is never buffered silently: each line is flushed to its log file
    as it arrives, and ``on_heartbeat`` fires at least once per
    ``heartbeat_interval`` seconds while the process is still running.
    """
    if _command_references_training_script(command):
        raise CalibrationError("refusing to invoke the training script from the calibration runner")

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    def _pump(stream, destination: Path) -> None:
        with destination.open("w", encoding="utf-8") as handle:
            for line in iter(stream.readline, ""):
                handle.write(line)
                handle.flush()
        stream.close()

    threads = [
        threading.Thread(target=_pump, args=(process.stdout, stdout_path), daemon=True),
        threading.Thread(target=_pump, args=(process.stderr, stderr_path), daemon=True),
    ]
    for thread in threads:
        thread.start()

    last_heartbeat = started
    poll_interval = min(0.5, heartbeat_interval) if heartbeat_interval > 0 else 0.5
    while process.poll() is None:
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_interval:
            on_heartbeat(now - started)
            last_heartbeat = now
        time.sleep(poll_interval)

    for thread in threads:
        thread.join(timeout=5)

    elapsed = time.monotonic() - started
    return {"returncode": process.returncode, "wall_seconds": elapsed}


# =============================================================================
# Trial execution
# =============================================================================

def _username() -> str:
    return os.environ.get("USER", os.environ.get("USERNAME", "unknown"))


def execute_trial(
    trial: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    *,
    config_sha256: str,
    trial_index: int = 1,
    total_trials: int = 1,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Run one trial's prediction + strict evaluation, isolated to its own directory.

    Any stale directory from a previous attempt at this exact trial_id (its
    own output directory and its own deterministic scratch prediction
    directory) is removed before anything runs, so a rerun can never merge
    new predictions on top of stale ones.
    """
    _print_trial_start(trial, trial_index, total_trials)

    trial_dir = output_dir / "trials" / trial["trial_id"]
    _remove_trial_dir(trial_dir, output_dir, trial["trial_id"])
    trial_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(trial["checkpoint_path"])
    checkpoint_sha256, checkpoint_config_sha256 = _verify_checkpoint_files(checkpoint_path)

    split_file_path = Path(trial["split_file"])
    split_file_sha256 = _safe_sha256(str(split_file_path))
    if split_file_sha256 is None:
        raise CalibrationError(f"split file not found for trial {trial['trial_id']}: {split_file_path}")
    expected_datasets = _expected_datasets(split_file_path, trial["split"])

    resolved_data_dir = Path(data_dir).resolve()

    method = f"calib_{trial['trial_id']}"
    username = _username()
    scratch_dir = _prediction_scratch_dir(REPO_ROOT, username, method, trial["split"])
    trial_predictions_dir = trial_dir / "predictions"
    metrics_path = trial_dir / "metrics.json"

    prediction_command = build_prediction_command(trial, resolved_data_dir, method)
    evaluation_command = build_evaluation_command(trial_predictions_dir, resolved_data_dir, metrics_path)

    # Clean any stale scratch directory left behind by an interrupted prior
    # attempt at this exact trial before running prediction.
    _remove_prediction_scratch_dir(scratch_dir, REPO_ROOT, username, method, trial["split"])

    def _heartbeat(phase: str) -> Callable[[float], None]:
        def _callback(elapsed: float) -> None:
            _print_heartbeat(trial, trial_index, total_trials, phase, elapsed)
        return _callback

    try:
        pred_result = _stream_subprocess(
            prediction_command, trial_dir / "prediction.stdout.log", trial_dir / "prediction.stderr.log",
            on_heartbeat=_heartbeat("predicting"), heartbeat_interval=heartbeat_interval,
        )
        if pred_result["returncode"] != 0:
            raise CalibrationError(
                f"prediction failed for trial {trial['trial_id']} "
                f"(see {trial_dir / 'prediction.stderr.log'})",
            )
        prediction_validation = _validate_predictions(scratch_dir, expected_datasets)
        _copy_path(scratch_dir, trial_predictions_dir)
    finally:
        # Cleaned up after success or any failure above (prediction,
        # coverage validation, or copy) so a stale scratch directory can
        # never be merged into a later attempt.
        _remove_prediction_scratch_dir(scratch_dir, REPO_ROOT, username, method, trial["split"])

    eval_result = _stream_subprocess(
        evaluation_command, trial_dir / "evaluation.stdout.log", trial_dir / "evaluation.stderr.log",
        on_heartbeat=_heartbeat("evaluating"), heartbeat_interval=heartbeat_interval,
    )
    if eval_result["returncode"] != 0:
        raise CalibrationError(
            f"evaluation failed for trial {trial['trial_id']} "
            f"(see {trial_dir / 'evaluation.stderr.log'})",
        )

    metrics = _validate_strict_metrics(metrics_path, expected_datasets)
    prediction_manifest = _prediction_manifest(trial_predictions_dir, expected_datasets)

    return {
        **trial,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_config_sha256": checkpoint_config_sha256,
        "split_file_sha256": split_file_sha256,
        "config_sha256": config_sha256,
        "data_dir": str(resolved_data_dir),
        "expected_datasets": expected_datasets,
        "status": "success",
        "error": None,
        "commands": {"prediction": prediction_command, "evaluation": evaluation_command},
        "prediction_dir": str(trial_predictions_dir),
        "metrics_path": str(metrics_path),
        "prediction_validation": prediction_validation,
        "prediction_manifest": prediction_manifest,
        "prediction_wall_seconds": pred_result["wall_seconds"],
        "evaluation_wall_seconds": eval_result["wall_seconds"],
        "summary_metrics": metrics["summary_metrics"],
        "per_dataset_metrics": metrics["per_dataset_metrics"],
        "resumed": False,
    }


# =============================================================================
# Resume validation
# =============================================================================

_RESUME_MATCH_FIELDS = (
    "stage", "checkpoint_label", "checkpoint_path", "det_threshold", "edge_threshold",
    "pool_kernel_um", "tracking", "split_file", "split",
)


def _resume_is_valid(
    prior: dict[str, Any] | None,
    trial: dict[str, Any],
    *,
    config_sha256: str,
    data_dir: Path,
) -> bool:
    """A prior trial record may be reused only if every invariant still holds.

    Every field that could scientifically invalidate a cached result is
    checked against what is on disk *right now* — the config hash, the
    split file's current content (recomputed, not merely trusted from the
    prior record), the checkpoint and its adjacent config.json, the data
    directory identity, all inference parameters, prediction coverage, and
    the strict metrics on disk matching the record byte-for-byte.
    """
    if not isinstance(prior, dict) or prior.get("status") != "success":
        return False
    for field in _RESUME_MATCH_FIELDS:
        if prior.get(field) != trial.get(field):
            return False

    if prior.get("config_sha256") != config_sha256:
        return False

    resolved_data_dir = str(Path(data_dir).resolve())
    if prior.get("data_dir") != resolved_data_dir:
        return False

    checkpoint_path = Path(trial["checkpoint_path"])
    if not checkpoint_path.is_file():
        return False
    if _sha256(checkpoint_path) != prior.get("checkpoint_sha256"):
        return False

    checkpoint_config_path = _checkpoint_config_path(checkpoint_path)
    if not checkpoint_config_path.is_file():
        return False
    if _sha256(checkpoint_config_path) != prior.get("checkpoint_config_sha256"):
        return False

    split_file_path = Path(trial["split_file"])
    if not split_file_path.is_file():
        return False
    if _sha256(split_file_path) != prior.get("split_file_sha256"):
        return False

    try:
        expected = _expected_datasets(split_file_path, trial["split"])
    except CalibrationError:
        return False
    if expected != prior.get("expected_datasets"):
        return False

    prediction_dir = prior.get("prediction_dir")
    if not prediction_dir:
        return False
    try:
        _validate_predictions(Path(prediction_dir), expected)
    except CalibrationError:
        return False

    prior_manifest = prior.get("prediction_manifest")
    if not isinstance(prior_manifest, dict):
        return False
    if _prediction_manifest(Path(prediction_dir), expected) != prior_manifest:
        return False

    metrics_path = prior.get("metrics_path")
    if not metrics_path or not Path(metrics_path).is_file():
        return False
    if not _metrics_match_record(Path(metrics_path), prior, expected):
        return False

    return True


# =============================================================================
# Runner
# =============================================================================

def run_trials(
    trials: list[dict[str, Any]],
    data_dir: Path,
    output_dir: Path,
    config_path: Path,
    stage_key: str,
    *,
    resume: bool = True,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
    build_selection: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute every trial, writing incrementally so completed work survives interruption.

    After all trials finish, ``build_selection`` (if given) computes the
    stage's deterministic selection from the final records and it is
    persisted at the top level of the report. If it raises (too few
    successful trials to select from), the error propagates *after* the
    per-trial results are still written to disk — the run fails loudly
    instead of silently omitting or fabricating a selection.
    """
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "calibration_results.json"
    config_sha256 = _require_sha256(Path(config_path))

    existing_by_id: dict[str, dict[str, Any]] = {}
    if resume and results_path.is_file():
        try:
            existing = json.loads(results_path.read_text(encoding="utf-8"))
            for record in existing.get("trials", []):
                if isinstance(record, dict) and "trial_id" in record:
                    existing_by_id[record["trial_id"]] = record
        except (OSError, json.JSONDecodeError):
            existing_by_id = {}

    def _report(records: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        base = {
            "schema_version": SCHEMA_VERSION,
            "stage": stage_key,
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "selection_metric": "summary_metrics.score",
            "environment": _environment_provenance(),
            "trials": records,
        }
        if extra:
            base.update(extra)
        return base

    total = len(trials)
    records: list[dict[str, Any]] = []
    for index, trial in enumerate(trials, start=1):
        prior = existing_by_id.get(trial["trial_id"])
        if resume and _resume_is_valid(prior, trial, config_sha256=config_sha256, data_dir=data_dir):
            record = dict(prior)
            record["resumed"] = True
            print(
                f"[stage={stage_key} trial {index}/{total} "
                f"checkpoint={trial['checkpoint_label']} det={trial['det_threshold']:.4f} "
                f"edge={trial['edge_threshold']:.4f}] resumed from prior results",
                flush=True,
            )
            records.append(record)
        else:
            try:
                record = execute_trial(
                    trial, data_dir, output_dir,
                    config_sha256=config_sha256, trial_index=index, total_trials=total,
                    heartbeat_interval=heartbeat_interval,
                )
            except Exception as exc:  # noqa: BLE001 - failures are data, not fatal to the run
                record = {
                    **trial,
                    "checkpoint_sha256": _safe_sha256(trial.get("checkpoint_path")),
                    "config_sha256": config_sha256,
                    "data_dir": str(data_dir),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "resumed": False,
                }
            records.append(record)
        _write_json_atomic(results_path, _report(records))

    selection_error: CalibrationError | None = None
    extra: dict[str, Any] = {}
    if build_selection is not None:
        try:
            extra = build_selection(records)
        except CalibrationError as exc:
            selection_error = exc

    _write_json_atomic(results_path, _report(records, extra))

    if selection_error is not None:
        raise selection_error

    return {"results_path": str(results_path), "trials": records, **extra}


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one stage of the inference-threshold calibration workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to the reviewed calibration config JSON.")
    parser.add_argument("--stage", choices=("A", "B", "C"), required=True)
    parser.add_argument("--checkpoint", action="append", metavar="LABEL=PATH", required=True,
                        help="Repeatable. Maps a config checkpoint_label to a local weights file.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-results", type=Path, default=None,
                        help="calibration_results.json from the prior stage (required for B and C).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any existing results in --output-dir and re-run every trial.")
    args = parser.parse_args()

    try:
        config_path = _resolve_cli_path(args.config)
        config = load_config(config_path)
        checkpoints = parse_checkpoint_args(args.checkpoint)
        checkpoints = {label: str(_resolve_cli_path(path)) for label, path in checkpoints.items()}
        data_dir = _resolve_cli_path(args.data_dir)
        output_dir = _resolve_cli_path(args.output_dir)
        previous_results_path = _resolve_cli_path(args.previous_results) if args.previous_results else None

        stage_key = _STAGE_KEY_BY_ARG[args.stage]
        if args.stage == "A":
            trials = build_stage_a_trials(config, checkpoints)

            def build_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
                return {
                    "selected_det_thresholds": select_best_det_thresholds(
                        records, config["stage_a"]["n_best_det_thresholds"],
                    ),
                }
        elif args.stage == "B":
            if previous_results_path is None:
                raise CalibrationError("--previous-results (Stage A results) is required for Stage B")
            _, best_det_thresholds = load_stage_a_report_for_stage_b(
                previous_results_path, config, config_path, checkpoints, data_dir,
            )
            trials = build_stage_b_trials(config, checkpoints, best_det_thresholds)

            def build_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
                pairs = select_best_pairs(records, config["stage_b"]["n_best_pairs"])
                return {"selected_pairs": [list(pair) for pair in pairs]}
        else:
            if previous_results_path is None:
                raise CalibrationError("--previous-results (Stage B results) is required for Stage C")
            _, best_pairs = load_stage_b_report_for_stage_c(
                previous_results_path, config, config_path, checkpoints, data_dir,
            )
            trials = build_stage_c_trials(config, checkpoints, best_pairs)

            def build_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
                best = select_final_best(records, checkpoint_label_order=config["stage_c"]["checkpoint_labels"])
                return {"selected_final_trial": _selected_final_trial_summary(best)}

        report = run_trials(
            trials, data_dir, output_dir, config_path, stage_key,
            resume=not args.no_resume, build_selection=build_selection,
        )
    except (CalibrationError, OSError, json.JSONDecodeError, KeyError) as exc:
        parser.error(str(exc))
        return

    screening_note = (
        " (SCREENING scores on a validation subset, not final validation scores)"
        if args.stage in ("A", "B") else ""
    )
    summary = {
        "results_path": report["results_path"],
        "n_trials": len(report["trials"]),
        "n_succeeded": sum(1 for r in report["trials"] if r["status"] == "success"),
        "n_failed": sum(1 for r in report["trials"] if r["status"] == "failed"),
    }
    for key in ("selected_det_thresholds", "selected_pairs", "selected_final_trial"):
        if key in report:
            summary[key] = report[key]
    print(json.dumps(summary, indent=2))
    print(f"Stage {args.stage} complete.{screening_note}")


if __name__ == "__main__":
    main()
