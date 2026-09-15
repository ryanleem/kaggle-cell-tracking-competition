#!/usr/bin/env python
"""Extended, resumable inference-calibration workflow (Stages A-D).

This is a separate workflow from ``scripts/calibrate_inference_thresholds.py``
and ``experiments/configs/inference_threshold_calibration.json`` — it neither
replaces nor modifies either. It starts from the calibrated baseline
(det_threshold=0.70, edge_threshold=0.35, pool_kernel_um=5.0, greedy
tracking) and, on both the 7500 and 10000 checkpoints independently:

    Stage A — screens a wider ``det_threshold`` range at the baseline edge
        threshold and pooling kernel, on the six-dataset calibration subset,
        and selects the best 2 det_thresholds *per checkpoint*.
    Stage B — screens a lower ``edge_threshold`` range at each Stage-A
        candidate, on the subset, and selects the best 2 (det, edge) pairs
        *per checkpoint*.
    Stage C — compares ``pool_kernel_um`` in {5.0, 6.0} at each Stage-B
        candidate, on the subset, and selects the best 2 (det, edge, pool)
        candidates *per checkpoint*. Gated on the preserved ground-truth
        detection-spacing audit (see ``scripts/audit_validation.py``): the
        6.0 kernel is only swept when the audit's pinned hash and findings
        still hold.
    Stage D — confirms the union of every checkpoint's Stage-C candidates on
        the *complete* fold-0 validation test partition and selects one
        global winner by official ``summary_metrics.score``.

This module never shells out itself and never invokes anything but
``predict_unet_transformer.py`` and ``evaluate.py`` — every trial is
executed via ``calibrate_inference_thresholds.execute_trial`` (through
``run_trials``), so the training-script refusal, prediction/metrics
validation, atomic incremental writes, scratch-directory isolation and
cleanup, heartbeat streaming, and per-trial resume revalidation are all
inherited unmodified from that module rather than re-implemented here.

The audit gate and the extended config's own hash are both re-checked at
the *start* of every stage invocation — including resumed ones — so a
resume is refused outright (before any trial, new or cached, is touched)
if either the audit file or the config has drifted since a prior run. Each
successful trial additionally records an aggregate ``diagnostics`` object
(score, edge/adjusted-edge Jaccard, node recall, division counts, and
predicted-edge-count-derived fractions) purely for inspection — selection
always uses official ``summary_metrics.score`` alone.

Staged usage — see
``experiments/configs/extended_inference_calibration.json``:

    python scripts/calibrate_extended_inference.py --stage A \\
        --config experiments/configs/extended_inference_calibration.json \\
        --checkpoint 7500=weights/.../edge_predictor_iter_007500.pth \\
        --checkpoint 10000=weights/.../edge_predictor_iter_010000.pth \\
        --data-dir data/train --output-dir experiments/calibration_runs/ext_stage_a

    python scripts/calibrate_extended_inference.py --stage B ... \\
        --output-dir experiments/calibration_runs/ext_stage_b \\
        --previous-results experiments/calibration_runs/ext_stage_a/calibration_results.json

    python scripts/calibrate_extended_inference.py --stage C ... \\
        --output-dir experiments/calibration_runs/ext_stage_c \\
        --previous-results experiments/calibration_runs/ext_stage_b/calibration_results.json

    python scripts/calibrate_extended_inference.py --stage D ... \\
        --output-dir experiments/calibration_runs/ext_stage_d \\
        --previous-results experiments/calibration_runs/ext_stage_c/calibration_results.json

Stages A-C run on the six-dataset calibration subset purely to limit GPU
use; their scores are SCREENING scores, not final validation scores. Only
Stage D runs on the complete fold-0 validation test partition and produces
the final validation score.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from numbers import Real
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_validation
import calibrate_inference_thresholds as base

REPO_ROOT = base.REPO_ROOT
SCHEMA_VERSION = 1

ALLOWED_CHECKPOINT_LABELS = ["7500", "10000"]

# The reviewed plan pins these values exactly: editing the config to sweep a
# different grid, a different selection metric, or a different tracker must
# never silently authorize a different experiment.
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_SELECTION_METRIC = "summary_metrics.score"
EXPECTED_TRACKING = "greedy"
EXPECTED_BASELINE = {
    "det_threshold": 0.70, "edge_threshold": 0.35, "pool_kernel_um": 5.0, "tracking": "greedy",
}
EXPECTED_STAGE_A_DET_THRESHOLDS = [0.70, 0.74, 0.78, 0.82, 0.86]
EXPECTED_STAGE_A_EDGE_THRESHOLD = 0.35
EXPECTED_STAGE_A_POOL_KERNEL_UM = 5.0
EXPECTED_STAGE_A_N_BEST_PER_CHECKPOINT = 2
EXPECTED_STAGE_B_EDGE_THRESHOLDS = [0.20, 0.25, 0.30, 0.35]
EXPECTED_STAGE_B_POOL_KERNEL_UM = 5.0
EXPECTED_STAGE_B_N_BEST_PER_CHECKPOINT = 2
EXPECTED_STAGE_C_POOL_KERNEL_UM_VALUES = [5.0, 6.0]
EXPECTED_STAGE_C_N_BEST_PER_CHECKPOINT = 2
EXPECTED_STAGE_D_CHECKPOINT_LABELS = ["7500", "10000"]

# The reviewed checkpoint and audit *identities* are pinned here too — not
# just their shape. A config edited to point at a different (but otherwise
# well-formed: correct hex length, internally self-consistent) checkpoint or
# audit input must never be silently authorized. These supplement, and never
# replace, the on-disk hash/structural checks in validate_pinned_checkpoints()
# and audit_validation.load_and_validate_audit().
EXPECTED_CHECKPOINT_PINS = {
    "7500": {
        "filename": "edge_predictor_iter_007500.pth",
        "sha256": "7cd66f8168647c38cca222604d88c4bc25f9c2b66f82b9ea3f552d1bb2b9c4c8",
    },
    "10000": {
        "filename": "edge_predictor_iter_010000.pth",
        "sha256": "d6a0221948a6335f581ec12b509bc529672b99f0dfcd6c393be3f1a01a9c7d96",
    },
    "adjacent_config_sha256": "b9c71baa3e2d5523b420e419eae8c56f5598917f36bbfecbe9cc2703598b8a6c",
}
EXPECTED_AUDIT_PATH = "experiments/audits/detection_spacing_fold0_downsample_1x4x4.json"
EXPECTED_AUDIT_SHA256 = "59c4e39e3136164a715f95623209d09013fcd09ef32b7e794f1bc5dd51f355a7"
EXPECTED_AUDIT_DATASET_COUNT = 19
EXPECTED_AUDIT_NODE_COUNT = 13335
EXPECTED_AUDIT_TOLERANCE = 0.001
EXPECTED_AUDIT_PRIMARY_POOL_KERNEL_UM = 5.0
EXPECTED_AUDIT_PRIMARY_VOXEL_KERNEL = [3, 3, 3]
EXPECTED_AUDIT_SECONDARY_POOL_KERNEL_UM = 6.0
EXPECTED_AUDIT_SECONDARY_VOXEL_KERNEL = [5, 5, 5]
EXPECTED_AUDIT_OBSERVED = {
    "primary": {"count": 0, "pair_count": 0, "fraction": 0.0},
    "secondary": {"count": 2, "pair_count": 1, "fraction": 0.00014998125234345707},  # == 2 / 13335
}

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_STAGE_KEY_BY_ARG = {"A": "stage_a", "B": "stage_b", "C": "stage_c", "D": "stage_d"}
_PREVIOUS_STAGE_KEY_BY_ARG = {"B": "stage_a", "C": "stage_b", "D": "stage_c"}

_REQUIRED_CONFIG_KEYS = {
    "schema_version", "selection_metric", "checkpoint_labels", "tracking",
    "subset_split_file", "subset_split", "subset_dataset_names",
    "full_split_file", "full_split", "baseline", "audit", "checkpoint_pins",
    "stage_a", "stage_b", "stage_c", "stage_d",
}
_REQUIRED_STAGE_KEYS = {
    "stage_a": {
        "checkpoint_labels", "det_thresholds", "edge_threshold", "pool_kernel_um",
        "n_best_det_thresholds_per_checkpoint", "det_threshold_upper_bound_warning",
    },
    "stage_b": {
        "checkpoint_labels", "edge_thresholds", "pool_kernel_um",
        "n_best_pairs_per_checkpoint", "edge_threshold_lower_bound_warning",
    },
    "stage_c": {"checkpoint_labels", "pool_kernel_um_values", "n_best_candidates_per_checkpoint"},
    "stage_d": {"checkpoint_labels"},
}
_REQUIRED_BASELINE_KEYS = {"det_threshold", "edge_threshold", "pool_kernel_um", "tracking"}
_REQUIRED_CHECKPOINT_PIN_ENTRY_KEYS = {"filename", "sha256"}


# =============================================================================
# Strict JSON parsing (no NaN/Infinity tokens, no duplicate keys)
# =============================================================================

def _reject_nonfinite_json_constant(token: str) -> float:
    raise base.CalibrationError(f"extended calibration config contains a non-finite JSON constant: {token}")


def _no_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise base.CalibrationError(f"extended calibration config contains a duplicate JSON key: {key!r}")
        seen[key] = value
    return seen


# =============================================================================
# Config loading and validation
# =============================================================================

def load_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate the extended-calibration config.

    Pins ``checkpoint_labels`` to the reviewed ``["7500", "10000"]`` plan at
    every stage, requires the pooling kernels declared across
    ``baseline``/``stage_a``/``stage_b``/``stage_c``/``audit`` to agree with
    each other exactly, requires the six-dataset subset split's test list to
    exactly match the configured dataset names (reusing
    ``calibrate_inference_thresholds._validate_subset_matches_split``), and
    requires the config's declared checkpoint and audit *identities*
    (filenames, hashes, path, dataset/node counts, observed findings,
    tolerance) to exactly equal the reviewed constants pinned at the top of
    this module — a well-formed but different replacement value is rejected
    even though it may be internally self-consistent.
    """
    try:
        config = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise base.CalibrationError(f"cannot read extended calibration config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise base.CalibrationError("extended calibration config must be a JSON object")

    missing = sorted(_REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise base.CalibrationError(f"extended calibration config missing required keys: {missing}")

    for stage_key, required in _REQUIRED_STAGE_KEYS.items():
        stage_cfg = config[stage_key]
        if not isinstance(stage_cfg, dict):
            raise base.CalibrationError(f"extended calibration config {stage_key} must be an object")
        stage_missing = sorted(required - set(stage_cfg))
        if stage_missing:
            raise base.CalibrationError(f"extended calibration config {stage_key} missing required keys: {stage_missing}")

    if not isinstance(config["audit"], dict):
        raise base.CalibrationError("extended calibration config 'audit' must be an object")
    if not isinstance(config["baseline"], dict):
        raise base.CalibrationError("extended calibration config 'baseline' must be an object")
    baseline_missing = sorted(_REQUIRED_BASELINE_KEYS - set(config["baseline"]))
    if baseline_missing:
        raise base.CalibrationError(f"extended calibration config baseline missing required keys: {baseline_missing}")

    if config["schema_version"] != EXPECTED_SCHEMA_VERSION:
        raise base.CalibrationError(
            f"schema_version {config['schema_version']!r} does not match the reviewed plan {EXPECTED_SCHEMA_VERSION!r}",
        )
    if config["selection_metric"] != EXPECTED_SELECTION_METRIC:
        raise base.CalibrationError(
            f"selection_metric {config['selection_metric']!r} does not match the reviewed plan "
            f"{EXPECTED_SELECTION_METRIC!r} (official score must remain the only selection metric)",
        )
    if config["checkpoint_labels"] != ALLOWED_CHECKPOINT_LABELS:
        raise base.CalibrationError(
            f"checkpoint_labels {config['checkpoint_labels']!r} does not match the reviewed plan "
            f"{ALLOWED_CHECKPOINT_LABELS}",
        )
    if config["tracking"] != EXPECTED_TRACKING:
        raise base.CalibrationError(f"tracking must be {EXPECTED_TRACKING!r}, got {config['tracking']!r}")

    for key in ("subset_split", "full_split"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise base.CalibrationError(f"{key} must be a non-negative integer, got {value!r}")

    if not isinstance(config["subset_dataset_names"], list) or not config["subset_dataset_names"]:
        raise base.CalibrationError("subset_dataset_names must be a non-empty list")
    if len(config["subset_dataset_names"]) != len(set(config["subset_dataset_names"])):
        raise base.CalibrationError("subset_dataset_names contains duplicates")

    baseline = config["baseline"]
    base._validate_finite_number("baseline.det_threshold", baseline["det_threshold"], lo=0.0, hi=1.0)
    base._validate_finite_number("baseline.edge_threshold", baseline["edge_threshold"], lo=0.0, hi=1.0)
    base._validate_finite_number("baseline.pool_kernel_um", baseline["pool_kernel_um"], lo=1e-9)
    if baseline["tracking"] != EXPECTED_TRACKING:
        raise base.CalibrationError(f"baseline.tracking must be {EXPECTED_TRACKING!r}, got {baseline['tracking']!r}")
    if baseline != EXPECTED_BASELINE:
        raise base.CalibrationError(
            f"baseline {baseline!r} does not match the reviewed plan {EXPECTED_BASELINE!r}",
        )

    stage_a = config["stage_a"]
    if stage_a["checkpoint_labels"] != config["checkpoint_labels"]:
        raise base.CalibrationError("stage_a.checkpoint_labels must equal the top-level checkpoint_labels")
    if not isinstance(stage_a["det_thresholds"], list) or not stage_a["det_thresholds"]:
        raise base.CalibrationError("stage_a.det_thresholds must be a non-empty list")
    for value in stage_a["det_thresholds"]:
        base._validate_finite_number("stage_a.det_thresholds[*]", value, lo=0.0, hi=1.0)
    if len(stage_a["det_thresholds"]) != len(set(stage_a["det_thresholds"])):
        raise base.CalibrationError(
            "stage_a.det_thresholds contains duplicates (duplicate thresholds would collide on the "
            "same trial_id and trial directory)",
        )
    if stage_a["det_thresholds"] != EXPECTED_STAGE_A_DET_THRESHOLDS:
        raise base.CalibrationError(
            f"stage_a.det_thresholds {stage_a['det_thresholds']!r} does not match the reviewed plan "
            f"{EXPECTED_STAGE_A_DET_THRESHOLDS!r}",
        )
    base._validate_finite_number("stage_a.edge_threshold", stage_a["edge_threshold"], lo=0.0, hi=1.0)
    base._validate_finite_number("stage_a.pool_kernel_um", stage_a["pool_kernel_um"], lo=1e-9)
    if stage_a["edge_threshold"] != EXPECTED_STAGE_A_EDGE_THRESHOLD:
        raise base.CalibrationError(
            f"stage_a.edge_threshold {stage_a['edge_threshold']!r} does not match the reviewed plan "
            f"{EXPECTED_STAGE_A_EDGE_THRESHOLD!r}",
        )
    if stage_a["pool_kernel_um"] != EXPECTED_STAGE_A_POOL_KERNEL_UM:
        raise base.CalibrationError(
            f"stage_a.pool_kernel_um {stage_a['pool_kernel_um']!r} does not match the reviewed plan "
            f"{EXPECTED_STAGE_A_POOL_KERNEL_UM!r}",
        )
    n_best_det = base._validate_positive_int(
        "stage_a.n_best_det_thresholds_per_checkpoint", stage_a["n_best_det_thresholds_per_checkpoint"],
    )
    if n_best_det != EXPECTED_STAGE_A_N_BEST_PER_CHECKPOINT:
        raise base.CalibrationError(
            f"stage_a.n_best_det_thresholds_per_checkpoint {n_best_det!r} does not match the reviewed "
            f"plan {EXPECTED_STAGE_A_N_BEST_PER_CHECKPOINT!r}",
        )
    unique_dets = len(set(stage_a["det_thresholds"]))
    if n_best_det > unique_dets:
        raise base.CalibrationError(
            f"stage_a.n_best_det_thresholds_per_checkpoint ({n_best_det}) exceeds the number of unique "
            f"det_thresholds available ({unique_dets})",
        )
    upper_bound = base._validate_finite_number(
        "stage_a.det_threshold_upper_bound_warning", stage_a["det_threshold_upper_bound_warning"], lo=0.0, hi=1.0,
    )
    if upper_bound != max(stage_a["det_thresholds"]):
        raise base.CalibrationError(
            "stage_a.det_threshold_upper_bound_warning must equal max(stage_a.det_thresholds) "
            f"({max(stage_a['det_thresholds'])!r}), got {upper_bound!r}",
        )

    stage_b = config["stage_b"]
    if stage_b["checkpoint_labels"] != config["checkpoint_labels"]:
        raise base.CalibrationError("stage_b.checkpoint_labels must equal the top-level checkpoint_labels")
    if not isinstance(stage_b["edge_thresholds"], list) or not stage_b["edge_thresholds"]:
        raise base.CalibrationError("stage_b.edge_thresholds must be a non-empty list")
    for value in stage_b["edge_thresholds"]:
        base._validate_finite_number("stage_b.edge_thresholds[*]", value, lo=0.0, hi=1.0)
    if len(stage_b["edge_thresholds"]) != len(set(stage_b["edge_thresholds"])):
        raise base.CalibrationError(
            "stage_b.edge_thresholds contains duplicates (duplicate thresholds would collide on the "
            "same trial_id and trial directory)",
        )
    if stage_b["edge_thresholds"] != EXPECTED_STAGE_B_EDGE_THRESHOLDS:
        raise base.CalibrationError(
            f"stage_b.edge_thresholds {stage_b['edge_thresholds']!r} does not match the reviewed plan "
            f"{EXPECTED_STAGE_B_EDGE_THRESHOLDS!r}",
        )
    base._validate_finite_number("stage_b.pool_kernel_um", stage_b["pool_kernel_um"], lo=1e-9)
    if stage_b["pool_kernel_um"] != EXPECTED_STAGE_B_POOL_KERNEL_UM:
        raise base.CalibrationError(
            f"stage_b.pool_kernel_um {stage_b['pool_kernel_um']!r} does not match the reviewed plan "
            f"{EXPECTED_STAGE_B_POOL_KERNEL_UM!r}",
        )
    n_best_pairs = base._validate_positive_int(
        "stage_b.n_best_pairs_per_checkpoint", stage_b["n_best_pairs_per_checkpoint"],
    )
    if n_best_pairs != EXPECTED_STAGE_B_N_BEST_PER_CHECKPOINT:
        raise base.CalibrationError(
            f"stage_b.n_best_pairs_per_checkpoint {n_best_pairs!r} does not match the reviewed plan "
            f"{EXPECTED_STAGE_B_N_BEST_PER_CHECKPOINT!r}",
        )
    max_pairs = n_best_det * len(set(stage_b["edge_thresholds"]))
    if n_best_pairs > max_pairs:
        raise base.CalibrationError(
            f"stage_b.n_best_pairs_per_checkpoint ({n_best_pairs}) exceeds the number of unique candidate "
            f"pairs available per checkpoint from the config ({max_pairs})",
        )
    lower_bound = base._validate_finite_number(
        "stage_b.edge_threshold_lower_bound_warning", stage_b["edge_threshold_lower_bound_warning"], lo=0.0, hi=1.0,
    )
    if lower_bound != min(stage_b["edge_thresholds"]):
        raise base.CalibrationError(
            "stage_b.edge_threshold_lower_bound_warning must equal min(stage_b.edge_thresholds) "
            f"({min(stage_b['edge_thresholds'])!r}), got {lower_bound!r}",
        )

    audit_cfg = config["audit"]

    # Hard-lock the reviewed audit *identity*: a config edited to point at a
    # different (but well-formed) audit path/hash, different dataset/node
    # counts, different observed findings, or a different tolerance must be
    # rejected outright — not merely checked for internal self-consistency.
    if audit_cfg.get("path") != EXPECTED_AUDIT_PATH:
        raise base.CalibrationError(
            f"audit.path {audit_cfg.get('path')!r} does not match the reviewed plan {EXPECTED_AUDIT_PATH!r}",
        )
    if audit_cfg.get("sha256") != EXPECTED_AUDIT_SHA256:
        raise base.CalibrationError(
            f"audit.sha256 {audit_cfg.get('sha256')!r} does not match the reviewed plan pinned in "
            "scripts/calibrate_extended_inference.py",
        )
    if audit_cfg.get("dataset_count") != EXPECTED_AUDIT_DATASET_COUNT:
        raise base.CalibrationError(
            f"audit.dataset_count {audit_cfg.get('dataset_count')!r} does not match the reviewed plan "
            f"{EXPECTED_AUDIT_DATASET_COUNT!r}",
        )
    if audit_cfg.get("node_count") != EXPECTED_AUDIT_NODE_COUNT:
        raise base.CalibrationError(
            f"audit.node_count {audit_cfg.get('node_count')!r} does not match the reviewed plan "
            f"{EXPECTED_AUDIT_NODE_COUNT!r}",
        )
    if audit_cfg.get("tolerance") != EXPECTED_AUDIT_TOLERANCE:
        raise base.CalibrationError(
            f"audit.tolerance {audit_cfg.get('tolerance')!r} does not match the reviewed plan "
            f"{EXPECTED_AUDIT_TOLERANCE!r}",
        )
    if audit_cfg.get("quantized_collision_fraction_max") != EXPECTED_AUDIT_TOLERANCE:
        raise base.CalibrationError(
            f"audit.quantized_collision_fraction_max {audit_cfg.get('quantized_collision_fraction_max')!r} "
            f"does not match the reviewed tolerance {EXPECTED_AUDIT_TOLERANCE!r}",
        )
    if audit_cfg.get("primary_pool_kernel_um") != EXPECTED_AUDIT_PRIMARY_POOL_KERNEL_UM:
        raise base.CalibrationError(
            f"audit.primary_pool_kernel_um {audit_cfg.get('primary_pool_kernel_um')!r} does not match the "
            f"reviewed plan {EXPECTED_AUDIT_PRIMARY_POOL_KERNEL_UM!r}",
        )
    if audit_cfg.get("primary_voxel_kernel") != EXPECTED_AUDIT_PRIMARY_VOXEL_KERNEL:
        raise base.CalibrationError(
            f"audit.primary_voxel_kernel {audit_cfg.get('primary_voxel_kernel')!r} does not match the "
            f"reviewed plan {EXPECTED_AUDIT_PRIMARY_VOXEL_KERNEL!r}",
        )
    if audit_cfg.get("secondary_pool_kernel_um") != EXPECTED_AUDIT_SECONDARY_POOL_KERNEL_UM:
        raise base.CalibrationError(
            f"audit.secondary_pool_kernel_um {audit_cfg.get('secondary_pool_kernel_um')!r} does not match "
            f"the reviewed plan {EXPECTED_AUDIT_SECONDARY_POOL_KERNEL_UM!r}",
        )
    if audit_cfg.get("secondary_voxel_kernel") != EXPECTED_AUDIT_SECONDARY_VOXEL_KERNEL:
        raise base.CalibrationError(
            f"audit.secondary_voxel_kernel {audit_cfg.get('secondary_voxel_kernel')!r} does not match the "
            f"reviewed plan {EXPECTED_AUDIT_SECONDARY_VOXEL_KERNEL!r}",
        )
    if audit_cfg.get("observed") != EXPECTED_AUDIT_OBSERVED:
        raise base.CalibrationError(
            f"audit.observed {audit_cfg.get('observed')!r} does not match the reviewed findings "
            f"{EXPECTED_AUDIT_OBSERVED!r}",
        )

    primary_pool = audit_cfg.get("primary_pool_kernel_um")
    secondary_pool = audit_cfg.get("secondary_pool_kernel_um")
    if baseline["pool_kernel_um"] != primary_pool:
        raise base.CalibrationError(
            f"baseline.pool_kernel_um ({baseline['pool_kernel_um']!r}) must equal audit.primary_pool_kernel_um "
            f"({primary_pool!r})",
        )
    if stage_a["pool_kernel_um"] != primary_pool:
        raise base.CalibrationError("stage_a.pool_kernel_um must equal audit.primary_pool_kernel_um")
    if stage_b["pool_kernel_um"] != primary_pool:
        raise base.CalibrationError("stage_b.pool_kernel_um must equal audit.primary_pool_kernel_um")

    stage_c = config["stage_c"]
    if stage_c["checkpoint_labels"] != config["checkpoint_labels"]:
        raise base.CalibrationError("stage_c.checkpoint_labels must equal the top-level checkpoint_labels")
    if stage_c["pool_kernel_um_values"] != [primary_pool, secondary_pool]:
        raise base.CalibrationError(
            "stage_c.pool_kernel_um_values must be exactly [audit.primary_pool_kernel_um, "
            f"audit.secondary_pool_kernel_um] = [{primary_pool!r}, {secondary_pool!r}], got "
            f"{stage_c['pool_kernel_um_values']!r}",
        )
    if stage_c["pool_kernel_um_values"] != EXPECTED_STAGE_C_POOL_KERNEL_UM_VALUES:
        raise base.CalibrationError(
            f"stage_c.pool_kernel_um_values {stage_c['pool_kernel_um_values']!r} does not match the "
            f"reviewed plan {EXPECTED_STAGE_C_POOL_KERNEL_UM_VALUES!r}",
        )
    n_best_candidates = base._validate_positive_int(
        "stage_c.n_best_candidates_per_checkpoint", stage_c["n_best_candidates_per_checkpoint"],
    )
    if n_best_candidates != EXPECTED_STAGE_C_N_BEST_PER_CHECKPOINT:
        raise base.CalibrationError(
            f"stage_c.n_best_candidates_per_checkpoint {n_best_candidates!r} does not match the reviewed "
            f"plan {EXPECTED_STAGE_C_N_BEST_PER_CHECKPOINT!r}",
        )
    max_candidates = n_best_pairs * len(set(stage_c["pool_kernel_um_values"]))
    if n_best_candidates > max_candidates:
        raise base.CalibrationError(
            f"stage_c.n_best_candidates_per_checkpoint ({n_best_candidates}) exceeds the number of unique "
            f"candidates available per checkpoint from the config ({max_candidates})",
        )

    stage_d = config["stage_d"]
    if stage_d["checkpoint_labels"] != config["checkpoint_labels"]:
        raise base.CalibrationError("stage_d.checkpoint_labels must equal the top-level checkpoint_labels")
    if stage_d["checkpoint_labels"] != EXPECTED_STAGE_D_CHECKPOINT_LABELS:
        raise base.CalibrationError(
            f"stage_d.checkpoint_labels {stage_d['checkpoint_labels']!r} does not match the reviewed "
            f"plan {EXPECTED_STAGE_D_CHECKPOINT_LABELS!r}",
        )

    _validate_checkpoint_pins_shape(config)

    # Hard-lock the reviewed checkpoint *identity*: a config edited to point
    # at a different (but well-formed 64-hex) checkpoint or adjacent-config
    # hash, or a different filename, must be rejected outright.
    if config["checkpoint_pins"] != EXPECTED_CHECKPOINT_PINS:
        raise base.CalibrationError(
            f"checkpoint_pins {config['checkpoint_pins']!r} does not match the reviewed checkpoint "
            f"identities pinned in scripts/calibrate_extended_inference.py: {EXPECTED_CHECKPOINT_PINS!r}",
        )

    base._validate_subset_matches_split(config)

    return config


# =============================================================================
# Checkpoint identity pinning — validated before any subprocess may run
# =============================================================================

def _validate_checkpoint_pins_shape(config: dict[str, Any]) -> dict[str, Any]:
    pins = config["checkpoint_pins"]
    if not isinstance(pins, dict):
        raise base.CalibrationError("checkpoint_pins must be an object")
    adjacent = pins.get("adjacent_config_sha256")
    if not isinstance(adjacent, str) or not _HEX64_RE.match(adjacent):
        raise base.CalibrationError(
            "checkpoint_pins.adjacent_config_sha256 must be a 64-character lowercase hex sha256 digest",
        )
    for label in config["checkpoint_labels"]:
        entry = pins.get(label)
        if not isinstance(entry, dict) or sorted(entry) != sorted(_REQUIRED_CHECKPOINT_PIN_ENTRY_KEYS):
            raise base.CalibrationError(
                f"checkpoint_pins[{label!r}] must be an object with exactly keys "
                f"{sorted(_REQUIRED_CHECKPOINT_PIN_ENTRY_KEYS)}",
            )
        filename = entry["filename"]
        sha = entry["sha256"]
        if not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename:
            raise base.CalibrationError(f"checkpoint_pins[{label!r}].filename must be a bare filename, got {filename!r}")
        if not isinstance(sha, str) or not _HEX64_RE.match(sha):
            raise base.CalibrationError(
                f"checkpoint_pins[{label!r}].sha256 must be a 64-character lowercase hex sha256 digest",
            )
    return pins


def validate_pinned_checkpoints(config: dict[str, Any], checkpoints: dict[str, str]) -> None:
    """Require every supplied checkpoint to exactly match its reviewed pin before any subprocess runs.

    Each checkpoint must (1) use the exact pinned filename, (2) be a regular,
    non-symlink file, (3) hash to the exact pinned sha256, and (4) have an
    adjacent ``config.json`` that is itself a regular, non-symlink file
    hashing to the exact pinned ``adjacent_config_sha256``. Any mismatch
    refuses to proceed — no trial, fresh or resumed, may run against
    unpinned checkpoint bytes.
    """
    pins = config["checkpoint_pins"]
    adjacent_expected = pins["adjacent_config_sha256"]
    for label, path_str in checkpoints.items():
        entry = pins.get(label)
        if entry is None:
            raise base.CalibrationError(f"no checkpoint_pins entry for label {label!r}")
        path = Path(path_str)
        if path.name != entry["filename"]:
            raise base.CalibrationError(
                f"checkpoint for label {label!r} has filename {path.name!r}, expected the pinned "
                f"filename {entry['filename']!r}",
            )
        if path.is_symlink():
            raise base.CalibrationError(f"checkpoint for label {label!r} must not be a symlink: {path}")
        if not path.is_file():
            raise base.CalibrationError(f"checkpoint for label {label!r} not found: {path}")
        actual_sha256 = base._sha256(path)
        if actual_sha256 != entry["sha256"]:
            raise base.CalibrationError(
                f"checkpoint for label {label!r} sha256 {actual_sha256} does not match the pinned "
                f"value {entry['sha256']} (refusing to run any trial against unpinned checkpoint bytes)",
            )
        config_path = path.parent / "config.json"
        if config_path.is_symlink():
            raise base.CalibrationError(
                f"checkpoint config.json for label {label!r} must not be a symlink: {config_path}",
            )
        if not config_path.is_file():
            raise base.CalibrationError(f"checkpoint config.json not found for label {label!r}: {config_path}")
        actual_config_sha256 = base._sha256(config_path)
        if actual_config_sha256 != adjacent_expected:
            raise base.CalibrationError(
                f"checkpoint config.json for label {label!r} sha256 {actual_config_sha256} does not "
                f"match the pinned adjacent_config_sha256 {adjacent_expected}",
            )


# =============================================================================
# Checkpoint labels
# =============================================================================

def parse_and_validate_checkpoints(values: list[str] | None) -> dict[str, str]:
    """Parse ``--checkpoint LABEL=PATH`` args, requiring the label set to be exactly
    ``{"7500", "10000"}`` — rejecting duplicates, unknown labels, and a missing label.
    """
    checkpoints = base.parse_checkpoint_args(values)
    unexpected = sorted(set(checkpoints) - set(ALLOWED_CHECKPOINT_LABELS))
    if unexpected:
        raise base.CalibrationError(
            f"--checkpoint label(s) outside the allowed set {ALLOWED_CHECKPOINT_LABELS}: {unexpected}",
        )
    missing = sorted(set(ALLOWED_CHECKPOINT_LABELS) - set(checkpoints))
    if missing:
        raise base.CalibrationError(
            f"--checkpoint is missing required label(s) {missing}; the supplied checkpoint label set must "
            f"be exactly {set(ALLOWED_CHECKPOINT_LABELS)}",
        )
    return checkpoints


# =============================================================================
# Trial construction (pool_kernel_um is part of the trial_id: Stage C sweeps
# it for the *same* checkpoint/det/edge combination, so it must be part of
# what makes a trial unique)
# =============================================================================

def _trial_id(stage: str, checkpoint_label: str, det_threshold: float, edge_threshold: float, pool_kernel_um: float) -> str:
    return (
        f"{stage}__ckpt-{checkpoint_label}__det-{det_threshold:.4f}"
        f"__edge-{edge_threshold:.4f}__pool-{pool_kernel_um:.4f}"
    )


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
    pool_kernel_um = float(pool_kernel_um)
    return {
        "trial_id": _trial_id(stage, checkpoint_label, det_threshold, edge_threshold, pool_kernel_um),
        "stage": stage,
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(checkpoint_path),
        "det_threshold": det_threshold,
        "edge_threshold": edge_threshold,
        "pool_kernel_um": pool_kernel_um,
        "tracking": tracking,
        "split_file": str(base._resolve_repo_path(split_file)),
        "split": int(split),
        "is_screening": bool(is_screening),
    }


def build_stage_a_trials(config: dict[str, Any], checkpoints: dict[str, str]) -> list[dict[str, Any]]:
    stage_cfg = config["stage_a"]
    trials = []
    for label in stage_cfg["checkpoint_labels"]:
        checkpoint_path = base._resolve_checkpoint_path(checkpoints, label)
        for det_threshold in stage_cfg["det_thresholds"]:
            trials.append(_make_trial(
                stage="stage_a", checkpoint_label=label, checkpoint_path=checkpoint_path,
                det_threshold=det_threshold, edge_threshold=stage_cfg["edge_threshold"],
                pool_kernel_um=stage_cfg["pool_kernel_um"], tracking=config["tracking"],
                split_file=config["subset_split_file"], split=config["subset_split"], is_screening=True,
            ))
    return trials


def build_stage_b_trials(
    config: dict[str, Any], checkpoints: dict[str, str], selected_det_by_checkpoint: dict[str, list[float]],
) -> list[dict[str, Any]]:
    stage_cfg = config["stage_b"]
    trials = []
    for label in stage_cfg["checkpoint_labels"]:
        checkpoint_path = base._resolve_checkpoint_path(checkpoints, label)
        for det_threshold in selected_det_by_checkpoint[label]:
            for edge_threshold in stage_cfg["edge_thresholds"]:
                trials.append(_make_trial(
                    stage="stage_b", checkpoint_label=label, checkpoint_path=checkpoint_path,
                    det_threshold=det_threshold, edge_threshold=edge_threshold,
                    pool_kernel_um=stage_cfg["pool_kernel_um"], tracking=config["tracking"],
                    split_file=config["subset_split_file"], split=config["subset_split"], is_screening=True,
                ))
    return trials


def build_stage_c_trials(
    config: dict[str, Any], checkpoints: dict[str, str],
    selected_pairs_by_checkpoint: dict[str, list[tuple[float, float]]],
) -> list[dict[str, Any]]:
    stage_cfg = config["stage_c"]
    trials = []
    for label in stage_cfg["checkpoint_labels"]:
        checkpoint_path = base._resolve_checkpoint_path(checkpoints, label)
        for det_threshold, edge_threshold in selected_pairs_by_checkpoint[label]:
            for pool_kernel_um in stage_cfg["pool_kernel_um_values"]:
                trials.append(_make_trial(
                    stage="stage_c", checkpoint_label=label, checkpoint_path=checkpoint_path,
                    det_threshold=det_threshold, edge_threshold=edge_threshold,
                    pool_kernel_um=pool_kernel_um, tracking=config["tracking"],
                    split_file=config["subset_split_file"], split=config["subset_split"], is_screening=True,
                ))
    return trials


def build_stage_d_trials(
    config: dict[str, Any], checkpoints: dict[str, str],
    selected_candidates_by_checkpoint: dict[str, list[tuple[float, float, float]]],
) -> list[dict[str, Any]]:
    stage_cfg = config["stage_d"]
    trials = []
    for label in stage_cfg["checkpoint_labels"]:
        checkpoint_path = base._resolve_checkpoint_path(checkpoints, label)
        for det_threshold, edge_threshold, pool_kernel_um in selected_candidates_by_checkpoint[label]:
            trials.append(_make_trial(
                stage="stage_d", checkpoint_label=label, checkpoint_path=checkpoint_path,
                det_threshold=det_threshold, edge_threshold=edge_threshold,
                pool_kernel_um=pool_kernel_um, tracking=config["tracking"],
                split_file=config["full_split_file"], split=config["full_split"], is_screening=False,
            ))
    return trials


# =============================================================================
# Per-checkpoint selection with deterministic, documented tie-breaking
# =============================================================================

def select_best_det_thresholds_by_checkpoint(
    trial_records: list[dict[str, Any]], n_per_checkpoint: int, checkpoint_labels: list[str],
) -> dict[str, list[float]]:
    """Per checkpoint: rank by score desc, ties broken by ascending det_threshold."""
    n_per_checkpoint = base._validate_positive_int("n_per_checkpoint", n_per_checkpoint)
    base._require_complete_successful_grid(trial_records, "stage A")
    result: dict[str, list[float]] = {}
    for label in checkpoint_labels:
        subset = [r for r in trial_records if r["checkpoint_label"] == label]
        if not subset:
            raise base.CalibrationError(f"stage A selection has no trial records for checkpoint {label!r}")
        best_by_threshold: dict[float, tuple[float, dict[str, Any]]] = {}
        for record in subset:
            score = base._finite_score(record)
            key = record["det_threshold"]
            if key not in best_by_threshold or score > best_by_threshold[key][0]:
                best_by_threshold[key] = (score, record)
        ranked = sorted(best_by_threshold.values(), key=lambda pair: (-pair[0], pair[1]["det_threshold"]))
        if len(ranked) < n_per_checkpoint:
            raise base.CalibrationError(
                f"stage A selection for checkpoint {label!r} requires {n_per_checkpoint} successful unique "
                f"det_threshold candidate(s) from a complete grid, found {len(ranked)}",
            )
        result[label] = [record["det_threshold"] for _, record in ranked[:n_per_checkpoint]]
    return result


def select_best_pairs_by_checkpoint(
    trial_records: list[dict[str, Any]], n_per_checkpoint: int, checkpoint_labels: list[str],
) -> dict[str, list[tuple[float, float]]]:
    """Per checkpoint: rank by score desc, ties broken by ascending edge_threshold then det_threshold."""
    n_per_checkpoint = base._validate_positive_int("n_per_checkpoint", n_per_checkpoint)
    base._require_complete_successful_grid(trial_records, "stage B")
    result: dict[str, list[tuple[float, float]]] = {}
    for label in checkpoint_labels:
        subset = [r for r in trial_records if r["checkpoint_label"] == label]
        if not subset:
            raise base.CalibrationError(f"stage B selection has no trial records for checkpoint {label!r}")
        best_by_pair: dict[tuple[float, float], tuple[float, dict[str, Any]]] = {}
        for record in subset:
            score = base._finite_score(record)
            key = (record["det_threshold"], record["edge_threshold"])
            if key not in best_by_pair or score > best_by_pair[key][0]:
                best_by_pair[key] = (score, record)
        ranked = sorted(
            best_by_pair.values(),
            key=lambda pair: (-pair[0], pair[1]["edge_threshold"], pair[1]["det_threshold"]),
        )
        if len(ranked) < n_per_checkpoint:
            raise base.CalibrationError(
                f"stage B selection for checkpoint {label!r} requires {n_per_checkpoint} successful unique "
                f"(det,edge) candidate(s) from a complete grid, found {len(ranked)}",
            )
        result[label] = [(record["det_threshold"], record["edge_threshold"]) for _, record in ranked[:n_per_checkpoint]]
    return result


def select_best_candidates_by_checkpoint(
    trial_records: list[dict[str, Any]], n_per_checkpoint: int, checkpoint_labels: list[str],
) -> dict[str, list[tuple[float, float, float]]]:
    """Per checkpoint: rank by score desc, ties broken by ascending pool, edge, then det."""
    n_per_checkpoint = base._validate_positive_int("n_per_checkpoint", n_per_checkpoint)
    base._require_complete_successful_grid(trial_records, "stage C")
    result: dict[str, list[tuple[float, float, float]]] = {}
    for label in checkpoint_labels:
        subset = [r for r in trial_records if r["checkpoint_label"] == label]
        if not subset:
            raise base.CalibrationError(f"stage C selection has no trial records for checkpoint {label!r}")
        best_by_candidate: dict[tuple[float, float, float], tuple[float, dict[str, Any]]] = {}
        for record in subset:
            score = base._finite_score(record)
            key = (record["det_threshold"], record["edge_threshold"], record["pool_kernel_um"])
            if key not in best_by_candidate or score > best_by_candidate[key][0]:
                best_by_candidate[key] = (score, record)
        ranked = sorted(
            best_by_candidate.values(),
            key=lambda pair: (-pair[0], pair[1]["pool_kernel_um"], pair[1]["edge_threshold"], pair[1]["det_threshold"]),
        )
        if len(ranked) < n_per_checkpoint:
            raise base.CalibrationError(
                f"stage C selection for checkpoint {label!r} requires {n_per_checkpoint} successful unique "
                f"(det,edge,pool) candidate(s) from a complete grid, found {len(ranked)}",
            )
        result[label] = [
            (record["det_threshold"], record["edge_threshold"], record["pool_kernel_um"])
            for _, record in ranked[:n_per_checkpoint]
        ]
    return result


def select_stage_d_winner(trial_records: list[dict[str, Any]], checkpoint_label_order: list[str]) -> dict[str, Any]:
    """Rank every Stage-D trial by score desc; ties broken by ascending pool, edge, det, then
    checkpoint position in ``checkpoint_label_order`` (earlier-listed checkpoints win ties).

    Unlike Stages A-C, this selects exactly one *global* winner across every
    checkpoint's candidates — Stage D is the workflow's single official
    result.
    """
    base._require_complete_successful_grid(trial_records, "stage D")
    rank_of = {label: index for index, label in enumerate(checkpoint_label_order)}
    ranked = sorted(
        trial_records,
        key=lambda record: (
            -base._finite_score(record),
            record["pool_kernel_um"],
            record["edge_threshold"],
            record["det_threshold"],
            rank_of.get(record["checkpoint_label"], len(rank_of)),
        ),
    )
    return ranked[0]


# =============================================================================
# Diagnostic metrics (informational only — never used for selection)
# =============================================================================

def _as_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _require_finite_nonneg_int(value: Any, *, context: str) -> int:
    """Require ``value`` to be a finite, integer-valued, nonnegative number.

    Missing or malformed diagnostic inputs (edge_tp/edge_fp, division
    counts) are never silently coerced to zero — a malformed successful
    trial must raise rather than publish misleading diagnostics.
    """
    number = _as_finite_number(value)
    if number is None:
        raise base.CalibrationError(f"{context} must be a finite number, got {value!r}")
    if number != int(number):
        raise base.CalibrationError(f"{context} must be an integer-valued number, got {value!r}")
    if number < 0:
        raise base.CalibrationError(f"{context} must be nonnegative, got {value!r}")
    return int(number)


def compute_diagnostics(trial_record: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate diagnostic metrics for one successful trial record.

    Derived entirely from ``summary_metrics``/``per_dataset_metrics`` already
    present on the record — no filesystem or subprocess access — so it can
    be computed both while building a trial's selection summary and as a
    pure post-processing pass over an already-written results file. Never
    used for selection, which is official ``summary_metrics.score`` alone.

    Raises :class:`base.CalibrationError` if a trial has ``summary_metrics``
    but its ``edge_tp``/``edge_fp``/division counts are missing or
    malformed — publishing a diagnostic derived from a silently-zeroed
    malformed input would be misleading.
    """
    summary = trial_record.get("summary_metrics")
    if not isinstance(summary, dict):
        return None
    trial_id = trial_record.get("trial_id", "<unknown>")
    per_dataset = trial_record.get("per_dataset_metrics")
    if not isinstance(per_dataset, list):
        raise base.CalibrationError(
            f"trial {trial_id!r} has summary_metrics but a missing or malformed per_dataset_metrics; "
            "refusing to compute diagnostics",
        )
    predicted_edge_count = 0
    for row in per_dataset:
        if not isinstance(row, dict):
            raise base.CalibrationError(f"trial {trial_id!r} has a malformed per_dataset_metrics row")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise base.CalibrationError(
                f"trial {trial_id!r} per_dataset_metrics row for dataset {row.get('dataset')!r} is missing 'metrics'",
            )
        dataset_name = row.get("dataset", "<unknown>")
        edge_tp = _require_finite_nonneg_int(
            metrics.get("edge_tp"), context=f"trial {trial_id!r} dataset {dataset_name!r} edge_tp",
        )
        edge_fp = _require_finite_nonneg_int(
            metrics.get("edge_fp"), context=f"trial {trial_id!r} dataset {dataset_name!r} edge_fp",
        )
        predicted_edge_count += edge_tp + edge_fp

    division_tp = _require_finite_nonneg_int(summary.get("division_tp"), context=f"trial {trial_id!r} summary division_tp")
    division_fp = _require_finite_nonneg_int(summary.get("division_fp"), context=f"trial {trial_id!r} summary division_fp")
    division_fn = _require_finite_nonneg_int(summary.get("division_fn"), context=f"trial {trial_id!r} summary division_fn")

    fraction = (division_fp / predicted_edge_count) if predicted_edge_count != 0 else 0.0

    return {
        "score": summary.get("score"),
        "edge_jaccard": summary.get("edge_jaccard"),
        "adj_edge_jaccard": summary.get("adj_edge_jaccard"),
        "node_recall": summary.get("node_recall"),
        "division_jaccard": summary.get("division_jaccard"),
        "division_tp": division_tp,
        "division_fp": division_fp,
        "division_fn": division_fn,
        "predicted_edge_count": predicted_edge_count,
        "division_fp_fraction_of_predicted_edges": fraction,
    }


def _require_valid_diagnostics_for_successful_trials(records: list[dict[str, Any]]) -> None:
    """Require every successful trial's diagnostics to be computable before any selection runs.

    Selection itself depends only on ``summary_metrics.score`` — never on
    ``edge_tp``/``edge_fp``/division counts. But a malformed successful trial
    must still block *publishing* a selection, not merely the informational
    diagnostics annotation pass that runs later: this raises the same
    :class:`base.CalibrationError` ``compute_diagnostics`` would, before
    ``build_selection`` computes or returns anything.
    """
    for record in records:
        if isinstance(record, dict) and record.get("status") == "success":
            compute_diagnostics(record)


def _annotate_diagnostics(results_path: Path) -> None:
    """Post-process a results file, attaching ``diagnostics`` to every successful trial.

    Pure read-modify-write over already-written official metrics; never
    touches selection, which is computed before this runs and depends only
    on ``summary_metrics.score``.
    """
    try:
        payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    trials = payload.get("trials")
    if not isinstance(trials, list):
        return
    changed = False
    for record in trials:
        if isinstance(record, dict) and record.get("status") == "success":
            diagnostics = compute_diagnostics(record)
            if diagnostics is not None and record.get("diagnostics") != diagnostics:
                record["diagnostics"] = diagnostics
                changed = True
    if changed:
        base._write_json_atomic(Path(results_path), payload)


def _annotate_audit_provenance(results_path: Path, audit_result: dict[str, Any]) -> None:
    """Ensure the full validated audit provenance is recorded in the report.

    Runs unconditionally after every stage invocation — including one whose
    selection failed because one or more trials failed — so a report never
    records only ``audit_sha256``: it always carries the audit's path, hash,
    schema/split/partition/downsample, dataset count, node count, tolerance,
    and per-kernel counts/pair-counts/fractions actually validated this run.
    """
    try:
        payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    provenance = {
        "path": audit_result["path"],
        "sha256": audit_result["sha256"],
        "schema_version": audit_result["schema_version"],
        "split": audit_result["split"],
        "partition": audit_result["partition"],
        "downsample": audit_result["downsample"],
        "dataset_count": audit_result["dataset_count"],
        "node_count": audit_result["node_count"],
        "tolerance": audit_result["tolerance"],
        "findings": audit_result["findings"],
    }
    if payload.get("audit_sha256") == audit_result["sha256"] and payload.get("audit") == provenance:
        return
    payload["audit_sha256"] = audit_result["sha256"]
    payload["audit"] = provenance
    base._write_json_atomic(Path(results_path), payload)


def _annotate_previous_results(results_path: Path, previous_results_block: dict[str, Any] | None) -> None:
    """Ensure the previous-results provenance is recorded in the report, unconditionally.

    ``previous_results_block`` (resolved path, sha256, previous stage, and
    exact consumed selection) is fully known before any trial runs — Stage
    B/C/D compute it directly from the already-validated prior-stage report,
    before ``run_trials`` is even called. It must never depend on
    ``build_selection`` succeeding: a report with one or more failed trials
    still needs the complete block recorded, so a later resume attempt (or a
    later stage's lineage revalidation) can be checked against it. A no-op
    for Stage A, which has no previous stage (``previous_results_block`` is
    ``None``).
    """
    if previous_results_block is None:
        return
    try:
        payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if payload.get("previous_results") == previous_results_block:
        return
    payload["previous_results"] = previous_results_block
    base._write_json_atomic(Path(results_path), payload)


def _selected_final_trial_summary(trial_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": trial_record["trial_id"],
        "checkpoint_label": trial_record["checkpoint_label"],
        "checkpoint_path": trial_record["checkpoint_path"],
        "checkpoint_sha256": trial_record["checkpoint_sha256"],
        "det_threshold": trial_record["det_threshold"],
        "edge_threshold": trial_record["edge_threshold"],
        "pool_kernel_um": trial_record["pool_kernel_um"],
        "tracking": trial_record["tracking"],
        "split_file": trial_record["split_file"],
        "split": trial_record["split"],
        "official_score": trial_record["summary_metrics"]["score"],
        "summary_metrics": trial_record["summary_metrics"],
        "per_dataset_metrics": trial_record["per_dataset_metrics"],
        "diagnostics": compute_diagnostics(trial_record),
    }


# =============================================================================
# Boundary warnings (informational; never block a run)
# =============================================================================

def stage_a_boundary_warnings(
    selected_det_by_checkpoint: dict[str, list[float]], stage_a_cfg: dict[str, Any],
) -> list[dict[str, str]]:
    bound = float(stage_a_cfg["det_threshold_upper_bound_warning"])
    warnings: list[dict[str, str]] = []
    for label, dets in selected_det_by_checkpoint.items():
        if dets and dets[0] == bound:
            warnings.append({
                "checkpoint_label": label,
                "kind": "det_threshold_upper_bound",
                "message": (
                    f"checkpoint {label}: Stage A's selected winner uses det_threshold={dets[0]:.4f}, at the "
                    f"configured detection upper bound ({bound:.4f}) — consider widening the sweep"
                ),
            })
    return warnings


def stage_b_boundary_warnings(
    selected_pairs_by_checkpoint: dict[str, list[tuple[float, float]]], stage_b_cfg: dict[str, Any],
) -> list[dict[str, str]]:
    bound = float(stage_b_cfg["edge_threshold_lower_bound_warning"])
    warnings: list[dict[str, str]] = []
    for label, pairs in selected_pairs_by_checkpoint.items():
        if pairs and pairs[0][1] == bound:
            warnings.append({
                "checkpoint_label": label,
                "kind": "edge_threshold_lower_bound",
                "message": (
                    f"checkpoint {label}: Stage B's selected winner uses edge_threshold={pairs[0][1]:.4f}, at "
                    f"the configured edge lower bound ({bound:.4f}) — consider widening the sweep"
                ),
            })
    return warnings


# =============================================================================
# Previous-stage result validation (strict — rejects tampering, drift, or an
# incomplete grid; reuses calibrate_inference_thresholds' envelope/grid/
# on-disk revalidation machinery, and additionally requires the previous
# report to have been produced under the audit hash pinned right now)
# =============================================================================

def _check_audit_sha256_matches(report: dict[str, Any], expected_audit_sha256: str, *, stage_label: str) -> None:
    if report.get("audit_sha256") != expected_audit_sha256:
        raise base.CalibrationError(
            f"previous-results ({stage_label}) was produced under a different audit hash than the one "
            f"pinned now ({report.get('audit_sha256')!r} != {expected_audit_sha256!r}); rerun the prior stage",
        )


def _normalize_selection_tuple(arity: int, value: Any) -> float | tuple[float, ...]:
    if arity == 1:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"expected a number, got {value!r}")
        return float(value)
    if not (isinstance(value, (list, tuple)) and len(value) == arity):
        raise ValueError(f"expected a {arity}-element list, got {value!r}")
    return tuple(float(v) for v in value)


def _validate_and_return_selection_by_checkpoint(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    selection_key: str,
    checkpoint_labels: list[str],
    expected_count_per_checkpoint: int,
    arity: int,
    selector: Callable[[list[dict[str, Any]], int, list[str]], dict[str, list[Any]]],
) -> dict[str, list[Any]]:
    recomputed = selector(records, expected_count_per_checkpoint, checkpoint_labels)
    stored = report.get(selection_key)
    if not isinstance(stored, dict):
        raise base.CalibrationError(f"previous-results {selection_key} is missing or malformed")
    if sorted(stored) != sorted(checkpoint_labels):
        raise base.CalibrationError(
            f"previous-results {selection_key} checkpoint labels {sorted(stored)} do not match the "
            f"expected checkpoint labels {sorted(checkpoint_labels)}",
        )
    normalized_stored: dict[str, list[Any]] = {}
    for label, value in stored.items():
        if not isinstance(value, list):
            raise base.CalibrationError(f"previous-results {selection_key}[{label!r}] must be a list")
        try:
            normalized_stored[label] = [_normalize_selection_tuple(arity, v) for v in value]
        except (ValueError, TypeError) as exc:
            raise base.CalibrationError(f"previous-results {selection_key}[{label!r}] is malformed: {exc}") from exc
        if len(normalized_stored[label]) != expected_count_per_checkpoint:
            raise base.CalibrationError(
                f"previous-results {selection_key}[{label!r}] must contain exactly "
                f"{expected_count_per_checkpoint} entries, found {len(normalized_stored[label])}",
            )
    normalized_recomputed = {
        label: [_normalize_selection_tuple(arity, v) for v in values] for label, values in recomputed.items()
    }
    if normalized_stored != normalized_recomputed:
        raise base.CalibrationError(
            f"previous-results {selection_key} does not match a deterministic recomputation from its own "
            "trial records (possible tampering or corruption)",
        )
    return recomputed


def _normalize_selection_by_checkpoint(
    arity: int, checkpoint_labels: list[str], selection: Any, *, context: str,
) -> dict[str, list[Any]]:
    if not isinstance(selection, dict) or sorted(selection) != sorted(checkpoint_labels):
        raise base.CalibrationError(
            f"{context}: selection must be an object keyed by exactly {sorted(checkpoint_labels)}, got "
            f"{sorted(selection) if isinstance(selection, dict) else selection!r}",
        )
    normalized: dict[str, list[Any]] = {}
    for label, values in selection.items():
        if not isinstance(values, list):
            raise base.CalibrationError(f"{context}: selection[{label!r}] must be a list")
        try:
            normalized[label] = [_normalize_selection_tuple(arity, v) for v in values]
        except (ValueError, TypeError) as exc:
            raise base.CalibrationError(f"{context}: selection[{label!r}] is malformed: {exc}") from exc
    return normalized


def _require_selection_matches(
    *, arity: int, checkpoint_labels: list[str], recorded: Any, recomputed: dict[str, list[Any]], context: str,
) -> None:
    normalized_recorded = _normalize_selection_by_checkpoint(arity, checkpoint_labels, recorded, context=context)
    normalized_recomputed = _normalize_selection_by_checkpoint(arity, checkpoint_labels, recomputed, context=context)
    if normalized_recorded != normalized_recomputed:
        raise base.CalibrationError(
            f"lineage validation failed: {context} do not match (possible tampering, drift, or a "
            "substituted but still in-grid selection)",
        )


_PREVIOUS_RESULTS_BLOCK_KEYS = {"path", "sha256", "stage", "consumed_selection"}


def _extract_previous_results_provenance(
    report: dict[str, Any], *, expected_previous_stage: str, stage_label: str,
) -> tuple[Path, str, Any]:
    """Read the ``previous_results`` provenance a report recorded about what it consumed.

    A prior stage's selected inputs must never be inferred solely from the
    current report's own trial records — this returns exactly what was
    recorded at the time the stage ran: the resolved path and sha256 of the
    prior-stage report it was pointed at, and the exact selection it
    consumed from it.
    """
    block = report.get("previous_results")
    if not isinstance(block, dict) or set(block) != _PREVIOUS_RESULTS_BLOCK_KEYS:
        raise base.CalibrationError(
            f"previous-results ({stage_label}) is missing a valid 'previous_results' provenance block "
            f"(must have exactly keys {sorted(_PREVIOUS_RESULTS_BLOCK_KEYS)})",
        )
    if block["stage"] != expected_previous_stage:
        raise base.CalibrationError(
            f"previous-results ({stage_label}) previous_results.stage is {block['stage']!r}, expected "
            f"{expected_previous_stage!r}",
        )
    recorded_path_str = block["path"]
    if not isinstance(recorded_path_str, str) or not recorded_path_str:
        raise base.CalibrationError(f"previous-results ({stage_label}) previous_results.path is invalid")
    recorded_sha256 = block["sha256"]
    if not isinstance(recorded_sha256, str) or not _HEX64_RE.match(recorded_sha256):
        raise base.CalibrationError(f"previous-results ({stage_label}) previous_results.sha256 is invalid")
    return Path(recorded_path_str), recorded_sha256, block["consumed_selection"]


def _selection_to_jsonable(selection: dict[str, list[Any]]) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for label, values in selection.items():
        jsonable: list[Any] = []
        for value in values:
            if isinstance(value, (tuple, list)):
                jsonable.append([float(v) for v in value])
            else:
                jsonable.append(float(value))
        result[label] = jsonable
    return result


def _build_previous_results_provenance(
    previous_results_path: Path, previous_stage_key: str, consumed_selection: dict[str, list[Any]],
) -> dict[str, Any]:
    return {
        "path": str(previous_results_path),
        "sha256": base._require_sha256(previous_results_path),
        "stage": previous_stage_key,
        "consumed_selection": _selection_to_jsonable(consumed_selection),
    }


def _refuse_stale_previous_results_on_resume(
    output_dir: Path, resume: bool, expected_previous_results: dict[str, Any],
) -> None:
    """Refuse to resume cached output whose recorded previous-results identity differs.

    A stage's cached ``calibration_results.json`` records exactly which
    previous-results file (path, hash, stage, consumed selection) it was
    built from — unconditionally, via ``_annotate_previous_results``, so it
    is present even in a report where one or more trials failed. If the
    caller now supplies a different previous-results file, per-trial resume
    validation alone cannot be trusted to catch every case (a different
    upstream run could coincidentally produce the same trial parameters) —
    so this stage-level identity is checked first, before any trial (fresh
    or resumed) may run. A results file that exists but has a missing or
    malformed ``previous_results`` block is never treated as acceptable: it
    is rejected exactly like a mismatched one.
    """
    if not resume:
        return
    results_path = output_dir / "calibration_results.json"
    if not results_path.is_file():
        return
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    recorded = payload.get("previous_results")
    if recorded != expected_previous_results:
        raise base.CalibrationError(
            f"cached results in {output_dir} have a missing, malformed, or different previous_results "
            f"provenance block than the one supplied now (recorded={recorded!r}, "
            f"expected={expected_previous_results!r}); rerun with --no-resume or point --previous-results "
            "at the file that produced the cached output",
        )


def load_stage_a_report_for_stage_b(
    path: Path, config: dict[str, Any], config_path: Path, checkpoints: dict[str, str], data_dir: Path,
    audit_sha256: str,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    report, records, config_sha256 = base._load_and_check_report_envelope(
        path, expected_stage_key="stage_a", config=config, config_path=config_path,
    )
    _check_audit_sha256_matches(report, audit_sha256, stage_label="stage A")
    expected_trials = build_stage_a_trials(config, checkpoints)
    base._validate_report_matches_expected_grid(
        report, records, expected_trials,
        expected_stage_key="stage_a", require_screening=True, config_sha256=config_sha256, data_dir=data_dir,
    )
    n = config["stage_a"]["n_best_det_thresholds_per_checkpoint"]
    labels = config["stage_a"]["checkpoint_labels"]
    selection = _validate_and_return_selection_by_checkpoint(
        report, records, selection_key="selected_det_thresholds_by_checkpoint",
        checkpoint_labels=labels, expected_count_per_checkpoint=n, arity=1,
        selector=select_best_det_thresholds_by_checkpoint,
    )
    return report, selection


def load_stage_b_report_for_stage_c(
    path: Path, config: dict[str, Any], config_path: Path, checkpoints: dict[str, str], data_dir: Path,
    audit_sha256: str,
) -> tuple[dict[str, Any], dict[str, list[tuple[float, float]]]]:
    """Strictly validate a Stage-B results file before Stage C may consume it.

    Recovers the exact Stage-A grid Stage B consumed not by inferring it
    from Stage B's own trial records, but from Stage B's recorded
    ``previous_results`` provenance: the referenced Stage-A report is
    independently revalidated and its selection freshly recomputed, and that
    recomputation is required to equal Stage B's recorded consumed
    selection before the exact Stage-B grid is constructed and checked.
    """
    report, records, config_sha256 = base._load_and_check_report_envelope(
        path, expected_stage_key="stage_b", config=config, config_path=config_path,
    )
    _check_audit_sha256_matches(report, audit_sha256, stage_label="stage B")

    stage_a_path, recorded_stage_a_sha256, recorded_consumed_selection = _extract_previous_results_provenance(
        report, expected_previous_stage="stage_a", stage_label="stage B",
    )
    actual_stage_a_sha256 = base._require_sha256(stage_a_path)
    if actual_stage_a_sha256 != recorded_stage_a_sha256:
        raise base.CalibrationError(
            f"previous-results (stage B) previous_results points at a stage-A report whose current "
            f"sha256 ({actual_stage_a_sha256}) no longer matches what stage B recorded when it consumed "
            f"it ({recorded_stage_a_sha256}); rerun stage B against the current stage-A report",
        )
    _, recomputed_a_selection = load_stage_a_report_for_stage_b(
        stage_a_path, config, config_path, checkpoints, data_dir, audit_sha256,
    )
    _require_selection_matches(
        arity=1, checkpoint_labels=config["stage_a"]["checkpoint_labels"],
        recorded=recorded_consumed_selection, recomputed=recomputed_a_selection,
        context="stage B's recorded consumed_selection vs. a fresh recomputation of the referenced stage-A report",
    )
    det_by_checkpoint = recomputed_a_selection

    expected_trials = build_stage_b_trials(config, checkpoints, det_by_checkpoint)
    base._validate_report_matches_expected_grid(
        report, records, expected_trials,
        expected_stage_key="stage_b", require_screening=True, config_sha256=config_sha256, data_dir=data_dir,
    )
    n = config["stage_b"]["n_best_pairs_per_checkpoint"]
    labels = config["stage_b"]["checkpoint_labels"]
    selection = _validate_and_return_selection_by_checkpoint(
        report, records, selection_key="selected_pairs_by_checkpoint",
        checkpoint_labels=labels, expected_count_per_checkpoint=n, arity=2,
        selector=select_best_pairs_by_checkpoint,
    )
    return report, selection


def load_stage_c_report_for_stage_d(
    path: Path, config: dict[str, Any], config_path: Path, checkpoints: dict[str, str], data_dir: Path,
    audit_sha256: str,
) -> tuple[dict[str, Any], dict[str, list[tuple[float, float, float]]]]:
    """Strictly validate a Stage-C results file before Stage D may consume it.

    Recovers the exact Stage-B grid Stage C consumed from Stage C's recorded
    ``previous_results`` provenance (never inferred from Stage C's own trial
    records). Loading that referenced Stage-B report in turn revalidates its
    own referenced Stage-A report — so this transitively validates the full
    A -> B -> C lineage before Stage D's grid is constructed.
    """
    report, records, config_sha256 = base._load_and_check_report_envelope(
        path, expected_stage_key="stage_c", config=config, config_path=config_path,
    )
    _check_audit_sha256_matches(report, audit_sha256, stage_label="stage C")

    stage_b_path, recorded_stage_b_sha256, recorded_consumed_selection = _extract_previous_results_provenance(
        report, expected_previous_stage="stage_b", stage_label="stage C",
    )
    actual_stage_b_sha256 = base._require_sha256(stage_b_path)
    if actual_stage_b_sha256 != recorded_stage_b_sha256:
        raise base.CalibrationError(
            f"previous-results (stage C) previous_results points at a stage-B report whose current "
            f"sha256 ({actual_stage_b_sha256}) no longer matches what stage C recorded when it consumed "
            f"it ({recorded_stage_b_sha256}); rerun stage C against the current stage-B report",
        )
    _, recomputed_b_selection = load_stage_b_report_for_stage_c(
        stage_b_path, config, config_path, checkpoints, data_dir, audit_sha256,
    )
    _require_selection_matches(
        arity=2, checkpoint_labels=config["stage_b"]["checkpoint_labels"],
        recorded=recorded_consumed_selection, recomputed=recomputed_b_selection,
        context="stage C's recorded consumed_selection vs. a fresh recomputation of the referenced stage-B report",
    )
    pairs_by_checkpoint = recomputed_b_selection

    expected_trials = build_stage_c_trials(config, checkpoints, pairs_by_checkpoint)
    base._validate_report_matches_expected_grid(
        report, records, expected_trials,
        expected_stage_key="stage_c", require_screening=True, config_sha256=config_sha256, data_dir=data_dir,
    )
    n = config["stage_c"]["n_best_candidates_per_checkpoint"]
    labels = config["stage_c"]["checkpoint_labels"]
    selection = _validate_and_return_selection_by_checkpoint(
        report, records, selection_key="selected_candidates_by_checkpoint",
        checkpoint_labels=labels, expected_count_per_checkpoint=n, arity=3,
        selector=select_best_candidates_by_checkpoint,
    )
    return report, selection


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one stage (A-D) of the extended inference-calibration workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to experiments/configs/extended_inference_calibration.json (or a variant).")
    parser.add_argument("--stage", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--checkpoint", action="append", metavar="LABEL=PATH", required=True,
                        help="Repeatable. LABEL must be 7500 or 10000.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-results", type=Path, default=None,
                        help="calibration_results.json from the prior stage (required for B, C, and D).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any existing results in --output-dir and re-run every trial.")
    args = parser.parse_args()

    warnings_list: list[dict[str, str]] = []
    report: dict[str, Any] | None = None

    try:
        config_path = base._resolve_cli_path(args.config)
        config = load_config(config_path)

        # Audit gate — checked before any --checkpoint/--previous-results
        # handling and before any trial (fresh or resumed) may run. The
        # audit's own dataset coverage must exactly match the configured
        # full-fold split's test list.
        full_split_datasets = base._expected_datasets(
            base._resolve_repo_path(config["full_split_file"]), config["full_split"],
        )
        audit_result = audit_validation.load_and_validate_audit(
            config["audit"], full_split_datasets=full_split_datasets,
        )
        audit_sha256 = audit_result["sha256"]

        checkpoints = parse_and_validate_checkpoints(args.checkpoint)
        checkpoints = {label: str(base._resolve_cli_path(path)) for label, path in checkpoints.items()}

        # Checkpoint identity gate — filename, non-symlink, exact checkpoint
        # hash, and exact adjacent config.json hash, all checked before any
        # trial (fresh or resumed) may run.
        validate_pinned_checkpoints(config, checkpoints)

        data_dir = base._resolve_cli_path(args.data_dir)
        output_dir = base._resolve_cli_path(args.output_dir)
        previous_results_path = base._resolve_cli_path(args.previous_results) if args.previous_results else None
        resume = not args.no_resume

        stage_key = _STAGE_KEY_BY_ARG[args.stage]
        # Known (Stage B/C/D) or None (Stage A, no previous stage) before any
        # trial runs, and recorded unconditionally via _annotate_previous_results
        # regardless of whether selection later succeeds or fails.
        previous_results_block: dict[str, Any] | None = None

        if args.stage == "A":
            trials = build_stage_a_trials(config, checkpoints)

            def build_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
                _require_valid_diagnostics_for_successful_trials(records)
                n = config["stage_a"]["n_best_det_thresholds_per_checkpoint"]
                labels = config["stage_a"]["checkpoint_labels"]
                selection = select_best_det_thresholds_by_checkpoint(records, n, labels)
                warnings_list.extend(stage_a_boundary_warnings(selection, config["stage_a"]))
                return {
                    "selected_det_thresholds_by_checkpoint": dict(selection),
                    "warnings": list(warnings_list),
                }
        elif args.stage == "B":
            if previous_results_path is None:
                raise base.CalibrationError("--previous-results (Stage A results) is required for Stage B")
            _, det_by_checkpoint = load_stage_a_report_for_stage_b(
                previous_results_path, config, config_path, checkpoints, data_dir, audit_sha256,
            )
            previous_results_block = _build_previous_results_provenance(
                previous_results_path, "stage_a", det_by_checkpoint,
            )
            _refuse_stale_previous_results_on_resume(output_dir, resume, previous_results_block)
            trials = build_stage_b_trials(config, checkpoints, det_by_checkpoint)

            def build_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
                _require_valid_diagnostics_for_successful_trials(records)
                n = config["stage_b"]["n_best_pairs_per_checkpoint"]
                labels = config["stage_b"]["checkpoint_labels"]
                selection = select_best_pairs_by_checkpoint(records, n, labels)
                warnings_list.extend(stage_b_boundary_warnings(selection, config["stage_b"]))
                return {
                    "selected_pairs_by_checkpoint": {k: [list(p) for p in v] for k, v in selection.items()},
                    "warnings": list(warnings_list),
                }
        elif args.stage == "C":
            if previous_results_path is None:
                raise base.CalibrationError("--previous-results (Stage B results) is required for Stage C")
            _, pairs_by_checkpoint = load_stage_b_report_for_stage_c(
                previous_results_path, config, config_path, checkpoints, data_dir, audit_sha256,
            )
            previous_results_block = _build_previous_results_provenance(
                previous_results_path, "stage_b", pairs_by_checkpoint,
            )
            _refuse_stale_previous_results_on_resume(output_dir, resume, previous_results_block)
            trials = build_stage_c_trials(config, checkpoints, pairs_by_checkpoint)

            def build_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
                _require_valid_diagnostics_for_successful_trials(records)
                n = config["stage_c"]["n_best_candidates_per_checkpoint"]
                labels = config["stage_c"]["checkpoint_labels"]
                selection = select_best_candidates_by_checkpoint(records, n, labels)
                return {
                    "selected_candidates_by_checkpoint": {k: [list(c) for c in v] for k, v in selection.items()},
                }
        else:
            if previous_results_path is None:
                raise base.CalibrationError("--previous-results (Stage C results) is required for Stage D")
            _, candidates_by_checkpoint = load_stage_c_report_for_stage_d(
                previous_results_path, config, config_path, checkpoints, data_dir, audit_sha256,
            )
            previous_results_block = _build_previous_results_provenance(
                previous_results_path, "stage_c", candidates_by_checkpoint,
            )
            _refuse_stale_previous_results_on_resume(output_dir, resume, previous_results_block)
            trials = build_stage_d_trials(config, checkpoints, candidates_by_checkpoint)

            def build_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
                _require_valid_diagnostics_for_successful_trials(records)
                best = select_stage_d_winner(records, config["stage_d"]["checkpoint_labels"])
                return {
                    "selected_final_trial": _selected_final_trial_summary(best),
                }

        selection_error: base.CalibrationError | None = None
        try:
            report = base.run_trials(
                trials, data_dir, output_dir, config_path, stage_key,
                resume=resume, build_selection=build_selection,
            )
        except base.CalibrationError as exc:
            selection_error = exc

        results_path = output_dir / "calibration_results.json"
        # Order matters: previous-results provenance and audit provenance are
        # both fully known before any trial ran and must survive a selection
        # or diagnostics failure, so they are written first and unconditionally.
        # Diagnostics are computed last, over already-written official metrics —
        # if a malformed successful trial makes diagnostics computation raise,
        # that failure must not erase or prevent the provenance already recorded.
        _annotate_previous_results(results_path, previous_results_block)
        _annotate_audit_provenance(results_path, audit_result)
        _annotate_diagnostics(results_path)

        if selection_error is not None:
            raise selection_error
    except (base.CalibrationError, audit_validation.AuditValidationError, OSError, json.JSONDecodeError, KeyError) as exc:
        parser.error(str(exc))
        return

    for warning in warnings_list:
        print(f"WARNING: {warning['message']}", flush=True)

    screening_note = (
        " (SCREENING scores on a validation subset, not final validation scores)"
        if args.stage in ("A", "B", "C") else ""
    )
    summary = {
        "results_path": report["results_path"],
        "n_trials": len(report["trials"]),
        "n_succeeded": sum(1 for r in report["trials"] if r["status"] == "success"),
        "n_failed": sum(1 for r in report["trials"] if r["status"] == "failed"),
    }
    for key in (
        "selected_det_thresholds_by_checkpoint", "selected_pairs_by_checkpoint",
        "selected_candidates_by_checkpoint", "selected_final_trial",
    ):
        if key in report:
            summary[key] = report[key]
    print(json.dumps(summary, indent=2, default=str))
    print(f"Stage {args.stage} complete.{screening_note}")


if __name__ == "__main__":
    main()
