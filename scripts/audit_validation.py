#!/usr/bin/env python
"""Strict validation for the preserved ground-truth detection-spacing audit.

The extended inference-calibration workflow pins the audit's repository
path, sha256 hash, and observed findings in its config (see
``experiments/configs/extended_inference_calibration.json``). This module
loads the audit, requires its bytes to still match the pinned hash, requires
its structure to be strict and unambiguous JSON (no NaN/Infinity tokens, no
duplicate keys), requires its dataset coverage to exactly match the
configured full-fold test partition, requires its aggregate figures to be
exactly reproducible from its own per-dataset rows, and requires its
findings to still match what was pinned as reviewed and compatible with the
plan — all before any calibration subprocess is allowed to run. It never
shells out and never imports model or training code, so these checks are
always cheap and safe to run first.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCHEMA_VERSION = 1
EXPECTED_SPLIT = 0
EXPECTED_PARTITION = "test"
EXPECTED_DOWNSAMPLE = [1, 4, 4]

_REQUIRED_AUDIT_CONFIG_KEYS = {
    "path", "sha256", "primary_pool_kernel_um", "primary_voxel_kernel",
    "secondary_pool_kernel_um", "secondary_voxel_kernel", "quantized_collision_fraction_max",
    "dataset_count", "node_count", "observed", "tolerance",
}
_REQUIRED_OBSERVED_KEYS = {"primary", "secondary"}
_REQUIRED_OBSERVED_ENTRY_KEYS = {"count", "pair_count", "fraction"}

_FRACTION_EPSILON = 1e-9


class AuditValidationError(RuntimeError):
    """Raised when the preserved audit's hash, structure, or findings are incompatible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_nonfinite_constant(token: str) -> float:
    raise AuditValidationError(f"audit file contains a non-finite JSON constant: {token}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise AuditValidationError(f"audit file contains a duplicate JSON key: {key!r}")
        seen[key] = value
    return seen


def _require_nonneg_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditValidationError(f"{context} must be an integer, got {value!r}")
    if value < 0:
        raise AuditValidationError(f"{context} must be nonnegative, got {value!r}")
    return value


def _require_unit_fraction(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditValidationError(f"{context} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise AuditValidationError(f"{context} must be finite, got {value!r}")
    if not (0.0 <= value <= 1.0):
        raise AuditValidationError(f"{context} must be in [0, 1], got {value!r}")
    return value


def _find_requested_entry(rows: Any, requested_um: float, *, context: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise AuditValidationError(f"{context} must be a list")
    matches = [
        row for row in rows
        if isinstance(row, dict) and isinstance(row.get("requested_um"), (int, float))
        and not isinstance(row.get("requested_um"), bool)
        and float(row["requested_um"]) == requested_um
    ]
    if len(matches) != 1:
        raise AuditValidationError(
            f"{context} must contain exactly one entry for requested_um={requested_um}, found {len(matches)}",
        )
    return matches[0]


def _collect_dataset_rows(audit: dict[str, Any], requested_um: float) -> dict[str, dict[str, Any]]:
    datasets = audit.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise AuditValidationError("audit 'datasets' must be a non-empty object")
    rows: dict[str, dict[str, Any]] = {}
    for name, entry in datasets.items():
        if not isinstance(entry, dict):
            raise AuditValidationError(f"audit dataset {name!r} entry is malformed")
        row = _find_requested_entry(
            entry.get("quantized_target_collision_risk"), requested_um,
            context=f"audit dataset {name!r} quantized_target_collision_risk",
        )
        kernel = row.get("voxel_kernel")
        if not (isinstance(kernel, list) and len(kernel) == 3 and all(isinstance(v, int) and not isinstance(v, bool) for v in kernel)):
            raise AuditValidationError(
                f"audit dataset {name!r} voxel_kernel for requested_um={requested_um} is malformed: {kernel!r}",
            )
        rows[name] = {
            "voxel_kernel": kernel,
            "count": _require_nonneg_int(row.get("count"), context=f"audit dataset {name!r} count (requested_um={requested_um})"),
            "pair_count": _require_nonneg_int(
                row.get("pair_count"), context=f"audit dataset {name!r} pair_count (requested_um={requested_um})",
            ),
            "fraction": _require_unit_fraction(
                row.get("fraction"), context=f"audit dataset {name!r} fraction (requested_um={requested_um})",
            ),
        }
    return rows


def load_and_validate_audit(
    audit_config: dict[str, Any], *, full_split_datasets: list[str], repo_root: Path | None = None,
) -> dict[str, Any]:
    """Strictly validate the preserved audit against a pinned path/hash/findings.

    ``audit_config`` is the ``"audit"`` object from the extended-calibration
    config: ``path`` (repository-relative), ``sha256``,
    ``primary_pool_kernel_um``/``primary_voxel_kernel`` (the calibrated
    baseline's pooling kernel — required to show zero quantized target
    collisions), ``secondary_pool_kernel_um``/``secondary_voxel_kernel`` (the
    swept alternative — allowed a small quantized-collision fraction, capped
    by ``quantized_collision_fraction_max``/``tolerance``, which must agree),
    ``dataset_count``/``node_count`` (the audit's own reviewed totals), and
    ``observed`` (the reviewed ``primary``/``secondary`` count/pair_count/
    fraction values). ``full_split_datasets`` is the exact test-dataset list
    of the configured full fold split — the audit's own dataset coverage
    must equal it exactly.

    Raises :class:`AuditValidationError` — before any subprocess may run —
    if the file is missing, its bytes no longer match the pinned sha256, it
    is not strict unambiguous JSON (no NaN/Infinity tokens, no duplicate
    keys), or its findings are no longer compatible with the reviewed plan:
    wrong schema/split/partition/downsample, a dataset-coverage mismatch
    against the full-fold split, any collapsed or duplicate target nodes,
    aggregate figures that are not exactly reproduced by their own
    per-dataset rows, a voxel kernel that disagrees with what's pinned,
    nonzero quantized collisions at the primary kernel, a quantized-
    collision fraction at the secondary kernel above the configured
    tolerance, or any observed value that no longer matches what is pinned
    in the config.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    missing_keys = sorted(_REQUIRED_AUDIT_CONFIG_KEYS - set(audit_config))
    if missing_keys:
        raise AuditValidationError(f"audit config missing required keys: {missing_keys}")

    tolerance = audit_config["tolerance"]
    fraction_max = audit_config["quantized_collision_fraction_max"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not (0.0 <= float(tolerance) < 1.0):
        raise AuditValidationError(f"audit config 'tolerance' must be in [0, 1), got {tolerance!r}")
    if float(tolerance) != float(fraction_max):
        raise AuditValidationError(
            f"audit config 'tolerance' ({tolerance!r}) must equal 'quantized_collision_fraction_max' "
            f"({fraction_max!r})",
        )
    fraction_max = float(fraction_max)

    if not isinstance(full_split_datasets, list) or not full_split_datasets:
        raise AuditValidationError("full_split_datasets must be a non-empty list")

    observed = audit_config["observed"]
    if not isinstance(observed, dict) or set(observed) != _REQUIRED_OBSERVED_KEYS:
        raise AuditValidationError(f"audit config 'observed' must have exactly keys {_REQUIRED_OBSERVED_KEYS}")
    for label, entry in observed.items():
        if not isinstance(entry, dict) or set(entry) != _REQUIRED_OBSERVED_ENTRY_KEYS:
            raise AuditValidationError(f"audit config observed[{label!r}] must have exactly keys {_REQUIRED_OBSERVED_ENTRY_KEYS}")

    pinned_dataset_count = _require_nonneg_int(audit_config["dataset_count"], context="audit config 'dataset_count'")
    pinned_node_count = _require_nonneg_int(audit_config["node_count"], context="audit config 'node_count'")

    audit_path = Path(audit_config["path"])
    resolved_path = audit_path if audit_path.is_absolute() else (root / audit_path)
    if not resolved_path.is_file():
        raise AuditValidationError(f"pinned audit file not found: {resolved_path}")

    actual_sha256 = sha256_file(resolved_path)
    expected_sha256 = audit_config["sha256"]
    if actual_sha256 != expected_sha256:
        raise AuditValidationError(
            "audit file sha256 does not match the pinned value in the calibration config (the audit "
            f"file may have been regenerated or edited): {actual_sha256} != {expected_sha256}",
        )

    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditValidationError(f"cannot read audit file {resolved_path}: {exc}") from exc
    try:
        audit = json.loads(raw_text, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_nonfinite_constant)
    except json.JSONDecodeError as exc:
        raise AuditValidationError(f"audit file is not valid JSON: {resolved_path}: {exc}") from exc
    if not isinstance(audit, dict):
        raise AuditValidationError(f"audit file must be a JSON object: {resolved_path}")

    if audit.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise AuditValidationError(f"audit schema_version {audit.get('schema_version')!r} != {EXPECTED_SCHEMA_VERSION}")
    if audit.get("split") != EXPECTED_SPLIT:
        raise AuditValidationError(f"audit split {audit.get('split')!r} != {EXPECTED_SPLIT}")
    if audit.get("partition") != EXPECTED_PARTITION:
        raise AuditValidationError(f"audit partition {audit.get('partition')!r} != {EXPECTED_PARTITION!r}")
    if audit.get("downsample") != EXPECTED_DOWNSAMPLE:
        raise AuditValidationError(f"audit downsample {audit.get('downsample')!r} != {EXPECTED_DOWNSAMPLE}")

    aggregate = audit.get("aggregate")
    if not isinstance(aggregate, dict):
        raise AuditValidationError("audit 'aggregate' must be an object")
    if aggregate.get("collapsed_node_count") != 0:
        raise AuditValidationError(
            f"audit aggregate.collapsed_node_count must be 0, got {aggregate.get('collapsed_node_count')!r}",
        )
    if aggregate.get("duplicate_target_voxel_count") != 0:
        raise AuditValidationError(
            "audit aggregate.duplicate_target_voxel_count must be 0, got "
            f"{aggregate.get('duplicate_target_voxel_count')!r}",
        )

    datasets = audit.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise AuditValidationError("audit 'datasets' must be a non-empty object")
    if sorted(datasets) != sorted(full_split_datasets):
        missing = sorted(set(full_split_datasets) - set(datasets))
        extra = sorted(set(datasets) - set(full_split_datasets))
        raise AuditValidationError(
            "audit dataset coverage does not exactly match the configured full-fold split test list: "
            f"missing={missing}, extra={extra}",
        )

    audit_dataset_count = audit.get("dataset_count")
    if (
        isinstance(audit_dataset_count, bool) or not isinstance(audit_dataset_count, int)
        or audit_dataset_count != len(datasets)
    ):
        raise AuditValidationError(
            f"audit dataset_count {audit_dataset_count!r} does not match len(datasets)={len(datasets)}",
        )
    if audit_dataset_count != pinned_dataset_count:
        raise AuditValidationError(
            f"audit dataset_count {audit_dataset_count!r} does not match the pinned config value {pinned_dataset_count!r}",
        )

    agg_node_count = aggregate.get("node_count")
    if isinstance(agg_node_count, bool) or not isinstance(agg_node_count, int) or agg_node_count < 0:
        raise AuditValidationError(f"audit aggregate.node_count must be a nonnegative integer, got {agg_node_count!r}")
    if agg_node_count != pinned_node_count:
        raise AuditValidationError(
            f"audit aggregate.node_count {agg_node_count!r} does not match the pinned config value {pinned_node_count!r}",
        )

    dataset_node_total = 0
    for name, entry in datasets.items():
        if not isinstance(entry, dict):
            raise AuditValidationError(f"audit dataset {name!r} entry is malformed")
        dataset_node_total += _require_nonneg_int(entry.get("node_count"), context=f"audit dataset {name!r} node_count")
    if dataset_node_total != agg_node_count:
        raise AuditValidationError(
            f"audit dataset-level node_count totals ({dataset_node_total}) do not reproduce "
            f"aggregate.node_count ({agg_node_count})",
        )

    findings: dict[str, Any] = {}
    for label, um_key, kernel_key in (
        ("primary", "primary_pool_kernel_um", "primary_voxel_kernel"),
        ("secondary", "secondary_pool_kernel_um", "secondary_voxel_kernel"),
    ):
        requested_um = audit_config[um_key]
        if isinstance(requested_um, bool) or not isinstance(requested_um, (int, float)):
            raise AuditValidationError(f"audit config {um_key} must be a number, got {requested_um!r}")
        requested_um = float(requested_um)
        expected_kernel = audit_config[kernel_key]
        if not (isinstance(expected_kernel, list) and len(expected_kernel) == 3):
            raise AuditValidationError(f"audit config {kernel_key} must be a 3-element list, got {expected_kernel!r}")
        expected_kernel = [int(v) for v in expected_kernel]

        agg_row = _find_requested_entry(
            aggregate.get("quantized_target_collision_risk"), requested_um,
            context="audit aggregate.quantized_target_collision_risk",
        )
        dataset_rows = _collect_dataset_rows(audit, requested_um)
        mismatched = {name: r["voxel_kernel"] for name, r in dataset_rows.items() if r["voxel_kernel"] != expected_kernel}
        if mismatched:
            raise AuditValidationError(
                f"audit datasets do not unanimously report voxel_kernel={expected_kernel} for "
                f"requested_um={requested_um} ({label} pooling kernel): {mismatched}",
            )

        count = _require_nonneg_int(
            agg_row.get("count"), context=f"audit aggregate count (requested_um={requested_um})",
        )
        pair_count = _require_nonneg_int(
            agg_row.get("pair_count"), context=f"audit aggregate pair_count (requested_um={requested_um})",
        )
        fraction = _require_unit_fraction(
            agg_row.get("fraction"), context=f"audit aggregate fraction (requested_um={requested_um})",
        )

        dataset_count_total = sum(r["count"] for r in dataset_rows.values())
        dataset_pair_total = sum(r["pair_count"] for r in dataset_rows.values())
        if dataset_count_total != count:
            raise AuditValidationError(
                f"audit dataset-level count totals ({dataset_count_total}) do not reproduce aggregate "
                f"count ({count}) for requested_um={requested_um} ({label} pooling kernel)",
            )
        if dataset_pair_total != pair_count:
            raise AuditValidationError(
                f"audit dataset-level pair_count totals ({dataset_pair_total}) do not reproduce aggregate "
                f"pair_count ({pair_count}) for requested_um={requested_um} ({label} pooling kernel)",
            )

        expected_fraction = (count / agg_node_count) if agg_node_count else 0.0
        if abs(fraction - expected_fraction) > _FRACTION_EPSILON:
            raise AuditValidationError(
                f"audit aggregate fraction ({fraction!r}) does not equal count/node_count "
                f"({expected_fraction!r}) for requested_um={requested_um} ({label} pooling kernel)",
            )

        if label == "primary":
            # The primary kernel is the calibrated baseline's own kernel: the
            # reviewed plan requires it to show zero quantized collisions,
            # with no tolerance.
            if count != 0 or pair_count != 0:
                raise AuditValidationError(
                    f"audit primary pooling kernel (requested_um={requested_um}) has nonzero quantized "
                    f"target collisions (count={count}, pair_count={pair_count}); refusing to proceed",
                )
        elif fraction > fraction_max:
            raise AuditValidationError(
                f"audit secondary pooling kernel (requested_um={requested_um}) quantized-collision "
                f"fraction {fraction!r} exceeds the configured tolerance {fraction_max!r} "
                f"(count={count}, pair_count={pair_count})",
            )

        observed_pin = observed[label]
        pinned_triple = (
            _require_nonneg_int(observed_pin["count"], context=f"audit config observed[{label!r}].count"),
            _require_nonneg_int(observed_pin["pair_count"], context=f"audit config observed[{label!r}].pair_count"),
            _require_unit_fraction(observed_pin["fraction"], context=f"audit config observed[{label!r}].fraction"),
        )
        if pinned_triple != (count, pair_count, fraction):
            raise AuditValidationError(
                f"observed {label} audit values (count={count}, pair_count={pair_count}, fraction={fraction!r}) "
                f"do not match those pinned in the config (observed[{label!r}]={pinned_triple!r})",
            )

        findings[label] = {
            "requested_um": requested_um, "voxel_kernel": expected_kernel,
            "count": count, "pair_count": pair_count, "fraction": fraction,
        }

    return {
        "path": str(resolved_path),
        "sha256": actual_sha256,
        "schema_version": audit["schema_version"],
        "split": audit["split"],
        "partition": audit["partition"],
        "downsample": audit["downsample"],
        "dataset_count": audit_dataset_count,
        "node_count": agg_node_count,
        "aggregate_collapsed_node_count": aggregate["collapsed_node_count"],
        "aggregate_duplicate_target_voxel_count": aggregate["duplicate_target_voxel_count"],
        "quantized_collision_fraction_max": fraction_max,
        "tolerance": float(tolerance),
        "findings": findings,
    }
