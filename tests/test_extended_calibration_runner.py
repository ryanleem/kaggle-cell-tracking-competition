"""Tests for scripts/calibrate_extended_inference.py and scripts/audit_validation.py.

All prediction/evaluation subprocesses are faked by monkeypatching
``calibrate_inference_thresholds._stream_subprocess`` — no real inference or
training ever runs. Fake predictions write placeholder ``.geff`` files into
the exact scratch directory the runner computes; fake evaluations write a
strict-schema ``metrics.json`` whose ``summary_metrics.score`` is looked up
by trial_id from an injectable ``scores_by_trial_id`` map, so tests can pin
exactly which trial "wins" a selection.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import audit_validation as av
import calibrate_extended_inference as ext
import calibrate_inference_thresholds as base

REPO_ROOT = Path(__file__).parent.parent
REAL_CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "extended_inference_calibration.json"
REAL_AUDIT_PATH = REPO_ROOT / "experiments" / "audits" / "detection_spacing_fold0_downsample_1x4x4.json"
ORIGINAL_CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "inference_threshold_calibration.json"

SUBSET_NAMES = ["dsA", "dsB", "dsC", "dsD", "dsE", "dsF"]
FULL_NAMES = SUBSET_NAMES + ["dsG"]
CHECKPOINT_LABELS = ["7500", "10000"]


# =============================================================================
# Synthetic fixture builders
# =============================================================================

def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_checkpoints(base_dir: Path, labels: list[str]) -> dict[str, Path]:
    """Both checkpoints share one directory and one adjacent config.json, matching production."""
    ckpt_dir = base_dir / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps({"pool_kernel_um": 5.0}), encoding="utf-8")
    paths = {}
    for label in labels:
        p = ckpt_dir / f"edge_predictor_iter_{label}.pth"
        p.write_bytes(f"fake-checkpoint-{label}".encode())
        paths[label] = p
    return paths


def _checkpoint_pins_block(paths: dict[str, Path]) -> dict:
    config_paths = {p.parent / "config.json" for p in paths.values()}
    assert len(config_paths) == 1, "test fixture checkpoints must share one adjacent config.json"
    adjacent_sha256 = hashlib.sha256(next(iter(config_paths)).read_bytes()).hexdigest()
    pins = {"adjacent_config_sha256": adjacent_sha256}
    for label, path in paths.items():
        pins[label] = {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return pins


def _make_split_file(path: Path, names: list[str], split: int = 0) -> None:
    _write_json(path, [{"split": split, "train": [], "test": list(names)}])


def _quant_row(um: float, kernel: list[int], count: int = 0, pair_count: int = 0, node_count: int = 1000) -> dict:
    return {
        "requested_um": um, "count": count, "pair_count": pair_count,
        "fraction": (count / node_count) if node_count else 0.0, "voxel_kernel": list(kernel),
    }


def _provenance_from_payload(payload: dict) -> dict:
    """Derive dataset_count/node_count/observed pin values from a raw audit payload."""
    datasets = payload["datasets"]
    node_count = sum(ds["node_count"] for ds in datasets.values())
    agg_rows = payload["aggregate"]["quantized_target_collision_risk"]
    row5 = next(r for r in agg_rows if r["requested_um"] == 5.0)
    row6 = next(r for r in agg_rows if r["requested_um"] == 6.0)
    return {
        "dataset_count": len(datasets),
        "node_count": node_count,
        "observed": {
            "primary": {"count": row5["count"], "pair_count": row5["pair_count"], "fraction": row5["fraction"]},
            "secondary": {"count": row6["count"], "pair_count": row6["pair_count"], "fraction": row6["fraction"]},
        },
    }


def _make_audit(path: Path, *, secondary_count: int = 0, secondary_pair_count: int = 0,
                 collapsed: int = 0, duplicate: int = 0, schema_version: int = 1,
                 split: int = 0, partition: str = "test", downsample: list[int] | None = None,
                 dataset_names: list[str] | None = None, per_dataset_node_count: int = 1000) -> tuple[str, dict]:
    downsample = downsample if downsample is not None else [1, 4, 4]
    names = dataset_names if dataset_names is not None else FULL_NAMES
    node_count = per_dataset_node_count * len(names)

    def dataset_entry(is_last: bool) -> dict:
        sc = secondary_count if is_last else 0
        sp = secondary_pair_count if is_last else 0
        return {
            "node_count": per_dataset_node_count,
            "quantized_target_collision_risk": [
                _quant_row(5.0, [3, 3, 3], 0, 0, per_dataset_node_count),
                _quant_row(6.0, [5, 5, 5], sc, sp, per_dataset_node_count),
            ],
        }

    datasets = {name: dataset_entry(i == len(names) - 1) for i, name in enumerate(names)}
    payload = {
        "schema_version": schema_version, "split": split, "partition": partition, "downsample": downsample,
        "dataset_count": len(names),
        "aggregate": {
            "collapsed_node_count": collapsed, "duplicate_target_voxel_count": duplicate,
            "node_count": node_count,
            "quantized_target_collision_risk": [
                {"requested_um": 5.0, "count": 0, "pair_count": 0, "fraction": 0.0},
                {"requested_um": 6.0, "count": secondary_count, "pair_count": secondary_pair_count,
                 "fraction": (secondary_count / node_count) if node_count else 0.0},
            ],
        },
        "datasets": datasets,
    }
    _write_json(path, payload)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return sha, _provenance_from_payload(payload)


def _audit_config_block(audit_path: Path, audit_sha256: str, provenance: dict, *, fraction_max: float = 0.001) -> dict:
    return {
        "path": str(audit_path), "sha256": audit_sha256,
        "primary_pool_kernel_um": 5.0, "primary_voxel_kernel": [3, 3, 3],
        "secondary_pool_kernel_um": 6.0, "secondary_voxel_kernel": [5, 5, 5],
        "quantized_collision_fraction_max": fraction_max,
        "dataset_count": provenance["dataset_count"],
        "node_count": provenance["node_count"],
        "observed": provenance["observed"],
        "tolerance": fraction_max,
    }


def _make_config(
    tmp_path: Path, *, audit_path: Path, audit_sha256: str, audit_provenance: dict, checkpoint_pins: dict,
    subset_split_path: Path, full_split_path: Path, fraction_max: float = 0.001,
    name: str = "extended_config.json", mutate=None,
) -> Path:
    config = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
    config["subset_split_file"] = str(subset_split_path)
    config["subset_dataset_names"] = SUBSET_NAMES
    config["full_split_file"] = str(full_split_path)
    config["audit"] = _audit_config_block(audit_path, audit_sha256, audit_provenance, fraction_max=fraction_max)
    config["checkpoint_pins"] = checkpoint_pins
    if mutate is not None:
        mutate(config)
    path = tmp_path / name
    _write_json(path, config)
    return path


class Environment:
    """A complete, self-contained set of fixture files for one test.

    ``calibrate_extended_inference.load_config()`` hard-locks the reviewed
    checkpoint and audit *identities* against module-level constants (see
    ``EXPECTED_CHECKPOINT_PINS``/``EXPECTED_AUDIT_*``) so a config cannot
    authorize a different checkpoint or audit merely by supplying another
    well-formed hash. Since these unit tests use synthetic checkpoints/audit
    files (never the real ones), a ``monkeypatch`` fixture is required and
    used to point those constants at this environment's own synthetic
    identities for the duration of the test — the few dedicated tests that
    exercise the *real* reviewed constants do not use this fixture.
    """

    def __init__(
        self, tmp_path: Path, monkeypatch, *,
        secondary_count: int = 0, secondary_pair_count: int = 0, mutate_config=None,
    ):
        self.tmp_path = tmp_path
        self.data_dir = tmp_path / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_paths = _make_checkpoints(tmp_path, CHECKPOINT_LABELS)
        self.checkpoint_paths = checkpoint_paths
        self.checkpoints = {label: str(path) for label, path in checkpoint_paths.items()}
        self.checkpoint_pins = _checkpoint_pins_block(checkpoint_paths)

        self.audit_path = tmp_path / "audit.json"
        self.audit_sha256, self.audit_provenance = _make_audit(
            self.audit_path, secondary_count=secondary_count, secondary_pair_count=secondary_pair_count,
        )
        self.subset_split_path = tmp_path / "subset_split.json"
        _make_split_file(self.subset_split_path, SUBSET_NAMES)
        self.full_split_path = tmp_path / "full_split.json"
        _make_split_file(self.full_split_path, FULL_NAMES)

        monkeypatch.setattr(ext, "EXPECTED_CHECKPOINT_PINS", self.checkpoint_pins)
        monkeypatch.setattr(ext, "EXPECTED_AUDIT_PATH", str(self.audit_path))
        monkeypatch.setattr(ext, "EXPECTED_AUDIT_SHA256", self.audit_sha256)
        monkeypatch.setattr(ext, "EXPECTED_AUDIT_DATASET_COUNT", self.audit_provenance["dataset_count"])
        monkeypatch.setattr(ext, "EXPECTED_AUDIT_NODE_COUNT", self.audit_provenance["node_count"])
        monkeypatch.setattr(ext, "EXPECTED_AUDIT_OBSERVED", self.audit_provenance["observed"])

        self.config_path = _make_config(
            tmp_path, audit_path=self.audit_path, audit_sha256=self.audit_sha256,
            audit_provenance=self.audit_provenance, checkpoint_pins=self.checkpoint_pins,
            subset_split_path=self.subset_split_path, full_split_path=self.full_split_path,
            mutate=mutate_config,
        )
        self.config = ext.load_config(self.config_path)
        self.full_split_datasets = sorted(FULL_NAMES)

    def remake(self, mutate, name: str = "mutated.json") -> Path:
        return _make_config(
            self.tmp_path, audit_path=self.audit_path, audit_sha256=self.audit_sha256,
            audit_provenance=self.audit_provenance, checkpoint_pins=self.checkpoint_pins,
            subset_split_path=self.subset_split_path, full_split_path=self.full_split_path,
            name=name, mutate=mutate,
        )


@pytest.fixture
def env(tmp_path: Path, monkeypatch) -> Environment:
    return Environment(tmp_path, monkeypatch)


# =============================================================================
# Fake subprocess harness (no real inference/training ever runs)
# =============================================================================

def _parse_flags(command: list[str]) -> dict[str, str]:
    """Parse ``--flag value`` and bare boolean ``--flag`` tokens (e.g. ``--strict``)."""
    args = command[2:]
    result: dict[str, str] = {}
    index = 0
    while index < len(args):
        flag = args[index].lstrip("-")
        has_value = index + 1 < len(args) and not args[index + 1].startswith("--")
        if has_value:
            result[flag] = args[index + 1]
            index += 2
        else:
            result[flag] = True
            index += 1
    return result


def install_fake_subprocess(
    monkeypatch, *,
    scores_by_trial_id: dict[str, float] | None = None,
    default_score: float = 0.6,
    fail_predict_trial_ids: frozenset[str] = frozenset(),
    fail_evaluate_trial_ids: frozenset[str] = frozenset(),
    missing_dataset_trial_ids: frozenset[str] = frozenset(),
    per_dataset_overrides: dict[str, dict] | None = None,
    summary_overrides: dict[str, dict] | None = None,
    heartbeat_log: list[tuple[str, float]] | None = None,
) -> None:
    scores_by_trial_id = scores_by_trial_id or {}
    per_dataset_overrides = per_dataset_overrides or {}
    summary_overrides = summary_overrides or {}

    def fake_stream_subprocess(command, stdout_path, stderr_path, *, on_heartbeat=lambda e: None, heartbeat_interval=60.0):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        on_heartbeat(1.0)
        if heartbeat_log is not None:
            heartbeat_log.append((str(command[1]), 1.0))
        assert command[0] == str(sys.executable)
        script = command[1]
        if script == str(base.PREDICT_SCRIPT):
            flags = _parse_flags(command)
            method = flags["method"]
            assert method.startswith("calib_")
            trial_id = method[len("calib_"):]
            split = int(flags["split"])
            scratch_dir = base._prediction_scratch_dir(base.REPO_ROOT, base._username(), method, split)
            if trial_id in fail_predict_trial_ids:
                stderr_path.write_text("fake prediction failure\n", encoding="utf-8")
                return {"returncode": 1, "wall_seconds": 0.01}
            scratch_dir.mkdir(parents=True, exist_ok=True)
            expected = base._expected_datasets(Path(flags["splits"]), split)
            skip_last = trial_id in missing_dataset_trial_ids
            for index, name in enumerate(expected):
                if skip_last and index == len(expected) - 1:
                    continue
                (scratch_dir / f"{name}.geff").write_bytes(f"geff-{trial_id}-{name}".encode())
            return {"returncode": 0, "wall_seconds": 0.01}
        if script == str(base.EVALUATE_SCRIPT):
            flags = _parse_flags(command)
            pred_dir = Path(flags["pred-dir"])
            json_out = Path(flags["json-out"])
            trial_id = json_out.parent.name
            if trial_id in fail_evaluate_trial_ids:
                stderr_path.write_text("fake evaluation failure\n", encoding="utf-8")
                return {"returncode": 1, "wall_seconds": 0.01}
            names = sorted(p.stem for p in pred_dir.glob("*.geff"))
            row_metrics = {
                "edge_tp": 10, "edge_fp": 2, "edge_fn": 1, "num_pred_nodes": 20,
                "total_node_ratio": 0.1, "node_recall": 0.9, "edge_jaccard": 0.77, "adj_edge_jaccard": 0.75,
            }
            row_metrics.update(per_dataset_overrides.get(trial_id, {}))
            per_dataset = [{"dataset": n, "metrics": dict(row_metrics)} for n in names]
            summary = {
                "n": len(names), "edge_jaccard": 0.77, "division_jaccard": 0.5,
                "division_tp": 3, "division_fp": 1, "division_fn": 1, "node_recall": 0.9,
                "adj_edge_jaccard": 0.75, "n_adj": len(names),
                "score": scores_by_trial_id.get(trial_id, default_score),
            }
            summary.update(summary_overrides.get(trial_id, {}))
            report = {
                "schema_version": 1, "evaluated_datasets": names, "skipped_datasets": [],
                "per_dataset_metrics": per_dataset, "summary_metrics": summary,
            }
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps(report), encoding="utf-8")
            return {"returncode": 0, "wall_seconds": 0.01}
        raise AssertionError(f"unexpected command script: {script}")

    monkeypatch.setattr(base, "_stream_subprocess", fake_stream_subprocess)


def _main_args(
    env: Environment, *, stage: str, output_dir: Path, previous_results: Path | None = None, no_resume: bool = False,
) -> list[str]:
    args = [
        "--config", str(env.config_path), "--stage", stage,
        "--data-dir", str(env.data_dir), "--output-dir", str(output_dir),
    ]
    for label, path in env.checkpoints.items():
        args += ["--checkpoint", f"{label}={path}"]
    if previous_results is not None:
        args += ["--previous-results", str(previous_results)]
    if no_resume:
        args.append("--no-resume")
    return args


def _run_main(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["calibrate_extended_inference.py", *argv])
    ext.main()


def _run_stage_a_happy(env: Environment, monkeypatch, out_dir: Path) -> dict:
    scores = {}
    for label in CHECKPOINT_LABELS:
        for i, det in enumerate(env.config["stage_a"]["det_thresholds"]):
            scores[ext._trial_id("stage_a", label, det, 0.35, 5.0)] = 0.5 + 0.01 * i
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)
    return base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a", build_selection=lambda r: {
        "audit_sha256": env.audit_sha256,
        "selected_det_thresholds_by_checkpoint": ext.select_best_det_thresholds_by_checkpoint(r, 2, CHECKPOINT_LABELS),
    })


# =============================================================================
# audit_validation module
# =============================================================================

def test_real_audit_file_passes_validation() -> None:
    config = ext.load_config(REAL_CONFIG_PATH)
    full_split_datasets = base._expected_datasets(
        base._resolve_repo_path(config["full_split_file"]), config["full_split"],
    )
    result = av.load_and_validate_audit(config["audit"], full_split_datasets=full_split_datasets)
    assert result["findings"]["primary"]["count"] == 0
    assert result["findings"]["secondary"]["voxel_kernel"] == [5, 5, 5]
    assert result["schema_version"] == 1
    assert result["split"] == 0
    assert result["partition"] == "test"
    assert result["downsample"] == [1, 4, 4]
    assert result["dataset_count"] == 19
    assert result["node_count"] == 13335
    assert result["tolerance"] == 0.001


def test_real_audit_file_is_strict_json() -> None:
    raw = REAL_AUDIT_PATH.read_text(encoding="utf-8")
    json.loads(raw)  # raises if not strict JSON


def test_real_audit_sha256_matches_pinned_config_value() -> None:
    config = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
    assert av.sha256_file(REAL_AUDIT_PATH) == config["audit"]["sha256"]


def test_audit_rejects_hash_mismatch(env: Environment) -> None:
    tampered = dict(env.config["audit"])
    tampered["sha256"] = "0" * 64
    with pytest.raises(av.AuditValidationError, match="sha256"):
        av.load_and_validate_audit(tampered, full_split_datasets=env.full_split_datasets)


def test_audit_rejects_changed_bytes_after_pinning(env: Environment) -> None:
    audit_cfg = dict(env.config["audit"])
    env.audit_path.write_text(env.audit_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(av.AuditValidationError, match="sha256"):
        av.load_and_validate_audit(audit_cfg, full_split_datasets=env.full_split_datasets)


@pytest.mark.parametrize(("field", "bad_value"), [
    ("schema_version", 2), ("split", 1), ("partition", "train"), ("downsample", [1, 2, 2]),
])
def test_audit_rejects_wrong_envelope_fields(tmp_path: Path, field: str, bad_value) -> None:
    audit_path = tmp_path / "audit.json"
    kwargs = {field: bad_value}
    sha, provenance = _make_audit(audit_path, **kwargs)
    cfg = _audit_config_block(audit_path, sha, provenance)
    with pytest.raises(av.AuditValidationError):
        av.load_and_validate_audit(cfg, full_split_datasets=sorted(FULL_NAMES))


def test_audit_rejects_nonzero_collapsed_or_duplicate(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    sha, provenance = _make_audit(audit_path, collapsed=1)
    with pytest.raises(av.AuditValidationError, match="collapsed_node_count"):
        av.load_and_validate_audit(_audit_config_block(audit_path, sha, provenance), full_split_datasets=sorted(FULL_NAMES))

    audit_path2 = tmp_path / "audit2.json"
    sha2, provenance2 = _make_audit(audit_path2, duplicate=1)
    with pytest.raises(av.AuditValidationError, match="duplicate_target_voxel_count"):
        av.load_and_validate_audit(
            _audit_config_block(audit_path2, sha2, provenance2), full_split_datasets=sorted(FULL_NAMES),
        )


def test_audit_rejects_nonzero_primary_kernel_collisions(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    names = FULL_NAMES

    def dataset_entry(is_last: bool) -> dict:
        return {
            "node_count": 1000,
            "quantized_target_collision_risk": [
                _quant_row(5.0, [3, 3, 3], 2, 1, 1000), _quant_row(6.0, [5, 5, 5], 0, 0, 1000),
            ],
        }

    datasets = {name: dataset_entry(i == len(names) - 1) for i, name in enumerate(names)}
    payload = {
        "schema_version": 1, "split": 0, "partition": "test", "downsample": [1, 4, 4],
        "dataset_count": len(names),
        "aggregate": {
            "collapsed_node_count": 0, "duplicate_target_voxel_count": 0, "node_count": 1000 * len(names),
            "quantized_target_collision_risk": [
                {"requested_um": 5.0, "count": 2 * len(names), "pair_count": 1 * len(names), "fraction": 0.002},
                {"requested_um": 6.0, "count": 0, "pair_count": 0, "fraction": 0.0},
            ],
        },
        "datasets": datasets,
    }
    _write_json(audit_path, payload)
    sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    provenance = _provenance_from_payload(payload)
    with pytest.raises(av.AuditValidationError, match="primary pooling kernel"):
        av.load_and_validate_audit(
            _audit_config_block(audit_path, sha, provenance), full_split_datasets=sorted(FULL_NAMES),
        )


def test_audit_accepts_secondary_kernel_below_tolerance(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    sha, provenance = _make_audit(audit_path, secondary_count=1, secondary_pair_count=1)
    result = av.load_and_validate_audit(
        _audit_config_block(audit_path, sha, provenance, fraction_max=0.5), full_split_datasets=sorted(FULL_NAMES),
    )
    assert result["findings"]["secondary"]["count"] == 1


def test_audit_rejects_secondary_kernel_above_tolerance(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    sha, provenance = _make_audit(audit_path, secondary_count=999, secondary_pair_count=500, per_dataset_node_count=1000)
    with pytest.raises(av.AuditValidationError, match="tolerance"):
        av.load_and_validate_audit(
            _audit_config_block(audit_path, sha, provenance, fraction_max=0.001), full_split_datasets=sorted(FULL_NAMES),
        )


def test_audit_rejects_voxel_kernel_mismatch(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    names = FULL_NAMES

    def dataset_entry() -> dict:
        return {
            "node_count": 1000,
            "quantized_target_collision_risk": [
                _quant_row(5.0, [3, 3, 3], 0, 0, 1000), _quant_row(6.0, [7, 7, 7], 0, 0, 1000),  # wrong kernel
            ],
        }

    payload = {
        "schema_version": 1, "split": 0, "partition": "test", "downsample": [1, 4, 4],
        "dataset_count": len(names),
        "aggregate": {
            "collapsed_node_count": 0, "duplicate_target_voxel_count": 0, "node_count": 1000 * len(names),
            "quantized_target_collision_risk": [
                {"requested_um": 5.0, "count": 0, "pair_count": 0, "fraction": 0.0},
                {"requested_um": 6.0, "count": 0, "pair_count": 0, "fraction": 0.0},
            ],
        },
        "datasets": {name: dataset_entry() for name in names},
    }
    _write_json(audit_path, payload)
    sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    provenance = _provenance_from_payload(payload)
    with pytest.raises(av.AuditValidationError, match="voxel_kernel"):
        av.load_and_validate_audit(
            _audit_config_block(audit_path, sha, provenance), full_split_datasets=sorted(FULL_NAMES),
        )


def test_audit_rejects_dataset_mismatch_with_full_split(env: Environment) -> None:
    with pytest.raises(av.AuditValidationError, match="full-fold split"):
        av.load_and_validate_audit(dict(env.config["audit"]), full_split_datasets=SUBSET_NAMES)


def test_audit_rejects_tolerance_fraction_max_mismatch(env: Environment) -> None:
    cfg = json.loads(json.dumps(env.config["audit"]))
    cfg["tolerance"] = 0.5
    with pytest.raises(av.AuditValidationError, match="tolerance"):
        av.load_and_validate_audit(cfg, full_split_datasets=env.full_split_datasets)


def test_audit_rejects_observed_values_not_matching_config_pin(env: Environment) -> None:
    cfg = json.loads(json.dumps(env.config["audit"]))
    cfg["observed"]["secondary"]["count"] = 999
    with pytest.raises(av.AuditValidationError, match="do not match those pinned"):
        av.load_and_validate_audit(cfg, full_split_datasets=env.full_split_datasets)


def test_audit_rejects_inconsistent_dataset_level_node_count_totals(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    sha, provenance = _make_audit(audit_path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    first_name = next(iter(payload["datasets"]))
    payload["datasets"][first_name]["node_count"] += 5
    _write_json(audit_path, payload)
    sha2 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    cfg = _audit_config_block(audit_path, sha2, provenance)
    with pytest.raises(av.AuditValidationError, match="node_count totals"):
        av.load_and_validate_audit(cfg, full_split_datasets=sorted(FULL_NAMES))


def test_audit_rejects_inconsistent_aggregate_count_totals(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    sha, provenance = _make_audit(audit_path, secondary_count=2, secondary_pair_count=1)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    row6 = next(r for r in payload["aggregate"]["quantized_target_collision_risk"] if r["requested_um"] == 6.0)
    row6["count"] = 999
    _write_json(audit_path, payload)
    sha2 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    cfg = _audit_config_block(audit_path, sha2, provenance, fraction_max=0.5)
    with pytest.raises(av.AuditValidationError, match="count totals"):
        av.load_and_validate_audit(cfg, full_split_datasets=sorted(FULL_NAMES))


def test_audit_rejects_fraction_not_equal_to_count_over_node_count(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    sha, provenance = _make_audit(audit_path, secondary_count=2, secondary_pair_count=1)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    row6 = next(r for r in payload["aggregate"]["quantized_target_collision_risk"] if r["requested_um"] == 6.0)
    row6["fraction"] = 0.5
    _write_json(audit_path, payload)
    sha2 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    cfg = _audit_config_block(audit_path, sha2, provenance, fraction_max=0.9)
    with pytest.raises(av.AuditValidationError, match="does not equal count/node_count"):
        av.load_and_validate_audit(cfg, full_split_datasets=sorted(FULL_NAMES))


def test_audit_rejects_nan_and_infinity_tokens(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    raw = (
        '{"schema_version": 1, "split": 0, "partition": "test", "downsample": [1,4,4], '
        '"dataset_count": 0, "aggregate": {"collapsed_node_count": 0, "duplicate_target_voxel_count": 0, '
        '"node_count": NaN, "quantized_target_collision_risk": []}, "datasets": {}}'
    )
    audit_path.write_text(raw, encoding="utf-8")
    sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    cfg = _audit_config_block(audit_path, sha, {
        "dataset_count": 0, "node_count": 0,
        "observed": {"primary": {"count": 0, "pair_count": 0, "fraction": 0.0},
                     "secondary": {"count": 0, "pair_count": 0, "fraction": 0.0}},
    })
    with pytest.raises(av.AuditValidationError, match="non-finite"):
        av.load_and_validate_audit(cfg, full_split_datasets=["x"])


def test_audit_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    raw = (
        '{"schema_version": 1, "schema_version": 1, "split": 0, "partition": "test", '
        '"downsample": [1,4,4], "dataset_count": 0, "aggregate": {}, "datasets": {}}'
    )
    audit_path.write_text(raw, encoding="utf-8")
    sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    cfg = _audit_config_block(audit_path, sha, {
        "dataset_count": 0, "node_count": 0,
        "observed": {"primary": {"count": 0, "pair_count": 0, "fraction": 0.0},
                     "secondary": {"count": 0, "pair_count": 0, "fraction": 0.0}},
    })
    with pytest.raises(av.AuditValidationError, match="duplicate JSON key"):
        av.load_and_validate_audit(cfg, full_split_datasets=["x"])


# =============================================================================
# Config loading — locked plan
# =============================================================================

def test_committed_extended_config_is_valid_and_loadable() -> None:
    config = ext.load_config(REAL_CONFIG_PATH)
    assert config["schema_version"] == 1
    assert config["selection_metric"] == "summary_metrics.score"
    assert config["tracking"] == "greedy"
    assert config["checkpoint_labels"] == ["7500", "10000"]
    assert config["baseline"] == {
        "det_threshold": 0.70, "edge_threshold": 0.35, "pool_kernel_um": 5.0, "tracking": "greedy",
    }
    assert config["stage_a"]["det_thresholds"] == [0.70, 0.74, 0.78, 0.82, 0.86]
    assert config["stage_a"]["edge_threshold"] == 0.35
    assert config["stage_a"]["pool_kernel_um"] == 5.0
    assert config["stage_a"]["n_best_det_thresholds_per_checkpoint"] == 2
    assert config["stage_b"]["edge_thresholds"] == [0.20, 0.25, 0.30, 0.35]
    assert config["stage_b"]["pool_kernel_um"] == 5.0
    assert config["stage_b"]["n_best_pairs_per_checkpoint"] == 2
    assert config["stage_c"]["pool_kernel_um_values"] == [5.0, 6.0]
    assert config["stage_c"]["n_best_candidates_per_checkpoint"] == 2
    assert config["stage_d"]["checkpoint_labels"] == ["7500", "10000"]


def test_original_config_and_runner_still_load() -> None:
    """Backward compatibility: the original workflow is untouched by this module."""
    config = base.load_config(ORIGINAL_CONFIG_PATH)
    assert config["stage_c"]["checkpoint_labels"] == ["7500", "10000"]
    assert "checkpoint_labels" not in config  # original schema has no top-level field of this name


@pytest.mark.parametrize(("mutate", "match"), [
    (lambda c: c.update(schema_version=2), "schema_version"),
    (lambda c: c.update(selection_metric="other_metric"), "selection_metric"),
    (lambda c: c.update(tracking="hungarian"), "tracking"),
    (lambda c: c["baseline"].update(det_threshold=0.5), "baseline"),
    (lambda c: c["baseline"].update(edge_threshold=0.5), "baseline"),
    (lambda c: c["baseline"].update(pool_kernel_um=7.0), "pool_kernel_um"),
    (lambda c: c["stage_a"].update(det_thresholds=[0.70, 0.74, 0.78, 0.82, 0.90]), "det_thresholds"),
    (lambda c: c["stage_a"].update(edge_threshold=0.5), "edge_threshold"),
    (lambda c: c["stage_a"].update(pool_kernel_um=6.0), "pool_kernel_um"),
    (lambda c: c["stage_a"].update(n_best_det_thresholds_per_checkpoint=1), "n_best_det_thresholds_per_checkpoint"),
    (lambda c: c["stage_b"].update(edge_thresholds=[0.15, 0.25, 0.30, 0.35]), "edge_thresholds"),
    (lambda c: c["stage_b"].update(pool_kernel_um=6.0), "pool_kernel_um"),
    (lambda c: c["stage_b"].update(n_best_pairs_per_checkpoint=1), "n_best_pairs_per_checkpoint"),
    (lambda c: c["stage_c"].update(pool_kernel_um_values=[5.0, 8.0]), "pool_kernel_um_values"),
    (lambda c: c["stage_c"].update(n_best_candidates_per_checkpoint=1), "n_best_candidates_per_checkpoint"),
    (lambda c: c["stage_d"].update(checkpoint_labels=["7500"]), "checkpoint_labels"),
])
def test_config_rejects_locked_value_mutation(env: Environment, mutate, match: str) -> None:
    path = env.remake(mutate, name="locked_mutation.json")
    with pytest.raises(base.CalibrationError, match=match):
        ext.load_config(path)


def test_config_rejects_pool_3_and_7(env: Environment) -> None:
    def mutate(config):
        config["stage_c"]["pool_kernel_um_values"] = [3.0, 7.0]
    path = env.remake(mutate, name="bad_pool.json")
    with pytest.raises(base.CalibrationError, match="pool_kernel_um_values"):
        ext.load_config(path)


def test_config_accepts_pool_5_and_6() -> None:
    config = ext.load_config(REAL_CONFIG_PATH)
    assert config["stage_c"]["pool_kernel_um_values"] == [5.0, 6.0]


def test_config_rejects_unexpected_checkpoint_labels(env: Environment) -> None:
    def mutate(config):
        config["checkpoint_labels"] = ["7500", "12500"]
    path = env.remake(mutate, name="bad_ckpt.json")
    with pytest.raises(base.CalibrationError, match="checkpoint_labels"):
        ext.load_config(path)


def test_config_requires_top_level_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_json(path, {"tracking": "greedy"})
    with pytest.raises(base.CalibrationError, match="missing required keys"):
        ext.load_config(path)


def test_config_rejects_upper_bound_warning_mismatch(env: Environment) -> None:
    def mutate(config):
        config["stage_a"]["det_threshold_upper_bound_warning"] = 0.99
    path = env.remake(mutate, name="bad_bound.json")
    with pytest.raises(base.CalibrationError, match="det_threshold_upper_bound_warning"):
        ext.load_config(path)


# =============================================================================
# Checkpoint identity pinning
# =============================================================================

def test_config_rejects_malformed_checkpoint_pin_hash(env: Environment) -> None:
    def mutate(config):
        config["checkpoint_pins"]["7500"]["sha256"] = "not-a-hash"
    path = env.remake(mutate, name="bad_pin_hash.json")
    with pytest.raises(base.CalibrationError, match="sha256"):
        ext.load_config(path)


def test_config_rejects_checkpoint_pin_filename_with_path_separator(env: Environment) -> None:
    def mutate(config):
        config["checkpoint_pins"]["7500"]["filename"] = "sub/dir/name.pth"
    path = env.remake(mutate, name="bad_pin_filename.json")
    with pytest.raises(base.CalibrationError, match="filename"):
        ext.load_config(path)


def test_config_rejects_missing_adjacent_config_sha256(env: Environment) -> None:
    def mutate(config):
        del config["checkpoint_pins"]["adjacent_config_sha256"]
    path = env.remake(mutate, name="missing_adjacent.json")
    with pytest.raises(base.CalibrationError, match="adjacent_config_sha256"):
        ext.load_config(path)


def test_validate_pinned_checkpoints_accepts_correct_files(env: Environment) -> None:
    ext.validate_pinned_checkpoints(env.config, env.checkpoints)  # must not raise


def test_validate_pinned_checkpoints_rejects_wrong_bytes(env: Environment) -> None:
    Path(env.checkpoints["7500"]).write_bytes(b"corrupted-checkpoint-bytes")
    with pytest.raises(base.CalibrationError, match="sha256"):
        ext.validate_pinned_checkpoints(env.config, env.checkpoints)


def test_validate_pinned_checkpoints_rejects_wrong_filename(env: Environment) -> None:
    original = Path(env.checkpoints["7500"])
    renamed = original.with_name("renamed_checkpoint.pth")
    original.rename(renamed)
    checkpoints = dict(env.checkpoints)
    checkpoints["7500"] = str(renamed)
    with pytest.raises(base.CalibrationError, match="filename"):
        ext.validate_pinned_checkpoints(env.config, checkpoints)


def test_validate_pinned_checkpoints_rejects_missing_file(env: Environment) -> None:
    checkpoints = dict(env.checkpoints)
    missing = Path(env.checkpoints["7500"]).with_name("edge_predictor_iter_7500.pth.missing")
    # Preserve the pinned filename but point at a nonexistent path with that basename.
    pinned_name = env.checkpoint_pins["7500"]["filename"]
    missing = Path(env.checkpoints["7500"]).parent / "does_not_exist" / pinned_name
    checkpoints["7500"] = str(missing)
    with pytest.raises(base.CalibrationError, match="not found"):
        ext.validate_pinned_checkpoints(env.config, checkpoints)


def test_validate_pinned_checkpoints_rejects_adjacent_config_hash_mismatch(env: Environment) -> None:
    config_path = Path(env.checkpoints["7500"]).parent / "config.json"
    config_path.write_text(json.dumps({"pool_kernel_um": 999.0}), encoding="utf-8")
    with pytest.raises(base.CalibrationError, match="adjacent_config_sha256"):
        ext.validate_pinned_checkpoints(env.config, env.checkpoints)


def test_validate_pinned_checkpoints_rejects_symlinked_checkpoint(env: Environment, tmp_path: Path) -> None:
    original = Path(env.checkpoints["7500"])
    alt_dir = tmp_path / "alt"
    alt_dir.mkdir()
    link = alt_dir / original.name
    try:
        link.symlink_to(original)
        (alt_dir / "config.json").symlink_to(original.parent / "config.json")
    except OSError:
        pytest.skip("symlink creation not permitted in this environment")
    checkpoints = dict(env.checkpoints)
    checkpoints["7500"] = str(link)
    with pytest.raises(base.CalibrationError, match="symlink"):
        ext.validate_pinned_checkpoints(env.config, checkpoints)


# =============================================================================
# Checkpoint label handling
# =============================================================================

def test_duplicate_checkpoint_label_rejected() -> None:
    with pytest.raises(base.CalibrationError, match="duplicate"):
        ext.parse_and_validate_checkpoints(["7500=/a/x.pth", "7500=/a/y.pth"])


def test_unexpected_checkpoint_label_rejected() -> None:
    with pytest.raises(base.CalibrationError, match="allowed set"):
        ext.parse_and_validate_checkpoints(["7500=/a/x.pth", "99999=/a/y.pth"])


def test_allowed_checkpoint_labels_accepted() -> None:
    result = ext.parse_and_validate_checkpoints(["7500=/a/x.pth", "10000=/a/y.pth"])
    assert result == {"7500": "/a/x.pth", "10000": "/a/y.pth"}


# =============================================================================
# Trial grid construction
# =============================================================================

def test_stage_a_trial_grid_is_exact(env: Environment) -> None:
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)
    assert len(trials) == 2 * 5  # 2 checkpoints x 5 det_thresholds
    for t in trials:
        assert t["edge_threshold"] == 0.35
        assert t["pool_kernel_um"] == 5.0
        assert t["is_screening"] is True
        assert t["split"] == 0
    trial_ids = {t["trial_id"] for t in trials}
    assert len(trial_ids) == len(trials)  # no collisions


def test_stage_c_trial_grid_includes_pool_5_and_6_not_3_or_7(env: Environment) -> None:
    pairs_by_checkpoint = {label: [(0.74, 0.30), (0.78, 0.25)] for label in CHECKPOINT_LABELS}
    trials = ext.build_stage_c_trials(env.config, env.checkpoints, pairs_by_checkpoint)
    pools = {t["pool_kernel_um"] for t in trials}
    assert pools == {5.0, 6.0}
    assert 3.0 not in pools
    assert 7.0 not in pools
    assert len(trials) == 2 * 2 * 2  # 2 checkpoints x 2 pairs x 2 pool values


def test_trial_id_disambiguates_pool_kernel(env: Environment) -> None:
    t5 = ext._make_trial(
        stage="stage_c", checkpoint_label="7500", checkpoint_path="x", det_threshold=0.7,
        edge_threshold=0.3, pool_kernel_um=5.0, tracking="greedy",
        split_file=env.config["subset_split_file"], split=0, is_screening=True,
    )
    t6 = ext._make_trial(
        stage="stage_c", checkpoint_label="7500", checkpoint_path="x", det_threshold=0.7,
        edge_threshold=0.3, pool_kernel_um=6.0, tracking="greedy",
        split_file=env.config["subset_split_file"], split=0, is_screening=True,
    )
    assert t5["trial_id"] != t6["trial_id"]


# =============================================================================
# End-to-end Stage A -> B -> C -> D (fully mocked subprocess)
# =============================================================================

def test_stage_a_selects_top_2_per_checkpoint(env: Environment, monkeypatch, tmp_path: Path) -> None:
    scores = {}
    for label in CHECKPOINT_LABELS:
        for i, det in enumerate(env.config["stage_a"]["det_thresholds"]):
            trial_id = ext._trial_id("stage_a", label, det, 0.35, 5.0)
            scores[trial_id] = 0.5 + 0.01 * i  # increasing with det_threshold -> highest det wins
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)

    out_dir = tmp_path / "stage_a_out"
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)
    report = base.run_trials(
        trials, env.data_dir, out_dir, env.config_path, "stage_a",
        build_selection=lambda records: {
            "audit_sha256": env.audit_sha256,
            "selected_det_thresholds_by_checkpoint": ext.select_best_det_thresholds_by_checkpoint(
                records, 2, CHECKPOINT_LABELS,
            ),
        },
    )
    selection = report["selected_det_thresholds_by_checkpoint"]
    for label in CHECKPOINT_LABELS:
        assert selection[label] == [0.86, 0.82]  # top-2 highest-scoring dets, best first
    assert len(report["trials"]) == 10
    assert all(r["status"] == "success" for r in report["trials"])


def test_full_pipeline_stage_a_through_d(env: Environment, monkeypatch, tmp_path: Path) -> None:
    """Drive the CLI's internal logic stage by stage via the module functions,
    verifying resumability, per-checkpoint selection, previous-results
    lineage provenance, and the Stage D global winner — all subprocess
    calls are faked.
    """
    scores: dict[str, float] = {}

    def set_score(stage, label, det, edge, pool, value):
        scores[ext._trial_id(stage, label, det, edge, pool)] = value

    for label, best_dets in (("7500", [0.74, 0.78]), ("10000", [0.82, 0.86])):
        for det in env.config["stage_a"]["det_thresholds"]:
            if det == best_dets[0]:
                value = 0.95
            elif det == best_dets[1]:
                value = 0.9
            else:
                value = 0.1
            set_score("stage_a", label, det, 0.35, 5.0, value)
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)

    stage_a_dir = tmp_path / "a"
    trials_a = ext.build_stage_a_trials(env.config, env.checkpoints)

    def sel_a(records):
        selection = ext.select_best_det_thresholds_by_checkpoint(records, 2, CHECKPOINT_LABELS)
        return {"audit_sha256": env.audit_sha256, "selected_det_thresholds_by_checkpoint": selection,
                "warnings": ext.stage_a_boundary_warnings(selection, env.config["stage_a"])}

    report_a = base.run_trials(trials_a, env.data_dir, stage_a_dir, env.config_path, "stage_a", build_selection=sel_a)
    assert report_a["selected_det_thresholds_by_checkpoint"]["7500"] == [0.74, 0.78]
    assert report_a["selected_det_thresholds_by_checkpoint"]["10000"] == [0.82, 0.86]

    # Stage B: edge=0.20 always best (to exercise the lower-bound warning).
    det_by_checkpoint = report_a["selected_det_thresholds_by_checkpoint"]
    for label, dets in det_by_checkpoint.items():
        for det in dets:
            for i, edge in enumerate(env.config["stage_b"]["edge_thresholds"]):
                set_score("stage_b", label, det, edge, 5.0, 0.5 - 0.01 * i)
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)

    stage_b_dir = tmp_path / "b"
    trials_b = ext.build_stage_b_trials(env.config, env.checkpoints, det_by_checkpoint)
    previous_results_a = ext._build_previous_results_provenance(
        Path(report_a["results_path"]), "stage_a", det_by_checkpoint,
    )

    def sel_b(records):
        selection = ext.select_best_pairs_by_checkpoint(records, 2, CHECKPOINT_LABELS)
        return {"previous_results": previous_results_a,
                "selected_pairs_by_checkpoint": {k: [list(p) for p in v] for k, v in selection.items()},
                "warnings": ext.stage_b_boundary_warnings(selection, env.config["stage_b"])}

    report_b = base.run_trials(trials_b, env.data_dir, stage_b_dir, env.config_path, "stage_b", build_selection=sel_b)
    for label in CHECKPOINT_LABELS:
        pairs = [tuple(p) for p in report_b["selected_pairs_by_checkpoint"][label]]
        assert pairs[0][1] == 0.20  # winner uses the lowest edge threshold
    assert any(w["kind"] == "edge_threshold_lower_bound" for w in report_b["warnings"])

    # Stage C: pool=6.0 slightly better than 5.0, for every pair.
    pairs_by_checkpoint = {
        k: [tuple(p) for p in v] for k, v in report_b["selected_pairs_by_checkpoint"].items()
    }
    for label, pairs in pairs_by_checkpoint.items():
        for det, edge in pairs:
            set_score("stage_c", label, det, edge, 5.0, 0.80)
            set_score("stage_c", label, det, edge, 6.0, 0.85)
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)

    stage_c_dir = tmp_path / "c"
    trials_c = ext.build_stage_c_trials(env.config, env.checkpoints, pairs_by_checkpoint)
    previous_results_b = ext._build_previous_results_provenance(
        Path(report_b["results_path"]), "stage_b", pairs_by_checkpoint,
    )

    def sel_c(records):
        selection = ext.select_best_candidates_by_checkpoint(records, 2, CHECKPOINT_LABELS)
        return {"previous_results": previous_results_b,
                "selected_candidates_by_checkpoint": {k: [list(c) for c in v] for k, v in selection.items()}}

    report_c = base.run_trials(trials_c, env.data_dir, stage_c_dir, env.config_path, "stage_c", build_selection=sel_c)
    for label in CHECKPOINT_LABELS:
        candidates = [tuple(c) for c in report_c["selected_candidates_by_checkpoint"][label]]
        assert candidates[0][2] == 6.0  # winner uses the better pooling kernel

    # Stage D: full-fold confirmation, one global winner across every checkpoint's candidates.
    candidates_by_checkpoint = {
        k: [tuple(c) for c in v] for k, v in report_c["selected_candidates_by_checkpoint"].items()
    }
    previous_results_c = ext._build_previous_results_provenance(
        Path(report_c["results_path"]), "stage_c", candidates_by_checkpoint,
    )
    global_best_trial_id = None
    for label, candidates in candidates_by_checkpoint.items():
        for det, edge, pool in candidates:
            value = 0.99 if label == "10000" and det == candidates[0][0] else 0.5
            set_score("stage_d", label, det, edge, pool, value)
            if value == 0.99:
                global_best_trial_id = ext._trial_id("stage_d", label, det, edge, pool)
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)

    stage_d_dir = tmp_path / "d"
    trials_d = ext.build_stage_d_trials(env.config, env.checkpoints, candidates_by_checkpoint)

    def sel_d(records):
        best = ext.select_stage_d_winner(records, env.config["stage_d"]["checkpoint_labels"])
        return {"previous_results": previous_results_c, "selected_final_trial": ext._selected_final_trial_summary(best)}

    report_d = base.run_trials(trials_d, env.data_dir, stage_d_dir, env.config_path, "stage_d", build_selection=sel_d)
    winner = report_d["selected_final_trial"]
    assert winner["trial_id"] == global_best_trial_id
    assert winner["checkpoint_label"] == "10000"
    assert winner["split_file"] == str(env.full_split_path)
    assert "diagnostics" in winner
    assert winner["diagnostics"]["score"] == winner["official_score"]
    for r in trials_d:
        assert r["is_screening"] is False  # full-fold, not screening


def test_stage_d_lineage_recursively_validates_full_chain(env: Environment, monkeypatch, tmp_path: Path) -> None:
    """Stage D's lineage loader recursively revalidates C -> B -> A end to end."""
    out_dir_a = tmp_path / "a"
    report_a = _run_stage_a_happy(env, monkeypatch, out_dir_a)
    det_by_checkpoint = report_a["selected_det_thresholds_by_checkpoint"]

    scores = {}
    for label, dets in det_by_checkpoint.items():
        for det in dets:
            for i, edge in enumerate(env.config["stage_b"]["edge_thresholds"]):
                scores[ext._trial_id("stage_b", label, det, edge, 5.0)] = 0.5 + 0.01 * i
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)
    trials_b = ext.build_stage_b_trials(env.config, env.checkpoints, det_by_checkpoint)
    out_dir_b = tmp_path / "b"
    previous_results_a = ext._build_previous_results_provenance(
        Path(report_a["results_path"]), "stage_a", det_by_checkpoint,
    )
    report_b = base.run_trials(trials_b, env.data_dir, out_dir_b, env.config_path, "stage_b", build_selection=lambda r: {
        "audit_sha256": env.audit_sha256,
        "previous_results": previous_results_a,
        "selected_pairs_by_checkpoint": {
            k: [list(p) for p in v] for k, v in ext.select_best_pairs_by_checkpoint(r, 2, CHECKPOINT_LABELS).items()
        },
    })
    pairs_by_checkpoint = {k: [tuple(p) for p in v] for k, v in report_b["selected_pairs_by_checkpoint"].items()}

    scores2 = {}
    for label, pairs in pairs_by_checkpoint.items():
        for det, edge in pairs:
            scores2[ext._trial_id("stage_c", label, det, edge, 5.0)] = 0.80
            scores2[ext._trial_id("stage_c", label, det, edge, 6.0)] = 0.85
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores2)
    trials_c = ext.build_stage_c_trials(env.config, env.checkpoints, pairs_by_checkpoint)
    out_dir_c = tmp_path / "c"
    previous_results_b = ext._build_previous_results_provenance(
        Path(report_b["results_path"]), "stage_b", pairs_by_checkpoint,
    )
    report_c = base.run_trials(trials_c, env.data_dir, out_dir_c, env.config_path, "stage_c", build_selection=lambda r: {
        "audit_sha256": env.audit_sha256,
        "previous_results": previous_results_b,
        "selected_candidates_by_checkpoint": {
            k: [list(c) for c in v] for k, v in ext.select_best_candidates_by_checkpoint(r, 2, CHECKPOINT_LABELS).items()
        },
    })
    results_path_c = Path(report_c["results_path"])

    _, candidates_by_checkpoint = ext.load_stage_c_report_for_stage_d(
        results_path_c, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
    )
    assert set(candidates_by_checkpoint) == {"7500", "10000"}
    for label, candidates in candidates_by_checkpoint.items():
        for det, edge, pool in candidates:
            assert pool in (5.0, 6.0)


# =============================================================================
# Diagnostics
# =============================================================================

def test_diagnostics_computed_from_summary_and_per_dataset() -> None:
    record = {
        "status": "success",
        "summary_metrics": {
            "score": 0.81, "edge_jaccard": 0.7, "adj_edge_jaccard": 0.68, "node_recall": 0.9,
            "division_jaccard": 0.4, "division_tp": 4, "division_fp": 2, "division_fn": 1,
        },
        "per_dataset_metrics": [
            {"dataset": "dsA", "metrics": {"edge_tp": 10, "edge_fp": 3}},
            {"dataset": "dsB", "metrics": {"edge_tp": 5, "edge_fp": 2}},
        ],
    }
    diagnostics = ext.compute_diagnostics(record)
    assert diagnostics["predicted_edge_count"] == 20
    assert diagnostics["division_fp_fraction_of_predicted_edges"] == pytest.approx(2 / 20)
    assert diagnostics["score"] == 0.81


def test_diagnostics_zero_predicted_edges_gives_zero_fraction() -> None:
    record = {
        "status": "success",
        "summary_metrics": {
            "score": 0.0, "edge_jaccard": 0.0, "adj_edge_jaccard": 0.0, "node_recall": 0.0,
            "division_jaccard": float("nan"), "division_tp": 0, "division_fp": 0, "division_fn": 0,
        },
        "per_dataset_metrics": [{"dataset": "dsA", "metrics": {"edge_tp": 0, "edge_fp": 0}}],
    }
    diagnostics = ext.compute_diagnostics(record)
    assert diagnostics["predicted_edge_count"] == 0
    assert diagnostics["division_fp_fraction_of_predicted_edges"] == 0.0


def test_diagnostics_do_not_affect_selection(env: Environment, monkeypatch, tmp_path: Path) -> None:
    """A trial with a terrible diagnostic breakdown but the best official score still wins."""
    scores = {}
    overrides = {}
    dets = env.config["stage_a"]["det_thresholds"]
    for label in CHECKPOINT_LABELS:
        for det in dets:
            trial_id = ext._trial_id("stage_a", label, det, 0.35, 5.0)
            scores[trial_id] = 0.5
    winner_id = ext._trial_id("stage_a", "7500", dets[0], 0.35, 5.0)
    scores[winner_id] = 0.99
    overrides[winner_id] = {"edge_tp": 0, "edge_fp": 0}  # terrible diagnostics, best official score
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores, per_dataset_overrides=overrides)

    out_dir = tmp_path / "stage_a_diag"
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)
    report = base.run_trials(
        trials, env.data_dir, out_dir, env.config_path, "stage_a",
        build_selection=lambda records: {
            "selected_det_thresholds_by_checkpoint": ext.select_best_det_thresholds_by_checkpoint(records, 2, CHECKPOINT_LABELS),
        },
    )
    ext._annotate_diagnostics(Path(report["results_path"]))
    payload = json.loads(Path(report["results_path"]).read_text(encoding="utf-8"))
    winner_record = next(r for r in payload["trials"] if r["trial_id"] == winner_id)
    assert winner_record["diagnostics"]["predicted_edge_count"] == 0
    assert report["selected_det_thresholds_by_checkpoint"]["7500"][0] == dets[0]  # still selected first


def test_as_finite_number_rejects_nan_and_infinities() -> None:
    assert ext._as_finite_number(float("nan")) is None
    assert ext._as_finite_number(float("inf")) is None
    assert ext._as_finite_number(float("-inf")) is None
    assert ext._as_finite_number(3) == 3.0
    assert ext._as_finite_number(True) is None


def _malformed_diagnostics_record(**metric_overrides) -> dict:
    summary = {"score": 0.5, "division_tp": 0, "division_fp": 0, "division_fn": 0}
    summary.update({k: v for k, v in metric_overrides.items() if k.startswith("division_")})
    metrics = {"edge_tp": 1, "edge_fp": 0}
    metrics.update({k: v for k, v in metric_overrides.items() if k in ("edge_tp", "edge_fp")})
    return {
        "trial_id": "malformed", "status": "success", "summary_metrics": summary,
        "per_dataset_metrics": [{"dataset": "dsA", "metrics": metrics}],
    }


def test_compute_diagnostics_rejects_nan_edge_tp() -> None:
    with pytest.raises(base.CalibrationError, match="finite"):
        ext.compute_diagnostics(_malformed_diagnostics_record(edge_tp=float("nan")))


def test_compute_diagnostics_rejects_infinite_edge_fp() -> None:
    with pytest.raises(base.CalibrationError, match="finite"):
        ext.compute_diagnostics(_malformed_diagnostics_record(edge_fp=float("inf")))


def test_compute_diagnostics_rejects_noninteger_division_fp() -> None:
    with pytest.raises(base.CalibrationError, match="integer-valued"):
        ext.compute_diagnostics(_malformed_diagnostics_record(division_fp=1.5))


def test_compute_diagnostics_rejects_negative_division_tp() -> None:
    with pytest.raises(base.CalibrationError, match="nonnegative"):
        ext.compute_diagnostics(_malformed_diagnostics_record(division_tp=-1))


def test_compute_diagnostics_rejects_missing_edge_tp() -> None:
    record = {
        "trial_id": "malformed", "status": "success",
        "summary_metrics": {"score": 0.5, "division_tp": 0, "division_fp": 0, "division_fn": 0},
        "per_dataset_metrics": [{"dataset": "dsA", "metrics": {"edge_fp": 0}}],  # edge_tp missing
    }
    with pytest.raises(base.CalibrationError, match="finite"):
        ext.compute_diagnostics(record)


def test_compute_diagnostics_rejects_missing_per_dataset_metrics() -> None:
    record = {
        "trial_id": "malformed", "status": "success",
        "summary_metrics": {"score": 0.5, "division_tp": 0, "division_fp": 0, "division_fn": 0},
    }
    with pytest.raises(base.CalibrationError, match="per_dataset_metrics"):
        ext.compute_diagnostics(record)


# =============================================================================
# Boundary warnings
# =============================================================================

def test_stage_a_upper_bound_warning_triggered(env: Environment) -> None:
    selection = {"7500": [0.86, 0.82], "10000": [0.70, 0.74]}
    warnings = ext.stage_a_boundary_warnings(selection, env.config["stage_a"])
    assert len(warnings) == 1
    assert warnings[0]["checkpoint_label"] == "7500"
    assert warnings[0]["kind"] == "det_threshold_upper_bound"


def test_stage_a_upper_bound_warning_absent_when_not_at_bound(env: Environment) -> None:
    selection = {"7500": [0.78, 0.82], "10000": [0.70, 0.74]}
    warnings = ext.stage_a_boundary_warnings(selection, env.config["stage_a"])
    assert warnings == []


def test_stage_b_lower_bound_warning_triggered(env: Environment) -> None:
    selection = {"7500": [(0.74, 0.20), (0.74, 0.25)], "10000": [(0.82, 0.30), (0.82, 0.35)]}
    warnings = ext.stage_b_boundary_warnings(selection, env.config["stage_b"])
    assert len(warnings) == 1
    assert warnings[0]["checkpoint_label"] == "7500"
    assert warnings[0]["kind"] == "edge_threshold_lower_bound"


# =============================================================================
# Complete-grid enforcement and failure handling
# =============================================================================

def test_failed_trial_blocks_selection_but_stays_recorded(env: Environment, monkeypatch, tmp_path: Path) -> None:
    dets = env.config["stage_a"]["det_thresholds"]
    fail_id = ext._trial_id("stage_a", "7500", dets[0], 0.35, 5.0)
    install_fake_subprocess(monkeypatch, fail_predict_trial_ids=frozenset({fail_id}))

    out_dir = tmp_path / "stage_a_fail"
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)
    with pytest.raises(base.CalibrationError, match="succeeded"):
        base.run_trials(
            trials, env.data_dir, out_dir, env.config_path, "stage_a",
            build_selection=lambda records: {
                "selected_det_thresholds_by_checkpoint": ext.select_best_det_thresholds_by_checkpoint(records, 2, CHECKPOINT_LABELS),
            },
        )
    payload = json.loads((out_dir / "calibration_results.json").read_text(encoding="utf-8"))
    failed = [r for r in payload["trials"] if r["trial_id"] == fail_id]
    assert failed and failed[0]["status"] == "failed"
    assert "selected_det_thresholds_by_checkpoint" not in payload


def test_failed_trial_retries_on_rerun(env: Environment, monkeypatch, tmp_path: Path) -> None:
    dets = env.config["stage_a"]["det_thresholds"]
    fail_id = ext._trial_id("stage_a", "7500", dets[0], 0.35, 5.0)
    out_dir = tmp_path / "stage_a_retry"
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)

    install_fake_subprocess(monkeypatch, fail_predict_trial_ids=frozenset({fail_id}))
    with pytest.raises(base.CalibrationError):
        base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a", build_selection=lambda r: {
            "selected_det_thresholds_by_checkpoint": ext.select_best_det_thresholds_by_checkpoint(r, 2, CHECKPOINT_LABELS),
        })

    install_fake_subprocess(monkeypatch)  # no more failures
    report = base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a", build_selection=lambda r: {
        "selected_det_thresholds_by_checkpoint": ext.select_best_det_thresholds_by_checkpoint(r, 2, CHECKPOINT_LABELS),
    })
    assert all(r["status"] == "success" for r in report["trials"])
    retried = next(r for r in report["trials"] if r["trial_id"] == fail_id)
    assert retried["resumed"] is False  # this exact trial had to rerun, not resume


def test_stale_trial_outputs_do_not_survive_rerun(env: Environment, monkeypatch, tmp_path: Path) -> None:
    """A trial directory left over from a failed attempt must be wiped, not merged into a retry."""
    dets = env.config["stage_a"]["det_thresholds"]
    trial_id = ext._trial_id("stage_a", "7500", dets[0], 0.35, 5.0)
    out_dir = tmp_path / "stage_a_stale"
    stale_trial_dir = out_dir / "trials" / trial_id
    stale_trial_dir.mkdir(parents=True, exist_ok=True)
    (stale_trial_dir / "predictions").mkdir(exist_ok=True)
    (stale_trial_dir / "predictions" / "stale_leftover.geff").write_bytes(b"stale")

    install_fake_subprocess(monkeypatch)
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)
    report = base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a", build_selection=lambda r: {
        "selected_det_thresholds_by_checkpoint": ext.select_best_det_thresholds_by_checkpoint(r, 2, CHECKPOINT_LABELS),
    })
    record = next(r for r in report["trials"] if r["trial_id"] == trial_id)
    assert record["status"] == "success"
    assert not (Path(record["prediction_dir"]) / "stale_leftover.geff").exists()


def test_cleanup_after_prediction_failure_removes_scratch_dir(env: Environment, monkeypatch, tmp_path: Path) -> None:
    dets = env.config["stage_a"]["det_thresholds"]
    fail_id = ext._trial_id("stage_a", "7500", dets[0], 0.35, 5.0)
    install_fake_subprocess(monkeypatch, missing_dataset_trial_ids=frozenset({fail_id}))
    out_dir = tmp_path / "stage_a_partial"
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)
    method = f"calib_{fail_id}"
    scratch_dir = base._prediction_scratch_dir(base.REPO_ROOT, base._username(), method, 0)
    try:
        with pytest.raises(base.CalibrationError):
            base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a", build_selection=lambda r: {
                "selected_det_thresholds_by_checkpoint": ext.select_best_det_thresholds_by_checkpoint(r, 2, CHECKPOINT_LABELS),
            })
        assert not scratch_dir.exists()
    finally:
        if scratch_dir.exists():
            import shutil
            shutil.rmtree(scratch_dir)


def test_cleanup_after_evaluation_failure(env: Environment, monkeypatch, tmp_path: Path) -> None:
    dets = env.config["stage_a"]["det_thresholds"]
    fail_id = ext._trial_id("stage_a", "7500", dets[0], 0.35, 5.0)
    install_fake_subprocess(monkeypatch, fail_evaluate_trial_ids=frozenset({fail_id}))
    out_dir = tmp_path / "stage_a_eval_fail"
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)
    method = f"calib_{fail_id}"
    scratch_dir = base._prediction_scratch_dir(base.REPO_ROOT, base._username(), method, 0)
    try:
        with pytest.raises(base.CalibrationError):
            base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a", build_selection=lambda r: {
                "selected_det_thresholds_by_checkpoint": ext.select_best_det_thresholds_by_checkpoint(r, 2, CHECKPOINT_LABELS),
            })
        assert not scratch_dir.exists()
        record = json.loads((out_dir / "calibration_results.json").read_text())
        failed = next(r for r in record["trials"] if r["trial_id"] == fail_id)
        assert failed["status"] == "failed"
    finally:
        if scratch_dir.exists():
            import shutil
            shutil.rmtree(scratch_dir)


# =============================================================================
# Training-script refusal
# =============================================================================

def test_no_training_script_reference_possible() -> None:
    import inspect
    source = inspect.getsource(ext)
    assert "train_unet_transformer" not in source


def test_execute_trial_refuses_command_referencing_training_script(env: Environment, monkeypatch, tmp_path: Path) -> None:
    def evil_build_prediction_command(trial, data_dir, method):
        return [str(sys.executable), str(base.TRAIN_SCRIPT), "--bogus"]

    monkeypatch.setattr(base, "build_prediction_command", evil_build_prediction_command)
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)[:1]
    out_dir = tmp_path / "evil"
    report = base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a")
    record = report["trials"][0]
    assert record["status"] == "failed"
    assert "training script" in record["error"]


# =============================================================================
# CWD independence
# =============================================================================

def test_cwd_independent_config_loading(env: Environment, tmp_path: Path) -> None:
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    original_cwd = Path.cwd()
    import os
    os.chdir(other_dir)
    try:
        config = ext.load_config(env.config_path)
        assert config["checkpoint_labels"] == ["7500", "10000"]
        av.load_and_validate_audit(config["audit"], full_split_datasets=env.full_split_datasets)
    finally:
        os.chdir(original_cwd)


def test_repo_root_independent_of_cwd() -> None:
    assert ext.REPO_ROOT == REPO_ROOT.resolve()
    assert av.REPO_ROOT == REPO_ROOT.resolve()


# =============================================================================
# Previous-stage validation and tamper rejection
# =============================================================================

def test_stage_b_rejects_tampered_stage_a_selection(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "a"
    _run_stage_a_happy(env, monkeypatch, out_dir)
    results_path = out_dir / "calibration_results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["selected_det_thresholds_by_checkpoint"]["7500"] = [0.70, 0.74]  # tamper
    results_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(base.CalibrationError, match="tampering|recomputation"):
        ext.load_stage_a_report_for_stage_b(
            results_path, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
        )


def test_stage_b_rejects_stale_config_hash(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "a"
    _run_stage_a_happy(env, monkeypatch, out_dir)
    results_path = out_dir / "calibration_results.json"

    def mutate(config):
        config["stage_a"]["det_thresholds"] = [0.70, 0.74, 0.78, 0.82, 0.90]
        config["stage_a"]["det_threshold_upper_bound_warning"] = 0.90
    new_config_path = env.remake(mutate, name="mutated_stale_config.json")
    with pytest.raises(base.CalibrationError, match="det_thresholds"):
        ext.load_config(new_config_path)  # the reviewed plan itself refuses a different det_thresholds grid


def test_stage_b_rejects_stale_audit_hash(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "a"
    _run_stage_a_happy(env, monkeypatch, out_dir)
    results_path = out_dir / "calibration_results.json"
    with pytest.raises(base.CalibrationError, match="audit hash"):
        ext.load_stage_a_report_for_stage_b(
            results_path, env.config, env.config_path, env.checkpoints, env.data_dir, "0" * 64,
        )


def test_stage_b_rejects_checkpoint_hash_tampering(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "a"
    _run_stage_a_happy(env, monkeypatch, out_dir)
    results_path = out_dir / "calibration_results.json"
    # Mutate the checkpoint bytes on disk after Stage A ran.
    ckpt_path = Path(env.checkpoints["7500"])
    ckpt_path.write_bytes(b"tampered-checkpoint-bytes")

    with pytest.raises(base.CalibrationError, match="revalidation"):
        ext.load_stage_a_report_for_stage_b(
            results_path, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
        )


def test_stage_b_rejects_checkpoint_config_hash_tampering(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "a"
    _run_stage_a_happy(env, monkeypatch, out_dir)
    results_path = out_dir / "calibration_results.json"
    ckpt_config_path = Path(env.checkpoints["7500"]).parent / "config.json"
    ckpt_config_path.write_text(json.dumps({"pool_kernel_um": 999.0}), encoding="utf-8")

    with pytest.raises(base.CalibrationError, match="revalidation"):
        ext.load_stage_a_report_for_stage_b(
            results_path, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
        )


def test_stage_b_rejects_split_hash_tampering(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "a"
    _run_stage_a_happy(env, monkeypatch, out_dir)
    results_path = out_dir / "calibration_results.json"
    _make_split_file(env.subset_split_path, SUBSET_NAMES[:-1] + ["dsZ"])  # change the test list

    with pytest.raises(base.CalibrationError, match="revalidation"):
        ext.load_stage_a_report_for_stage_b(
            results_path, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
        )


def test_stage_b_rejects_prediction_manifest_tampering(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "a"
    report = _run_stage_a_happy(env, monkeypatch, out_dir)
    record = report["trials"][0]
    pred_dir = Path(record["prediction_dir"])
    geff_files = sorted(pred_dir.glob("*.geff"))
    geff_files[0].write_bytes(b"tampered-geff-bytes")

    results_path = out_dir / "calibration_results.json"
    with pytest.raises(base.CalibrationError, match="revalidation"):
        ext.load_stage_a_report_for_stage_b(
            results_path, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
        )


def test_stage_b_rejects_metrics_tampering(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "a"
    report = _run_stage_a_happy(env, monkeypatch, out_dir)
    record = report["trials"][0]
    metrics_path = Path(record["metrics_path"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["summary_metrics"]["score"] = 0.9999
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    results_path = out_dir / "calibration_results.json"
    with pytest.raises(base.CalibrationError, match="revalidation"):
        ext.load_stage_a_report_for_stage_b(
            results_path, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
        )


def test_stage_b_rejects_incomplete_grid(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "a_partial"
    scores = {}
    for label in CHECKPOINT_LABELS:
        for i, det in enumerate(env.config["stage_a"]["det_thresholds"]):
            scores[ext._trial_id("stage_a", label, det, 0.35, 5.0)] = 0.5 + 0.01 * i
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)[:-1]  # drop one planned trial
    base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a")  # no selection requested

    results_path = out_dir / "calibration_results.json"
    with pytest.raises(base.CalibrationError, match="does not exactly match|audit hash"):
        ext.load_stage_a_report_for_stage_b(
            results_path, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
        )


def test_stage_c_reconstructs_and_validates_stage_b_grid(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir_a = tmp_path / "a"
    report_a = _run_stage_a_happy(env, monkeypatch, out_dir_a)
    det_by_checkpoint = report_a["selected_det_thresholds_by_checkpoint"]

    scores = {}
    for label, dets in det_by_checkpoint.items():
        for det in dets:
            for i, edge in enumerate(env.config["stage_b"]["edge_thresholds"]):
                scores[ext._trial_id("stage_b", label, det, edge, 5.0)] = 0.5 + 0.01 * i
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)
    trials_b = ext.build_stage_b_trials(env.config, env.checkpoints, det_by_checkpoint)
    out_dir_b = tmp_path / "b"
    previous_results_a = ext._build_previous_results_provenance(
        out_dir_a / "calibration_results.json", "stage_a", det_by_checkpoint,
    )
    report_b = base.run_trials(trials_b, env.data_dir, out_dir_b, env.config_path, "stage_b", build_selection=lambda r: {
        "audit_sha256": env.audit_sha256,
        "previous_results": previous_results_a,
        "selected_pairs_by_checkpoint": {
            k: [list(p) for p in v]
            for k, v in ext.select_best_pairs_by_checkpoint(r, 2, CHECKPOINT_LABELS).items()
        },
    })
    results_path_b = out_dir_b / "calibration_results.json"
    _, pairs = ext.load_stage_b_report_for_stage_c(
        results_path_b, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
    )
    assert set(pairs) == {"7500", "10000"}
    for label, dets in det_by_checkpoint.items():
        for det, edge in pairs[label]:
            assert det in dets


def test_stage_c_rejects_pairs_outside_reviewed_grid(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir_a = tmp_path / "a"
    report_a = _run_stage_a_happy(env, monkeypatch, out_dir_a)
    det_by_checkpoint = report_a["selected_det_thresholds_by_checkpoint"]

    scores = {}
    for label, dets in det_by_checkpoint.items():
        for det in dets:
            for i, edge in enumerate(env.config["stage_b"]["edge_thresholds"]):
                scores[ext._trial_id("stage_b", label, det, edge, 5.0)] = 0.5 + 0.01 * i
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)
    trials_b = ext.build_stage_b_trials(env.config, env.checkpoints, det_by_checkpoint)
    out_dir_b = tmp_path / "b"
    previous_results_a = ext._build_previous_results_provenance(
        out_dir_a / "calibration_results.json", "stage_a", det_by_checkpoint,
    )
    base.run_trials(trials_b, env.data_dir, out_dir_b, env.config_path, "stage_b", build_selection=lambda r: {
        "audit_sha256": env.audit_sha256,
        "previous_results": previous_results_a,
        "selected_pairs_by_checkpoint": {
            k: [list(p) for p in v]
            for k, v in ext.select_best_pairs_by_checkpoint(r, 2, CHECKPOINT_LABELS).items()
        },
    })
    results_path_b = out_dir_b / "calibration_results.json"
    payload = json.loads(results_path_b.read_text(encoding="utf-8"))
    # Inject a record using an edge_threshold outside the reviewed stage_b grid.
    tampered_record = dict(payload["trials"][0])
    tampered_record["edge_threshold"] = 0.999
    tampered_record["trial_id"] = "stage_b__tampered"
    payload["trials"].append(tampered_record)
    results_path_b.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(base.CalibrationError):
        ext.load_stage_b_report_for_stage_c(
            results_path_b, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
        )


def test_stage_c_rejects_different_but_in_grid_consumed_selection(env: Environment, monkeypatch, tmp_path: Path) -> None:
    """A tampered previous_results.consumed_selection that is still fully in-grid must be rejected."""
    out_dir_a = tmp_path / "a"
    report_a = _run_stage_a_happy(env, monkeypatch, out_dir_a)
    det_by_checkpoint = report_a["selected_det_thresholds_by_checkpoint"]  # actual winners: [0.86, 0.82]

    scores = {}
    for label, dets in det_by_checkpoint.items():
        for det in dets:
            for i, edge in enumerate(env.config["stage_b"]["edge_thresholds"]):
                scores[ext._trial_id("stage_b", label, det, edge, 5.0)] = 0.5 + 0.01 * i
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)
    trials_b = ext.build_stage_b_trials(env.config, env.checkpoints, det_by_checkpoint)
    out_dir_b = tmp_path / "b"
    previous_results_a = ext._build_previous_results_provenance(
        out_dir_a / "calibration_results.json", "stage_a", det_by_checkpoint,
    )
    report_b = base.run_trials(trials_b, env.data_dir, out_dir_b, env.config_path, "stage_b", build_selection=lambda r: {
        "audit_sha256": env.audit_sha256,
        "previous_results": previous_results_a,
        "selected_pairs_by_checkpoint": {
            k: [list(p) for p in v]
            for k, v in ext.select_best_pairs_by_checkpoint(r, 2, CHECKPOINT_LABELS).items()
        },
    })
    results_path_b = Path(report_b["results_path"])

    payload = json.loads(results_path_b.read_text(encoding="utf-8"))
    # A different, still fully in-grid, det_threshold pair (not the actual [0.86, 0.82] winners).
    substitute_selection = {label: [0.86, 0.78] for label in CHECKPOINT_LABELS}
    assert substitute_selection != det_by_checkpoint
    payload["previous_results"]["consumed_selection"] = substitute_selection
    results_path_b.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(base.CalibrationError, match="lineage validation failed"):
        ext.load_stage_b_report_for_stage_c(
            results_path_b, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
        )


def test_stage_d_rejects_different_but_in_grid_consumed_selection(env: Environment, monkeypatch, tmp_path: Path) -> None:
    """The same tamper protection applies one link further up the chain (C's consumed selection)."""
    out_dir_a = tmp_path / "a"
    report_a = _run_stage_a_happy(env, monkeypatch, out_dir_a)
    det_by_checkpoint = report_a["selected_det_thresholds_by_checkpoint"]

    scores = {}
    for label, dets in det_by_checkpoint.items():
        for det in dets:
            for i, edge in enumerate(env.config["stage_b"]["edge_thresholds"]):
                scores[ext._trial_id("stage_b", label, det, edge, 5.0)] = 0.5 + 0.01 * i
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)
    trials_b = ext.build_stage_b_trials(env.config, env.checkpoints, det_by_checkpoint)
    out_dir_b = tmp_path / "b"
    previous_results_a = ext._build_previous_results_provenance(
        out_dir_a / "calibration_results.json", "stage_a", det_by_checkpoint,
    )
    report_b = base.run_trials(trials_b, env.data_dir, out_dir_b, env.config_path, "stage_b", build_selection=lambda r: {
        "audit_sha256": env.audit_sha256,
        "previous_results": previous_results_a,
        "selected_pairs_by_checkpoint": {
            k: [list(p) for p in v]
            for k, v in ext.select_best_pairs_by_checkpoint(r, 2, CHECKPOINT_LABELS).items()
        },
    })
    pairs_by_checkpoint = {k: [tuple(p) for p in v] for k, v in report_b["selected_pairs_by_checkpoint"].items()}

    scores2 = {}
    for label, pairs in pairs_by_checkpoint.items():
        for det, edge in pairs:
            scores2[ext._trial_id("stage_c", label, det, edge, 5.0)] = 0.80
            scores2[ext._trial_id("stage_c", label, det, edge, 6.0)] = 0.85
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores2)
    trials_c = ext.build_stage_c_trials(env.config, env.checkpoints, pairs_by_checkpoint)
    out_dir_c = tmp_path / "c"
    previous_results_b = ext._build_previous_results_provenance(
        Path(report_b["results_path"]), "stage_b", pairs_by_checkpoint,
    )
    report_c = base.run_trials(trials_c, env.data_dir, out_dir_c, env.config_path, "stage_c", build_selection=lambda r: {
        "audit_sha256": env.audit_sha256,
        "previous_results": previous_results_b,
        "selected_candidates_by_checkpoint": {
            k: [list(c) for c in v] for k, v in ext.select_best_candidates_by_checkpoint(r, 2, CHECKPOINT_LABELS).items()
        },
    })
    results_path_c = Path(report_c["results_path"])

    payload = json.loads(results_path_c.read_text(encoding="utf-8"))
    actual_pair = next(iter(pairs_by_checkpoint.values()))[0]
    substitute_pairs = {label: [list(actual_pair), [actual_pair[0], 0.30]] for label in CHECKPOINT_LABELS}
    assert substitute_pairs != {k: [list(p) for p in v] for k, v in pairs_by_checkpoint.items()}
    payload["previous_results"]["consumed_selection"] = substitute_pairs
    results_path_c.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(base.CalibrationError, match="lineage validation failed"):
        ext.load_stage_c_report_for_stage_d(
            results_path_c, env.config, env.config_path, env.checkpoints, env.data_dir, env.audit_sha256,
        )


# =============================================================================
# Resume revalidation (config/audit/split/checkpoint/manifest/metrics)
# =============================================================================

def test_resume_reuses_successful_trial_without_rerunning(env: Environment, monkeypatch, tmp_path: Path) -> None:
    install_fake_subprocess(monkeypatch)
    call_count = {"n": 0}
    original = base._stream_subprocess

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(base, "_stream_subprocess", counting)

    out_dir = tmp_path / "resume"
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)[:2]
    base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a")
    first_calls = call_count["n"]
    assert first_calls > 0

    base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a")
    assert call_count["n"] == first_calls  # fully resumed, no new subprocess calls


def test_audit_change_forces_full_stage_gate_failure_even_with_valid_cache(env: Environment, monkeypatch, tmp_path: Path) -> None:
    """Simulates the CLI-level guarantee: the audit gate is re-checked at the
    start of every stage invocation, so a changed audit blocks the whole
    stage before any resumed trial is even considered."""
    install_fake_subprocess(monkeypatch)
    out_dir = tmp_path / "gated"
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)[:2]
    base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a")

    # Now tamper with the audit file so its bytes no longer match the pinned hash.
    env.audit_path.write_text(env.audit_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(av.AuditValidationError, match="sha256"):
        av.load_and_validate_audit(env.config["audit"], full_split_datasets=env.full_split_datasets)


def test_stage_b_resume_refuses_when_previous_results_identity_changes(env: Environment, monkeypatch, tmp_path: Path) -> None:
    """A stage-B rerun pointed at a *different* --previous-results file must not silently
    resume a cache built from the old one, even if the new upstream run happens to select
    the exact same det_threshold values (so per-trial fields alone would otherwise match)."""
    out_dir_a1 = tmp_path / "a1"
    report_a1 = _run_stage_a_happy(env, monkeypatch, out_dir_a1)
    det_by_checkpoint_1 = report_a1["selected_det_thresholds_by_checkpoint"]

    out_dir_b = tmp_path / "b"
    argv = _main_args(env, stage="B", output_dir=out_dir_b, previous_results=Path(report_a1["results_path"]))
    _run_main(monkeypatch, argv)
    assert (out_dir_b / "calibration_results.json").is_file()

    out_dir_a2 = tmp_path / "a2"
    report_a2 = _run_stage_a_happy(env, monkeypatch, out_dir_a2)
    assert report_a2["selected_det_thresholds_by_checkpoint"] == det_by_checkpoint_1

    argv2 = _main_args(env, stage="B", output_dir=out_dir_b, previous_results=Path(report_a2["results_path"]))
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, argv2)


# =============================================================================
# Heartbeat / progress output
# =============================================================================

def test_heartbeat_fires_during_trial_execution(env: Environment, monkeypatch, tmp_path: Path, capsys) -> None:
    log: list[tuple[str, float]] = []
    install_fake_subprocess(monkeypatch, heartbeat_log=log)
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)[:1]
    out_dir = tmp_path / "heartbeat"
    base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a")
    assert len(log) >= 2  # prediction phase + evaluation phase both fired a heartbeat
    captured = capsys.readouterr()
    assert "starting" in captured.out


# =============================================================================
# Deterministic ties
# =============================================================================

def test_stage_a_tie_break_is_deterministic_ascending_det(env: Environment, monkeypatch, tmp_path: Path) -> None:
    scores = {}
    dets = env.config["stage_a"]["det_thresholds"]
    for label in CHECKPOINT_LABELS:
        for det in dets:
            scores[ext._trial_id("stage_a", label, det, 0.35, 5.0)] = 0.5  # exact tie
    install_fake_subprocess(monkeypatch, scores_by_trial_id=scores)
    trials = ext.build_stage_a_trials(env.config, env.checkpoints)
    out_dir = tmp_path / "ties"
    report = base.run_trials(trials, env.data_dir, out_dir, env.config_path, "stage_a", build_selection=lambda r: {
        "selected_det_thresholds_by_checkpoint": ext.select_best_det_thresholds_by_checkpoint(r, 2, CHECKPOINT_LABELS),
    })
    for label in CHECKPOINT_LABELS:
        assert report["selected_det_thresholds_by_checkpoint"][label] == sorted(dets)[:2]


def test_stage_d_tie_break_prefers_earlier_checkpoint() -> None:
    records = []
    for label in CHECKPOINT_LABELS:
        records.append({
            "trial_id": f"stage_d__{label}", "checkpoint_label": label, "det_threshold": 0.74,
            "edge_threshold": 0.20, "pool_kernel_um": 5.0, "status": "success",
            "summary_metrics": {"score": 0.777},
        })
    winner = ext.select_stage_d_winner(records, ["7500", "10000"])
    assert winner["checkpoint_label"] == "7500"


# =============================================================================
# main() end-to-end CLI integration
# =============================================================================

def test_main_stage_a_end_to_end_happy_path(env: Environment, monkeypatch, tmp_path: Path, capsys) -> None:
    install_fake_subprocess(monkeypatch)
    out_dir = tmp_path / "main_a_happy"
    argv = _main_args(env, stage="A", output_dir=out_dir)
    _run_main(monkeypatch, argv)  # must not raise SystemExit
    payload = json.loads((out_dir / "calibration_results.json").read_text(encoding="utf-8"))
    assert "selected_det_thresholds_by_checkpoint" in payload
    assert payload["audit"]["dataset_count"] == len(FULL_NAMES)
    captured = capsys.readouterr()
    assert "Stage A complete" in captured.out


def test_main_refuses_tampered_checkpoint_before_subprocess(env: Environment, monkeypatch, tmp_path: Path) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("subprocess must not run when checkpoint pin validation fails")
    monkeypatch.setattr(base, "_stream_subprocess", boom)
    Path(env.checkpoints["7500"]).write_bytes(b"corrupted-checkpoint-bytes")
    out_dir = tmp_path / "pin_fail"
    argv = _main_args(env, stage="A", output_dir=out_dir)
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, argv)
    assert not (out_dir / "calibration_results.json").exists()


def test_main_records_full_audit_provenance_even_when_a_trial_fails(env: Environment, monkeypatch, tmp_path: Path) -> None:
    dets = env.config["stage_a"]["det_thresholds"]
    fail_id = ext._trial_id("stage_a", "7500", dets[0], 0.35, 5.0)
    install_fake_subprocess(monkeypatch, fail_predict_trial_ids=frozenset({fail_id}))
    out_dir = tmp_path / "main_fail"
    argv = _main_args(env, stage="A", output_dir=out_dir)
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, argv)
    payload = json.loads((out_dir / "calibration_results.json").read_text(encoding="utf-8"))
    assert "selected_det_thresholds_by_checkpoint" not in payload
    assert payload["audit_sha256"] == env.audit_sha256
    audit_block = payload["audit"]
    assert audit_block["dataset_count"] == len(FULL_NAMES)
    assert audit_block["node_count"] > 0
    assert audit_block["tolerance"] == 0.001
    assert "primary" in audit_block["findings"] and "secondary" in audit_block["findings"]


# =============================================================================
# Encoding cleanup
# =============================================================================

def test_no_corrupted_encoding_sequences_in_source_files() -> None:
    corrupted_sequence = chr(0x0393) + chr(0x00C7) + chr(0x00F6)  # the literal "mojibake" em-dash sequence
    paths = [
        REPO_ROOT / "scripts" / "audit_validation.py",
        REPO_ROOT / "scripts" / "calibrate_extended_inference.py",
        REPO_ROOT / "tests" / "test_extended_calibration_runner.py",
        REAL_CONFIG_PATH,
        REAL_AUDIT_PATH,
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        # This check itself deliberately avoids containing the literal sequence
        # (it is built from chr() codepoints above), so scanning this test file
        # cannot trip a false positive against its own source.
        assert corrupted_sequence not in text, f"{path} contains a corrupted encoding sequence"


# =============================================================================
# Backward compatibility of the original module
# =============================================================================

def test_original_runner_functions_untouched() -> None:
    config = base.load_config(ORIGINAL_CONFIG_PATH)
    trials = base.build_stage_a_trials(config, {"10000": "/fake/path.pth"})
    assert len(trials) == len(config["stage_a"]["det_thresholds"]) * len(config["stage_a"]["edge_thresholds"])


def test_original_config_file_unchanged_on_disk() -> None:
    config = json.loads(ORIGINAL_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["stage_c"]["checkpoint_labels"] == ["7500", "10000"]
    assert "audit" not in config


# =============================================================================
# previous_results provenance survives a failed Stage B/C/D report, and
# resume enforces it strictly (missing/malformed/different is always
# rejected, before any subprocess)
# =============================================================================

def test_stage_b_failed_report_preserves_and_enforces_previous_results(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir_a = tmp_path / "a"
    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(env, stage="A", output_dir=out_dir_a))
    report_a = json.loads((out_dir_a / "calibration_results.json").read_text(encoding="utf-8"))
    det_by_checkpoint = report_a["selected_det_thresholds_by_checkpoint"]

    fail_label = CHECKPOINT_LABELS[0]
    fail_det = det_by_checkpoint[fail_label][0]
    fail_edge = env.config["stage_b"]["edge_thresholds"][0]
    fail_id = ext._trial_id("stage_b", fail_label, fail_det, fail_edge, 5.0)
    install_fake_subprocess(monkeypatch, fail_predict_trial_ids=frozenset({fail_id}))

    out_dir_b = tmp_path / "b"
    previous_results_path_a = out_dir_a / "calibration_results.json"
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, _main_args(env, stage="B", output_dir=out_dir_b, previous_results=previous_results_path_a))

    payload = json.loads((out_dir_b / "calibration_results.json").read_text(encoding="utf-8"))
    assert "selected_pairs_by_checkpoint" not in payload
    prev = payload["previous_results"]
    assert prev["path"] == str(previous_results_path_a.resolve())
    assert prev["sha256"] == hashlib.sha256(previous_results_path_a.read_bytes()).hexdigest()
    assert prev["stage"] == "stage_a"
    assert prev["consumed_selection"] == det_by_checkpoint
    assert payload["audit"]["dataset_count"] == len(FULL_NAMES)  # audit provenance present despite the failure

    # Resume with the SAME upstream report: succeeds, retrying only the failed trial.
    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(env, stage="B", output_dir=out_dir_b, previous_results=previous_results_path_a))
    payload2 = json.loads((out_dir_b / "calibration_results.json").read_text(encoding="utf-8"))
    assert "selected_pairs_by_checkpoint" in payload2
    resumed_flags = {r["trial_id"]: r["resumed"] for r in payload2["trials"]}
    assert resumed_flags[fail_id] is False
    assert all(resumed for trial_id, resumed in resumed_flags.items() if trial_id != fail_id)

    # A different upstream report (even one that happens to select the same
    # values) must be rejected before any subprocess call.
    out_dir_a2 = tmp_path / "a2"
    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(env, stage="A", output_dir=out_dir_a2))
    report_a2 = json.loads((out_dir_a2 / "calibration_results.json").read_text(encoding="utf-8"))
    assert report_a2["selected_det_thresholds_by_checkpoint"] == det_by_checkpoint

    def boom(*args, **kwargs):
        raise AssertionError("subprocess must not run when previous_results identity drift is detected")
    monkeypatch.setattr(base, "_stream_subprocess", boom)
    argv_drift = _main_args(env, stage="B", output_dir=out_dir_b, previous_results=out_dir_a2 / "calibration_results.json")
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, argv_drift)


def test_stage_c_failed_report_preserves_and_enforces_previous_results(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir_a = tmp_path / "a"
    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(env, stage="A", output_dir=out_dir_a))
    report_a = json.loads((out_dir_a / "calibration_results.json").read_text(encoding="utf-8"))

    out_dir_b = tmp_path / "b"
    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(
        env, stage="B", output_dir=out_dir_b, previous_results=out_dir_a / "calibration_results.json",
    ))
    report_b = json.loads((out_dir_b / "calibration_results.json").read_text(encoding="utf-8"))
    pairs_by_checkpoint = report_b["selected_pairs_by_checkpoint"]

    fail_label = CHECKPOINT_LABELS[0]
    fail_det, fail_edge = pairs_by_checkpoint[fail_label][0]
    fail_id = ext._trial_id("stage_c", fail_label, fail_det, fail_edge, 5.0)
    install_fake_subprocess(monkeypatch, fail_predict_trial_ids=frozenset({fail_id}))

    out_dir_c = tmp_path / "c"
    previous_results_path_b = out_dir_b / "calibration_results.json"
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, _main_args(env, stage="C", output_dir=out_dir_c, previous_results=previous_results_path_b))

    payload = json.loads((out_dir_c / "calibration_results.json").read_text(encoding="utf-8"))
    assert "selected_candidates_by_checkpoint" not in payload
    prev = payload["previous_results"]
    assert prev["path"] == str(previous_results_path_b.resolve())
    assert prev["sha256"] == hashlib.sha256(previous_results_path_b.read_bytes()).hexdigest()
    assert prev["stage"] == "stage_b"
    assert prev["consumed_selection"] == pairs_by_checkpoint
    assert payload["audit"]["dataset_count"] == len(FULL_NAMES)

    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(env, stage="C", output_dir=out_dir_c, previous_results=previous_results_path_b))
    payload2 = json.loads((out_dir_c / "calibration_results.json").read_text(encoding="utf-8"))
    assert "selected_candidates_by_checkpoint" in payload2
    resumed_flags = {r["trial_id"]: r["resumed"] for r in payload2["trials"]}
    assert resumed_flags[fail_id] is False
    assert all(resumed for trial_id, resumed in resumed_flags.items() if trial_id != fail_id)

    out_dir_b2 = tmp_path / "b2"
    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(
        env, stage="B", output_dir=out_dir_b2, previous_results=out_dir_a / "calibration_results.json",
    ))
    report_b2 = json.loads((out_dir_b2 / "calibration_results.json").read_text(encoding="utf-8"))
    assert report_b2["selected_pairs_by_checkpoint"] == pairs_by_checkpoint

    def boom(*args, **kwargs):
        raise AssertionError("subprocess must not run when previous_results identity drift is detected")
    monkeypatch.setattr(base, "_stream_subprocess", boom)
    argv_drift = _main_args(env, stage="C", output_dir=out_dir_c, previous_results=out_dir_b2 / "calibration_results.json")
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, argv_drift)


def test_stage_d_failed_report_preserves_and_enforces_previous_results(env: Environment, monkeypatch, tmp_path: Path) -> None:
    out_dir_a = tmp_path / "a"
    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(env, stage="A", output_dir=out_dir_a))

    out_dir_b = tmp_path / "b"
    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(
        env, stage="B", output_dir=out_dir_b, previous_results=out_dir_a / "calibration_results.json",
    ))

    out_dir_c = tmp_path / "c"
    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(
        env, stage="C", output_dir=out_dir_c, previous_results=out_dir_b / "calibration_results.json",
    ))
    report_c = json.loads((out_dir_c / "calibration_results.json").read_text(encoding="utf-8"))
    candidates_by_checkpoint = report_c["selected_candidates_by_checkpoint"]

    fail_label = CHECKPOINT_LABELS[0]
    fail_det, fail_edge, fail_pool = candidates_by_checkpoint[fail_label][0]
    fail_id = ext._trial_id("stage_d", fail_label, fail_det, fail_edge, fail_pool)
    install_fake_subprocess(monkeypatch, fail_predict_trial_ids=frozenset({fail_id}))

    out_dir_d = tmp_path / "d"
    previous_results_path_c = out_dir_c / "calibration_results.json"
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, _main_args(env, stage="D", output_dir=out_dir_d, previous_results=previous_results_path_c))

    payload = json.loads((out_dir_d / "calibration_results.json").read_text(encoding="utf-8"))
    assert "selected_final_trial" not in payload
    prev = payload["previous_results"]
    assert prev["path"] == str(previous_results_path_c.resolve())
    assert prev["sha256"] == hashlib.sha256(previous_results_path_c.read_bytes()).hexdigest()
    assert prev["stage"] == "stage_c"
    assert prev["consumed_selection"] == candidates_by_checkpoint
    assert payload["audit"]["dataset_count"] == len(FULL_NAMES)

    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(env, stage="D", output_dir=out_dir_d, previous_results=previous_results_path_c))
    payload2 = json.loads((out_dir_d / "calibration_results.json").read_text(encoding="utf-8"))
    assert "selected_final_trial" in payload2
    resumed_flags = {r["trial_id"]: r["resumed"] for r in payload2["trials"]}
    assert resumed_flags[fail_id] is False
    assert all(resumed for trial_id, resumed in resumed_flags.items() if trial_id != fail_id)

    out_dir_c2 = tmp_path / "c2"
    install_fake_subprocess(monkeypatch)
    _run_main(monkeypatch, _main_args(
        env, stage="C", output_dir=out_dir_c2, previous_results=out_dir_b / "calibration_results.json",
    ))
    report_c2 = json.loads((out_dir_c2 / "calibration_results.json").read_text(encoding="utf-8"))
    assert report_c2["selected_candidates_by_checkpoint"] == candidates_by_checkpoint

    def boom(*args, **kwargs):
        raise AssertionError("subprocess must not run when previous_results identity drift is detected")
    monkeypatch.setattr(base, "_stream_subprocess", boom)
    argv_drift = _main_args(env, stage="D", output_dir=out_dir_d, previous_results=out_dir_c2 / "calibration_results.json")
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, argv_drift)


# =============================================================================
# Hard-locked reviewed checkpoint/audit identities: a well-formed alternative
# value must never be silently authorized (uses the REAL committed config and
# the REAL (unmocked) reviewed constants — not the `env` fixture, which
# monkeypatches those constants to its own synthetic identities).
# =============================================================================

def _real_config_with_mutation(tmp_path: Path, mutate, name: str = "real_variant.json") -> Path:
    config = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(config)
    path = tmp_path / name
    _write_json(path, config)
    return path


def _mutate_tolerance(config: dict) -> None:
    config["audit"]["tolerance"] = 0.5
    config["audit"]["quantized_collision_fraction_max"] = 0.5


@pytest.mark.parametrize(("mutate", "match"), [
    (lambda c: c["checkpoint_pins"]["7500"].update(sha256="a" * 64), "checkpoint_pins"),
    (lambda c: c["checkpoint_pins"]["10000"].update(sha256="b" * 64), "checkpoint_pins"),
    (lambda c: c["checkpoint_pins"]["7500"].update(filename="edge_predictor_iter_999999.pth"), "checkpoint_pins"),
    (lambda c: c["checkpoint_pins"].update(adjacent_config_sha256="c" * 64), "checkpoint_pins"),
    (lambda c: c["audit"].update(sha256="d" * 64), "audit.sha256"),
    (lambda c: c["audit"].update(path="experiments/audits/some_other_audit.json"), "audit.path"),
    (lambda c: c["audit"].update(dataset_count=20), "audit.dataset_count"),
    (lambda c: c["audit"].update(node_count=99999), "audit.node_count"),
    (_mutate_tolerance, "audit.tolerance"),
    (lambda c: c["audit"]["observed"]["secondary"].update(count=999, pair_count=999, fraction=0.5), "audit.observed"),
    (lambda c: c["audit"]["observed"]["primary"].update(count=1, pair_count=1, fraction=0.001), "audit.observed"),
])
def test_load_config_rejects_well_formed_alternative_identities(tmp_path: Path, mutate, match: str) -> None:
    """A newly edited config cannot authorize a different checkpoint or audit input merely by
    supplying another well-formed (correct hex length / internally self-consistent) value."""
    path = _real_config_with_mutation(tmp_path, mutate)
    with pytest.raises(base.CalibrationError, match=match):
        ext.load_config(path)


def test_main_rejects_well_formed_alternative_audit_hash_before_subprocess(tmp_path: Path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("subprocess must not run before the config identity hard-lock gate passes")
    monkeypatch.setattr(base, "_stream_subprocess", boom)
    config = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
    config["audit"]["sha256"] = "a" * 64  # well-formed but not the reviewed hash
    config_path = tmp_path / "tampered_audit.json"
    _write_json(config_path, config)
    argv = [
        "--config", str(config_path), "--stage", "A",
        "--checkpoint", "7500=/nonexistent/edge_predictor_iter_007500.pth",
        "--checkpoint", "10000=/nonexistent/edge_predictor_iter_010000.pth",
        "--data-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, argv)
    assert not (tmp_path / "out").exists()


def test_main_rejects_well_formed_alternative_checkpoint_hash_before_subprocess(tmp_path: Path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("subprocess must not run before the config identity hard-lock gate passes")
    monkeypatch.setattr(base, "_stream_subprocess", boom)
    config = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
    config["checkpoint_pins"]["7500"]["sha256"] = "e" * 64  # well-formed but not the reviewed hash
    config_path = tmp_path / "tampered_checkpoint.json"
    _write_json(config_path, config)
    argv = [
        "--config", str(config_path), "--stage", "A",
        "--checkpoint", "7500=/nonexistent/edge_predictor_iter_007500.pth",
        "--checkpoint", "10000=/nonexistent/edge_predictor_iter_010000.pth",
        "--data-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, argv)
    assert not (tmp_path / "out").exists()


def test_load_config_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    raw = REAL_CONFIG_PATH.read_text(encoding="utf-8").rstrip()
    assert raw.endswith("}")
    tampered = raw[:-1] + ', "schema_version": 1}'
    path = tmp_path / "dup_key.json"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(base.CalibrationError, match="duplicate JSON key"):
        ext.load_config(path)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_load_config_rejects_nonfinite_json_constants(tmp_path: Path, token: str) -> None:
    raw = REAL_CONFIG_PATH.read_text(encoding="utf-8")
    tampered = raw.replace(
        '"n_best_det_thresholds_per_checkpoint": 2', f'"n_best_det_thresholds_per_checkpoint": {token}', 1,
    )
    assert tampered != raw
    path = tmp_path / f"nonfinite_{token.strip('-')}.json"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(base.CalibrationError, match="non-finite"):
        ext.load_config(path)


def test_parse_and_validate_checkpoints_rejects_missing_label() -> None:
    with pytest.raises(base.CalibrationError, match="missing"):
        ext.parse_and_validate_checkpoints(["7500=/a/x.pth"])


def test_parse_and_validate_checkpoints_requires_exact_label_set() -> None:
    result = ext.parse_and_validate_checkpoints(["7500=/a/x.pth", "10000=/a/y.pth"])
    assert set(result) == {"7500", "10000"}


# =============================================================================
# Audit provenance is always written first, and survives a diagnostics
# failure caused by malformed successful-trial metrics; no selection is
# published from malformed metrics.
# =============================================================================

def test_malformed_successful_trial_metrics_fails_closed_and_preserves_audit(
    env: Environment, monkeypatch, tmp_path: Path,
) -> None:
    dets = env.config["stage_a"]["det_thresholds"]
    malformed_id = ext._trial_id("stage_a", "7500", dets[0], 0.35, 5.0)
    install_fake_subprocess(monkeypatch, per_dataset_overrides={malformed_id: {"edge_tp": -5}})
    out_dir = tmp_path / "malformed"
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, _main_args(env, stage="A", output_dir=out_dir))

    payload = json.loads((out_dir / "calibration_results.json").read_text(encoding="utf-8"))
    assert "selected_det_thresholds_by_checkpoint" not in payload  # no selection published
    assert payload["audit_sha256"] == env.audit_sha256  # audit provenance preserved despite the failure
    audit_block = payload["audit"]
    assert audit_block["dataset_count"] == len(FULL_NAMES)
    assert audit_block["node_count"] > 0
    assert "primary" in audit_block["findings"] and "secondary" in audit_block["findings"]

    malformed_record = next(r for r in payload["trials"] if r["trial_id"] == malformed_id)
    assert malformed_record["status"] == "success"  # the trial itself succeeded on disk
    assert "diagnostics" not in malformed_record  # but diagnostics annotation also failed closed
