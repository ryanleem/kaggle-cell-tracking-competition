import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import build_calibration_subset_split as subset_builder
import calibrate_inference_thresholds as calib

CONFIG_PATH = Path(__file__).parent.parent / "experiments" / "configs" / "inference_threshold_calibration.json"
SUBSET_SPLIT_PATH = Path(__file__).parent.parent / "experiments" / "splits" / "calibration_subset_fold0.json"
FULL_SPLIT_PATH = Path(__file__).parent.parent / "experiments" / "splits" / "baseline_fold0_seed0.json"

_KNOWN_SUBSET_NAMES = [
    "44b6_d5e7d891",
    "44b6_d754aa59",
    "44b6_e57ff5c6",
    "6bba_268e1230",
    "6bba_3a1849c2",
    "6bba_afb141ff",
]


# =============================================================================
# Config + subset split
# =============================================================================

def test_committed_config_is_valid_and_loadable() -> None:
    config = calib.load_config(CONFIG_PATH)
    assert config["stage_a"]["det_thresholds"] == [0.52, 0.58, 0.64, 0.70]
    assert config["stage_a"]["edge_thresholds"] == [0.50]
    assert config["stage_a"]["checkpoint_labels"] == ["10000"]
    assert config["stage_b"]["edge_thresholds"] == [0.35, 0.45, 0.50, 0.55, 0.65]
    assert config["stage_c"]["checkpoint_labels"] == ["7500", "10000"]


def test_load_config_requires_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"pool_kernel_um": 5.0}))
    with pytest.raises(calib.CalibrationError, match="missing required keys"):
        calib.load_config(path)


def test_load_config_requires_stage_keys(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text())
    del config["stage_a"]["det_thresholds"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(calib.CalibrationError, match="stage_a missing required keys"):
        calib.load_config(path)


def _write_config(tmp_path: Path, mutate) -> Path:
    config = json.loads(CONFIG_PATH.read_text())
    mutate(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_load_config_rejects_non_finite_pool_kernel_um(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text())
    config["pool_kernel_um"] = float("nan")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, allow_nan=True))
    with pytest.raises(calib.CalibrationError, match="pool_kernel_um"):
        calib.load_config(path)


def test_load_config_rejects_non_positive_pool_kernel_um(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda c: c.update(pool_kernel_um=0.0))
    with pytest.raises(calib.CalibrationError, match="pool_kernel_um"):
        calib.load_config(path)


def test_load_config_rejects_out_of_range_threshold(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda c: c["stage_a"].update(det_thresholds=[0.5, 1.5]))
    with pytest.raises(calib.CalibrationError, match="det_thresholds"):
        calib.load_config(path)


def test_load_config_rejects_duplicate_stage_a_det_thresholds(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda c: c["stage_a"].update(det_thresholds=[0.52, 0.58, 0.58, 0.70]))
    with pytest.raises(calib.CalibrationError, match="stage_a.det_thresholds contains duplicates"):
        calib.load_config(path)


def test_load_config_rejects_duplicate_stage_a_edge_thresholds(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda c: c["stage_a"].update(edge_thresholds=[0.50, 0.50]))
    with pytest.raises(calib.CalibrationError, match="stage_a.edge_thresholds contains duplicates"):
        calib.load_config(path)


def test_load_config_rejects_duplicate_stage_b_edge_thresholds(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path, lambda c: c["stage_b"].update(edge_thresholds=[0.35, 0.45, 0.45, 0.55, 0.65]),
    )
    with pytest.raises(calib.CalibrationError, match="stage_b.edge_thresholds contains duplicates"):
        calib.load_config(path)


def test_load_config_rejects_n_best_det_thresholds_exceeding_unique_candidates(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda c: c["stage_a"].update(n_best_det_thresholds=99))
    with pytest.raises(calib.CalibrationError, match="n_best_det_thresholds"):
        calib.load_config(path)


def test_load_config_rejects_n_best_pairs_exceeding_unique_candidates(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda c: c["stage_b"].update(n_best_pairs=99))
    with pytest.raises(calib.CalibrationError, match="n_best_pairs"):
        calib.load_config(path)


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, True])
def test_load_config_rejects_non_positive_integer_selection_counts(tmp_path: Path, bad_value) -> None:
    path = _write_config(tmp_path, lambda c: c["stage_a"].update(n_best_det_thresholds=bad_value))
    with pytest.raises(calib.CalibrationError):
        calib.load_config(path)


def test_load_config_rejects_stage_a_checkpoint_labels_not_matching_reviewed_plan(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda c: c["stage_a"].update(checkpoint_labels=["10000", "extra"]))
    with pytest.raises(calib.CalibrationError, match="reviewed plan"):
        calib.load_config(path)


def test_load_config_rejects_stage_c_checkpoint_labels_not_matching_reviewed_plan(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda c: c["stage_c"].update(checkpoint_labels=["10000"]))
    with pytest.raises(calib.CalibrationError, match="reviewed plan"):
        calib.load_config(path)


def test_load_config_rejects_duplicate_checkpoint_labels_in_stage(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda c: c["stage_c"].update(checkpoint_labels=["7500", "7500"]))
    with pytest.raises(calib.CalibrationError, match="duplicates"):
        calib.load_config(path)


def test_load_config_rejects_subset_dataset_names_mismatch(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda c: c.update(subset_dataset_names=["bogus_dataset"]))
    with pytest.raises(calib.CalibrationError, match="does not exactly match"):
        calib.load_config(path)


def test_subset_split_matches_existing_schema_and_full_fold_membership() -> None:
    full = json.loads(FULL_SPLIT_PATH.read_text())
    subset = json.loads(SUBSET_SPLIT_PATH.read_text())

    assert isinstance(full, list) and isinstance(subset, list)
    assert set(subset[0]) == {"split", "train", "test"} == set(full[0])
    assert isinstance(subset[0]["split"], int)
    assert subset[0]["test"] == _KNOWN_SUBSET_NAMES
    assert set(subset[0]["test"]).issubset(set(full[0]["test"]))
    assert subset[0]["train"] == full[0]["train"]


def test_config_subset_dataset_names_match_committed_split() -> None:
    config = calib.load_config(CONFIG_PATH)
    subset = json.loads(SUBSET_SPLIT_PATH.read_text())
    assert sorted(config["subset_dataset_names"]) == sorted(subset[0]["test"])
    assert config["subset_split_file"] == "experiments/splits/calibration_subset_fold0.json"


def test_build_subset_fold_rejects_names_outside_source_test() -> None:
    source_fold = {"split": 0, "train": ["a"], "test": ["b", "c"]}
    with pytest.raises(subset_builder.SubsetSplitError, match="not found"):
        subset_builder.build_subset_fold(source_fold, ["b", "d"], 0)


def test_build_subset_fold_rejects_duplicates_and_empty() -> None:
    source_fold = {"split": 0, "train": ["a"], "test": ["b", "c"]}
    with pytest.raises(subset_builder.SubsetSplitError, match="empty"):
        subset_builder.build_subset_fold(source_fold, [], 0)
    with pytest.raises(subset_builder.SubsetSplitError, match="duplicates"):
        subset_builder.build_subset_fold(source_fold, ["b", "b"], 0)


def test_build_subset_fold_preserves_train_and_reuses_schema() -> None:
    source_fold = {"split": 0, "train": ["a", "b"], "test": ["c", "d", "e"]}
    result = subset_builder.build_subset_fold(source_fold, ["c", "e"], 0)
    assert result == {"split": 0, "train": ["a", "b"], "test": ["c", "e"]}


# =============================================================================
# Checkpoint label parsing
# =============================================================================

def test_parse_checkpoint_args_valid() -> None:
    parsed = calib.parse_checkpoint_args(["10000=weights/a.pth", "7500=weights/b.pth"])
    assert parsed == {"10000": "weights/a.pth", "7500": "weights/b.pth"}


@pytest.mark.parametrize("bad", ["no-equals-sign", "=missing-label", "label="])
def test_parse_checkpoint_args_rejects_malformed(bad: str) -> None:
    with pytest.raises(calib.CalibrationError):
        calib.parse_checkpoint_args([bad])


def test_parse_checkpoint_args_rejects_duplicate_labels() -> None:
    with pytest.raises(calib.CalibrationError, match="duplicate"):
        calib.parse_checkpoint_args(["10000=weights/a.pth", "10000=weights/b.pth"])


def test_resolve_checkpoint_path_reports_missing_label() -> None:
    with pytest.raises(calib.CalibrationError, match="10000"):
        calib._resolve_checkpoint_path({"7500": "x.pth"}, "10000")


# =============================================================================
# Exact staged grid generation + robust path resolution
# =============================================================================

def _checkpoints() -> dict[str, str]:
    return {"7500": "weights/ckpt_7500.pth", "10000": "weights/ckpt_10000.pth"}


def test_stage_a_grid_is_exactly_four_trials_one_checkpoint() -> None:
    config = calib.load_config(CONFIG_PATH)
    trials = calib.build_stage_a_trials(config, _checkpoints())

    assert len(trials) == 4
    assert {t["det_threshold"] for t in trials} == {0.52, 0.58, 0.64, 0.70}
    assert {t["edge_threshold"] for t in trials} == {0.50}
    assert {t["checkpoint_label"] for t in trials} == {"10000"}
    assert all(t["checkpoint_path"] == "weights/ckpt_10000.pth" for t in trials)
    assert all(t["is_screening"] is True for t in trials)
    expected_split_file = str(calib._resolve_repo_path(config["subset_split_file"]))
    assert all(t["split_file"] == expected_split_file for t in trials)
    assert all(t["split"] == config["subset_split"] for t in trials)
    assert len({t["trial_id"] for t in trials}) == 4  # unique IDs


def test_stage_b_grid_is_exactly_ten_trials_two_dets_five_edges() -> None:
    config = calib.load_config(CONFIG_PATH)
    trials = calib.build_stage_b_trials(config, _checkpoints(), [0.58, 0.64])

    assert len(trials) == 10
    assert {t["det_threshold"] for t in trials} == {0.58, 0.64}
    assert {t["edge_threshold"] for t in trials} == {0.35, 0.45, 0.50, 0.55, 0.65}
    assert {t["checkpoint_label"] for t in trials} == {"10000"}
    assert all(t["is_screening"] is True for t in trials)
    expected_split_file = str(calib._resolve_repo_path(config["subset_split_file"]))
    assert all(t["split_file"] == expected_split_file for t in trials)


def test_stage_c_grid_is_exactly_four_trials_two_pairs_two_checkpoints() -> None:
    config = calib.load_config(CONFIG_PATH)
    best_pairs = [(0.58, 0.45), (0.64, 0.50)]
    trials = calib.build_stage_c_trials(config, _checkpoints(), best_pairs)

    assert len(trials) == 4
    assert {t["checkpoint_label"] for t in trials} == {"7500", "10000"}
    assert {(t["det_threshold"], t["edge_threshold"]) for t in trials} == set(best_pairs)
    assert all(t["is_screening"] is False for t in trials)
    expected_split_file = str(calib._resolve_repo_path(config["full_split_file"]))
    assert all(t["split_file"] == expected_split_file for t in trials)
    assert all(t["split"] == config["full_split"] for t in trials)


def test_stage_a_missing_checkpoint_label_raises() -> None:
    config = calib.load_config(CONFIG_PATH)
    with pytest.raises(calib.CalibrationError, match="10000"):
        calib.build_stage_a_trials(config, {"7500": "x.pth"})


def test_resolve_repo_path_is_independent_of_caller_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = calib._resolve_repo_path("experiments/splits/calibration_subset_fold0.json")
    assert resolved == calib.REPO_ROOT / "experiments" / "splits" / "calibration_subset_fold0.json"
    assert resolved.is_file()


def test_make_trial_resolves_split_file_independent_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    trial = calib._make_trial(
        stage="stage_a", checkpoint_label="10000", checkpoint_path="ckpt.pth",
        det_threshold=0.5, edge_threshold=0.5, pool_kernel_um=5.0, tracking="greedy",
        split_file="experiments/splits/calibration_subset_fold0.json", split=0, is_screening=True,
    )
    assert Path(trial["split_file"]).is_absolute()
    assert Path(trial["split_file"]).is_file()


# =============================================================================
# Official-score selection, deterministic tie-breaking, and hard failure
# on insufficient candidates
# =============================================================================

def _success_record(**overrides) -> dict:
    base = {
        "trial_id": "x", "status": "success", "checkpoint_label": "10000",
        "det_threshold": 0.52, "edge_threshold": 0.50,
        "summary_metrics": {"score": 0.5},
    }
    base.update(overrides)
    return base


def test_select_best_det_thresholds_orders_by_score_desc() -> None:
    records = [
        _success_record(det_threshold=0.52, summary_metrics={"score": 0.60}),
        _success_record(det_threshold=0.58, summary_metrics={"score": 0.62}),
        _success_record(det_threshold=0.64, summary_metrics={"score": 0.61}),
        _success_record(det_threshold=0.70, summary_metrics={"score": 0.55}),
    ]
    assert calib.select_best_det_thresholds(records, 2) == [0.58, 0.64]


def test_select_best_det_thresholds_tie_break_ascending_threshold() -> None:
    records = [
        _success_record(det_threshold=0.70, summary_metrics={"score": 0.60}),
        _success_record(det_threshold=0.52, summary_metrics={"score": 0.60}),
        _success_record(det_threshold=0.64, summary_metrics={"score": 0.60}),
    ]
    assert calib.select_best_det_thresholds(records, 2) == [0.52, 0.64]


def test_select_best_det_thresholds_raises_on_partial_grid() -> None:
    """A partial grid (any failed trial) must never be silently selected from."""
    records = [
        _success_record(det_threshold=0.52, summary_metrics={"score": 0.99}, status="failed"),
        _success_record(det_threshold=0.58, summary_metrics={"score": 0.10}),
    ]
    with pytest.raises(calib.CalibrationError, match="requires all 2 planned trial"):
        calib.select_best_det_thresholds(records, 1)


def test_select_best_det_thresholds_raises_when_insufficient_candidates() -> None:
    records = [_success_record(det_threshold=0.52, summary_metrics={"score": 0.5})]
    with pytest.raises(calib.CalibrationError, match="requires 2"):
        calib.select_best_det_thresholds(records, 2)


def test_select_best_pairs_orders_by_score_then_tie_breaks() -> None:
    records = [
        _success_record(det_threshold=0.58, edge_threshold=0.50, summary_metrics={"score": 0.60}),
        _success_record(det_threshold=0.58, edge_threshold=0.45, summary_metrics={"score": 0.60}),
        _success_record(det_threshold=0.64, edge_threshold=0.35, summary_metrics={"score": 0.70}),
    ]
    assert calib.select_best_pairs(records, 2) == [(0.64, 0.35), (0.58, 0.45)]


def test_select_best_pairs_raises_when_insufficient_candidates() -> None:
    records = [_success_record(det_threshold=0.58, edge_threshold=0.5, summary_metrics={"score": 0.5})]
    with pytest.raises(calib.CalibrationError, match="requires 2"):
        calib.select_best_pairs(records, 2)


def test_select_best_pairs_raises_on_partial_grid() -> None:
    """A partial grid (any failed trial) must never be silently selected from."""
    records = [
        _success_record(det_threshold=0.58, edge_threshold=0.50, summary_metrics={"score": 0.60}),
        _success_record(det_threshold=0.64, edge_threshold=0.35, status="failed"),
    ]
    with pytest.raises(calib.CalibrationError, match="requires all 2 planned trial"):
        calib.select_best_pairs(records, 1)


def test_select_final_best_tie_break_prefers_earlier_listed_checkpoint() -> None:
    records = [
        _success_record(checkpoint_label="7500", det_threshold=0.58, edge_threshold=0.45,
                         summary_metrics={"score": 0.60}),
        _success_record(checkpoint_label="10000", det_threshold=0.58, edge_threshold=0.45,
                         summary_metrics={"score": 0.60}),
    ]
    best = calib.select_final_best(records, checkpoint_label_order=["10000", "7500"])
    assert best["checkpoint_label"] == "10000"


def test_select_final_best_raises_when_all_failed() -> None:
    records = [_success_record(status="failed")]
    with pytest.raises(calib.CalibrationError, match="requires all 1 planned trial"):
        calib.select_final_best(records, ["10000", "7500"])


def test_select_final_best_raises_on_empty_records() -> None:
    with pytest.raises(calib.CalibrationError):
        calib.select_final_best([], ["10000", "7500"])


def test_select_final_best_raises_on_partial_grid() -> None:
    """A partial grid (any failed trial) must never be silently selected from."""
    records = [
        _success_record(checkpoint_label="10000", det_threshold=0.58, edge_threshold=0.45,
                         summary_metrics={"score": 0.90}),
        _success_record(checkpoint_label="7500", det_threshold=0.58, edge_threshold=0.45,
                         status="failed"),
    ]
    with pytest.raises(calib.CalibrationError, match="requires all 2 planned trial"):
        calib.select_final_best(records, ["10000", "7500"])


# =============================================================================
# Strict previous-stage validation: exact-grid matching + on-disk revalidation
# =============================================================================

def _stage_a_config() -> dict:
    return calib.load_config(CONFIG_PATH)


def _stage_report_env(tmp_path: Path) -> dict:
    """A minimal but real on-disk environment for previous-stage report tests:
    a checkpoint (+ adjacent config.json) for label '10000', and a data_dir.
    """
    checkpoint_dir = tmp_path / "weights"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "ckpt_10000.pth"
    checkpoint_path.write_bytes(b"weights-10000-v1")
    (checkpoint_dir / "config.json").write_text(json.dumps({"unet_out_channels": 32}))

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    return {"checkpoints": {"10000": str(checkpoint_path)}, "data_dir": data_dir}


def _full_success_record(
    trial: dict, output_dir: Path, *, config_sha256: str, data_dir: Path, expected: list[str], score: float,
) -> dict:
    """Build a success trial record with real on-disk predictions and metrics
    so it survives the runner's on-disk revalidation (checkpoint, checkpoint
    config, split, predictions, metrics) — not just its in-memory fields.
    """
    expected = sorted(expected)
    trial_dir = output_dir / "trials" / trial["trial_id"]
    predictions_dir = trial_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for name in expected:
        (predictions_dir / f"{name}.geff").write_bytes(f"geff-{trial['trial_id']}-{name}".encode())
    metrics_report = _valid_metrics_report(expected)
    metrics_report["summary_metrics"] = {"score": score}
    metrics_path = trial_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_report))

    checkpoint_path = Path(trial["checkpoint_path"])
    checkpoint_config_path = calib._checkpoint_config_path(checkpoint_path)
    return {
        **trial,
        "checkpoint_sha256": calib._sha256(checkpoint_path),
        "checkpoint_config_sha256": calib._sha256(checkpoint_config_path),
        "split_file_sha256": calib._sha256(Path(trial["split_file"])),
        "config_sha256": config_sha256,
        "data_dir": str(Path(data_dir).resolve()),
        "expected_datasets": expected,
        "status": "success",
        "error": None,
        "commands": {"prediction": [], "evaluation": []},
        "prediction_dir": str(predictions_dir),
        "metrics_path": str(metrics_path),
        "prediction_validation": {"datasets": expected, "missing": [], "extra": [], "empty": []},
        "prediction_manifest": calib._prediction_manifest(predictions_dir, expected),
        "prediction_wall_seconds": 0.1,
        "evaluation_wall_seconds": 0.1,
        "summary_metrics": metrics_report["summary_metrics"],
        "per_dataset_metrics": metrics_report["per_dataset_metrics"],
        "resumed": False,
    }


_STAGE_A_SCORES = {0.52: 0.5, 0.58: 0.7, 0.64: 0.6, 0.70: 0.4}  # top-2: 0.58, 0.64


def _build_stage_a_records(
    config: dict, checkpoints: dict, output_dir: Path, data_dir: Path, config_sha256: str,
    scores: dict[float, float],
) -> list[dict]:
    trials = calib.build_stage_a_trials(config, checkpoints)
    expected = config["subset_dataset_names"]
    return [
        _full_success_record(
            trial, output_dir, config_sha256=config_sha256, data_dir=data_dir,
            expected=expected, score=scores[trial["det_threshold"]],
        )
        for trial in trials
    ]


def _write_stage_a_report(tmp_path: Path, config: dict, env: dict, *, mutate=None) -> Path:
    config_sha256 = calib._require_sha256(CONFIG_PATH)
    output_dir = tmp_path / "stage_a_out"
    records = _build_stage_a_records(
        config, env["checkpoints"], output_dir, env["data_dir"], config_sha256, _STAGE_A_SCORES,
    )
    selected = calib.select_best_det_thresholds(records, config["stage_a"]["n_best_det_thresholds"])
    report = {
        "schema_version": 1,
        "stage": "stage_a",
        "config_sha256": config_sha256,
        "data_dir": str(Path(env["data_dir"]).resolve()),
        "trials": records,
        "selected_det_thresholds": selected,
    }
    if mutate is not None:
        mutate(report)
    path = tmp_path / "stage_a_results.json"
    path.write_text(json.dumps(report, allow_nan=True))
    return path


def _load_stage_a(path: Path, config: dict, env: dict):
    return calib.load_stage_a_report_for_stage_b(path, config, CONFIG_PATH, env["checkpoints"], env["data_dir"])


def test_load_stage_a_report_accepts_valid_report(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    path = _write_stage_a_report(tmp_path, config, env)
    report, selection = _load_stage_a(path, config, env)
    assert selection == [0.58, 0.64]
    assert report["stage"] == "stage_a"


def test_load_stage_a_report_rejects_missing_file(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    with pytest.raises(calib.CalibrationError, match="cannot read"):
        _load_stage_a(tmp_path / "nope.json", config, env)


def test_load_stage_a_report_rejects_malformed_json(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(calib.CalibrationError, match="not valid JSON"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_arbitrary_object(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    path = tmp_path / "arbitrary.json"
    path.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(calib.CalibrationError):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_wrong_schema_version(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    path = _write_stage_a_report(tmp_path, config, env, mutate=lambda r: r.update(schema_version=2))
    with pytest.raises(calib.CalibrationError, match="schema_version"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_wrong_stage(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    path = _write_stage_a_report(tmp_path, config, env, mutate=lambda r: r.update(stage="stage_b"))
    with pytest.raises(calib.CalibrationError, match="expected 'stage_a'"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_wrong_config_hash(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    path = _write_stage_a_report(tmp_path, config, env, mutate=lambda r: r.update(config_sha256="deadbeef"))
    with pytest.raises(calib.CalibrationError, match="different calibration config"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_wrong_data_dir(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    other_data_dir = tmp_path / "other-data"
    path = _write_stage_a_report(
        tmp_path, config, env, mutate=lambda r: r.update(data_dir=str(other_data_dir.resolve())),
    )
    with pytest.raises(calib.CalibrationError, match="does not match the current resolved data_dir"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_missing_trial_in_grid(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _drop_one(report):
        report["trials"] = report["trials"][:-1]

    path = _write_stage_a_report(tmp_path, config, env, mutate=_drop_one)
    with pytest.raises(calib.CalibrationError, match="does not exactly match"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_extra_trial_in_grid(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _add_extra(report):
        extra = dict(report["trials"][0])
        extra["trial_id"] = "stage_a__ckpt-10000__det-0.9900__edge-0.5000"
        extra["det_threshold"] = 0.99
        report["trials"].append(extra)

    path = _write_stage_a_report(tmp_path, config, env, mutate=_add_extra)
    with pytest.raises(calib.CalibrationError, match="does not exactly match"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_duplicate_trial_ids(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _duplicate(report):
        report["trials"][1]["trial_id"] = report["trials"][0]["trial_id"]

    path = _write_stage_a_report(tmp_path, config, env, mutate=_duplicate)
    with pytest.raises(calib.CalibrationError, match="duplicate trial_id"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_substituted_trial_id(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _substitute(report):
        report["trials"][0]["trial_id"] = "stage_a__ckpt-10000__det-0.9900__edge-0.5000"

    path = _write_stage_a_report(tmp_path, config, env, mutate=_substitute)
    with pytest.raises(calib.CalibrationError, match="does not exactly match"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_altered_det_threshold(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _alter(report):
        report["trials"][0]["det_threshold"] = 0.99

    path = _write_stage_a_report(tmp_path, config, env, mutate=_alter)
    with pytest.raises(calib.CalibrationError, match="does not match the expected grid"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_altered_checkpoint_label(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _relabel(report):
        report["trials"][0]["checkpoint_label"] = "not-a-configured-label"

    path = _write_stage_a_report(tmp_path, config, env, mutate=_relabel)
    with pytest.raises(calib.CalibrationError, match="does not match the expected grid"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_failed_trial_in_grid(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _fail_one(report):
        report["trials"][0]["status"] = "failed"

    path = _write_stage_a_report(tmp_path, config, env, mutate=_fail_one)
    with pytest.raises(calib.CalibrationError, match="did not succeed"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_non_finite_score(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _nan_score(report):
        report["trials"][0]["summary_metrics"]["score"] = float("nan")

    path = _write_stage_a_report(tmp_path, config, env, mutate=_nan_score)
    with pytest.raises(calib.CalibrationError, match="non-finite official score"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_non_screening_trial(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _not_screening(report):
        report["trials"][0]["is_screening"] = False

    path = _write_stage_a_report(tmp_path, config, env, mutate=_not_screening)
    with pytest.raises(calib.CalibrationError, match="does not match the expected grid"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_wrong_selection_count(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _short_selection(report):
        report["selected_det_thresholds"] = report["selected_det_thresholds"][:1]

    path = _write_stage_a_report(tmp_path, config, env, mutate=_short_selection)
    with pytest.raises(calib.CalibrationError, match="exactly 2 entries"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_tampered_selection(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _tamper(report):
        report["selected_det_thresholds"] = [0.70, 0.52]  # not the true top-2 by score

    path = _write_stage_a_report(tmp_path, config, env, mutate=_tamper)
    with pytest.raises(calib.CalibrationError, match="tampering or corruption"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_failed_status_value(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _bad_status(report):
        report["trials"][0]["status"] = "in_progress"

    path = _write_stage_a_report(tmp_path, config, env, mutate=_bad_status)
    with pytest.raises(calib.CalibrationError, match="invalid status"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_invalid_on_disk_checkpoint(tmp_path: Path) -> None:
    """A report whose fields all match the expected grid is still rejected if
    the checkpoint's on-disk bytes no longer match the recorded hash."""
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    path = _write_stage_a_report(tmp_path, config, env)

    Path(env["checkpoints"]["10000"]).write_bytes(b"tampered-weights")

    with pytest.raises(calib.CalibrationError, match="failed on-disk revalidation"):
        _load_stage_a(path, config, env)


def test_load_stage_a_report_rejects_invalid_on_disk_predictions(tmp_path: Path) -> None:
    """A GEFF that stays present and nonempty but whose bytes changed on disk
    must still be rejected, via the content manifest."""
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    path = _write_stage_a_report(tmp_path, config, env)
    report = json.loads(path.read_text())
    prediction_dir = Path(report["trials"][0]["prediction_dir"])
    geff = sorted(prediction_dir.glob("*.geff"))[0]
    geff.write_bytes(b"corrupted-but-still-nonempty")

    with pytest.raises(calib.CalibrationError, match="failed on-disk revalidation"):
        _load_stage_a(path, config, env)


def _build_stage_b_records(
    config: dict, checkpoints: dict, best_det_thresholds: list[float],
    output_dir: Path, data_dir: Path, config_sha256: str,
) -> list[dict]:
    trials = calib.build_stage_b_trials(config, checkpoints, best_det_thresholds)
    expected = config["subset_dataset_names"]
    return [
        _full_success_record(
            trial, output_dir, config_sha256=config_sha256, data_dir=data_dir,
            expected=expected, score=0.5 + 0.01 * i,
        )
        for i, trial in enumerate(trials)
    ]


def _write_stage_b_report(tmp_path: Path, config: dict, env: dict, *, mutate=None) -> Path:
    config_sha256 = calib._require_sha256(CONFIG_PATH)
    output_dir = tmp_path / "stage_b_out"
    best_det_thresholds = [0.58, 0.64]
    records = _build_stage_b_records(
        config, env["checkpoints"], best_det_thresholds, output_dir, env["data_dir"], config_sha256,
    )
    selected = calib.select_best_pairs(records, config["stage_b"]["n_best_pairs"])
    report = {
        "schema_version": 1,
        "stage": "stage_b",
        "config_sha256": config_sha256,
        "data_dir": str(Path(env["data_dir"]).resolve()),
        "trials": records,
        "selected_pairs": [list(p) for p in selected],
    }
    if mutate is not None:
        mutate(report)
    path = tmp_path / "stage_b_results.json"
    path.write_text(json.dumps(report, allow_nan=True))
    return path


def _load_stage_b(path: Path, config: dict, env: dict):
    return calib.load_stage_b_report_for_stage_c(path, config, CONFIG_PATH, env["checkpoints"], env["data_dir"])


def test_load_stage_b_report_accepts_valid_report(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    path = _write_stage_b_report(tmp_path, config, env)
    report, selection = _load_stage_b(path, config, env)
    assert len(selection) == config["stage_b"]["n_best_pairs"]
    assert report["stage"] == "stage_b"


def test_load_stage_b_report_rejects_wrong_stage(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    path = _write_stage_b_report(tmp_path, config, env, mutate=lambda r: r.update(stage="stage_a"))
    with pytest.raises(calib.CalibrationError, match="expected 'stage_b'"):
        _load_stage_b(path, config, env)


def test_load_stage_b_report_rejects_tampered_pairs(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _tamper(report):
        report["selected_pairs"] = [[0.99, 0.99], [0.01, 0.01]]

    path = _write_stage_b_report(tmp_path, config, env, mutate=_tamper)
    with pytest.raises(calib.CalibrationError, match="tampering or corruption"):
        _load_stage_b(path, config, env)


def test_load_stage_b_report_rejects_wrong_data_dir(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)
    other_data_dir = tmp_path / "other-data"
    path = _write_stage_b_report(
        tmp_path, config, env, mutate=lambda r: r.update(data_dir=str(other_data_dir.resolve())),
    )
    with pytest.raises(calib.CalibrationError, match="does not match the current resolved data_dir"):
        _load_stage_b(path, config, env)


def test_load_stage_b_report_rejects_altered_edge_threshold(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _alter(report):
        report["trials"][0]["edge_threshold"] = 0.99

    path = _write_stage_b_report(tmp_path, config, env, mutate=_alter)
    with pytest.raises(calib.CalibrationError, match="does not match the expected grid"):
        _load_stage_b(path, config, env)


def test_load_stage_b_report_rejects_extra_trial(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _add_extra(report):
        extra = dict(report["trials"][0])
        extra["trial_id"] = "stage_b__ckpt-10000__det-0.5800__edge-0.9900"
        extra["edge_threshold"] = 0.99
        report["trials"].append(extra)

    path = _write_stage_b_report(tmp_path, config, env, mutate=_add_extra)
    with pytest.raises(calib.CalibrationError, match="does not exactly match"):
        _load_stage_b(path, config, env)


def test_load_stage_b_report_rejects_det_threshold_count_mismatch(tmp_path: Path) -> None:
    """Stage C reconstructs Stage B's grid from the det_threshold values that
    actually appear in the report; that set must have exactly
    n_best_det_thresholds distinct values.
    """
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _collapse(report):
        for record in report["trials"]:
            record["det_threshold"] = 0.58

    path = _write_stage_b_report(tmp_path, config, env, mutate=_collapse)
    with pytest.raises(calib.CalibrationError, match="distinct det_threshold value"):
        _load_stage_b(path, config, env)


def test_load_stage_b_report_rejects_det_threshold_outside_reviewed_grid(tmp_path: Path) -> None:
    config = _stage_a_config()
    env = _stage_report_env(tmp_path)

    def _out_of_grid(report):
        for record in report["trials"]:
            if record["det_threshold"] == 0.64:
                record["det_threshold"] = 0.99

    path = _write_stage_b_report(tmp_path, config, env, mutate=_out_of_grid)
    with pytest.raises(calib.CalibrationError, match="outside the reviewed"):
        _load_stage_b(path, config, env)


# =============================================================================
# Command construction — strict evaluator, prediction, and no training
# =============================================================================

def _trial(**overrides) -> dict:
    base = {
        "trial_id": "stage_a__ckpt-10000__det-0.5200__edge-0.5000",
        "stage": "stage_a", "checkpoint_label": "10000",
        "checkpoint_path": "weights/ckpt.pth",
        "det_threshold": 0.52, "edge_threshold": 0.50,
        "pool_kernel_um": 5.0, "tracking": "greedy",
        "split_file": "experiments/splits/calibration_subset_fold0.json",
        "split": 0, "is_screening": True,
    }
    base.update(overrides)
    return base


def test_build_prediction_command_forwards_thresholds_and_never_references_training() -> None:
    command = calib.build_prediction_command(_trial(), Path("data/train"), "calib_x")
    assert command[0] == sys.executable
    assert command[1] == str(calib.PREDICT_SCRIPT)
    assert "--det-threshold" in command and command[command.index("--det-threshold") + 1] == "0.52"
    assert "--edge-threshold" in command and command[command.index("--edge-threshold") + 1] == "0.5"
    assert "--weights" in command and command[command.index("--weights") + 1] == "weights/ckpt.pth"
    assert str(calib.TRAIN_SCRIPT) not in command


def test_build_evaluation_command_is_strict_with_json_out() -> None:
    command = calib.build_evaluation_command(Path("pred"), Path("gt"), Path("metrics.json"))
    assert command[0] == sys.executable
    assert command[1] == str(calib.EVALUATE_SCRIPT)
    assert "--strict" in command
    assert "--json-out" in command and command[command.index("--json-out") + 1] == "metrics.json"
    assert "--pred-dir" in command and command[command.index("--pred-dir") + 1] == "pred"
    assert "--gt-dir" in command and command[command.index("--gt-dir") + 1] == "gt"
    assert str(calib.TRAIN_SCRIPT) not in command


def test_stream_subprocess_refuses_to_invoke_training_script(tmp_path: Path) -> None:
    command = [sys.executable, str(calib.TRAIN_SCRIPT), "--help"]
    with pytest.raises(calib.CalibrationError, match="training script"):
        calib._stream_subprocess(command, tmp_path / "out.log", tmp_path / "err.log")
    assert not (tmp_path / "out.log").exists()


@pytest.mark.parametrize(
    "make_arg",
    [
        lambda: str(calib.TRAIN_SCRIPT),
        lambda: "scripts/train_unet_transformer.py",
        lambda: "scripts\\train_unet_transformer.py",
        lambda: "./scripts/train_unet_transformer.py",
        lambda: str(calib.REPO_ROOT / "scripts" / ".." / "scripts" / "train_unet_transformer.py"),
        lambda: str(calib.REPO_ROOT) + "\\scripts\\train_unet_transformer.py",
    ],
    ids=["absolute", "repo-relative-posix", "repo-relative-windows", "dot-prefixed",
         "redundant-segments", "absolute-windows-backslash"],
)
def test_command_references_training_script_detects_every_spelling(make_arg) -> None:
    assert calib._command_references_training_script([sys.executable, make_arg(), "--help"]) is True


def test_command_references_training_script_ignores_unrelated_commands() -> None:
    command = [sys.executable, str(calib.PREDICT_SCRIPT), "--weights", "x.pth"]
    assert calib._command_references_training_script(command) is False


def test_stream_subprocess_refuses_repo_relative_training_script_path(tmp_path: Path) -> None:
    command = [sys.executable, "scripts/train_unet_transformer.py", "--help"]
    with pytest.raises(calib.CalibrationError, match="training script"):
        calib._stream_subprocess(command, tmp_path / "out.log", tmp_path / "err.log")
    assert not (tmp_path / "out.log").exists()


def test_stream_subprocess_refuses_windows_style_training_script_path(tmp_path: Path) -> None:
    command = [sys.executable, "scripts\\train_unet_transformer.py", "--help"]
    with pytest.raises(calib.CalibrationError, match="training script"):
        calib._stream_subprocess(command, tmp_path / "out.log", tmp_path / "err.log")
    assert not (tmp_path / "out.log").exists()


# =============================================================================
# Progress visibility: immediate start message, heartbeat, streaming
# =============================================================================

def test_print_trial_start_emits_immediately(capsys: pytest.CaptureFixture) -> None:
    trial = _trial()
    calib._print_trial_start(trial, 2, 4)
    out = capsys.readouterr().out
    assert "trial 2/4" in out
    assert "starting" in out
    assert "10000" in out


def test_print_heartbeat_contains_required_fields(capsys: pytest.CaptureFixture) -> None:
    trial = _trial()
    calib._print_heartbeat(trial, 1, 4, "predicting", 61.4)
    out = capsys.readouterr().out
    assert "stage=stage_a" in out
    assert "trial 1/4" in out
    assert "checkpoint=10000" in out
    assert "det=0.5200" in out
    assert "edge=0.5000" in out
    assert "phase=predicting" in out
    assert "elapsed=61s" in out


def test_stream_subprocess_emits_heartbeats_and_streams_output_live(tmp_path: Path) -> None:
    command = [
        sys.executable, "-c",
        "import sys, time\n"
        "for i in range(3):\n"
        "    print(f'line-{i}')\n"
        "    sys.stdout.flush()\n"
        "    time.sleep(0.12)\n",
    ]
    heartbeats: list[float] = []
    result = calib._stream_subprocess(
        command, tmp_path / "out.log", tmp_path / "err.log",
        on_heartbeat=heartbeats.append, heartbeat_interval=0.05,
    )
    assert result["returncode"] == 0
    assert len(heartbeats) >= 2
    out_content = (tmp_path / "out.log").read_text()
    assert "line-0" in out_content and "line-2" in out_content
    assert (tmp_path / "err.log").exists()


def test_stream_subprocess_keeps_stdout_and_stderr_separate(tmp_path: Path) -> None:
    command = [
        sys.executable, "-c",
        "import sys\nprint('to-stdout')\nprint('to-stderr', file=sys.stderr)\n",
    ]
    calib._stream_subprocess(command, tmp_path / "out.log", tmp_path / "err.log")
    assert "to-stdout" in (tmp_path / "out.log").read_text()
    assert "to-stdout" not in (tmp_path / "err.log").read_text()
    assert "to-stderr" in (tmp_path / "err.log").read_text()


# =============================================================================
# Prediction / metrics validation helpers
# =============================================================================

def _write_geff(path: Path, content: bytes = b"geff") -> None:
    path.write_bytes(content)


def test_validate_predictions_requires_exact_nonempty_coverage(tmp_path: Path) -> None:
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    _write_geff(pred_dir / "a.geff")
    _write_geff(pred_dir / "b.geff")

    result = calib._validate_predictions(pred_dir, ["a", "b"])
    assert result["missing"] == [] and result["extra"] == [] and result["empty"] == []


@pytest.mark.parametrize(
    "setup",
    [
        "missing",  # only 'a.geff' present, 'b' missing
        "extra",    # unexpected 'c.geff' present
        "empty",    # 'b.geff' is zero bytes
    ],
)
def test_validate_predictions_rejects_coverage_failures(tmp_path: Path, setup: str) -> None:
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    _write_geff(pred_dir / "a.geff")
    if setup == "missing":
        expected = ["a", "b"]
    elif setup == "extra":
        _write_geff(pred_dir / "b.geff")
        _write_geff(pred_dir / "c.geff")
        expected = ["a", "b"]
    else:
        _write_geff(pred_dir / "b.geff", b"")
        expected = ["a", "b"]

    with pytest.raises(calib.CalibrationError, match="coverage failure"):
        calib._validate_predictions(pred_dir, expected)


def test_hash_geff_entry_for_file_geff_reports_kind_count_bytes_and_sha256(tmp_path: Path) -> None:
    geff = tmp_path / "a.geff"
    _write_geff(geff, b"hello")
    entry = calib._hash_geff_entry(geff)
    assert entry == {
        "kind": "file", "member_count": 1, "total_bytes": 5, "sha256": calib._sha256(geff),
    }


def test_hash_geff_entry_for_dir_geff_hashes_member_paths_and_contents(tmp_path: Path) -> None:
    geff = tmp_path / "b.geff"
    geff.mkdir()
    (geff / "nodes.bin").write_bytes(b"node-bytes")
    (geff / "meta").mkdir()
    (geff / "meta" / "edges.bin").write_bytes(b"edge-bytes")

    entry = calib._hash_geff_entry(geff)
    assert entry["kind"] == "dir"
    assert entry["member_count"] == 2
    assert entry["total_bytes"] == len(b"node-bytes") + len(b"edge-bytes")

    # Same member paths and total size, but different content -> different hash.
    (geff / "nodes.bin").write_bytes(b"NODE-BYTES")
    assert calib._hash_geff_entry(geff)["sha256"] != entry["sha256"]

    # Same content and count, but a member renamed -> different hash (paths are hashed too).
    geff2 = tmp_path / "c.geff"
    geff2.mkdir()
    (geff2 / "nodes-renamed.bin").write_bytes(b"node-bytes")
    (geff2 / "meta").mkdir()
    (geff2 / "meta" / "edges.bin").write_bytes(b"edge-bytes")
    assert calib._hash_geff_entry(geff2)["sha256"] != entry["sha256"]


def test_prediction_manifest_covers_every_expected_dataset_deterministically(tmp_path: Path) -> None:
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    _write_geff(pred_dir / "a.geff", b"aaa")
    (pred_dir / "b.geff").mkdir()
    (pred_dir / "b.geff" / "part.bin").write_bytes(b"bbb")

    manifest = calib._prediction_manifest(pred_dir, ["a", "b"])
    assert set(manifest) == {"a", "b"}
    assert manifest["a"]["kind"] == "file"
    assert manifest["b"]["kind"] == "dir"
    # Deterministic across repeated computation.
    assert calib._prediction_manifest(pred_dir, ["a", "b"]) == manifest


def _valid_metrics_report(expected: list[str]) -> dict:
    return {
        "evaluated_datasets": expected,
        "skipped_datasets": [],
        "summary_metrics": {"score": 0.6},
        "per_dataset_metrics": [
            {"dataset": name, "metrics": {
                "edge_tp": 1, "edge_fp": 0, "edge_fn": 0, "num_pred_nodes": 1,
                "total_node_ratio": 0.0, "node_recall": 1.0, "edge_jaccard": 1.0,
                "adj_edge_jaccard": 1.0,
            }}
            for name in expected
        ],
    }


def test_validate_strict_metrics_accepts_well_formed_report(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_valid_metrics_report(["a", "b"])))
    result = calib._validate_strict_metrics(metrics_path, ["a", "b"])
    assert result["summary_metrics"]["score"] == 0.6


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r.update(skipped_datasets=["a"]), "skipped"),
        (lambda r: r["summary_metrics"].update(score=float("nan")), "not finite"),
        (lambda r: r.update(evaluated_datasets=["a"]), "differ from expected"),
        (lambda r: r.pop("per_dataset_metrics"), "missing or incomplete"),
    ],
)
def test_validate_strict_metrics_rejects_malformed_reports(tmp_path: Path, mutate, match: str) -> None:
    report = _valid_metrics_report(["a", "b"])
    mutate(report)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(report, allow_nan=True))
    with pytest.raises(calib.CalibrationError, match=match):
        calib._validate_strict_metrics(metrics_path, ["a", "b"])


# =============================================================================
# Path isolation and cleanup safety
# =============================================================================

def test_remove_prediction_scratch_dir_refuses_mismatched_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    actual_dir = repo_root / "predictions" / "user" / "calib_x" / "split_0"
    actual_dir.mkdir(parents=True)
    (actual_dir / "marker.txt").write_text("keep me")

    with pytest.raises(calib.CalibrationError, match="unexpected"):
        calib._remove_prediction_scratch_dir(actual_dir, repo_root, "user", "calib_other", 0)

    assert (actual_dir / "marker.txt").is_file()  # untouched


def test_remove_prediction_scratch_dir_removes_only_the_exact_expected_dir(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    target_dir = repo_root / "predictions" / "user" / "calib_x" / "split_0"
    sibling_dir = repo_root / "predictions" / "user" / "calib_y" / "split_0"
    target_dir.mkdir(parents=True)
    sibling_dir.mkdir(parents=True)
    (sibling_dir / "keep.txt").write_text("keep me")

    calib._remove_prediction_scratch_dir(target_dir, repo_root, "user", "calib_x", 0)

    assert not target_dir.exists()
    assert (sibling_dir / "keep.txt").is_file()


def test_remove_trial_dir_refuses_mismatched_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    actual_dir = output_dir / "trials" / "trial-a"
    actual_dir.mkdir(parents=True)
    (actual_dir / "marker.txt").write_text("keep me")

    with pytest.raises(calib.CalibrationError, match="unexpected"):
        calib._remove_trial_dir(actual_dir, output_dir, "trial-b")

    assert (actual_dir / "marker.txt").is_file()


def test_remove_trial_dir_removes_only_the_exact_expected_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    target_dir = output_dir / "trials" / "trial-a"
    sibling_dir = output_dir / "trials" / "trial-b"
    target_dir.mkdir(parents=True)
    sibling_dir.mkdir(parents=True)
    (sibling_dir / "keep.txt").write_text("keep me")

    calib._remove_trial_dir(target_dir, output_dir, "trial-a")

    assert not target_dir.exists()
    assert (sibling_dir / "keep.txt").is_file()


def _setup_trial_env(tmp_path: Path) -> dict:
    checkpoint_dir = tmp_path / "weights"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "ckpt.pth"
    checkpoint_path.write_bytes(b"weights-v1")
    (checkpoint_dir / "config.json").write_text(json.dumps({"unet_out_channels": 32}))

    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps([{"split": 0, "train": [], "test": ["a"]}]))

    config_path = tmp_path / "calib_config.json"
    config_path.write_text(json.dumps({"placeholder": True}))

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    return {
        "checkpoint_path": checkpoint_path,
        "split_path": split_path,
        "config_path": config_path,
        "data_dir": data_dir,
    }


def _make_env_trial(env: dict, **overrides) -> dict:
    kwargs = dict(
        stage="stage_a", checkpoint_label="10000", checkpoint_path=str(env["checkpoint_path"]),
        det_threshold=0.52, edge_threshold=0.50, pool_kernel_um=5.0, tracking="greedy",
        split_file=str(env["split_path"]), split=0, is_screening=True,
    )
    kwargs.update(overrides)
    return calib._make_trial(**kwargs)


def test_execute_trial_never_writes_outside_its_own_trial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    env = _setup_trial_env(tmp_path)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(calib, "REPO_ROOT", repo_root)
    monkeypatch.setattr(calib, "_username", lambda: "user")

    trial = _make_env_trial(env)
    scratch = repo_root / "predictions" / "user" / f"calib_{trial['trial_id']}" / "split_0"

    def fake_stream_subprocess(command, stdout_path, stderr_path, **kwargs):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok")
        stderr_path.write_text("")
        if command[1] == str(calib.PREDICT_SCRIPT):
            scratch.mkdir(parents=True, exist_ok=True)
            (scratch / "a.geff").write_bytes(b"geff")
        else:
            metrics_path = Path(command[command.index("--json-out") + 1])
            metrics_path.write_text(json.dumps(_valid_metrics_report(["a"])))
        return {"returncode": 0, "wall_seconds": 0.01}

    monkeypatch.setattr(calib, "_stream_subprocess", fake_stream_subprocess)

    record = calib.execute_trial(trial, env["data_dir"], output_dir, config_sha256="deadbeef")

    assert record["status"] == "success"
    assert record["checkpoint_config_sha256"] == calib._sha256(env["checkpoint_path"].parent / "config.json")
    assert record["config_sha256"] == "deadbeef"
    assert record["data_dir"] == str(env["data_dir"].resolve())
    # The only artifact outside output_dir is the scratch dir, and it must be gone.
    assert not scratch.exists()
    predictions_dir = Path(record["prediction_dir"])
    assert predictions_dir.is_relative_to(output_dir)
    assert (predictions_dir / "a.geff").is_file()


def test_execute_trial_requires_checkpoint_config_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _setup_trial_env(tmp_path)
    (env["checkpoint_path"].parent / "config.json").unlink()
    trial = _make_env_trial(env)
    monkeypatch.setattr(calib, "_username", lambda: "user")

    with pytest.raises(calib.CalibrationError, match="config.json"):
        calib.execute_trial(trial, env["data_dir"], tmp_path / "out", config_sha256="deadbeef")


def test_execute_trial_is_independent_of_caller_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    env = _setup_trial_env(tmp_path)
    output_dir = tmp_path / "out"
    trial = _make_env_trial(env)

    def fake_stream_subprocess(command, stdout_path, stderr_path, **kwargs):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok")
        stderr_path.write_text("")
        if command[1] == str(calib.PREDICT_SCRIPT):
            scratch_dir = Path(command[command.index("--data-dir") + 1])  # unused marker
            pred_dir = calib._prediction_scratch_dir(calib.REPO_ROOT, calib._username(), f"calib_{trial['trial_id']}", 0)
            pred_dir.mkdir(parents=True, exist_ok=True)
            (pred_dir / "a.geff").write_bytes(b"geff")
        else:
            metrics_path = Path(command[command.index("--json-out") + 1])
            metrics_path.write_text(json.dumps(_valid_metrics_report(["a"])))
        return {"returncode": 0, "wall_seconds": 0.01}

    monkeypatch.setattr(calib, "_stream_subprocess", fake_stream_subprocess)
    monkeypatch.chdir(other_cwd)

    record = calib.execute_trial(trial, env["data_dir"], output_dir, config_sha256="deadbeef")

    assert record["status"] == "success"
    assert Path(record["prediction_dir"]).is_relative_to(output_dir)
    assert record["data_dir"] == str(env["data_dir"].resolve())


def test_execute_trial_removes_stale_trial_directory_before_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(calib, "_username", lambda: "user")
    trial = _make_env_trial(env)

    stale_trial_dir = output_dir / "trials" / trial["trial_id"]
    stale_predictions = stale_trial_dir / "predictions"
    stale_predictions.mkdir(parents=True)
    (stale_predictions / "stale_extra.geff").write_bytes(b"stale")

    def fake_stream_subprocess(command, stdout_path, stderr_path, **kwargs):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok")
        stderr_path.write_text("")
        if command[1] == str(calib.PREDICT_SCRIPT):
            pred_dir = calib._prediction_scratch_dir(calib.REPO_ROOT, "user", f"calib_{trial['trial_id']}", 0)
            pred_dir.mkdir(parents=True, exist_ok=True)
            (pred_dir / "a.geff").write_bytes(b"fresh")
        else:
            metrics_path = Path(command[command.index("--json-out") + 1])
            metrics_path.write_text(json.dumps(_valid_metrics_report(["a"])))
        return {"returncode": 0, "wall_seconds": 0.01}

    monkeypatch.setattr(calib, "_stream_subprocess", fake_stream_subprocess)

    record = calib.execute_trial(trial, env["data_dir"], output_dir, config_sha256="deadbeef")

    predictions_dir = Path(record["prediction_dir"])
    assert (predictions_dir / "a.geff").is_file()
    assert not (predictions_dir / "stale_extra.geff").exists()


def test_execute_trial_cleans_scratch_dir_after_prediction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    monkeypatch.setattr(calib, "_username", lambda: "user")
    trial = _make_env_trial(env)
    scratch = calib._prediction_scratch_dir(calib.REPO_ROOT, "user", f"calib_{trial['trial_id']}", 0)

    def fake_stream_subprocess(command, stdout_path, stderr_path, **kwargs):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("")
        stderr_path.write_text("boom")
        if command[1] == str(calib.PREDICT_SCRIPT):
            scratch.mkdir(parents=True, exist_ok=True)
            (scratch / "partial.geff").write_bytes(b"partial")
            return {"returncode": 1, "wall_seconds": 0.01}
        return {"returncode": 0, "wall_seconds": 0.01}

    monkeypatch.setattr(calib, "_stream_subprocess", fake_stream_subprocess)

    with pytest.raises(calib.CalibrationError, match="prediction failed"):
        calib.execute_trial(trial, env["data_dir"], tmp_path / "out", config_sha256="deadbeef")

    assert not scratch.exists()


def test_execute_trial_cleans_scratch_dir_after_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    monkeypatch.setattr(calib, "_username", lambda: "user")
    trial = _make_env_trial(env)
    scratch = calib._prediction_scratch_dir(calib.REPO_ROOT, "user", f"calib_{trial['trial_id']}", 0)

    def fake_stream_subprocess(command, stdout_path, stderr_path, **kwargs):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok")
        stderr_path.write_text("")
        if command[1] == str(calib.PREDICT_SCRIPT):
            scratch.mkdir(parents=True, exist_ok=True)
            # 'a' expected but missing -> _validate_predictions raises.
        return {"returncode": 0, "wall_seconds": 0.01}

    monkeypatch.setattr(calib, "_stream_subprocess", fake_stream_subprocess)

    with pytest.raises(calib.CalibrationError, match="coverage failure"):
        calib.execute_trial(trial, env["data_dir"], tmp_path / "out", config_sha256="deadbeef")

    assert not scratch.exists()


def test_execute_trial_cleans_scratch_dir_after_copy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    monkeypatch.setattr(calib, "_username", lambda: "user")
    trial = _make_env_trial(env)
    scratch = calib._prediction_scratch_dir(calib.REPO_ROOT, "user", f"calib_{trial['trial_id']}", 0)

    def fake_stream_subprocess(command, stdout_path, stderr_path, **kwargs):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok")
        stderr_path.write_text("")
        if command[1] == str(calib.PREDICT_SCRIPT):
            scratch.mkdir(parents=True, exist_ok=True)
            (scratch / "a.geff").write_bytes(b"geff")
        return {"returncode": 0, "wall_seconds": 0.01}

    def fake_copy(source, destination):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(calib, "_stream_subprocess", fake_stream_subprocess)
    monkeypatch.setattr(calib, "_copy_path", fake_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        calib.execute_trial(trial, env["data_dir"], tmp_path / "out", config_sha256="deadbeef")

    assert not scratch.exists()


def test_execute_trial_scratch_already_clean_when_evaluation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    monkeypatch.setattr(calib, "_username", lambda: "user")
    trial = _make_env_trial(env)
    scratch = calib._prediction_scratch_dir(calib.REPO_ROOT, "user", f"calib_{trial['trial_id']}", 0)

    def fake_stream_subprocess(command, stdout_path, stderr_path, **kwargs):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok")
        stderr_path.write_text("")
        if command[1] == str(calib.PREDICT_SCRIPT):
            scratch.mkdir(parents=True, exist_ok=True)
            (scratch / "a.geff").write_bytes(b"geff")
            return {"returncode": 0, "wall_seconds": 0.01}
        return {"returncode": 1, "wall_seconds": 0.01}  # evaluation fails

    monkeypatch.setattr(calib, "_stream_subprocess", fake_stream_subprocess)

    with pytest.raises(calib.CalibrationError, match="evaluation failed"):
        calib.execute_trial(trial, env["data_dir"], tmp_path / "out", config_sha256="deadbeef")

    assert not scratch.exists()


# =============================================================================
# Atomic incremental records, deterministic selection persistence, and
# scientifically safe resume
# =============================================================================

def _fake_success_record(trial: dict, output_dir: Path, *, config_sha256: str, data_dir: Path) -> dict:
    trial_dir = output_dir / "trials" / trial["trial_id"]
    predictions_dir = trial_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (predictions_dir / "a.geff").write_bytes(b"geff")
    metrics_path = trial_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_valid_metrics_report(["a"])))
    checkpoint_path = Path(trial["checkpoint_path"])
    checkpoint_config_path = calib._checkpoint_config_path(checkpoint_path)
    return {
        **trial,
        "checkpoint_sha256": calib._sha256(checkpoint_path),
        "checkpoint_config_sha256": calib._sha256(checkpoint_config_path),
        "split_file_sha256": calib._sha256(Path(trial["split_file"])),
        "config_sha256": config_sha256,
        "data_dir": str(Path(data_dir).resolve()),
        "expected_datasets": ["a"],
        "status": "success",
        "error": None,
        "commands": {"prediction": [], "evaluation": []},
        "prediction_dir": str(predictions_dir),
        "metrics_path": str(metrics_path),
        "prediction_validation": {"datasets": ["a"], "missing": [], "extra": [], "empty": []},
        "prediction_manifest": calib._prediction_manifest(predictions_dir, ["a"]),
        "prediction_wall_seconds": 0.1,
        "evaluation_wall_seconds": 0.1,
        "summary_metrics": {"score": 0.6},
        "per_dataset_metrics": _valid_metrics_report(["a"])["per_dataset_metrics"],
        "resumed": False,
    }


def _fake_execute_factory(call_counter: dict[str, int]):
    def fake_execute(trial, data_dir, out_dir, **kwargs):
        call_counter["n"] += 1
        return _fake_success_record(trial, out_dir, config_sha256=kwargs["config_sha256"], data_dir=data_dir)
    return fake_execute


def test_run_trials_writes_atomically_after_every_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trials = [
        _make_env_trial(env, det_threshold=det)
        for det in (0.52, 0.58)
    ]
    output_dir = tmp_path / "out"
    write_calls: list[int] = []
    real_write = calib._write_json_atomic

    def counting_write(path, value):
        write_calls.append(len(value["trials"]))
        real_write(path, value)

    monkeypatch.setattr(calib, "_write_json_atomic", counting_write)
    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory({"n": 0}))

    report = calib.run_trials(
        trials, env["data_dir"], output_dir, env["config_path"], "stage_a",
    )

    # One atomic write after each of the 2 trials, plus a final write with selection.
    assert write_calls[:2] == [1, 2]
    assert len(report["trials"]) == 2
    on_disk = json.loads((output_dir / "calibration_results.json").read_text())
    assert len(on_disk["trials"]) == 2
    assert on_disk["schema_version"] == 1
    assert on_disk["stage"] == "stage_a"
    assert "environment" in on_disk and "git" in on_disk["environment"]


def test_run_trials_resumes_valid_trials_without_reexecuting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    output_dir = tmp_path / "out"
    call_count = {"n": 0}
    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory(call_count))

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 1

    report = calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 1
    assert report["trials"][0]["resumed"] is True


def test_run_trials_reexecutes_when_checkpoint_hash_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    output_dir = tmp_path / "out"
    call_count = {"n": 0}
    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory(call_count))

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 1

    env["checkpoint_path"].write_bytes(b"weights-v2-different-hash")
    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 2


def test_run_trials_reexecutes_when_checkpoint_config_json_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    output_dir = tmp_path / "out"
    call_count = {"n": 0}
    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory(call_count))

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 1

    (env["checkpoint_path"].parent / "config.json").write_text(json.dumps({"unet_out_channels": 64}))
    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 2


def test_run_trials_reexecutes_when_split_file_content_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    output_dir = tmp_path / "out"
    call_count = {"n": 0}
    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory(call_count))

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 1

    env["split_path"].write_text(json.dumps([{"split": 0, "train": [], "test": ["a", "b"]}]))
    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 2


def test_run_trials_reexecutes_when_config_hash_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    output_dir = tmp_path / "out"
    call_count = {"n": 0}
    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory(call_count))

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 1

    env["config_path"].write_text(json.dumps({"placeholder": False, "changed": True}))
    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 2


def test_run_trials_reexecutes_when_data_dir_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    output_dir = tmp_path / "out"
    call_count = {"n": 0}
    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory(call_count))

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 1

    other_data_dir = tmp_path / "data2"
    other_data_dir.mkdir()
    calib.run_trials([trial], other_data_dir, output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 2


def test_run_trials_reexecutes_when_predictions_are_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    output_dir = tmp_path / "out"
    call_count = {"n": 0}
    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory(call_count))

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 1

    predictions_dir = output_dir / "trials" / trial["trial_id"] / "predictions"
    (predictions_dir / "a.geff").unlink()

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 2


def test_run_trials_reexecutes_when_prediction_bytes_change_but_stay_nonempty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GEFF that keeps the same name and stays nonempty, but whose bytes
    changed, must still be rejected by resume via the content manifest.
    """
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    output_dir = tmp_path / "out"
    call_count = {"n": 0}
    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory(call_count))

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 1

    predictions_dir = output_dir / "trials" / trial["trial_id"] / "predictions"
    geff = predictions_dir / "a.geff"
    assert geff.is_file() and geff.stat().st_size > 0
    geff.write_bytes(b"different-nonempty-content")  # same name, still nonempty, different bytes

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 2


def test_run_trials_reexecutes_when_on_disk_metrics_are_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    output_dir = tmp_path / "out"
    call_count = {"n": 0}
    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory(call_count))

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 1

    metrics_path = output_dir / "trials" / trial["trial_id"] / "metrics.json"
    tampered = _valid_metrics_report(["a"])
    tampered["summary_metrics"]["score"] = 0.9999
    metrics_path.write_text(json.dumps(tampered))

    calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert call_count["n"] == 2


def test_run_trials_records_failed_trial_without_aborting_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trials = [
        _make_env_trial(env, det_threshold=det)
        for det in (0.52, 0.58)
    ]
    output_dir = tmp_path / "out"

    def fake_execute(trial, data_dir, out_dir, **kwargs):
        if trial["det_threshold"] == 0.52:
            raise calib.CalibrationError("simulated prediction failure")
        return _fake_success_record(trial, out_dir, config_sha256=kwargs["config_sha256"], data_dir=data_dir)

    monkeypatch.setattr(calib, "execute_trial", fake_execute)
    report = calib.run_trials(
        trials, env["data_dir"], output_dir, env["config_path"], "stage_a",
        build_selection=lambda records: {"selected_det_thresholds": [0.58]},
    )

    statuses = {t["det_threshold"]: t["status"] for t in report["trials"]}
    assert statuses == {0.52: "failed", 0.58: "success"}
    failed = next(t for t in report["trials"] if t["status"] == "failed")
    assert "simulated prediction failure" in failed["error"]
    assert failed["checkpoint_sha256"] == calib._sha256(env["checkpoint_path"])

    on_disk = json.loads((output_dir / "calibration_results.json").read_text())
    assert {t["status"] for t in on_disk["trials"]} == {"failed", "success"}
    assert on_disk["selected_det_thresholds"] == [0.58]


def test_run_trials_fails_stage_when_selection_unavailable_but_still_persists_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trials = [_make_env_trial(env, det_threshold=0.52)]
    output_dir = tmp_path / "out"
    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory({"n": 0}))

    def build_selection(records):
        return {"selected_det_thresholds": calib.select_best_det_thresholds(records, 2)}

    with pytest.raises(calib.CalibrationError, match="requires 2"):
        calib.run_trials(
            trials, env["data_dir"], output_dir, env["config_path"], "stage_a",
            build_selection=build_selection,
        )

    on_disk = json.loads((output_dir / "calibration_results.json").read_text())
    assert len(on_disk["trials"]) == 1
    assert on_disk["trials"][0]["status"] == "success"
    assert "selected_det_thresholds" not in on_disk


def test_run_trials_persists_selected_det_thresholds_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trials = [_make_env_trial(env, det_threshold=det) for det in (0.52, 0.58, 0.64)]
    output_dir = tmp_path / "out"

    def fake_execute(trial, data_dir, out_dir, **kwargs):
        record = _fake_success_record(trial, out_dir, config_sha256=kwargs["config_sha256"], data_dir=data_dir)
        record["summary_metrics"] = {"score": {0.52: 0.5, 0.58: 0.9, 0.64: 0.7}[trial["det_threshold"]]}
        return record

    monkeypatch.setattr(calib, "execute_trial", fake_execute)

    def build_selection(records):
        return {"selected_det_thresholds": calib.select_best_det_thresholds(records, 2)}

    report = calib.run_trials(
        trials, env["data_dir"], output_dir, env["config_path"], "stage_a",
        build_selection=build_selection,
    )
    assert report["selected_det_thresholds"] == [0.58, 0.64]

    on_disk = json.loads((output_dir / "calibration_results.json").read_text())
    assert on_disk["selected_det_thresholds"] == [0.58, 0.64]

    # Rerunning (full resume) must reproduce exactly the same selection.
    report2 = calib.run_trials(
        trials, env["data_dir"], output_dir, env["config_path"], "stage_a",
        build_selection=build_selection,
    )
    assert report2["selected_det_thresholds"] == [0.58, 0.64]


def test_run_trials_persists_selected_final_trial_with_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trials = [
        _make_env_trial(env, stage="stage_c", det_threshold=0.58, edge_threshold=0.45, checkpoint_label="10000"),
        _make_env_trial(env, stage="stage_c", det_threshold=0.64, edge_threshold=0.50, checkpoint_label="10000"),
    ]
    output_dir = tmp_path / "out"

    def fake_execute(trial, data_dir, out_dir, **kwargs):
        record = _fake_success_record(trial, out_dir, config_sha256=kwargs["config_sha256"], data_dir=data_dir)
        record["summary_metrics"] = {"score": 0.9 if trial["det_threshold"] == 0.64 else 0.5}
        return record

    monkeypatch.setattr(calib, "execute_trial", fake_execute)

    def build_selection(records):
        best = calib.select_final_best(records, checkpoint_label_order=["10000", "7500"])
        return {"selected_final_trial": calib._selected_final_trial_summary(best)}

    report = calib.run_trials(
        trials, env["data_dir"], output_dir, env["config_path"], "stage_c",
        build_selection=build_selection,
    )

    final = report["selected_final_trial"]
    for key in (
        "trial_id", "checkpoint_label", "checkpoint_path", "checkpoint_sha256",
        "det_threshold", "edge_threshold", "official_score", "summary_metrics",
    ):
        assert key in final
    assert final["det_threshold"] == 0.64
    assert final["official_score"] == 0.9

    on_disk = json.loads((output_dir / "calibration_results.json").read_text())
    assert on_disk["selected_final_trial"]["trial_id"] == final["trial_id"]


def test_run_trials_is_independent_of_caller_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    output_dir = tmp_path / "out"
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()

    monkeypatch.setattr(calib, "execute_trial", _fake_execute_factory({"n": 0}))
    monkeypatch.chdir(other_cwd)

    report = calib.run_trials([trial], env["data_dir"], output_dir, env["config_path"], "stage_a")
    assert Path(report["results_path"]).is_relative_to(output_dir)
    assert Path(report["results_path"]).is_file()


def test_resume_is_valid_rejects_prior_with_mismatched_parameters(tmp_path: Path) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    config_sha256 = calib._require_sha256(env["config_path"])
    prior = _fake_success_record(trial, tmp_path / "out", config_sha256=config_sha256, data_dir=env["data_dir"])
    prior["det_threshold"] = 0.58  # parameters changed since the prior run
    assert calib._resume_is_valid(prior, trial, config_sha256=config_sha256, data_dir=env["data_dir"]) is False


def test_resume_is_valid_rejects_failed_prior(tmp_path: Path) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    prior = {**trial, "status": "failed", "error": "boom"}
    assert calib._resume_is_valid(prior, trial, config_sha256="x", data_dir=env["data_dir"]) is False


def test_resume_is_valid_rejects_when_metrics_file_missing(tmp_path: Path) -> None:
    env = _setup_trial_env(tmp_path)
    trial = _make_env_trial(env)
    config_sha256 = calib._require_sha256(env["config_path"])
    prior = _fake_success_record(trial, tmp_path / "out", config_sha256=config_sha256, data_dir=env["data_dir"])
    Path(prior["metrics_path"]).unlink()
    assert calib._resume_is_valid(prior, trial, config_sha256=config_sha256, data_dir=env["data_dir"]) is False
