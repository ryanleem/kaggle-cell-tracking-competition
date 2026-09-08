import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import evaluate as evaluate_script
import run_baseline_experiment as runner
import train_unet_transformer as training_script


def _make_split_fixture(tmp_path: Path, count: int = 3) -> tuple[Path, Path]:
    data_dir = tmp_path / "train"
    data_dir.mkdir()
    names = [f"sample_{index:02d}" for index in range(count)]
    for name in names:
        (data_dir / f"{name}.zarr").mkdir()
        (data_dir / f"{name}.geff").mkdir()
    shuffled = sorted(names)
    random.Random(0).shuffle(shuffled)
    n_val = max(1, count // 10)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps([{
        "split": 0,
        "train": shuffled[n_val:],
        "test": shuffled[:n_val],
    }]))
    return data_dir, split_path


def test_baseline_pilot_500_config_changes_only_iterations_and_artifact_method() -> None:
    config_dir = Path(__file__).parent.parent / "experiments" / "configs"
    baseline = json.loads((config_dir / "baseline_smoke.json").read_text())
    pilot = json.loads((config_dir / "baseline_pilot_500.json").read_text())

    assert set(pilot) == set(baseline)
    assert baseline["max_iters"] == 2
    assert pilot["max_iters"] == 500
    assert baseline["method"] == "baseline"
    assert pilot["method"] == "baseline_pilot_500"
    for key in baseline:
        if key not in {"method", "max_iters"}:
            assert pilot[key] == baseline[key]
    assert pilot["split_file"] == "experiments/splits/baseline_fold0_seed0.json"
    assert pilot["seed"] == 0
    assert pilot["epochs"] == 1


def test_evaluate_json_out_converts_nonfinite_values_and_preserves_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    (pred_dir / "sample.geff").mkdir()
    (gt_dir / "sample.geff").mkdir()
    rows = [{"edge_tp": 1, "division_jaccard": float("nan")}]
    summary = {
        "n": 1, "score": 0.25, "edge_jaccard": 0.5,
        "adj_edge_jaccard": 0.5, "n_adj": 1,
        "division_jaccard": float("nan"), "division_tp": 0,
        "division_fp": 0, "division_fn": 0, "node_recall": 1.0,
    }
    monkeypatch.setattr(evaluate_script, "evaluate_pairs", lambda *args, **kwargs: (rows, []))
    monkeypatch.setattr(evaluate_script, "summarise", lambda rows: summary)
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys, "argv", [
            "evaluate.py", "--pred-dir", str(pred_dir), "--gt-dir", str(gt_dir),
            "--strict", "--json-out", str(output),
        ],
    )

    evaluate_script.main()

    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["evaluated_datasets"] == ["sample"]
    assert report["skipped_datasets"] == []
    assert report["per_dataset_metrics"][0]["metrics"]["division_jaccard"] is None
    assert report["summary_metrics"]["division_jaccard"] is None
    assert 'NaN' not in output.read_text()


def test_evaluate_json_report_keeps_allow_nan_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate_script, "_json_safe", lambda value: value)

    with pytest.raises(ValueError, match="Out of range float values"):
        evaluate_script._write_json_report(
            tmp_path / "metrics.json", ["sample"], [], [{}],
            {"division_jaccard": float("nan")},
        )


def test_split_validation_matches_fallback_and_exact_coverage(tmp_path: Path) -> None:
    data_dir, split_path = _make_split_fixture(tmp_path)

    fold, validation = runner._load_and_validate_split(split_path, data_dir, 0)

    assert fold["split"] == 0
    assert validation["paired_count"] == 3
    assert validation["train_count"] == 2
    assert validation["test_count"] == 1
    assert validation["missing"] == []
    assert validation["extra"] == []
    assert validation["algorithm_reproduction"] is True


def test_build_commands_uses_current_cli_flags_only(tmp_path: Path) -> None:
    config = {
        "split": 0, "seed": 17, "epochs": 1, "lr": 0.0001, "batch_size": 2, "num_workers": 2,
        "unet_out_channels": 32, "unet_layers": "32,64,128", "downsample": "1,4,4",
        "det_loss_weight": 1.0, "det_neg_weight": 0.01, "max_iters": 2,
        "window_size": 2, "pool_kernel_um": 5.0, "det_threshold": 0.52,
    }
    commands = runner.build_commands(
        config, "baseline_unique", tmp_path / "data", tmp_path / "split.json",
        tmp_path / "weights" / "edge_predictor_best.pth", tmp_path / "pred", tmp_path,
    )

    assert commands["training"][0] == sys.executable
    seed_index = commands["training"].index("--seed")
    assert commands["training"][seed_index + 1] == "17"
    assert "--use-ilp" not in commands["prediction"]
    assert "--strict" in commands["evaluation"]
    assert "--require-all-gt" not in commands["evaluation"]
    assert "baseline_unique" in commands["prediction"]


def test_validate_csv_enforces_repository_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n"
        "0,sample,node,0,0,1,2,3,-1,-1\n"
        "1,sample,node,1,1,2,3,4,-1,-1\n"
        "2,sample,edge,-1,-1,-1,-1,-1,0,1\n",
    )

    report = runner.validate_csv(csv_path, ["sample"])

    assert report["columns"] == runner.EXPECTED_CSV_COLUMNS
    assert report["row_count"] == 3
    assert report["node_count"] == 2
    assert report["edge_count"] == 1


def test_runner_refuses_dirty_worktree_without_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir, split_path = _make_split_fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "split": 0, "seed": 0, "epochs": 1, "max_iters": 2, "batch_size": 2,
        "num_workers": 2, "lr": 0.0001, "det_loss_weight": 1.0, "det_neg_weight": 0.01,
        "unet_out_channels": 32, "unet_layers": "32,64,128", "downsample": "1,4,4",
        "window_size": 2, "pool_kernel_um": 5.0, "det_threshold": 0.52,
        "tracking": "greedy", "split_file": str(split_path),
    }))
    monkeypatch.setattr(runner, "_git_provenance", lambda: {
        "commit": "deadbeef", "dirty": True, "status_porcelain": [" M file.py"],
    })

    with pytest.raises(runner.ExperimentError, match="dirty Git worktree"):
        runner.run_experiment(config_path, data_dir, tmp_path / "runs")


def test_training_cli_passes_seed_to_existing_train_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(training_script, "train", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys, "argv", [
            "train_unet_transformer.py",
            "--data-dir", str(tmp_path),
            "--splits", str(tmp_path / "split.json"),
            "--seed", "123",
        ],
    )

    training_script.main()

    assert calls[0]["seed"] == 123
    assert "Effective seed: 123" in capsys.readouterr().out


def test_runner_passes_and_records_seed_in_config_and_provenance_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, split_path = _make_split_fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "method": "baseline", "split": 0, "seed": 123, "epochs": 1, "max_iters": 2,
        "batch_size": 2, "num_workers": 0, "lr": 0.0001, "det_loss_weight": 1.0,
        "det_neg_weight": 0.01, "unet_out_channels": 32, "unet_layers": "32,64,128",
        "downsample": "1,4,4", "window_size": 2, "pool_kernel_um": 5.0,
        "det_threshold": 0.52, "tracking": "greedy", "split_file": str(split_path),
    }))
    monkeypatch.setattr(runner, "_git_provenance", lambda: {
        "commit": "deadbeef", "dirty": False, "status_porcelain": [],
    })
    captured: dict[str, object] = {}

    def stop_before_training(stage, command, run_dir, provenance, provenance_path):
        captured["stage"] = stage
        captured["command"] = command
        raise runner.ExperimentError("test stop before training")

    monkeypatch.setattr(runner, "_run_stage", stop_before_training)
    output_root = tmp_path / "runs"
    with pytest.raises(runner.ExperimentError, match="test stop"):
        runner.run_experiment(config_path, data_dir, output_root)

    run_dir = next(output_root.iterdir())
    command = captured["command"]
    assert captured["stage"] == "training"
    assert command[command.index("--seed") + 1] == "123"
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert resolved["seed"] == provenance["seed"] == 123
    assert resolved["seed_passed_to_training"] is True


def test_training_saves_seed_in_checkpoint_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights_root = tmp_path / "weights"
    monkeypatch.setattr(training_script, "WEIGHTS_PATH", weights_root)

    def stop_after_config(*args, **kwargs):
        raise RuntimeError("stop after config")

    monkeypatch.setattr(training_script, "load_dataset_windows", stop_after_config)
    with pytest.raises(RuntimeError, match="stop after config"):
        training_script.train(
            data_dir=tmp_path,
            fold=0,
            splits_file=tmp_path / "missing-splits.json",
            method="seed_check",
            debug_video=tmp_path / "debug.zarr",
            seed=123,
        )

    config_path = weights_root / "seed_check" / "split_0" / "config.json"
    assert json.loads(config_path.read_text())["seed"] == 123
