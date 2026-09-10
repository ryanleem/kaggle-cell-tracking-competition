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


def _make_runner_config(split_path: Path, checkpoint_iters: list[int] | None = None) -> dict[str, object]:
    config: dict[str, object] = {
        "method": "baseline",
        "split": 0,
        "seed": 0,
        "epochs": 1,
        "max_iters": 2,
        "batch_size": 2,
        "num_workers": 0,
        "lr": 0.0001,
        "det_loss_weight": 1.0,
        "det_neg_weight": 0.01,
        "unet_out_channels": 32,
        "unet_layers": "32,64,128",
        "downsample": "1,4,4",
        "window_size": 2,
        "pool_kernel_um": 5.0,
        "det_threshold": 0.52,
        "tracking": "greedy",
        "split_file": str(split_path),
    }
    if checkpoint_iters is not None:
        config["checkpoint_iters"] = checkpoint_iters
    return config


def _install_fake_runner_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    expected_datasets: list[str],
    checkpoint_iters: list[int],
    missing_checkpoint: int | None = None,
    checkpoint_scores: dict[int, float] | None = None,
    stage_calls: list[tuple[str, list[str]]] | None = None,
) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", repo_root)
    checkpoint_config_bytes = b'{"model": "periodic-regression", "version": 1}\n'
    monkeypatch.setattr(runner, "_git_provenance", lambda: {
        "commit": "deadbeef", "dirty": False, "status_porcelain": [],
    })

    def command_value(command: list[str], flag: str) -> str:
        return command[command.index(flag) + 1]

    def fake_run_stage(
        stage: str,
        command: list[str],
        run_dir: Path,
        provenance: dict[str, object],
        provenance_path: Path,
    ) -> None:
        if stage_calls is not None:
            stage_calls.append((stage, command))
        provenance.setdefault("stages", {})[stage] = {"returncode": 0, "wall_seconds": 0.0}
        (run_dir / f"{stage}.stdout.log").write_text("fake stdout")
        (run_dir / f"{stage}.stderr.log").write_text("")
        _write = runner._write_json
        _write(provenance_path, provenance)
        run_method = command_value(command, "--method") if "--method" in command else ""

        if stage == "training":
            output_dir = repo_root / "weights" / run_method / "split_0"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "edge_predictor_best.pth").write_bytes(b"best")
            (output_dir / "config.json").write_bytes(checkpoint_config_bytes)
            history_records = []
            for iteration in checkpoint_iters:
                filename = f"edge_predictor_iter_{iteration:06d}.pth"
                if iteration != missing_checkpoint:
                    (output_dir / filename).write_bytes(filename.encode())
                history_records.append({
                    "iteration": iteration,
                    "running_avg_edge_loss": 0.1,
                    "running_avg_detection_loss": 0.2,
                    "checkpoint_filename": filename,
                    "elapsed_training_seconds": float(iteration),
                })
            (output_dir / "training_history.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in history_records),
            )
        elif stage == "prediction" or stage.startswith("checkpoint_iter_") and stage.endswith("_prediction"):
            if stage.startswith("checkpoint_iter_"):
                checkpoint_path = Path(command_value(command, "--weights"))
                periodic_config_path = checkpoint_path.parent / "config.json"
                assert periodic_config_path.is_file()
                assert periodic_config_path.read_bytes() == checkpoint_config_bytes
            username = runner.os.environ.get("USER", runner.os.environ.get("USERNAME", "unknown"))
            prediction_dir = repo_root / "predictions" / username / run_method / "split_0"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            for dataset in expected_datasets:
                (prediction_dir / f"{dataset}.geff").write_text("fake geff")
        elif stage == "evaluation" or stage.startswith("checkpoint_iter_") and stage.endswith("_evaluation"):
            metrics_path = Path(command_value(command, "--json-out"))
            iteration = int(stage.split("_")[2]) if stage.endswith("_evaluation") else None
            metrics = {
                "edge_tp": 1, "edge_fp": 0, "edge_fn": 0,
                "num_pred_nodes": 1, "total_node_ratio": 0.0,
                "node_recall": 1.0, "edge_jaccard": 1.0,
                "adj_edge_jaccard": 1.0,
            }
            metrics_path.write_text(json.dumps({
                "evaluated_datasets": expected_datasets,
                "skipped_datasets": [],
                "per_dataset_metrics": [
                    {"dataset": dataset, "metrics": metrics}
                    for dataset in expected_datasets
                ],
                "summary_metrics": {
                    "score": checkpoint_scores.get(iteration, 1.0)
                    if checkpoint_scores is not None and iteration is not None else 1.0,
                },
            }))
        elif stage == "geffs_to_csv":
            csv_path = Path(command_value(command, "--csv"))
            rows = ["id,dataset,row_type,node_id,t,z,y,x,source_id,target_id"]
            rows.extend(
                f"{index},{dataset},node,0,0,1,2,3,-1,-1"
                for index, dataset in enumerate(expected_datasets)
            )
            csv_path.write_text("\n".join(rows) + "\n")
        elif stage == "csv_to_geffs":
            output_dir = Path(command_value(command, "--out-dir"))
            output_dir.mkdir(parents=True, exist_ok=True)
            for dataset in expected_datasets:
                (output_dir / f"{dataset}.geff").write_text("fake geff")
        elif stage == "roundtrip_evaluation":
            metrics_path = Path(command_value(command, "--json-out"))
            metrics_path.write_text(json.dumps({
                "evaluated_datasets": expected_datasets,
                "skipped_datasets": [],
                "summary_metrics": {"score": 1.0},
            }))

    monkeypatch.setattr(runner, "_run_stage", fake_run_stage)


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


def test_baseline_pilot_5000_config_changes_only_iterations_and_artifact_method() -> None:
    config_dir = Path(__file__).parent.parent / "experiments" / "configs"
    baseline = json.loads((config_dir / "baseline_pilot_500.json").read_text())
    pilot = json.loads((config_dir / "baseline_pilot_5000.json").read_text())

    assert baseline["max_iters"] == 500
    assert pilot["max_iters"] == 5000
    assert baseline["method"] == "baseline_pilot_500"
    assert pilot["method"] == "baseline_pilot_5000"
    assert set(pilot) == set(baseline) | {"checkpoint_iters"}
    for key in baseline:
        if key not in {"method", "max_iters"}:
            assert pilot[key] == baseline[key]
    assert pilot["checkpoint_iters"] == [500, 1000, 2000, 5000]
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
    assert "--checkpoint-iters" not in commands["training"]
    assert "--use-ilp" not in commands["prediction"]
    assert "--strict" in commands["evaluation"]
    assert "--require-all-gt" not in commands["evaluation"]
    assert "baseline_unique" in commands["prediction"]


def test_build_commands_forwards_checkpoint_iters(tmp_path: Path) -> None:
    config = {
        "split": 0, "seed": 17, "epochs": 1, "lr": 0.0001, "batch_size": 2, "num_workers": 2,
        "unet_out_channels": 32, "unet_layers": "32,64,128", "downsample": "1,4,4",
        "det_loss_weight": 1.0, "det_neg_weight": 0.01, "max_iters": 5000,
        "checkpoint_iters": [500, 1000, 2000, 5000], "window_size": 2,
        "pool_kernel_um": 5.0, "det_threshold": 0.52,
    }
    commands = runner.build_commands(
        config, "baseline_unique", tmp_path / "data", tmp_path / "split.json",
        tmp_path / "weights" / "edge_predictor_best.pth", tmp_path / "pred", tmp_path,
    )

    checkpoint_index = commands["training"].index("--checkpoint-iters")
    assert commands["training"][checkpoint_index + 1] == "500,1000,2000,5000"


@pytest.mark.parametrize(
    "checkpoint_iters",
    [
        [],
        [500, "bad"],
        [500, 500],
        [0, 500],
        [1000, 500],
        [500, 1001],
    ],
)
def test_runner_rejects_invalid_checkpoint_schedules(checkpoint_iters: list[object]) -> None:
    config = {"checkpoint_iters": checkpoint_iters, "max_iters": 1000}
    with pytest.raises(runner.ExperimentError):
        runner._validate_checkpoint_iters(config)


def test_runner_keeps_older_configs_compatible() -> None:
    config_dir = Path(__file__).parent.parent / "experiments" / "configs"
    older_config = json.loads((config_dir / "baseline_pilot_500.json").read_text())
    runner._validate_checkpoint_iters(older_config)


def test_runner_fails_when_requested_periodic_checkpoint_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, split_path = _make_split_fixture(tmp_path)
    fold, _ = runner._load_and_validate_split(split_path, data_dir, 0)
    checkpoint_iters = [1, 2]
    _install_fake_runner_pipeline(
        monkeypatch, tmp_path / "repo", fold["test"], checkpoint_iters,
        missing_checkpoint=2,
    )
    config_path = tmp_path / "config.json"
    config = _make_runner_config(split_path, checkpoint_iters)
    config["max_iters"] = 3
    config_path.write_text(json.dumps(config))

    with pytest.raises(runner.ExperimentError, match="periodic checkpoints missing"):
        runner.run_experiment(config_path, data_dir, tmp_path / "runs")


def test_runner_preserves_periodic_checkpoints_history_and_final_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, split_path = _make_split_fixture(tmp_path)
    fold, _ = runner._load_and_validate_split(split_path, data_dir, 0)
    checkpoint_iters = [1, 2]
    expected_datasets = fold["test"]
    _install_fake_runner_pipeline(
        monkeypatch, tmp_path / "repo", expected_datasets, checkpoint_iters,
    )
    config_path = tmp_path / "config.json"
    config = _make_runner_config(split_path, checkpoint_iters)
    config["max_iters"] = 3
    config_path.write_text(json.dumps(config))

    run_dir = runner.run_experiment(config_path, data_dir, tmp_path / "runs")

    expected_names = [f"edge_predictor_iter_{iteration:06d}.pth" for iteration in checkpoint_iters]
    preserved = run_dir / "checkpoints"
    assert [path.name for path in sorted(preserved.glob("*.pth"))] == expected_names
    original_checkpoint_config = next((tmp_path / "repo" / "weights").glob("baseline_*/split_0/config.json"))
    periodic_checkpoint_config = preserved / "config.json"
    assert periodic_checkpoint_config.is_file()
    assert periodic_checkpoint_config.read_bytes() == original_checkpoint_config.read_bytes()
    history_path = run_dir / "training_history.jsonl"
    assert history_path.is_file()
    records = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert [record["checkpoint_filename"] for record in records] == expected_names

    final_result = json.loads((run_dir / "final_result.json").read_text())
    assert final_result["artifacts"]["periodic_checkpoints"] == [
        str(preserved / name) for name in expected_names
    ]
    assert final_result["artifacts"]["training_history"] == str(history_path)
    assert final_result["artifacts"]["periodic_checkpoint_config"] == str(periodic_checkpoint_config)


def test_runner_evaluates_all_periodic_checkpoints_selects_official_best_and_cleans_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, split_path = _make_split_fixture(tmp_path)
    fold, _ = runner._load_and_validate_split(split_path, data_dir, 0)
    checkpoint_iters = [1, 2, 3]
    stage_calls: list[tuple[str, list[str]]] = []
    _install_fake_runner_pipeline(
        monkeypatch,
        tmp_path / "repo",
        fold["test"],
        checkpoint_iters,
        checkpoint_scores={1: 0.8, 2: 0.9, 3: 0.9},
        stage_calls=stage_calls,
    )
    config_path = tmp_path / "config.json"
    config = _make_runner_config(split_path, checkpoint_iters)
    config["max_iters"] = 3
    config_path.write_text(json.dumps(config))

    run_dir = runner.run_experiment(config_path, data_dir, tmp_path / "runs")

    evaluations = json.loads((run_dir / "checkpoint_evaluations.json").read_text())
    periodic_checkpoint_config = run_dir / "checkpoints" / "config.json"
    original_checkpoint_config = next((tmp_path / "repo" / "weights").glob("baseline_*/split_0/config.json"))
    assert evaluations["periodic_checkpoint_config_path"] == str(periodic_checkpoint_config)
    assert periodic_checkpoint_config.is_file()
    assert periodic_checkpoint_config.read_bytes() == original_checkpoint_config.read_bytes()
    assert evaluations["selection_metric"] == "summary_metrics.score"
    assert evaluations["selected_iteration"] == 3
    assert evaluations["selected_checkpoint"] == "edge_predictor_iter_000003.pth"
    assert [row["iteration"] for row in evaluations["evaluations"]] == checkpoint_iters
    assert [row["summary_metrics"]["score"] for row in evaluations["evaluations"]] == [0.8, 0.9, 0.9]
    assert all("edge_tp" in row["per_dataset_metrics"][0]["metrics"] for row in evaluations["evaluations"])
    assert all(not Path(row["prediction_dir"]).exists() for row in evaluations["evaluations"])
    assert all(Path(row["metrics_path"]).is_file() for row in evaluations["evaluations"])
    assert all(
        (Path(row["prediction_log_dir"]) / f"{row['prediction_stage']}.stdout.log").is_file()
        and (Path(row["prediction_log_dir"]) / f"{row['evaluation_stage']}.stdout.log").is_file()
        for row in evaluations["evaluations"]
    )
    periodic_prediction_commands = [
        command for stage, command in stage_calls
        if stage.startswith("checkpoint_iter_") and stage.endswith("_prediction")
    ]
    assert len(periodic_prediction_commands) == len(checkpoint_iters)
    assert all(
        Path(command[command.index("--weights") + 1]).parent / "config.json" == periodic_checkpoint_config
        for command in periodic_prediction_commands
    )
    assert all(
        row["periodic_checkpoint_config_path"] == str(periodic_checkpoint_config)
        for row in evaluations["evaluations"]
    )

    official = run_dir / "checkpoint" / "edge_predictor_official_best.pth"
    training_selected = run_dir / "checkpoint" / "edge_predictor_best.pth"
    periodic_selected = run_dir / "checkpoints" / "edge_predictor_iter_000003.pth"
    assert official.read_bytes() == periodic_selected.read_bytes()
    assert official.read_bytes() != training_selected.read_bytes()

    final_prediction_commands = [command for stage, command in stage_calls if stage == "prediction"]
    assert len(final_prediction_commands) == 1
    assert Path(final_prediction_commands[0][final_prediction_commands[0].index("--weights") + 1]) == official

    provenance = json.loads((run_dir / "provenance.json").read_text())
    final_result = json.loads((run_dir / "final_result.json").read_text())
    for artifact in (provenance, final_result):
        assert artifact["checkpoint_selection_metric"] == "summary_metrics.score"
        assert artifact["selected_checkpoint_iteration"] == 3
        assert artifact["selected_checkpoint_score"] == 0.9
        assert artifact["selected_checkpoint_path"] == str(run_dir / "checkpoints" / "edge_predictor_iter_000003.pth")
        assert artifact["checkpoint_evaluations_artifact_path"] == str(run_dir / "checkpoint_evaluations.json")
        assert artifact["periodic_checkpoint_config_path"] == str(periodic_checkpoint_config)
        assert artifact["training_selected_checkpoint_path"] == str(training_selected)
        assert artifact["official_selected_checkpoint_path"] == str(official)
    assert final_result["artifacts"]["periodic_checkpoint_config"] == str(periodic_checkpoint_config)


def _valid_checkpoint_metrics() -> dict[str, object]:
    return {
        "evaluated_datasets": ["sample"],
        "skipped_datasets": [],
        "summary_metrics": {"score": 0.5},
        "per_dataset_metrics": [{
            "dataset": "sample",
            "metrics": {
                "edge_tp": 1, "edge_fp": 0, "edge_fn": 0,
                "num_pred_nodes": 1, "total_node_ratio": 0.0,
                "node_recall": 1.0, "edge_jaccard": 1.0,
                "adj_edge_jaccard": 1.0,
            },
        }],
    }


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda report: report.pop("summary_metrics"), "summary_metrics"),
        (lambda report: report["summary_metrics"].update(score=float("nan")), "not finite"),
        (lambda report: report["summary_metrics"].update(score=None), "not finite"),
        (lambda report: report.pop("per_dataset_metrics"), "per_dataset_metrics"),
        (lambda report: report["per_dataset_metrics"][0]["metrics"].pop("edge_tp"), "per-dataset"),
    ],
)
def test_checkpoint_metrics_validation_rejects_malformed_missing_or_nonfinite_reports(
    tmp_path: Path, mutate, match: str,
) -> None:
    report = _valid_checkpoint_metrics()
    mutate(report)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(report, allow_nan=True))
    with pytest.raises(runner.ExperimentError, match=match):
        runner._validate_checkpoint_metrics(metrics_path, ["sample"])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("evaluated_datasets", ["other"], "datasets differ"),
        ("skipped_datasets", ["sample"], "skipped"),
    ],
)
def test_checkpoint_metrics_validation_rejects_dataset_mismatch_and_skips(
    tmp_path: Path, field: str, value: object, match: str,
) -> None:
    report = _valid_checkpoint_metrics()
    report[field] = value
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(report))
    with pytest.raises(runner.ExperimentError, match=match):
        runner._validate_checkpoint_metrics(metrics_path, ["sample"])


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
