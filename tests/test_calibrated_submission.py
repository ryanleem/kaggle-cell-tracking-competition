"""Mocked submission orchestration tests: no inference or training is executed."""
import ast
import builtins
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_calibrated_submission as runner


def write_csv(path, names, mutation=None):
    rows = []
    for name in names:
        rows.extend([
            [str(len(rows)), name, "node", "0", "0", "1", "2", "3", "-1", "-1"],
            [str(len(rows) + 1), name, "edge", "-1", "-1", "-1", "-1", "-1", "0", "0"],
        ])
    columns = runner.COLUMNS.copy()
    if mutation:
        mutation(columns, rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    data = tmp_path / "test"
    data.mkdir()
    for name in ("private_alpha", "hidden_beta"):
        (data / f"{name}.zarr").mkdir()
    checkpoint = tmp_path / "edge_predictor_iter_010000.pth"
    checkpoint.write_bytes(b"mock weights")
    checkpoint_config = tmp_path / "config.json"
    checkpoint_config.write_text("{}")
    output = tmp_path / "out"
    calls = []
    real_hash = runner.sha256
    def fake_hash(p):
        if p == checkpoint:
            return runner.CHECKPOINT_SHA256
        if p == checkpoint_config:
            return runner.CHECKPOINT_CONFIG_SHA256
        return real_hash(p)
    monkeypatch.setattr(runner, "sha256", fake_hash)
    monkeypatch.setattr(runner, "git_commit", lambda: "a" * 40)

    def command(cmd, log):
        calls.append(cmd)
        if Path(cmd[2]).name == "predict_unet_transformer.py":
            target = Path(cmd[cmd.index("--output-dir") + 1])
            target.mkdir()
            names = json.loads(Path(cmd[cmd.index("--splits") + 1]).read_text())[0]["test"]
            for name in names:
                (target / f"{name}.geff").mkdir()
        elif Path(cmd[2]).name == "geffs_to_csv.py":
            source = Path(cmd[cmd.index("--in-dir") + 1])
            write_csv(Path(cmd[cmd.index("--csv") + 1]), sorted(p.stem for p in source.glob("*.geff")))
        else:
            pytest.fail(f"unapproved command {cmd}")
        return 0.25

    monkeypatch.setattr(runner, "run_command", command)
    return data, checkpoint, output, calls, command


def assert_clean(output):
    assert not (output / "submission.csv").exists()
    assert not (output / "submission_provenance.json").exists()
    assert not list(output.glob(".calibrated-submission*"))


def test_success_hidden_discovery_path_independence_safe_cleanup(pipeline, tmp_path, monkeypatch):
    data, checkpoint, output, calls, _ = pipeline
    nested = data / "substituted"
    nested.mkdir()
    (nested / "unknown_2026.zarr").mkdir()
    # Zarr internals must never be mistaken for additional datasets.
    (nested / "unknown_2026.zarr" / "internal.zarr").mkdir()
    output.mkdir()
    sentinel = output / "unrelated.txt"
    sentinel.write_text("keep")
    other_scratch = output / ".calibrated-submission-unrelated"
    other_scratch.mkdir()
    checkpoint_bytes = checkpoint.read_bytes()
    monkeypatch.chdir(tmp_path)
    final = runner.run_submission(Path("test"), Path(checkpoint.name), Path("out"))
    assert final == output / "submission.csv"
    record = json.loads((output / "submission_provenance.json").read_text())
    assert record["discovered_datasets"] == ["hidden_beta", "private_alpha", "unknown_2026"]
    assert record["counts"]["rows"] == 6
    assert record["counts"]["nodes"] == record["counts"]["edges"] == 3
    assert record["training_performed"] is False
    assert record["checkpoint_sha256"] == runner.CHECKPOINT_SHA256
    assert record["checkpoint_config_sha256"] == runner.CHECKPOINT_CONFIG_SHA256
    assert record["output_sha256"] == runner.sha256(final)
    assert record["git_commit"] == "a" * 40
    assert record["elapsed_seconds"]["conversion"] == 0.25
    assert record["total_elapsed_seconds"] >= 0
    assert record["commands"] == calls
    assert record["command_lines"]
    for cmd in calls[:-1]:
        for flag, value in (("--det-threshold", "0.70"), ("--edge-threshold", "0.35"),
                            ("--split", "0"), ("--tracking", "greedy")):
            assert cmd[cmd.index(flag) + 1] == value
        assert Path(cmd[2]).is_absolute()
    assert sentinel.read_text() == "keep"
    assert list(output.glob(".calibrated-submission*")) == [other_scratch]
    assert checkpoint.read_bytes() == checkpoint_bytes


@pytest.mark.parametrize("problem", ["missing_checkpoint", "bad_hash", "missing_config", "invalid_config",
                                     "config_array", "config_hash_mismatch", "bad_reviewed_config",
                                     "reviewed_config_hash_mismatch", "missing_data", "empty_data", "duplicate_dataset"])
def test_input_validation_removes_stale_outputs(pipeline, monkeypatch, problem, tmp_path):
    data, checkpoint, output, calls, _ = pipeline
    output.mkdir()
    (output / "submission.csv").write_text("stale")
    (output / "submission_provenance.json").write_text('{"status":"success"}')
    if problem == "missing_checkpoint":
        checkpoint.unlink()
    elif problem == "bad_hash":
        monkeypatch.setattr(runner, "sha256", lambda p: "0" * 64)
    elif problem == "missing_config":
        (checkpoint.parent / "config.json").unlink()
    elif problem in ("invalid_config", "config_array"):
        (checkpoint.parent / "config.json").write_text("{" if problem == "invalid_config" else "[]")
    elif problem == "config_hash_mismatch":
        # Byte-modified but otherwise well-formed JSON object: only the hash check catches this.
        monkeypatch.setattr(runner, "sha256",
                             lambda p: runner.CHECKPOINT_SHA256 if p == checkpoint
                             else hashlib.sha256(p.read_bytes()).hexdigest())
        (checkpoint.parent / "config.json").write_text('{"tampered": true}')
    elif problem == "bad_reviewed_config":
        config = json.loads(runner.CONFIG_PATH.read_text(encoding="utf-8-sig"))
        config["det_threshold"] = 0.5
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(config))
        monkeypatch.setattr(runner, "CONFIG_PATH", path)
    elif problem == "reviewed_config_hash_mismatch":
        config = json.loads(runner.CONFIG_PATH.read_text(encoding="utf-8-sig"))
        config["checkpoint_config_sha256"] = "0" * 64
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(config))
        monkeypatch.setattr(runner, "CONFIG_PATH", path)
    elif problem == "missing_data":
        data = data / "missing"
    elif problem == "empty_data":
        data = tmp_path / "empty"
        data.mkdir()
    elif problem == "duplicate_dataset":
        (data / "nested" / "hidden_beta.zarr").mkdir(parents=True)
    with pytest.raises((runner.SubmissionError, ValueError)):
        runner.run_submission(data, checkpoint, output)
    assert calls == []
    assert_clean(output)


@pytest.mark.parametrize("problem", ["missing", "unexpected", "prediction_failure", "conversion_failure", "invalid_csv", "publish_failure", "interrupt"])
def test_pipeline_failures_never_publish_stale_csv(pipeline, monkeypatch, problem):
    data, checkpoint, output, calls, fake = pipeline
    output.mkdir()
    (output / "submission.csv").write_text("stale")
    sentinel = output / "keep"
    sentinel.mkdir()

    def command(cmd, log):
        assert not (output / "submission.csv").exists()
        prediction = Path(cmd[2]).name == "predict_unet_transformer.py"
        if problem == "interrupt":
            raise KeyboardInterrupt()
        if (prediction and problem == "prediction_failure") or (not prediction and problem == "conversion_failure"):
            raise runner.SubmissionError("subprocess failure")
        result = fake(cmd, log)
        if prediction and problem in ("missing", "unexpected"):
            target = Path(cmd[cmd.index("--output-dir") + 1])
            if problem == "missing":
                next(target.glob("*.geff")).rmdir()
            else:
                (target / "unexpected.geff").mkdir()
        if not prediction and problem == "invalid_csv":
            Path(cmd[cmd.index("--csv") + 1]).write_text("wrong,columns\n")
        return result

    monkeypatch.setattr(runner, "run_command", command)
    original_replace = Path.replace
    if problem == "publish_failure":
        def replace(path, target):
            if target == output / "submission.csv":
                raise OSError("publication denied")
            return original_replace(path, target)
        monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises((runner.SubmissionError, OSError, KeyboardInterrupt)):
        runner.run_submission(data, checkpoint, output)
    assert_clean(output)
    assert sentinel.is_dir()


@pytest.mark.parametrize("problem", ["columns", "duplicate_column", "empty_id", "duplicate_id", "type", "unexpected", "missing_dataset", "no_nodes", "source", "target", "negative_source", "extra_field", "short_row", "node_coordinate"])
def test_strict_csv_validation(tmp_path, problem):
    def mutate(columns, rows):
        if problem == "columns": columns.reverse()
        elif problem == "duplicate_column": columns[1] = "id"
        elif problem == "empty_id": rows[0][0] = " "
        elif problem == "duplicate_id": rows[1][0] = rows[0][0]
        elif problem == "type": rows[0][2] = "other"
        elif problem == "unexpected": rows[0][1] = "intruder"
        elif problem == "missing_dataset": rows[:] = rows[:2]
        elif problem == "no_nodes": rows[:] = [r for r in rows if r[2] == "edge"]
        elif problem == "source": rows[1][8] = ""
        elif problem == "target": rows[1][9] = "null"
        elif problem == "negative_source": rows[1][8] = "-1"
        elif problem == "extra_field": rows[0].append("extra")
        elif problem == "short_row": rows[0].pop()
        elif problem == "node_coordinate": rows[0][4] = "nan"
    path = tmp_path / "submission.csv"
    write_csv(path, ["a", "b"], mutate)
    with pytest.raises(runner.SubmissionError):
        runner.validate_csv(path, ["a", "b"])


@pytest.mark.parametrize("command", [
    [sys.executable, "-u", str(runner.REPO_ROOT / "scripts/train_unet_transformer.py")],
    ["uv", "run", "train"],
    [sys.executable, "-c", "print('not permitted')"],
    [sys.executable, "-u", "geffs_to_csv.py"],
])
def test_command_allowlist_cannot_launch_training(command, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(runner.SubmissionError, match="approved"):
        runner.run_command(command, tmp_path / "log")


def test_predictor_import_never_imports_training(monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if "train_unet_transformer" in name:
            pytest.fail("inference imported training")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    namespace = runpy.run_path(str(runner.SCRIPTS["predict_unet_transformer.py"]), run_name="inference_import_test")
    assert namespace["UNetNodeTransformer"].__module__ == "tracking_cellmot.unet_transformer"
    shared = ast.parse((runner.REPO_ROOT / "src/tracking_cellmot/unet_transformer.py").read_text())
    assert not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("train")
                   for n in ast.walk(shared))


@pytest.mark.parametrize("returncode", [0, 7])
def test_subprocess_stream_heartbeat_failure_and_cwd(tmp_path, monkeypatch, capsys, returncode):
    class Output(io.StringIO):
        def __next__(self):
            time.sleep(0.02)
            return super().__next__()
    class Process:
        stdout = Output("inference progress\n")
        def wait(self, **kwargs): return returncode
        def poll(self): return returncode
    def popen(command, **kwargs):
        assert kwargs["cwd"] == runner.REPO_ROOT
        assert kwargs["stderr"] == runner.subprocess.STDOUT
        assert "shell" not in kwargs
        return Process()
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    command = [sys.executable, "-u", str(runner.SCRIPTS["geffs_to_csv.py"])]
    log = tmp_path / "log"
    if returncode:
        with pytest.raises(runner.SubmissionError, match="7"):
            runner.run_command(command, log, heartbeat_seconds=0.005)
    else:
        assert runner.run_command(command, log, heartbeat_seconds=0.005) >= 0
    stdout = capsys.readouterr().out
    assert "inference progress" in stdout
    assert "Progress:" in stdout
    assert log.read_text() == "inference progress\n"

@pytest.mark.parametrize("suffix", ["", ".zarr"])
def test_dotted_dataset_path_resolves_without_truncation(tmp_path, monkeypatch, suffix):
    from tracking_cellmot import io as dataset_io
    store = tmp_path / "private.v2.zarr"
    store.mkdir()
    class StopBeforeLoading(Exception):
        pass
    def open_group(path, **kwargs):
        assert Path(path) == store
        raise StopBeforeLoading()
    monkeypatch.setattr(dataset_io.zarr, "open_group", open_group)
    with pytest.raises(StopBeforeLoading):
        dataset_io.open_dataset(tmp_path / f"private.v2{suffix}", load_image=False)


def test_explicit_prediction_output_stays_in_scratch(tmp_path, monkeypatch):
    import numpy as np
    import predict_unet_transformer as predictor
    target = tmp_path / "scratch" / "predictions"
    target.mkdir(parents=True)
    unrelated = tmp_path / "keep.geff"
    unrelated.mkdir()
    saved = []
    monkeypatch.setattr(predictor, "load_model", lambda *a: (object(), 2, (1, 1, 1)))
    monkeypatch.setattr(predictor, "predict_video", lambda *a, **kw: (np.empty((0, 4)), []))
    monkeypatch.setattr(predictor, "build_graph", lambda *a: object())
    monkeypatch.setattr(predictor, "save_graph", lambda graph, path: saved.append(path))
    predictor.predict(tmp_path, 0, tmp_path / "unused", tmp_path / "mock.pth",
                      predictor.PredictConfig(), debug_video=tmp_path / "hidden.v2.zarr",
                      output_dir=target)
    assert saved == [target / "hidden.v2.geff"]
    assert unrelated.is_dir()


def test_model_shared_with_existing_training_interface():
    # Importing the legacy interface here only checks identity; no training is run.
    import train_unet_transformer as legacy
    from tracking_cellmot import unet_transformer as shared
    assert legacy.UNetNodeTransformer is shared.UNetNodeTransformer
    assert legacy.extract_pos_features is shared.extract_pos_features
    assert legacy._POS_EMBED_DIM == shared._POS_EMBED_DIM


def test_interrupted_subprocess_is_terminated(tmp_path, monkeypatch):
    events = []
    class Output:
        def __iter__(self):
            raise KeyboardInterrupt()
        def close(self):
            events.append("close")
    class Process:
        stdout = Output()
        def poll(self): return None
        def terminate(self): events.append("terminate")
        def wait(self, **kwargs): events.append("wait")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: Process())
    with pytest.raises(KeyboardInterrupt):
        runner.run_command([sys.executable, "-u", str(runner.SCRIPTS["geffs_to_csv.py"])], tmp_path / "log")
    assert events == ["terminate", "wait", "close"]
