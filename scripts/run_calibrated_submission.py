#!/usr/bin/env python
"""Fail-closed, inference-only runner for the reviewed competition calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiments/configs/calibrated_submission_t070_e035.json"
CHECKPOINT_SHA256 = "d6a0221948a6335f581ec12b509bc529672b99f0dfcd6c393be3f1a01a9c7d96"
CHECKPOINT_CONFIG_SHA256 = "b9c71baa3e2d5523b420e419eae8c56f5598917f36bbfecbe9cc2703598b8a6c"
COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
SCRIPTS = {name: REPO_ROOT / "scripts" / name for name in (
    "predict_unet_transformer.py", "geffs_to_csv.py",
)}


class SubmissionError(RuntimeError):
    """An input, subprocess, or output failed validation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_datasets(data_dir: Path) -> dict[str, Path]:
    """Recursively discover Zarr stores, without walking their internal chunks."""
    if not data_dir.is_dir():
        raise SubmissionError(f"data directory does not exist: {data_dir}")
    found = {}
    def walk_error(error):
        raise error
    for root, dirs, files in os.walk(data_dir, onerror=walk_error):
        for name in sorted(dirs + files):
            if name.endswith(".zarr"):
                path = Path(root) / name
                dataset = name.removesuffix(".zarr")
                if not dataset.strip() or dataset in found:
                    raise SubmissionError(f"empty or duplicate dataset name: {dataset!r}")
                if path.is_symlink() or not path.is_dir():
                    raise SubmissionError(f"dataset must be a real Zarr directory: {path}")
                found[dataset] = path
        dirs[:] = sorted(d for d in dirs if not d.endswith(".zarr"))
    if not found:
        raise SubmissionError("no test .zarr datasets discovered")
    return dict(sorted(found.items()))


def validate_coverage(directory: Path, expected: list[str]) -> None:
    actual = {p.stem for p in directory.glob("*.geff")}
    if actual != set(expected):
        raise SubmissionError(f"prediction coverage mismatch: missing={sorted(set(expected) - actual)}, "
                              f"unexpected={sorted(actual - set(expected))}")


def validate_csv(path: Path, datasets: list[str]) -> dict:
    counts = {name: {"rows": 0, "nodes": 0, "edges": 0} for name in datasets}
    ids = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise SubmissionError(f"unexpected CSV columns: {reader.fieldnames}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise SubmissionError(f"malformed CSV row {line}")
            identifier = row["id"].strip()
            if not identifier or identifier in ids:
                raise SubmissionError(f"empty or duplicate id at row {line}")
            ids.add(identifier)
            name, kind = row["dataset"], row["row_type"]
            if name not in counts:
                raise SubmissionError(f"unexpected dataset at row {line}: {name!r}")
            if kind not in ("node", "edge"):
                raise SubmissionError(f"invalid row type at row {line}: {kind!r}")
            required = ("source_id", "target_id") if kind == "edge" else ("node_id", "t", "z", "y", "x")
            for key in required:
                try:
                    value = int(row[key])
                except ValueError as exc:
                    raise SubmissionError(f"missing or invalid {key} at row {line}") from exc
                if value < 0:
                    raise SubmissionError(f"negative {key} at row {line}")
            counts[name]["rows"] += 1
            counts[name]["nodes" if kind == "node" else "edges"] += 1
    if any(count["nodes"] == 0 for count in counts.values()):
        raise SubmissionError("every discovered dataset must have a positive node count")
    return {"datasets": counts, **{key: sum(c[key] for c in counts.values())
                                  for key in ("rows", "nodes", "edges")}}


def run_command(command: list[str], log_path: Path, heartbeat_seconds: float = 30) -> float:
    """Only the two fixed inference/conversion entry points may be launched."""
    if (len(command) < 3 or command[:2] != [sys.executable, "-u"]
            or command[2] not in {str(p) for p in SCRIPTS.values()}):
        raise SubmissionError("command is not an approved inference/conversion entry point")
    started = time.monotonic()
    print(f"Running: {subprocess.list2cmdline(command)}", flush=True)
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(heartbeat_seconds):
            print(f"Progress: {Path(command[2]).name} running ({time.monotonic() - started:.0f}s)", flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        thread.start()
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            if code:
                raise SubmissionError(f"subprocess failed ({code}); see {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
            stop.set()
            thread.join()
    return time.monotonic() - started


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def run_submission(data_dir: Path, checkpoint: Path, output_dir: Path) -> Path:
    started = time.monotonic()
    # Resolve the directory, but never follow an existing final-file symlink.
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / "submission.csv"
    provenance = output_dir / "submission_provenance.json"
    lock = output_dir / ".calibrated-submission.lock"
    # An exclusive lock prevents one run deleting another run's successful output.
    with lock.open("x"):
        pass
    try:
        final.unlink(missing_ok=True)
        provenance.unlink(missing_ok=True)
        data_dir, checkpoint = data_dir.resolve(), checkpoint.resolve()
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        if (config["checkpoint_sha256"] != CHECKPOINT_SHA256
                or config["checkpoint_config_sha256"] != CHECKPOINT_CONFIG_SHA256
                or config["det_threshold"] != 0.70 or config["edge_threshold"] != 0.35
                or config["split"] != 0 or config["tracking"] != "greedy"
                or config["training_performed"] is not False):
            raise SubmissionError("reviewed config differs from locked calibration")
        if not checkpoint.is_file():
            raise SubmissionError(f"checkpoint does not exist: {checkpoint}")
        print("Validating checkpoint SHA256...", flush=True)
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != CHECKPOINT_SHA256:
            raise SubmissionError("checkpoint SHA256 does not match reviewed checkpoint")
        checkpoint_config = checkpoint.parent / "config.json"
        if checkpoint_config.is_symlink() or not checkpoint_config.is_file():
            raise SubmissionError("checkpoint requires adjacent config.json as a regular file")
        print("Validating checkpoint config.json SHA256...", flush=True)
        checkpoint_config_hash = sha256(checkpoint_config)
        if checkpoint_config_hash != CHECKPOINT_CONFIG_SHA256:
            raise SubmissionError("checkpoint config.json SHA256 does not match reviewed checkpoint config")
        if not isinstance(json.loads(checkpoint_config.read_text(encoding="utf-8")), dict):
            raise SubmissionError("checkpoint config.json must contain an object")
        datasets = discover_datasets(data_dir)
        print(f"Discovered {len(datasets)} test datasets: {', '.join(datasets)}", flush=True)
        commit = git_commit()
        commands, timings = [], {}
        # TemporaryDirectory cleans only this invocation's uniquely named child.
        with tempfile.TemporaryDirectory(prefix=".calibrated-submission-", dir=output_dir) as scratch_name:
            scratch = Path(scratch_name)
            predictions = scratch / "predictions"
            predictions.mkdir()
            # Group nested datasets by parent to load the model only once per directory.
            groups = {}
            for name, path in datasets.items():
                groups.setdefault(path.parent, []).append(name)
            for index, (parent, names) in enumerate(groups.items()):
                split = scratch / f"split-{index}.json"
                split.write_text(json.dumps([{"split": 0, "train": [], "test": names}]), encoding="utf-8")
                group_output = scratch / f"group-{index}"
                command = [sys.executable, "-u", str(SCRIPTS["predict_unet_transformer.py"]),
                           "--data-dir", str(parent), "--weights", str(checkpoint),
                           "--splits", str(split), "--split", "0", "--det-threshold", "0.70",
                           "--edge-threshold", "0.35", "--tracking", "greedy",
                           "--output-dir", str(group_output)]
                commands.append(command)
                timings[f"prediction_{index}"] = run_command(command, output_dir / f"prediction-{index}.log")
                validate_coverage(group_output, names)
                for name in names:
                    shutil.move(str(group_output / f"{name}.geff"), str(predictions / f"{name}.geff"))
            validate_coverage(predictions, list(datasets))
            candidate = scratch / "submission.csv"
            command = [sys.executable, "-u", str(SCRIPTS["geffs_to_csv.py"]),
                       "--in-dir", str(predictions), "--csv", str(candidate)]
            commands.append(command)
            timings["conversion"] = run_command(command, output_dir / "conversion.log")
            counts = validate_csv(candidate, list(datasets))
            record = {"status": "success", "training_performed": False,
                      "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                      "checkpoint_config_sha256": checkpoint_config_hash,
                      "reviewed_config": config, "det_threshold": 0.70, "edge_threshold": 0.35,
                      "tracking": "greedy", "split": 0, "git_commit": commit,
                      "discovered_datasets": list(datasets),
                      "dataset_paths": {k: str(v) for k, v in datasets.items()},
                      "counts": counts, "elapsed_seconds": timings,
                      "total_elapsed_seconds": time.monotonic() - started,
                      "commands": commands, "command_lines": [subprocess.list2cmdline(c) for c in commands],
                      "output_sha256": sha256(candidate), "output": str(final)}
            record_path = scratch / "provenance.json"
            record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            # Publish the CSV last: it is never visible while inference/validation runs.
            record_path.replace(provenance)
            candidate.replace(final)
    except BaseException:
        final.unlink(missing_ok=True)
        provenance.unlink(missing_ok=True)
        raise
    finally:
        try:
            lock.unlink()
        except OSError:
            final.unlink(missing_ok=True)
            provenance.unlink(missing_ok=True)
            raise
    print(f"Success: {final} ({counts['rows']} rows)", flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Competition test directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_submission(args.data_dir, args.checkpoint, args.output_dir)
    except (SubmissionError, OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Submission failed: {exc}\n")


if __name__ == "__main__":
    main()
