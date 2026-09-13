import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import evaluate as evaluate_script
import predict_unet_transformer as prediction_script


@pytest.mark.parametrize(
    ("debug_name", "expected_name"),
    [
        ("44b6_0113de3b.zarr", "44b6_0113de3b.geff"),
        ("sample.v1.zarr", "sample.v1.geff"),
    ],
)
def test_debug_video_output_strips_only_final_zarr_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    debug_name: str,
    expected_name: str,
) -> None:
    saved_paths: list[Path] = []
    debug_video = tmp_path / debug_name

    monkeypatch.setattr(prediction_script, "load_model", lambda *_: (object(), 2, (1, 1, 1)))
    monkeypatch.setattr(
        prediction_script,
        "predict_video",
        lambda *args, **kwargs: (np.empty((0, 4), dtype=np.int16), []),
    )
    monkeypatch.setattr(prediction_script, "build_graph", lambda *args: object())
    monkeypatch.setattr(prediction_script, "save_graph", lambda graph, path: saved_paths.append(path))
    monkeypatch.setattr("dataspec.PREDICTIONS_PATH", tmp_path / "predictions")

    prediction_script.predict(
        data_dir=tmp_path,
        fold=0,
        splits_file=tmp_path / "splits.json",
        weights_path=tmp_path / "weights.pth",
        cfg=prediction_script.PredictConfig(),
        debug_video=debug_video,
    )

    assert saved_paths[0].name == expected_name
    assert saved_paths[0].parent.name == "split_0"


@pytest.mark.parametrize(
    ("cli_override", "checkpoint_config", "has_checkpoint_pool", "expected", "source"),
    [
        (9.0, {"pool_kernel_um": 7.0}, True, 9.0, "CLI override"),
        (None, {"pool_kernel_um": 7.0}, True, 7.0, "checkpoint config"),
        (None, {}, False, 3.0, "backward-compatible default"),
    ],
)
def test_pool_kernel_precedence(
    cli_override: float | None,
    checkpoint_config: dict,
    has_checkpoint_pool: bool,
    expected: float,
    source: str,
) -> None:
    assert prediction_script.resolve_pool_kernel_um(
        cli_override, checkpoint_config, has_checkpoint_pool,
    ) == (expected, source)


def test_pool_kernel_resolution_reads_training_checkpoint_config(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    weights_path = checkpoint_dir / "edge_predictor_best.pth"
    torch.save({}, weights_path)
    (checkpoint_dir / "config.json").write_text(json.dumps({
        "unet_out_channels": 32,
        "unet_layers": [32, 64, 128],
        "downsample": [1, 4, 4],
        "window_size": 2,
        "pool_kernel_um": 5.0,
    }))

    checkpoint_config, has_pool_kernel = prediction_script._load_model_config(weights_path)

    assert prediction_script.resolve_pool_kernel_um(
        None, checkpoint_config, has_pool_kernel,
    ) == (5.0, "checkpoint config")
    assert prediction_script.resolve_pool_kernel_um(
        9.0, checkpoint_config, has_pool_kernel,
    ) == (9.0, "CLI override")


def test_prediction_cli_passes_checkpoint_pool_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        prediction_script,
        "_load_model_config",
        lambda path: ({"pool_kernel_um": 7.0}, True),
    )
    monkeypatch.setattr(prediction_script, "predict", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        ["predict_unet_transformer.py", "--weights", str(tmp_path / "weights.pth")],
    )

    prediction_script.main()

    assert calls[0]["cfg"].pool_kernel_um == 7.0
    assert calls[0]["pool_kernel_source"] == "checkpoint config"


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        (0.0, True), (1.0, True), (0.5, True), (0.52, True),
        (-0.0001, False), (1.0001, False),
        (float("nan"), False), (float("inf"), False), (float("-inf"), False),
    ],
)
def test_validate_unit_threshold_bounds(value: float, valid: bool) -> None:
    if valid:
        assert prediction_script._validate_unit_threshold("--x", value) == value
    else:
        with pytest.raises(ValueError, match="--x"):
            prediction_script._validate_unit_threshold("--x", value)


def test_validate_unit_threshold_rejects_non_numeric_types() -> None:
    with pytest.raises(ValueError):
        prediction_script._validate_unit_threshold("--x", True)
    with pytest.raises(ValueError):
        prediction_script._validate_unit_threshold("--x", "0.5")


def test_edge_threshold_cli_forwards_into_predict_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        prediction_script, "_load_model_config", lambda path: ({}, False),
    )
    monkeypatch.setattr(prediction_script, "predict", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys, "argv",
        [
            "predict_unet_transformer.py",
            "--weights", str(tmp_path / "weights.pth"),
            "--det-threshold", "0.64",
            "--edge-threshold", "0.35",
        ],
    )

    prediction_script.main()

    cfg = calls[0]["cfg"]
    assert cfg.threshold == 0.35
    assert cfg.det_threshold == 0.64


def test_edge_threshold_omitted_preserves_default_predict_config_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting --edge-threshold must not change any existing inference behavior."""
    calls: list[dict] = []
    monkeypatch.setattr(
        prediction_script, "_load_model_config", lambda path: ({}, False),
    )
    monkeypatch.setattr(prediction_script, "predict", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys, "argv",
        ["predict_unet_transformer.py", "--weights", str(tmp_path / "weights.pth")],
    )

    prediction_script.main()

    cfg = calls[0]["cfg"]
    default_cfg = prediction_script.PredictConfig()
    assert cfg.threshold == default_cfg.threshold == 0.5
    assert cfg.det_threshold == 0.99  # unchanged CLI default


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--edge-threshold", "1.5"],
        ["--edge-threshold", "-0.1"],
        ["--det-threshold", "1.5"],
    ],
)
def test_invalid_thresholds_rejected_by_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_args: list[str],
) -> None:
    monkeypatch.setattr(prediction_script, "predict", lambda **kwargs: None)
    monkeypatch.setattr(
        sys, "argv",
        ["predict_unet_transformer.py", "--weights", str(tmp_path / "weights.pth"), *extra_args],
    )

    with pytest.raises(SystemExit) as exc_info:
        prediction_script.main()
    assert exc_info.value.code != 0


_SUMMARY = {
    "n": 1,
    "score": 0.0,
    "edge_jaccard": 0.0,
    "adj_edge_jaccard": 0.0,
    "n_adj": 0,
    "division_jaccard": 0.0,
    "division_tp": 0,
    "division_fp": 0,
    "division_fn": 0,
    "node_recall": 0.0,
}


def _run_evaluate_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_args: list[str],
    rows: list[dict],
    skipped: list[str],
) -> None:
    monkeypatch.setattr(evaluate_script, "evaluate_pairs", lambda *args, **kwargs: (rows, skipped))
    monkeypatch.setattr(evaluate_script, "summarise", lambda rows: _SUMMARY)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--pred-dir", str(tmp_path / "pred"),
            "--gt-dir", str(tmp_path / "gt"),
            *extra_args,
        ],
    )
    evaluate_script.main()


def test_evaluate_strict_fails_with_zero_evaluated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _run_evaluate_main(monkeypatch, tmp_path, ["--strict"], [], [])
    assert exc_info.value.code != 0


def test_evaluate_strict_fails_when_matched_dataset_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _run_evaluate_main(monkeypatch, tmp_path, ["--strict"], [{}], ["broken"])
    assert exc_info.value.code != 0


def test_evaluate_require_all_gt_is_opt_in_for_partial_debug_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "debug.geff").touch()
    (gt_dir / "unselected.geff").touch()
    (pred_dir / "debug.geff").touch()

    _run_evaluate_main(monkeypatch, tmp_path, [], [{}], [])

    with pytest.raises(SystemExit) as exc_info:
        _run_evaluate_main(monkeypatch, tmp_path, ["--require-all-gt"], [{}], [])
    assert exc_info.value.code != 0
