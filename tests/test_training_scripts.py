"""End-to-end integration test: train → predict → evaluate for UNet transformer."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import tracksdata as td

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import train_unet_transformer as training_script
from train_unet_transformer import train, DEFAULT_AUGMENTATIONS
from predict_unet_transformer import predict_video, build_graph, load_model, PredictConfig
from tracking_cellmot.io import open_dataset
from tracking_cellmot.metrics import evaluate, node_recall


# 5-frame clip extracted from the full dataset: frames 26–30, division at t=2.
_FIXTURE_DIR = Path(__file__).parent / "data" / "division_clip"
DS_PATH = _FIXTURE_DIR / "division_clip"

# Architecture used in the test (small for speed).
_TEST_CONFIG = {
    "unet_out_channels": 16,
    "unet_layers": [16, 32],
    "unet_stem": True,
    "downsample": [4, 4, 4],
    "pool_kernel_um": 5.0,
    "predict": {
        "threshold": 0.5,
        "det_tta": False,
        "max_parents_per_node": 1,
        "max_children_per_node": 2,
        "use_ilp": False,
        "edge_activation": "softmax",
    },
}


def test_baseline_pilot_10000_config_matches_pilot_5000_except_schedule() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "experiments" / "configs"
    config_path = config_dir / "baseline_pilot_10000.json"
    assert config_path.is_file()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = json.loads((config_dir / "baseline_pilot_5000.json").read_text(encoding="utf-8"))
    assert config["max_iters"] == 10000
    assert config["checkpoint_iters"] == [5000, 7500, 10000]
    assert config["method"] == "baseline_pilot_10000"

    changed_keys = {"method", "max_iters", "checkpoint_iters"}
    assert {key: value for key, value in config.items() if key not in changed_keys} == {
        key: value for key, value in source.items() if key not in changed_keys
    }


def test_checkpoint_iters_parse_as_one_based_update_numbers() -> None:
    assert training_script.parse_checkpoint_iters("500, 1000,2000, 5000") == [500, 1000, 2000, 5000]


@pytest.mark.parametrize(
    "value",
    [
        "500,abc",
        "",
        "500,500",
        "0,500",
        "1000,500",
        "500,1001",
    ],
)
def test_checkpoint_iters_reject_invalid_schedules(value: str) -> None:
    if value == "500,1001":
        with pytest.raises(ValueError, match="exceed max_iters"):
            training_script.validate_checkpoint_iters(
                training_script.parse_checkpoint_iters(value), max_iters=1000,
            )
    else:
        with pytest.raises(ValueError):
            training_script.parse_checkpoint_iters(value)


def test_periodic_checkpoints_have_exact_names_normalized_keys_and_history(tmp_path: Path) -> None:
    model = torch.nn.Module()
    model.unet = torch.nn.Module()
    model.unet.module = torch.nn.Linear(1, 1)
    history_path = tmp_path / "training_history.jsonl"

    with history_path.open("w", encoding="utf-8") as history_file:
        for checkpoint_count, iteration in enumerate([500, 1000, 2000, 5000], start=1):
            training_script._save_periodic_checkpoint(
                model, tmp_path, history_file, iteration,
                running_avg_edge_loss=0.1,
                running_avg_detection_loss=0.2,
                elapsed_training_seconds=float(iteration),
            )
            assert history_path.read_text(encoding="utf-8").count("\n") == checkpoint_count

    expected_names = [
        "edge_predictor_iter_000500.pth",
        "edge_predictor_iter_001000.pth",
        "edge_predictor_iter_002000.pth",
        "edge_predictor_iter_005000.pth",
    ]
    assert [path.name for path in sorted(tmp_path.glob("edge_predictor_iter_*.pth"))] == expected_names
    for name in expected_names:
        state = torch.load(tmp_path / name, weights_only=True)
        assert set(state) == {"unet.weight", "unet.bias"}
        assert not any(key.startswith("unet.module.") for key in state)

    records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert set(records[0]) == {
        "iteration",
        "running_avg_edge_loss",
        "running_avg_detection_loss",
        "checkpoint_filename",
        "elapsed_training_seconds",
    }
    assert [record["iteration"] for record in records] == [500, 1000, 2000, 5000]
    assert [record["checkpoint_filename"] for record in records] == expected_names


def test_periodic_checkpoint_save_is_atomic_and_failure_leaves_no_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1)
    history_path = tmp_path / "training_history.jsonl"
    real_save = training_script.torch.save
    saved_paths: list[Path] = []

    def recording_save(state: object, path: Path) -> None:
        saved_paths.append(path)
        real_save(state, path)

    monkeypatch.setattr(training_script.torch, "save", recording_save)
    with history_path.open("w+", encoding="utf-8") as history_file:
        training_script._save_periodic_checkpoint(
            model, tmp_path, history_file, 500, 0.1, 0.2, 1.0,
        )
    final_path = tmp_path / "edge_predictor_iter_000500.pth"
    assert final_path.is_file()
    assert saved_paths[0] != final_path
    assert saved_paths[0].parent == tmp_path
    assert not list(tmp_path.glob("*.tmp"))
    assert len(history_path.read_text(encoding="utf-8").splitlines()) == 1

    failure_dir = tmp_path / "failure"
    failure_dir.mkdir()
    failure_history = failure_dir / "training_history.jsonl"

    def failing_save(state: object, path: Path) -> None:
        raise RuntimeError("save failed")

    monkeypatch.setattr(training_script.torch, "save", failing_save)
    with failure_history.open("w+", encoding="utf-8") as history_file:
        with pytest.raises(RuntimeError, match="save failed"):
            training_script._save_periodic_checkpoint(
                model, failure_dir, history_file, 1000, 0.1, 0.2, 2.0,
            )
    assert not (failure_dir / "edge_predictor_iter_001000.pth").exists()
    assert failure_history.read_text(encoding="utf-8") == ""
    assert not list(failure_dir.glob("*.tmp"))


def test_train_rejects_periodic_checkpoints_for_multiple_epochs() -> None:
    with pytest.raises(ValueError, match="n_epochs == 1"):
        train(
            data_dir=Path("missing"),
            fold=0,
            splits_file=Path("missing-splits.json"),
            n_epochs=2,
            max_iters=10,
            checkpoint_iters=[5],
        )


def _seed_everything(seed: int = 42) -> None:
    # Warm up CUDA to ensure consistent kernel selection across parametrized runs.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _ = torch.zeros(1, device="cuda")  # trigger lazy CUDA init
        torch.cuda.synchronize()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.mark.slow
@pytest.mark.parametrize("window_size", [2])
def test_unet_transformer_overfit_and_evaluate(
    tmp_path: Path, window_size: int, division_clip_fixture: None
) -> None:
    """Train UNet transformer on a single video, predict, and assert jaccard == 1."""
    _seed_everything()

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()

    model = train(
        data_dir=_FIXTURE_DIR,
        fold=0,
        splits_file=_FIXTURE_DIR / "dataset_splits.json",  # ignored when debug_video is set
        method="unet_transformer",
        n_epochs=40,
        lr=1e-3,
        batch_size=32,
        num_workers=8,
        max_iters=400,
        unet_out_channels=_TEST_CONFIG["unet_out_channels"],
        unet_layers=_TEST_CONFIG["unet_layers"],
        downsample=tuple(_TEST_CONFIG["downsample"]),
        det_loss_weight=1e0,
        det_neg_weight=5e-2,
        debug_video=DS_PATH,
        seed=42,
        window_size=window_size,
        augmentations=None,
        pool_kernel_um=_TEST_CONFIG["pool_kernel_um"],
    )

    # Save weights + config so load_model can reconstruct the architecture.
    save_path = weights_dir / "edge_predictor_best.pth"
    torch.save(model.state_dict(), save_path)
    test_config = {**_TEST_CONFIG, "window_size": window_size}
    (weights_dir / "config.json").write_text(json.dumps(test_config))

    # Reload from disk (tests the save/load round-trip).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model, window_size, downsample = load_model(save_path, device)

    # Predict on the same video.
    cfg = PredictConfig(**_TEST_CONFIG["predict"], pool_kernel_um=_TEST_CONFIG["pool_kernel_um"])
    coords, edges = predict_video(loaded_model, DS_PATH, device, cfg, window_size=window_size, downsample=downsample)
    pred_graph = build_graph(coords, edges)

    # Load GT graph for evaluation.
    ds = open_dataset(DS_PATH, require_tracks=True)
    gt_graph = ds.tracks

    er = evaluate(pred_graph, gt_graph, scale=ds.scale)
    edge_denom = er.edge_tp + er.edge_fp + er.edge_fn
    jaccard = er.edge_tp / edge_denom if edge_denom > 0 else float("nan")
    div_denom = er.division_tp + er.division_fp + er.division_fn
    div_jaccard = er.division_tp / div_denom if div_denom > 0 else float("nan")
    # node_recall requires graph.match() to have been called, which evaluate()
    # skips for empty predicted graphs — guard against that here.
    if pred_graph.num_nodes() > 0 and pred_graph.num_edges() > 0:
        recall = node_recall(pred_graph, gt_graph)
    else:
        recall = 0.0
    n_pred_nodes = pred_graph.num_nodes()
    n_pred_edges = pred_graph.num_edges()
    n_gt_nodes = gt_graph.num_nodes()
    n_gt_edges = gt_graph.num_edges()
    print(
        f"Nodes: {n_pred_nodes} pred / {n_gt_nodes} GT | "
        f"Edges: {n_pred_edges} pred / {n_gt_edges} GT | "
        f"Jaccard: {jaccard:.4f} | Node recall: {recall:.4f} | Division Jaccard: {div_jaccard:.4f}"
    )
    assert jaccard >= 0.95, f"Expected jaccard >= 0.95 after overfitting, got {jaccard:.4f}"
    assert recall == 1.0, f"Expected node_recall=1.0, got {recall:.4f}"
    assert div_jaccard == 1.0 or np.isnan(div_jaccard), f"Expected division_jaccard=1.0 (or NaN if no divisions), got {div_jaccard:.4f}"
