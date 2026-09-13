"""Exercise the real update loop without model training or image I/O."""

import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_unet_transformer as training


@pytest.mark.parametrize("length,max_iters", [(5002, 10000), (3, 5001), (3, None), (5002, 5000)])
def test_update_loop_releases_batches_and_restarts_loader(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    length: int, max_iters: int | None,
) -> None:
    live_images = weakref.WeakValueDictionary()
    reads = 0

    class Windows(Dataset):
        def __len__(self):
            return length

        def __getitem__(self, index):
            nonlocal reads
            reads += 1
            # Test the retention defect beyond the observed checkpoint boundary.
            if reads == 5001:
                assert len(live_images) <= 2, "training retained thousands of image batches"
            imgs = torch.zeros(1, 2, 1, 1, 1)
            live_images[reads] = imgs
            return {
                "imgs": imgs,
                "coords": torch.zeros(1, 2, 1, 3),
                "pos_feats": torch.zeros(1, 2, 1, 32),
                "masks": torch.ones(1, 2, 1, dtype=torch.bool),
                "targets": torch.ones(1, 1, 1, 1),
                "image_shape": torch.tensor([[2, 1, 1, 1]]),
                "voxel_size": torch.ones(1, 3),
                "downsample": torch.ones(1, 3),
            }

    # No parameters, learned forward pass, gradients, or optimizer updates.
    model = SimpleNamespace(
        train=lambda: None,
        encode=lambda imgs: (torch.zeros(1, 2, 1), [None, None]),
        _index_features=lambda *a: None,
        predict_edges=lambda *a: None,
        parameters=lambda: (),
    )
    optimizer = Mock()
    monkeypatch.setattr(training, "compute_detection_loss", lambda *a: torch.tensor(2.0))
    monkeypatch.setattr(training, "compute_batch_loss", lambda *a: torch.tensor(3.0))
    monkeypatch.setattr(training, "build_matched_edge_targets", lambda *a: None)
    monkeypatch.setattr(training, "detect_and_match", lambda *a, **kw: (
        torch.zeros(1, 1, 3), None, torch.ones(1, 1, dtype=torch.bool), [],
    ))
    monkeypatch.setattr(torch.Tensor, "backward", lambda self: None)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a: None)
    monkeypatch.setattr(training, "tqdm", lambda iterable, **kw: iterable)
    progress_print = Mock(wraps=print)
    monkeypatch.setattr(training, "print", progress_print, raising=False)
    checkpoint_iters = (17, 5000, 7500, 10000)
    checkpoints = []

    def checkpoint(iteration, edge, det):
        assert optimizer.step.call_count == iteration
        assert f"checkpoint_start update={iteration} " in progress_print.call_args.args[0]
        assert progress_print.call_args.kwargs == {"flush": True}
        checkpoints.append((iteration, edge, det))

    result = training.train_epoch(
        model, DataLoader(Windows(), batch_size=None, num_workers=0),
        optimizer, torch.device("cpu"), max_iters=max_iters,
        checkpoint_iters=checkpoint_iters, checkpoint_callback=checkpoint,
    )
    expected_updates = max_iters if max_iters is not None else length
    assert optimizer.step.call_count == expected_updates
    assert reads == expected_updates  # Fresh samples/augmentations after every exhaustion.
    assert len(live_images) == 0
    assert result == (3.0, 2.0)
    assert checkpoints == [
        (i, 3.0, 2.0) for i in checkpoint_iters if i <= expected_updates
    ]
    log = capsys.readouterr().out
    lines = [line for line in log.splitlines() if "[progress]" in line]
    milestones = {1, expected_updates, *range(100, expected_updates + 1, 100)}
    milestones.update(i for i in checkpoint_iters if i <= expected_updates)
    completed = [line for line in lines if line.startswith("  [progress] completed_update=")]
    assert [int(line.split("completed_update=")[1].split()[0]) for line in completed] == sorted(milestones)
    for line in lines:
        for field in ("completed_update=", "phase=", "elapsed=", "data=", "forward=", "backward=", "edge=", "det="):
            assert field in line
    for iteration, _, _ in checkpoints:
        start = next(i for i, line in enumerate(lines) if f"checkpoint_start update={iteration} " in line)
        assert f"checkpoint_complete update={iteration} " in lines[start + 1]
        assert f"completed_update={iteration} " in lines[start + 1]
    restarts = [line for line in lines if "restarting_loader" in line]
    assert len(restarts) == (expected_updates - 1) // length
    for line, iteration in zip(restarts, range(length + 1, expected_updates + 1, length)):
        assert f"next_update={iteration} completed_update={iteration - 1} phase=data " in line
    assert len(lines) == 4 * len(milestones) + len(restarts) + 2 * len(checkpoints)
    assert progress_print.call_count == len(lines) + 1  # Final timing summary.
    assert all(call.kwargs == {"flush": True} for call in progress_print.call_args_list[:-1])
    if max_iters == 10000:
        assert len(lines) < 500
    assert "edge=3.000000 det=2.000000" in completed[-1]


@pytest.mark.parametrize("empty", [True, False])
def test_update_loop_does_not_hide_loader_failure(empty: bool, capsys) -> None:
    class BrokenLoader:
        def __iter__(self):
            if not empty:
                raise RuntimeError("worker failed")
            return iter(())

    optimizer = Mock()
    expected = ValueError if empty else RuntimeError
    message = "no batches" if empty else "worker failed"
    with pytest.raises(expected, match=message):
        training.train_epoch(Mock(), BrokenLoader(), optimizer, torch.device("cpu"), max_iters=10000)
    optimizer.step.assert_not_called()
    log = capsys.readouterr().out
    assert "checkpoint_start" not in log
    assert "phase=complete" not in log
    if empty:
        assert "restarting_loader next_update=1 completed_update=0 phase=data" in log
