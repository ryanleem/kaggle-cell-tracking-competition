"""Synthetic spacing tests; no training data, image processing, or inference."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tracksdata as td
import zarr

import audit_detection_spacing as audit
from tracking_cellmot.io import Dataset


def dataset(rows, scale=(1, 1, 1), frames=4):
    graph = td.graph.IndexedRXGraph()
    for axis in ("z", "y", "x"):
        graph.add_node_attr_key(axis, pl.Float64, 0.0)
    for t, z, y, x in rows:
        graph.add_node(attrs={"t": t, "z": float(z), "y": float(y), "x": float(x)})
    return Dataset(Path("synthetic"), None, graph, scale, image_shape=(frames, 16, 16, 16))


def test_same_frame_singletons_and_percentiles():
    ds = dataset([(0, 0, 0, 0), (0, 0, 0, 2), (0, 0, 0, 8), (1, 0, 0, 0)])
    result, distances = audit.audit_dataset(ds, (1, 1, 1), [5])
    np.testing.assert_array_equal(distances, [2, 2, 6])
    assert result["node_count"] == 4
    assert result["frame_count"] == 4
    assert result["annotated_frame_count"] == 2
    assert result["singleton_frame_count"] == 1
    assert result["nodes_with_neighbor"] == 3
    assert result["nearest_neighbor_um"] == dict(zip(
        ["minimum", "p1", "p5", "p10", "p25", "p50", "p75"], [2, 2, 2, 2, 2, 2, 4]))
    assert result["nearest_neighbor_collision_risk"][0]["count"] == 2
    assert result["nearest_neighbor_collision_risk"][0]["fraction"] == 0.5
    assert result["any_neighbor_collision_risk"][0]["count"] == 2
    assert result["any_neighbor_collision_risk"][0]["fraction"] == 0.5
    assert result["any_neighbor_collision_risk"][0]["pair_count"] == 1


def test_all_singleton_frames():
    result, nn = audit.audit_dataset(dataset([(0, 0, 0, 0), (1, 0, 0, 0)]), (1, 1, 1), [100])
    assert len(nn) == 0
    assert all(v is None for v in result["nearest_neighbor_um"].values())
    assert result["nearest_neighbor_collision_risk"][0]["count"] == 0
    assert result["any_neighbor_collision_risk"][0]["count"] == 0
    assert result["any_neighbor_collision_risk"][0]["pair_count"] == 0


def test_physical_scaling_is_independent_of_downsample():
    ds = dataset([(0, 0, 0, 0), (0, 1, 2, 2)], scale=(2, 3, 4))
    for downsample in [(1, 1, 1), (1, 4, 4)]:
        result, nn = audit.audit_dataset(ds, downsample, [7])
        np.testing.assert_allclose(nn, np.sqrt(104))
        assert result["effective_voxel_size_um"] == (np.array(ds.scale) * downsample).tolist()


def test_anisotropic_axis_window_and_boundary():
    ds = dataset([(0, 0, 0, 0), (0, 1, 1, 0), (1, 0, 0, 0), (1, 0, 0, 1)], scale=(2, 1, 4))
    result, _ = audit.audit_dataset(ds, (1, 1, 1), [4])
    risk = result["nearest_neighbor_collision_risk"][0]
    assert risk["voxel_kernel"] == [3, 5, 1]
    assert risk["physical_half_width_um"] == [2, 2, 0]
    # A diagonal within the box qualifies despite distance > 2; z half-width
    # cannot be used as a spherical radius or as the x half-width.
    assert risk["count"] == 2
    assert risk["fraction"] == 0.5
    any_risk = result["any_neighbor_collision_risk"][0]
    assert any_risk["count"] == 2
    assert any_risk["pair_count"] == 1


def test_nearest_neighbor_selection_uses_physical_distance():
    ds = dataset([(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 3)], scale=(10, 1, 1))
    _, nn = audit.audit_dataset(ds, (1, 1, 1), [5])
    np.testing.assert_array_equal(nn, [3, 3, 10])


def test_risk_checks_euclidean_nearest_not_any_neighbor_in_box():
    # A=(0,0,0), B=(2,0,0), C=(0,0,1) physical. For A the x neighbor (C) is
    # nearest (distance 1) but outside the zero-width x window; the farther z
    # neighbor (B, distance 2) is inside the box. nearest_neighbor_collision_risk
    # only checks each node's own Euclidean-nearest neighbor, so it only counts
    # B (whose nearest neighbor A happens to be the in-box one) and misses that
    # A also has an in-box neighbor (B) that just isn't its Euclidean-nearest.
    # any_neighbor_collision_risk checks every same-frame neighbor and must
    # therefore identify both endpoints (A and B) of the qualifying A-B pair.
    ds = dataset([(0, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, .25)], scale=(1, 1, 4))
    result, _ = audit.audit_dataset(ds, (1, 1, 1), [4])
    assert result["nearest_neighbor_collision_risk"][0]["count"] == 1
    any_risk = result["any_neighbor_collision_risk"][0]
    assert any_risk["count"] == 2
    assert any_risk["pair_count"] == 1


@pytest.mark.parametrize("um,expected", [(0.1, 1), (1.5, 3), (2.5, 3), (3.5, 5), (4.5, 5)])
def test_production_rounding(um, expected):
    result, _ = audit.audit_dataset(dataset([(0, 0, 0, 0)]), (1, 1, 1), [um])
    assert result["nearest_neighbor_collision_risk"][0]["voxel_kernel"] == [expected] * 3


def test_equivalent_kernels():
    result, _ = audit.audit_dataset(dataset([(0, 0, 0, 0)], scale=(1.625, .40625, .40625)),
                                    (1, 4, 4), [6., 7.])
    assert result["equivalent_kernels"] == [{"requested_um": [6., 7.], "voxel_kernel": [5, 5, 5]}]
    assert result["nearest_neighbor_collision_risk"][0]["physical_half_width_um"] == [3.25] * 3


def test_duplicate_coordinates_and_order_independence():
    rows = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0)]
    first, _ = audit.audit_dataset(dataset(rows), (1, 1, 1), [1, 5])
    second, _ = audit.audit_dataset(dataset(rows[::-1]), (1, 1, 1), [1, 5])
    assert first == second
    assert [r["count"] for r in first["nearest_neighbor_collision_risk"]] == [2, 4]
    # um=5's half-window (2,2,2) is wide enough that every one of the 4 nodes
    # has some in-box neighbor either way, so any-neighbor node counts match
    # the nearest-neighbor counts here; but any-neighbor also finds the extra
    # P2-P3 pair that no node's *nearest* neighbor captures, so its unique
    # pair count (6, i.e. all C(4,2) pairs) exceeds the 1 duplicate-only pair
    # nearest-neighbor risk would imply.
    assert [r["count"] for r in first["any_neighbor_collision_risk"]] == [2, 4]
    assert [r["pair_count"] for r in first["any_neighbor_collision_risk"]] == [1, 6]


@pytest.mark.parametrize("scale", [(0, 1, 1), (-1, 1, 1), (np.nan, 1, 1), (1, np.inf, 1), (1, 1)])
def test_invalid_scales(scale):
    with pytest.raises(audit.AuditError, match="scale"):
        audit.audit_dataset(dataset([(0, 0, 0, 0)], scale), (1, 1, 1), [5])


@pytest.mark.parametrize("downsample", [(0, 1, 1), (1, 1), (1, 1.5, 1)])
def test_invalid_downsample(downsample):
    with pytest.raises(audit.AuditError, match="downsample"):
        audit.audit_dataset(dataset([(0, 0, 0, 0)]), downsample, [5])


@pytest.mark.parametrize("requested", [[], [0], [-1], [np.nan], [np.inf]])
def test_invalid_pooling(requested):
    with pytest.raises(audit.AuditError, match="pool-kernel"):
        audit.audit_dataset(dataset([(0, 0, 0, 0)]), (1, 1, 1), requested)


def test_empty_missing_nonfinite_and_bad_frame():
    for ds, message in [(dataset([]), "empty"),
                        (dataset([(0, np.nan, 0, 0)]), "nonfinite"),
                        (dataset([(5, 0, 0, 0)]), "frame indices")]:
        with pytest.raises(audit.AuditError, match=message):
            audit.audit_dataset(ds, (1, 1, 1), [5])
    ds.tracks = None
    with pytest.raises(audit.AuditError, match="missing annotations"):
        audit.audit_dataset(ds, (1, 1, 1), [5])


@pytest.mark.parametrize("folds", [{}, [], [{"split": 0, "train": [], "test": []}],
    [{"split": 0, "train": [], "test": ["a", "a"]}],
    [{"split": 0, "train": ["a"], "test": ["a"]}],
    [{"split": 0, "train": [], "test": ["../a"]}],
    [{"split": 0, "train": [], "test": "a"}],
    [{"split": 0, "test": ["a"]}],
    [{"split": 0, "train": [], "test": ["a"]}] * 2])
def test_malformed_splits(tmp_path, folds):
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(folds))
    with pytest.raises(audit.AuditError):
        audit.audit(tmp_path, splits, 0, (1, 1, 1), [5])


def test_real_loader_cli_aggregation_and_determinism(tmp_path, capsys):
    for name, rows in [("a", [(0, 0, 0, 0), (0, 0, 0, 2)]), ("b", [(1, 0, 0, 0)])]:
        ds = dataset(rows)
        group = zarr.open_group(str(tmp_path / f"{name}.zarr"), mode="w")
        group.create_array("0", shape=(4, 16, 16, 16), dtype="uint8")
        group.attrs["multiscales"] = [{"datasets": [{"coordinateTransformations": [
            {"type": "scale", "scale": [1, 1, 1, 1]}]}]}]
        ds.tracks.to_geff(str(tmp_path / f"{name}.geff"))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps([{"split": 0, "train": ["not_loaded"], "test": ["b", "a"]}]))
    out = tmp_path / "report.json"
    args = ["--data-dir", str(tmp_path), "--splits", str(splits), "--split", "0",
            "--downsample", "1,1,1", "--pool-kernel-um", "5", "6", "--json-out", str(out)]
    audit.main(args)
    before = out.read_bytes()
    audit.main(args)
    assert out.read_bytes() == before
    result = json.loads(before)
    assert result["dataset_count"] == 2
    assert result["aggregate"]["node_count"] == 3
    assert result["aggregate"]["frame_count"] == 8
    assert result["aggregate"]["nearest_neighbor_collision_risk"][0]["fraction"] == 2 / 3
    assert result["aggregate"]["nearest_neighbor_um"]["p50"] == 2
    # Aggregation across datasets a (one qualifying pair) and b (singleton, no
    # pairs) must sum node counts and unique-pair counts across both kernels.
    assert [r["count"] for r in result["aggregate"]["any_neighbor_collision_risk"]] == [2, 2]
    assert [r["pair_count"] for r in result["aggregate"]["any_neighbor_collision_risk"]] == [1, 1]
    assert result["aggregate"]["any_neighbor_collision_risk"][0]["fraction"] == 2 / 3
    assert "not guaranteed merging" in capsys.readouterr().out


def test_quantization_boundary_truncates_adjacent_buckets():
    # x=7 and x=8 straddle a downsample-4 boundary: truncation toward zero
    # (torch.long() semantics) puts them in adjacent, not the same, buckets.
    raw = np.array([[0, 0, 7], [0, 0, 8]], dtype=float)
    spatial_shape = audit.downsampled_spatial_shape((16, 16, 16), (1, 1, 4))
    quantized = audit.quantize_target_coords(raw, (1, 1, 4), spatial_shape)
    np.testing.assert_array_equal(quantized, [[0, 0, 1], [0, 0, 2]])


def test_quantization_clips_to_downsampled_bounds():
    # x=100 is far outside the raw image, but the audit does not validate z,y,x
    # bounds (only t); this mirrors compute_detection_loss's own defensive
    # clamp(0, spatial_axis - 1) rather than relying on well-formed input.
    raw = np.array([[0, 0, 100]], dtype=float)
    spatial_shape = audit.downsampled_spatial_shape((16, 16, 16), (1, 1, 4))
    assert spatial_shape == (16, 16, 4)
    quantized = audit.quantize_target_coords(raw, (1, 1, 4), spatial_shape)
    assert quantized.tolist() == [[0, 0, 3]]


def test_quantize_matches_torch_float32_semantics():
    import torch

    raw = np.array([[0, 0, 0], [0, 0, 7], [16777217, 0, 0], [3, 5, 11]], dtype=float)
    downsample = (1, 3, 4)
    huge_spatial_shape = (10**8, 10**8, 10**8)  # large enough that clamp never triggers
    torch_quantized = (torch.from_numpy(raw.astype(np.float32))
                        / torch.tensor(downsample, dtype=torch.float32)).long().numpy()
    quantized = audit.quantize_target_coords(raw, downsample, huge_spatial_shape)
    np.testing.assert_array_equal(quantized, torch_quantized)
    # float32 precision is load-bearing here: 16777217 == 2**24 + 1 is not
    # exactly representable in float32 and rounds down to 2**24 before
    # truncation; a naive float64 division would not reproduce this.
    assert quantized[2, 0] == 16_777_216


def test_quantized_collision_collapses_nodes_continuous_risk_misses():
    # x=0 and x=1 are 1 unit apart physically; with downsample=4 they truncate
    # to the same target voxel (index 0) and so always collide on the
    # quantized grid, regardless of kernel. any_neighbor_collision_risk, which
    # stays in continuous physical space, sees no collision at this kernel.
    ds = dataset([(0, 0, 0, 0), (0, 0, 0, 1)])
    result, _ = audit.audit_dataset(ds, (1, 1, 4), [1.0])
    assert result["any_neighbor_collision_risk"][0]["count"] == 0
    assert result["any_neighbor_collision_risk"][0]["pair_count"] == 0
    assert result["quantized_target_collision_risk"][0]["count"] == 2
    assert result["quantized_target_collision_risk"][0]["pair_count"] == 1
    assert result["unique_target_voxels"] == 1
    assert result["duplicate_target_voxel_count"] == 1
    assert result["collapsed_node_count"] == 1


def test_quantized_aggregation_across_datasets(tmp_path):
    # Dataset a: x=0 and x=1 collapse into one target voxel (downsample 4);
    # x=9 lands in a separate voxel. Dataset b is a lone singleton frame and
    # contributes no pairs or collapses. The aggregate must sum node counts,
    # pair counts, unique-voxel counts, and collapse counts across both.
    for name, rows in [("a", [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 9)]),
                        ("b", [(1, 0, 0, 0)])]:
        ds = dataset(rows)
        group = zarr.open_group(str(tmp_path / f"{name}.zarr"), mode="w")
        group.create_array("0", shape=(4, 16, 16, 16), dtype="uint8")
        group.attrs["multiscales"] = [{"datasets": [{"coordinateTransformations": [
            {"type": "scale", "scale": [1, 1, 1, 1]}]}]}]
        ds.tracks.to_geff(str(tmp_path / f"{name}.geff"))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps([{"split": 0, "train": [], "test": ["a", "b"]}]))
    report = audit.audit(tmp_path, splits, 0, (1, 1, 4), [1.0])
    assert report["datasets"]["a"]["duplicate_target_voxel_count"] == 1
    assert report["datasets"]["a"]["collapsed_node_count"] == 1
    assert report["datasets"]["b"]["duplicate_target_voxel_count"] == 0
    assert report["datasets"]["b"]["collapsed_node_count"] == 0
    assert report["aggregate"]["unique_target_voxels"] == 3
    assert report["aggregate"]["duplicate_target_voxel_count"] == 1
    assert report["aggregate"]["collapsed_node_count"] == 1
    assert report["aggregate"]["quantized_target_collision_risk"][0]["count"] == 2
    assert report["aggregate"]["quantized_target_collision_risk"][0]["pair_count"] == 1


def test_atomic_failure_preserves_previous_output(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    out.write_text("original")
    def fail(*args):
        raise OSError("replace failed")
    monkeypatch.setattr(audit.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        audit.write_json_atomic(out, {"value": 1})
    assert out.read_text() == "original"
    assert list(tmp_path.iterdir()) == [out]


def test_missing_annotation_file_is_reported(tmp_path):
    group = zarr.open_group(str(tmp_path / "a.zarr"), mode="w")
    group.create_array("0", shape=(1, 1, 1, 1), dtype="uint8")
    splits = tmp_path / "splits.json"
    splits.write_text('[{"split": 0, "train": [], "test": ["a"]}]')
    with pytest.raises(audit.AuditError, match="dataset a: Tracks file not found"):
        audit.audit(tmp_path, splits, 0, (1, 1, 1), [5])


def test_cli_cannot_overwrite_inputs(tmp_path, capsys):
    splits = tmp_path / "splits.json"
    splits.write_text("original")
    for output in [splits, tmp_path / "a.geff" / "report.json"]:
        with pytest.raises(SystemExit):
            audit.main(["--splits", str(splits), "--pool-kernel-um", "5", "--json-out", str(output)])
        assert "must not overwrite" in capsys.readouterr().err
    assert splits.read_text() == "original"
