#!/usr/bin/env python
"""Read-only GT spacing audit of a fold's test datasets; no image pixels are read.

Example: uv run scripts/audit_detection_spacing.py --data-dir data/train
    --splits experiments/splits/baseline_fold0_seed0.json --split 0
    --downsample 1,4,4 --pool-kernel-um 5 6 7 --json-out spacing.json

Distances use original GEFF z,y,x coordinates times dataset scale. Singleton
nodes have no nearest neighbor and are excluded from percentiles, but included
in the collision-risk denominator. Half-windows are inclusive, with no coordinate
quantization. This measures collision risk, not guaranteed merging. A cKDTree
per frame avoids a quadratic distance matrix. Ties use the tree's choice on
lexicographically sorted coordinates, making results independent of graph order.

Two collision-risk metrics are reported per pooling kernel:
nearest_neighbor_collision_risk only checks whether each node's Euclidean-
nearest same-frame neighbor lies inside its axis-aligned half-window, and can
undercount when that nearest neighbor is outside the box but another neighbor
is inside it. any_neighbor_collision_risk instead checks every same-frame
neighbor: cKDTree.query_pairs(r=norm(half_width)) finds Euclidean-radius
candidates (a safe superset, since a box corner is the farthest point in the
box from its center), and an exact per-axis inclusive box test on those
candidates decides which pairs actually qualify. Both metrics use continuous
physical GT coordinates (frame-relative Euclidean nearest neighbor times
dataset scale); training's own coordinate quantization happens later, against
the network's downsampled spatial grid (see compute_detection_loss in
train_unet_transformer.py), which this audit does not have access to without
loading images, so it is not reproduced here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial import cKDTree

from tracking_cellmot.io import open_dataset
from build_calibration_subset_split import _load_fold, SubsetSplitError
from dataspec import DATASET_PATH
from predict_unet_transformer import pool_kernel_from_um


class AuditError(ValueError):
    """Invalid audit input."""


def positive_triplet(values, name, *, integer=False):
    arr = np.asarray(values, dtype=float)
    if (arr.shape != (3,) or not np.isfinite(arr).all() or (arr <= 0).any()
            or (integer and (arr != np.floor(arr)).any())):
        raise AuditError(f"{name} must contain three finite positive {'integers' if integer else 'values'}")
    return arr


def summarize(distances, counts, nearest_risks, any_node_counts, any_pair_counts, requested):
    percentiles = [0, 1, 5, 10, 25, 50, 75]
    values = np.percentile(distances, percentiles).tolist() if len(distances) else [None] * 7
    return {
        **counts,
        "nodes_with_neighbor": len(distances),
        "nearest_neighbor_um": dict(zip(["minimum", "p1", "p5", "p10", "p25", "p50", "p75"], values)),
        "nearest_neighbor_collision_risk": [
            {"requested_um": um, "count": int(count), "fraction": int(count) / counts["node_count"]}
            for um, count in zip(requested, nearest_risks)
        ],
        "any_neighbor_collision_risk": [
            {"requested_um": um, "count": int(count), "fraction": int(count) / counts["node_count"],
             "pair_count": int(pairs)}
            for um, count, pairs in zip(requested, any_node_counts, any_pair_counts)
        ],
    }


def audit_dataset(dataset, downsample, requested):
    """Return serializable dataset statistics and finite NN distances for aggregation."""
    scale = positive_triplet(dataset.scale, "dataset scale")
    strides = positive_triplet(downsample, "downsample", integer=True)
    effective = positive_triplet(scale * strides, "effective voxel size")
    requested = [float(v) for v in requested]
    if not requested or not all(np.isfinite(v) and v > 0 for v in requested):
        raise AuditError("pool-kernel-um values must be finite and positive")
    kernels = [pool_kernel_from_um(v, tuple(effective)) for v in requested]
    halves = np.asarray(kernels) // 2 * effective
    if dataset.tracks is None:
        raise AuditError("missing annotations")
    try:
        attrs = dataset.tracks.node_attrs(attr_keys=["t", "z", "y", "x"])
        nodes = attrs.select("t", "z", "y", "x").to_numpy().astype(float)
    except Exception as exc:
        raise AuditError(f"annotations require numeric t,z,y,x attributes: {exc}") from exc
    if not len(nodes):
        raise AuditError("empty dataset: no annotated nodes")
    if not np.isfinite(nodes).all():
        raise AuditError("nonfinite annotation coordinates or frame indices")
    shape = dataset.image_shape
    if shape is None or len(shape) != 4 or any(s <= 0 for s in shape):
        raise AuditError("empty or invalid dataset image shape")
    if ((nodes[:, 0] != np.floor(nodes[:, 0])).any()
            or (nodes[:, 0] < 0).any() or (nodes[:, 0] >= shape[0]).any()):
        raise AuditError("annotation frame indices must be integers within image frames")
    nodes = nodes[np.lexsort((nodes[:, 3], nodes[:, 2], nodes[:, 1], nodes[:, 0]))]
    with np.errstate(over="ignore"):
        coords = nodes[:, 1:] * scale
    if not np.isfinite(coords).all():
        raise AuditError("nonfinite physical coordinates")
    radii = [float(np.linalg.norm(half)) for half in halves]
    boundaries = np.r_[0, np.flatnonzero(np.diff(nodes[:, 0])) + 1, len(nodes)]
    distances = []
    nearest_risks = np.zeros(len(requested), dtype=np.int64)
    any_node_counts = np.zeros(len(requested), dtype=np.int64)
    any_pair_counts = np.zeros(len(requested), dtype=np.int64)
    singletons = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        points = coords[start:end]
        if len(points) == 1:
            singletons += 1
            continue
        tree = cKDTree(points)
        dist, indices = tree.query(points, k=2, workers=1)
        # With duplicate coordinates the first returned neighbor need not be self.
        choice = np.where(indices[:, 0] == np.arange(len(points)), 1, 0)
        neighbor = indices[np.arange(len(points)), choice]
        nearest = dist[np.arange(len(points)), choice]
        if not np.isfinite(nearest).all():
            raise AuditError("nonfinite nearest-neighbor distances")
        distances.append(nearest)
        delta = np.abs(points - points[neighbor])
        for i, (half, radius) in enumerate(zip(halves, radii)):
            nearest_risks[i] += np.count_nonzero(np.all(delta <= half, axis=1))
            # Euclidean radius = norm(half) is a safe superset of the box: a box
            # corner is the farthest point in the box from its center. The exact
            # per-axis inclusive check below then filters candidates down to the
            # true box membership, avoiding an O(n^2) all-pairs comparison.
            candidates = tree.query_pairs(r=radius, output_type="ndarray")
            if len(candidates):
                pair_delta = np.abs(points[candidates[:, 0]] - points[candidates[:, 1]])
                qualifying = candidates[np.all(pair_delta <= half, axis=1)]
            else:
                qualifying = candidates
            any_pair_counts[i] += len(qualifying)
            if len(qualifying):
                any_node_counts[i] += len(np.unique(qualifying))
    distances = np.concatenate(distances) if distances else np.empty(0)
    counts = {"node_count": len(nodes), "frame_count": int(shape[0]),
              "annotated_frame_count": len(boundaries) - 1, "singleton_frame_count": singletons}
    result = summarize(distances, counts, nearest_risks, any_node_counts, any_pair_counts, requested)
    result.update(dataset_scale_um=scale.tolist(), effective_voxel_size_um=effective.tolist())
    for row, kernel, half in zip(result["nearest_neighbor_collision_risk"], kernels, halves):
        row.update(voxel_kernel=list(kernel), physical_half_width_um=half.tolist())
    for row, kernel, half in zip(result["any_neighbor_collision_risk"], kernels, halves):
        row.update(voxel_kernel=list(kernel), physical_half_width_um=half.tolist())
    groups = {}
    for um, kernel in zip(requested, kernels):
        groups.setdefault(kernel, []).append(um)
    result["equivalent_kernels"] = [
        {"voxel_kernel": list(k), "requested_um": v}
        for k, v in sorted(groups.items()) if len(v) > 1
    ]
    return result, distances


def audit(data_dir, splits, split, downsample, requested):
    try:
        fold = _load_fold(Path(splits), split)
    except SubsetSplitError as exc:
        raise AuditError(str(exc)) from exc
    for key in ("train", "test"):
        names = fold[key]
        if len(set(names)) != len(names):
            raise AuditError(f"malformed splits: duplicate {key} datasets")
        if any(Path(n).name != n or n in (".", "..") or "/" in n or "\\" in n for n in names):
            raise AuditError("malformed splits: dataset names must be basenames")
    if set(fold["train"]) & set(fold["test"]):
        raise AuditError("malformed splits: train/test overlap")
    if not fold["test"]:
        raise AuditError("empty dataset selection in split test list")
    datasets, distances = {}, []
    for name in sorted(fold["test"]):
        try:
            ds = open_dataset(Path(data_dir) / name, require_tracks=True, load_image=False, normalize=False)
            result, nn = audit_dataset(ds, downsample, requested)
        except Exception as exc:
            raise AuditError(f"dataset {name}: {exc}") from exc
        datasets[name] = result
        distances.append(nn)
    counts = {key: sum(d[key] for d in datasets.values()) for key in
              ("node_count", "frame_count", "annotated_frame_count", "singleton_frame_count")}
    nearest_risks = [sum(d["nearest_neighbor_collision_risk"][i]["count"] for d in datasets.values())
                      for i in range(len(requested))]
    any_node_counts = [sum(d["any_neighbor_collision_risk"][i]["count"] for d in datasets.values())
                        for i in range(len(requested))]
    any_pair_counts = [sum(d["any_neighbor_collision_risk"][i]["pair_count"] for d in datasets.values())
                        for i in range(len(requested))]
    return {
        "schema_version": 1, "split": split, "partition": "test", "axis_order": ["z", "y", "x"],
        "downsample": list(downsample), "dataset_count": len(datasets),
        "notes": ["Collision risk, not guaranteed merging; inclusive axis-aligned half-window around each GT node.",
                  "Nearest neighbor is selected by physical Euclidean distance within the same frame only.",
                  "nearest_neighbor_collision_risk only checks the Euclidean-nearest same-frame neighbor; "
                  "any_neighbor_collision_risk checks every same-frame neighbor via an exact axis-aligned box "
                  "test on cKDTree.query_pairs candidates, and also reports unique qualifying pairs.",
                  "Singletons contribute to node counts and risk denominators, but not distance percentiles.",
                  "Percentiles use linear interpolation over nodes; frame_count includes unannotated image frames.",
                  "Equal-distance ties use cKDTree choice after lexicographic coordinate sorting.",
                  "Dataset scale follows open_dataset, including its default when scale metadata is absent."],
        "aggregate": summarize(np.concatenate(distances), counts, nearest_risks, any_node_counts,
                                any_pair_counts, requested),
        "datasets": datasets,
    }


def write_json_atomic(path, report):
    """Replace output only after strict JSON serialization and a complete flushed write."""
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATASET_PATH)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--split", type=int, default=0, help="Fold split ID; audits its test list")
    parser.add_argument("--downsample", default="1,4,4", help="Positive integer Z,Y,X strides")
    parser.add_argument("--pool-kernel-um", type=float, nargs="+", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        splits = args.splits or args.data_dir / "dataset_splits.json"
        output = args.json_out.resolve()
        if output == splits.resolve() or any(
            part.suffix.lower() in (".geff", ".zarr") for part in (output, *output.parents)
        ):
            raise AuditError("json-out must not overwrite input splits or write inside GEFF/Zarr datasets")
        downsample = tuple(int(v) for v in args.downsample.split(","))
        positive_triplet(downsample, "downsample", integer=True)
        report = audit(args.data_dir, splits,
                       args.split, downsample, args.pool_kernel_um)
        write_json_atomic(args.json_out, report)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
