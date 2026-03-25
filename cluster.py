from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Dict, List, Tuple
import multiprocessing as mp

import numpy as np
import cv2

from train_clean import run_training
from test import run_test


def _load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        default_path = Path(__file__).resolve().parent / "config.json"
        if not default_path.exists():
            return {}
        config_path = default_path
    else:
        config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_worker_count(
    value: object,
    min_workers: int = 1,
    max_workers: int | None = None,
) -> int:
    if isinstance(value, str):
        if value.strip().lower() in {"auto", "max", "all"}:
            value = 0
        else:
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = 0
    if value is None:
        value = 0
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        value_int = 0
    if value_int <= 0:
        value_int = os.cpu_count() or 1
    if max_workers is not None:
        value_int = min(value_int, max_workers)
    return max(min_workers, value_int)


def _build_frame_timestamp_cache(frames_dir: str) -> Dict[str, float]:
    cache: Dict[str, float] = {}
    try:
        for entry in os.scandir(frames_dir):
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith(".json"):
                continue
            try:
                with open(entry.path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if "cam_tstamp" in payload:
                    cache[Path(entry.name).stem] = float(payload["cam_tstamp"])
            except Exception:
                continue
    except OSError:
        return cache
    return cache


def _apply_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _normalize_dataset_kind(value: object) -> str:
    kind = str(value or "A2D2").strip().upper()
    if kind not in {"A2D2", "KITTI"}:
        raise ValueError("Invalid dataset. Use 'A2D2' or 'KITTI'.")
    return kind


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster runner for Speed Estimation")
    parser.add_argument("--config", help="Path to JSON config")
    parser.add_argument("--mode", choices=["train", "test"], help="Run mode")
    parser.add_argument("--dataset", help="Dataset type: A2D2 or KITTI")
    parser.add_argument("--dataset-path", help="Path to training dataset")
    parser.add_argument("--testdata-path", help="Path to test dataset")
    parser.add_argument("--output-path", help="Output path for training artifacts")
    parser.add_argument("--test-output-path", help="Output path for test artifacts")
    parser.add_argument("--pth-path", help="Model checkpoint path for testing")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--learning-rate", help="Learning rate")
    parser.add_argument("--resolution", help="Resolution like 224x224")
    parser.add_argument("--batch-size", help="Batch size")
    parser.add_argument("--epochs", type=int, help="Epoch count")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Early stopping patience (epochs without val loss improvement)",
    )
    parser.add_argument("--cutmix", type=int, help="Cutmix probability percent")
    parser.add_argument("--train-val-split", type=int, help="Training percent")
    parser.add_argument(
        "--gutter",
        type=int,
        help="Percent of frame pairs to skip (0-100)",
    )
    return parser.parse_args()


def _validate_config(config: Dict[str, Any]) -> None:
    mode = config.get("mode", "train")
    _normalize_dataset_kind(config.get("dataset", "A2D2"))
    missing: List[str] = []

    if mode == "train":
        if not config.get("dataset_path"):
            missing.append("dataset_path")
        if not config.get("output_path"):
            missing.append("output_path")
        if not config.get("model"):
            missing.append("model")
        if not config.get("learning_rate"):
            missing.append("learning_rate")
        if not config.get("resolution"):
            missing.append("resolution")
        if not config.get("batch_size"):
            missing.append("batch_size")
        if not config.get("epochs"):
            missing.append("epochs")
    else:
        if not config.get("testdata_path"):
            missing.append("testdata_path")
        if not config.get("test_output_path"):
            missing.append("test_output_path")
        if not config.get("pth_path"):
            missing.append("pth_path")
        if not config.get("model"):
            missing.append("model")
        if not config.get("resolution"):
            missing.append("resolution")

    if missing:
        raise ValueError(
            "Missing required config keys: "
            f"{', '.join(missing)}. Provide --config or pass them via CLI flags."
        )


def _prepare_common_paths(dataset_path: str, mode: str) -> Tuple[str, str, str]:
    temp_labels_dir = os.path.join(
        dataset_path, "temp_labels" if mode == "train" else "temp_labels_test"
    )
    labels_filename = "labels.npy" if mode == "train" else "labels_test.npy"
    labels_path = os.path.join(temp_labels_dir, labels_filename)
    manifest_path = os.path.join(
        dataset_path,
        "temp_pairs.json" if mode == "train" else "temp_pairs_test.json",
    )
    return temp_labels_dir, labels_path, manifest_path


def _clear_prepare_outputs(temp_labels_dir: str, manifest_path: str) -> None:
    os.makedirs(temp_labels_dir, exist_ok=True)
    for name in os.listdir(temp_labels_dir):
        if name.lower().endswith(".npy"):
            try:
                os.remove(os.path.join(temp_labels_dir, name))
            except OSError:
                pass
    if os.path.isfile(manifest_path):
        try:
            os.remove(manifest_path)
        except OSError:
            pass


def _prepare_data_a2d2(config: Dict[str, Any], mode: str, dataset_path: str) -> None:
    frames_dir = os.path.join(dataset_path, "frames")
    canbus_dir = os.path.join(dataset_path, "canbus")
    if not os.path.isdir(frames_dir) or not os.path.isdir(canbus_dir):
        raise ValueError("frames or canbus directory missing.")

    temp_labels_dir, labels_path, manifest_path = _prepare_common_paths(
        dataset_path, mode
    )

    skip_prepare = bool(config.get("skip_prepare", False))
    if skip_prepare:
        print("[Cluster] skip_prepare enabled. Skipping all preparation steps.")
        return

    _clear_prepare_outputs(temp_labels_dir, manifest_path)

    def list_pngs() -> List[str]:
        return sorted(
            f
            for f in os.listdir(frames_dir)
            if f.lower().endswith(".png")
            and os.path.isfile(os.path.join(frames_dir, f))
        )

    png_files = list_pngs()
    if len(png_files) < 2:
        raise ValueError("Not enough frames to build pairs.")

    total_pairs = len(png_files) - 1
    gutter = max(0, min(100, int(config.get("gutter", 0) or 0)))
    keep_ratio = max(0.0, 1.0 - (gutter / 100.0))

    def select_pair_indices(total: int) -> List[int]:
        if total <= 0 or keep_ratio <= 0.0:
            return []
        indices: List[int] = []
        error = 0.0
        for idx in range(total):
            error += keep_ratio
            if error >= 1.0:
                indices.append(idx)
                error -= 1.0
        return indices

    selected_indices = select_pair_indices(total_pairs)
    expected_pairs = len(selected_indices)

    frame_ts_cache = _build_frame_timestamp_cache(frames_dir)

    def load_canbus_values() -> List[Tuple[int, float]]:
        values: List[Tuple[int, float]] = []
        for name in os.listdir(canbus_dir):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(canbus_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                vehicle_speed = data.get("vehicle_speed", {})
                for item in vehicle_speed.get("values", []):
                    if isinstance(item, list) and len(item) >= 2:
                        values.append((int(item[0]), float(item[1])))
            except Exception:
                continue
        values.sort(key=lambda x: x[0])
        return values

    canbus_values = load_canbus_values()
    if not canbus_values:
        raise ValueError("No canbus speed values found.")

    canbus_ts = [item[0] for item in canbus_values]
    canbus_speeds = [item[1] for item in canbus_values]
    if canbus_ts:
        print(
            "[Cluster] CAN-bus timestamp range: "
            f"min={min(canbus_ts):.3f}, max={max(canbus_ts):.3f}"
        )

    pairs_payload: List[Dict[str, str]] = []
    labels: List[float] = []
    removed_no_interval = 0
    removed_missing_timestamp = 0
    frame_ts_min = None
    frame_ts_max = None
    for idx, pair_index in enumerate(selected_indices, start=1):
        png1 = png_files[pair_index]
        png2 = png_files[pair_index + 1]
        base_name = os.path.splitext(png1)[0]
        base_name2 = os.path.splitext(png2)[0]

        ts1 = frame_ts_cache.get(base_name)
        ts2 = frame_ts_cache.get(base_name2)
        if ts1 is None or ts2 is None:
            removed_missing_timestamp += 1
            continue
        frame_ts_min = ts1 if frame_ts_min is None else min(frame_ts_min, ts1)
        frame_ts_min = ts2 if frame_ts_min is None else min(frame_ts_min, ts2)
        frame_ts_max = ts1 if frame_ts_max is None else max(frame_ts_max, ts1)
        frame_ts_max = ts2 if frame_ts_max is None else max(frame_ts_max, ts2)
        start_ts = min(ts1, ts2)
        end_ts = max(ts1, ts2)
        speed, count = _mean_speed_in_interval(
            canbus_ts, canbus_speeds, start_ts, end_ts
        )
        if speed is None or count == 0:
            removed_no_interval += 1
            continue

        label_index = len(labels)
        labels.append(float(speed))

        frame1_rel = os.path.relpath(os.path.join(frames_dir, png1), dataset_path)
        frame2_rel = os.path.relpath(os.path.join(frames_dir, png2), dataset_path)
        pairs_payload.append(
            {
                "frame1": frame1_rel,
                "frame2": frame2_rel,
                "label_index": label_index,
            }
        )

        if idx % 50 == 0 or idx == expected_pairs:
            print(f"[Cluster] Prepared {idx}/{expected_pairs} pairs")

    if frame_ts_min is not None and frame_ts_max is not None:
        print(
            "[Cluster] Frame timestamp range: "
            f"min={frame_ts_min:.3f}, max={frame_ts_max:.3f}"
        )

    if removed_no_interval or removed_missing_timestamp:
        print(
            "[Cluster] Pairs removed by sync rules: "
            f"no_interval={removed_no_interval}, "
            f"missing_timestamp={removed_missing_timestamp}"
        )

    if not pairs_payload:
        raise ValueError("No valid frame/label pairs found.")

    np.save(labels_path, np.array(labels, dtype=np.float32))
    labels_rel = os.path.relpath(labels_path, dataset_path)

    manifest_payload = {
        "version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "labels_file": labels_rel,
        "pairs": pairs_payload,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle)


def _find_kitti_scene_roots(scenes_dir: str) -> List[Path]:
    roots: List[Path] = []
    for root, _, _ in os.walk(scenes_dir):
        root_path = Path(root)
        image_data = root_path / "image_02" / "data"
        oxts_data = root_path / "oxts" / "data"
        if image_data.is_dir() and oxts_data.is_dir():
            roots.append(root_path)
    return roots


def _read_kitti_speed_kmh(oxts_file: Path) -> float | None:
    try:
        with oxts_file.open("r", encoding="utf-8") as handle:
            line = handle.readline().strip()
    except OSError:
        return None
    if not line:
        return None
    parts = line.split()
    if len(parts) <= 8:
        return None
    try:
        speed_ms = float(parts[8])
    except ValueError:
        return None
    return speed_ms * 3.6


def _prepare_data_kitti(config: Dict[str, Any], mode: str, dataset_path: str) -> None:
    scenes_dir = os.path.join(dataset_path, "scenes")
    if not os.path.isdir(scenes_dir):
        raise ValueError("scenes directory missing for KITTI dataset.")

    temp_labels_dir, labels_path, manifest_path = _prepare_common_paths(
        dataset_path, mode
    )

    skip_prepare = bool(config.get("skip_prepare", False))
    if skip_prepare:
        print("[Cluster] skip_prepare enabled. Skipping all preparation steps.")
        return

    _clear_prepare_outputs(temp_labels_dir, manifest_path)

    scene_roots = _find_kitti_scene_roots(scenes_dir)
    if not scene_roots:
        raise ValueError(
            "No KITTI scene roots found. Expected scenes with image_02/data and oxts/data."
        )

    candidate_pairs: List[Tuple[str, str, float]] = []
    skipped_missing_oxts = 0
    skipped_bad_oxts = 0

    for scene_root in scene_roots:
        image_dir = scene_root / "image_02" / "data"
        oxts_dir = scene_root / "oxts" / "data"
        png_files = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        if len(png_files) < 2:
            continue

        for idx in range(len(png_files) - 1):
            frame1 = png_files[idx]
            frame2 = png_files[idx + 1]
            oxts1 = oxts_dir / f"{frame1.stem}.txt"
            oxts2 = oxts_dir / f"{frame2.stem}.txt"
            if not oxts1.is_file() or not oxts2.is_file():
                skipped_missing_oxts += 1
                continue
            speed_kmh = _read_kitti_speed_kmh(oxts1)
            if speed_kmh is None:
                skipped_bad_oxts += 1
                continue

            frame1_rel = os.path.relpath(str(frame1), dataset_path)
            frame2_rel = os.path.relpath(str(frame2), dataset_path)
            candidate_pairs.append((frame1_rel, frame2_rel, float(speed_kmh)))

    if not candidate_pairs:
        raise ValueError("No valid KITTI frame/label pairs found.")

    total_pairs = len(candidate_pairs)
    gutter = max(0, min(100, int(config.get("gutter", 0) or 0)))
    keep_ratio = max(0.0, 1.0 - (gutter / 100.0))

    def select_pair_indices(total: int) -> List[int]:
        if total <= 0 or keep_ratio <= 0.0:
            return []
        indices: List[int] = []
        error = 0.0
        for index in range(total):
            error += keep_ratio
            if error >= 1.0:
                indices.append(index)
                error -= 1.0
        return indices

    selected_indices = select_pair_indices(total_pairs)
    if not selected_indices:
        raise ValueError("No KITTI pairs kept after gutter filtering.")

    labels: List[float] = []
    pairs_payload: List[Dict[str, str]] = []
    expected_pairs = len(selected_indices)
    for idx, pair_index in enumerate(selected_indices, start=1):
        frame1_rel, frame2_rel, speed_kmh = candidate_pairs[pair_index]
        label_index = len(labels)
        labels.append(speed_kmh)
        pairs_payload.append(
            {
                "frame1": frame1_rel,
                "frame2": frame2_rel,
                "label_index": label_index,
            }
        )
        if idx % 50 == 0 or idx == expected_pairs:
            print(f"[Cluster] Prepared {idx}/{expected_pairs} KITTI pairs")

    if skipped_missing_oxts or skipped_bad_oxts:
        print(
            "[Cluster] KITTI pairs removed during prepare: "
            f"missing_oxts={skipped_missing_oxts}, bad_oxts={skipped_bad_oxts}"
        )

    np.save(labels_path, np.array(labels, dtype=np.float32))
    labels_rel = os.path.relpath(labels_path, dataset_path)
    manifest_payload = {
        "version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "labels_file": labels_rel,
        "pairs": pairs_payload,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle)


def _prepare_data(config: Dict[str, Any]) -> None:
    mode = config.get("mode", "train")
    dataset_path = (
        config.get("dataset_path") if mode == "train" else config.get("testdata_path")
    )
    dataset_kind = _normalize_dataset_kind(config.get("dataset", "A2D2"))
    if not dataset_path or not os.path.isdir(dataset_path):
        raise ValueError("Invalid dataset path.")

    if dataset_kind == "KITTI":
        _prepare_data_kitti(config, mode, dataset_path)
    else:
        _prepare_data_a2d2(config, mode, dataset_path)


def _dataset_check_marker_path(dataset_path: str, mode: str) -> Path:
    filename = (
        "temp_dataset_check.json" if mode == "train" else "temp_dataset_check_test.json"
    )
    return Path(dataset_path) / filename


def _latest_mtime_for_exts(path: str, exts: Tuple[str, ...]) -> float:
    mt = 0.0
    for root, _, files in os.walk(path):
        for name in files:
            if name.lower().endswith(exts):
                try:
                    mt = max(mt, os.path.getmtime(os.path.join(root, name)))
                except OSError:
                    continue
    return mt


def _dataset_inputs_mtime_a2d2(
    frames_dir: str, canbus_dir: str, manifest_path: str, temp_labels_dir: str
) -> float:
    mt = 0.0
    if os.path.isfile(manifest_path):
        try:
            mt = max(mt, os.path.getmtime(manifest_path))
        except OSError:
            pass
    mt = max(mt, _latest_mtime_for_exts(frames_dir, (".png", ".json")))
    mt = max(mt, _latest_mtime_for_exts(canbus_dir, (".json",)))
    mt = max(mt, _latest_mtime_for_exts(temp_labels_dir, (".npy",)))
    return mt


def _dataset_inputs_mtime_kitti(
    scenes_dir: str, manifest_path: str, temp_labels_dir: str
) -> float:
    mt = 0.0
    if os.path.isfile(manifest_path):
        try:
            mt = max(mt, os.path.getmtime(manifest_path))
        except OSError:
            pass
    mt = max(mt, _latest_mtime_for_exts(scenes_dir, (".png", ".txt")))
    mt = max(mt, _latest_mtime_for_exts(temp_labels_dir, (".npy",)))
    return mt


def _load_manifest_payload(manifest_path: str) -> Dict[str, Any]:
    if not os.path.isfile(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _load_cam_timestamp(json_path: str) -> float | None:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "cam_tstamp" not in data:
            return None
        return float(data.get("cam_tstamp"))
    except Exception:
        return None


def _load_canbus_values(canbus_dir: str) -> List[Tuple[float, float]]:
    values: List[Tuple[float, float]] = []
    for name in os.listdir(canbus_dir):
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(canbus_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            vehicle_speed = data.get("vehicle_speed", {})
            for item in vehicle_speed.get("values", []):
                if isinstance(item, list) and len(item) >= 2:
                    values.append((float(item[0]), float(item[1])))
        except Exception:
            continue
    values.sort(key=lambda x: x[0])
    return values


def _nearest_canbus_gap(
    ts: float, canbus_values: List[Tuple[float, float]]
) -> float | None:
    if not canbus_values:
        return None
    canbus_ts = [item[0] for item in canbus_values]
    idx = bisect_left(canbus_ts, ts)
    if idx <= 0:
        return abs(ts - canbus_ts[0])
    if idx >= len(canbus_ts):
        return abs(ts - canbus_ts[-1])
    gap0 = abs(ts - canbus_ts[idx - 1])
    gap1 = abs(ts - canbus_ts[idx])
    return min(gap0, gap1)


def _mean_speed_in_interval(
    canbus_ts: List[float],
    canbus_speeds: List[float],
    start_ts: float,
    end_ts: float,
) -> Tuple[float | None, int]:
    if not canbus_ts:
        return None, 0
    left = bisect_left(canbus_ts, start_ts)
    right = bisect_right(canbus_ts, end_ts)
    if right <= left:
        return None, 0
    values = canbus_speeds[left:right]
    if not values:
        return None, 0
    return float(sum(values) / len(values)), len(values)


_CHECK_DATASET_PATH: str | None = None
_CHECK_FRAME_TS: Dict[str, float] = {}
_CHECK_CANBUS_TS: List[float] = []
_CHECK_CANBUS_SPEEDS: List[float] = []
_CHECK_MAX_FRAME_DT_MS: int = 0
_CHECK_LIGHTWEIGHT: bool = False
_CHECK_LABELS_CACHE: Dict[str, np.ndarray] = {}
_CHECK_LABELS_DEFAULT: str | None = None
_CHECK_DATASET_KIND: str = "A2D2"


def _init_dataset_check_state(
    dataset_kind: str,
    dataset_path: str,
    frame_ts_cache: Dict[str, float],
    canbus_ts: List[float],
    canbus_speeds: List[float],
    max_frame_dt_ms: int,
    lightweight_check: bool,
    labels_default: str | None,
) -> None:
    global _CHECK_DATASET_PATH
    global _CHECK_FRAME_TS
    global _CHECK_CANBUS_TS
    global _CHECK_CANBUS_SPEEDS
    global _CHECK_MAX_FRAME_DT_MS
    global _CHECK_LIGHTWEIGHT
    global _CHECK_LABELS_CACHE
    global _CHECK_LABELS_DEFAULT
    global _CHECK_DATASET_KIND

    _CHECK_DATASET_KIND = dataset_kind
    _CHECK_DATASET_PATH = dataset_path
    _CHECK_FRAME_TS = frame_ts_cache
    _CHECK_CANBUS_TS = canbus_ts
    _CHECK_CANBUS_SPEEDS = canbus_speeds
    _CHECK_MAX_FRAME_DT_MS = max_frame_dt_ms
    _CHECK_LIGHTWEIGHT = lightweight_check
    _CHECK_LABELS_CACHE = {}
    _CHECK_LABELS_DEFAULT = labels_default
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _get_labels_array(labels_path: str) -> np.ndarray | None:
    cached = _CHECK_LABELS_CACHE.get(labels_path)
    if cached is not None:
        return cached
    try:
        labels = np.load(labels_path, mmap_mode="r")
        _CHECK_LABELS_CACHE[labels_path] = labels
        return labels
    except Exception:
        return None


def _dataset_check_pair(
    task: Tuple[int, Dict[str, Any]],
) -> Tuple[int, bool, str | None, Dict[str, Any]]:
    idx, item = task
    frame1_rel = item.get("frame1")
    frame2_rel = item.get("frame2")
    label_rel = item.get("label")
    label_index = item.get("label_index")
    label_file = item.get("label_file")

    if not isinstance(frame1_rel, str) or not isinstance(frame2_rel, str):
        return idx, False, "bad_pair_format", item
    if label_index is None and not isinstance(label_rel, str):
        return idx, False, "bad_label_format", item

    dataset_path = _CHECK_DATASET_PATH or ""
    frame1_path = os.path.join(dataset_path, frame1_rel)
    frame2_path = os.path.join(dataset_path, frame2_rel)

    if not os.path.isfile(frame1_path) or not os.path.isfile(frame2_path):
        return idx, False, "missing_frame", item

    if not _CHECK_LIGHTWEIGHT:
        img1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return idx, False, "unreadable_frame", item

    if label_index is None:
        label_path = os.path.join(dataset_path, label_rel)
        if not os.path.isfile(label_path):
            return idx, False, "missing_label", item
        try:
            label_value = np.load(label_path)
            if not np.isfinite(label_value).all():
                return idx, False, "bad_label", item
        except Exception:
            return idx, False, "bad_label", item
    else:
        labels_path = label_file or _CHECK_LABELS_DEFAULT
        if not isinstance(labels_path, str):
            return idx, False, "missing_label", item
        labels = _get_labels_array(labels_path)
        if labels is None:
            return idx, False, "missing_label", item
        try:
            label_index = int(label_index)
        except (TypeError, ValueError):
            return idx, False, "bad_label", item
        if label_index < 0 or label_index >= len(labels):
            return idx, False, "bad_label", item
        label_value = labels[label_index]
        if not np.isfinite(label_value).all():
            return idx, False, "bad_label", item

    if _CHECK_DATASET_KIND == "A2D2":
        ts1 = _CHECK_FRAME_TS.get(Path(frame1_rel).stem)
        ts2 = _CHECK_FRAME_TS.get(Path(frame2_rel).stem)
        if ts1 is None or ts2 is None:
            return idx, False, "missing_timestamp", item

        frame_dt_ms = _timestamp_diff_ms(ts1, ts2)
        if _CHECK_MAX_FRAME_DT_MS > 0 and frame_dt_ms > _CHECK_MAX_FRAME_DT_MS:
            return idx, False, "frame_dt", item

        start_ts = min(ts1, ts2)
        end_ts = max(ts1, ts2)
        _, count = _mean_speed_in_interval(
            _CHECK_CANBUS_TS, _CHECK_CANBUS_SPEEDS, start_ts, end_ts
        )
        if count == 0:
            return idx, False, "canbus_interval_empty", item

    return idx, True, None, item


def _timestamp_diff_ms(ts1: float, ts2: float) -> float:
    diff = abs(ts2 - ts1)
    magnitude = max(abs(ts1), abs(ts2))
    # Heuristic: detect timestamp unit based on absolute magnitude.
    if magnitude >= 1e17:
        return diff / 1e6  # ns -> ms
    if magnitude >= 1e14:
        return diff / 1e3  # us -> ms
    return diff  # already ms


def _check_and_clean_dataset(config: Dict[str, Any]) -> None:
    mode = str(config.get("mode", "train"))
    dataset_kind = _normalize_dataset_kind(config.get("dataset", "A2D2"))
    dataset_path = (
        config.get("dataset_path") if mode == "train" else config.get("testdata_path")
    )
    if not dataset_path or not os.path.isdir(dataset_path):
        raise ValueError("Invalid dataset path.")

    frames_dir = os.path.join(dataset_path, "frames")
    canbus_dir = os.path.join(dataset_path, "canbus")
    scenes_dir = os.path.join(dataset_path, "scenes")
    if dataset_kind == "A2D2":
        if not os.path.isdir(frames_dir) or not os.path.isdir(canbus_dir):
            raise ValueError("frames or canbus directory missing.")
    else:
        if not os.path.isdir(scenes_dir):
            raise ValueError("scenes directory missing for KITTI dataset.")

    temp_labels_dir = os.path.join(
        dataset_path, "temp_labels" if mode == "train" else "temp_labels_test"
    )
    manifest_path = os.path.join(
        dataset_path,
        "temp_pairs.json" if mode == "train" else "temp_pairs_test.json",
    )

    max_frame_dt_ms = int(config.get("max_frame_dt_ms", 300) or 300)
    max_drop_ratio = float(config.get("max_drop_ratio", 0.30) or 0.30)
    log_every = int(config.get("dataset_check_log_every", 100) or 100)
    log_every = max(1, log_every)
    debug_samples = int(config.get("dataset_check_debug_samples", 5) or 5)
    debug_samples = max(0, debug_samples)

    marker_path = _dataset_check_marker_path(dataset_path, mode)
    if dataset_kind == "A2D2":
        inputs_mtime = _dataset_inputs_mtime_a2d2(
            frames_dir, canbus_dir, manifest_path, temp_labels_dir
        )
    else:
        inputs_mtime = _dataset_inputs_mtime_kitti(
            scenes_dir, manifest_path, temp_labels_dir
        )
    manifest_mtime = (
        os.path.getmtime(manifest_path) if os.path.isfile(manifest_path) else 0.0
    )

    if marker_path.exists():
        try:
            with marker_path.open("r", encoding="utf-8") as handle:
                marker = json.load(handle)
            if (
                isinstance(marker, dict)
                and float(marker.get("inputs_mtime", 0.0)) >= inputs_mtime
                and float(marker.get("manifest_mtime", -1.0)) == manifest_mtime
                and int(marker.get("max_frame_dt_ms", -1)) == max_frame_dt_ms
                and float(marker.get("max_drop_ratio", -1.0)) == max_drop_ratio
            ):
                print("[Cluster] Dataset check skipped (already verified).")
                return
        except Exception:
            pass

    payload = _load_manifest_payload(manifest_path)
    pairs = payload.get("pairs", []) if isinstance(payload, dict) else []
    labels_rel = payload.get("labels_file") if isinstance(payload, dict) else None
    if isinstance(labels_rel, str):
        labels_default = os.path.join(dataset_path, labels_rel)
    else:
        labels_default = None
    if not pairs:
        raise ValueError("No pairs found for dataset check.")

    total = len(pairs)
    progress_every = max(1, total // 20)
    t_start = time.perf_counter()
    print(f"[Cluster] Dataset check started for {total} pairs.")

    if dataset_kind == "A2D2":
        canbus_values = _load_canbus_values(canbus_dir)
        if not canbus_values:
            raise ValueError("No canbus speed values found.")
        canbus_ts = [item[0] for item in canbus_values]
        canbus_speeds = [item[1] for item in canbus_values]
        frame_ts_cache = _build_frame_timestamp_cache(frames_dir)
    else:
        canbus_ts = []
        canbus_speeds = []
        frame_ts_cache = {}

    valid_pairs: List[Dict[str, Any]] = []
    invalid_count = 0
    invalid_reasons: Dict[str, int] = {}
    lightweight_check = bool(config.get("dataset_check_lightweight", False))
    worker_count = _resolve_worker_count(config.get("dataset_check_workers", 0))

    _init_dataset_check_state(
        dataset_kind,
        dataset_path,
        frame_ts_cache,
        canbus_ts,
        canbus_speeds,
        max_frame_dt_ms,
        lightweight_check,
        labels_default,
    )

    tasks = [(idx, item) for idx, item in enumerate(pairs, start=1)]
    if worker_count <= 1 or total <= 1:
        for idx, item in tasks:
            if idx % log_every == 0 or idx == total:
                frame1_rel = item.get("frame1")
                frame2_rel = item.get("frame2")
                if isinstance(frame1_rel, str) and isinstance(frame2_rel, str):
                    print(
                        f"[Cluster] Checking pair {idx}/{total}: "
                        f"{frame1_rel} | {frame2_rel}"
                    )
                else:
                    print(f"[Cluster] Checking pair {idx}/{total}")
            _, ok, reason, _ = _dataset_check_pair((idx, item))
            if ok:
                valid_pairs.append(item)
            else:
                invalid_count += 1
                if reason:
                    invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                debug_reason_set = {"missing_frame", "missing_label", "bad_label"}
                if dataset_kind == "A2D2":
                    debug_reason_set.update(
                        {"missing_timestamp", "frame_dt", "canbus_interval_empty"}
                    )
                if idx <= debug_samples and reason in debug_reason_set:
                    print(f"[Cluster] Debug pair {idx}: reason={reason}")
            if idx % progress_every == 0 or idx == total:
                elapsed = time.perf_counter() - t_start
                pct = (idx / total) * 100.0
                print(
                    f"[Cluster] Dataset check progress: {pct:.1f}% "
                    f"({idx}/{total}) | elapsed {elapsed:.1f}s"
                )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=_init_dataset_check_state,
            initargs=(
                dataset_kind,
                dataset_path,
                frame_ts_cache,
                canbus_ts,
                canbus_speeds,
                max_frame_dt_ms,
                lightweight_check,
                labels_default,
            ),
        ) as executor:
            futures = [executor.submit(_dataset_check_pair, task) for task in tasks]
            completed = 0
            results: Dict[int, Dict[str, Any]] = {}
            for future in as_completed(futures):
                idx, ok, reason, item = future.result()
                completed += 1
                if ok:
                    results[idx] = item
                else:
                    invalid_count += 1
                    if reason:
                        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                    debug_reason_set = {"missing_frame", "missing_label", "bad_label"}
                    if dataset_kind == "A2D2":
                        debug_reason_set.update(
                            {"missing_timestamp", "frame_dt", "canbus_interval_empty"}
                        )
                    if idx <= debug_samples and reason in debug_reason_set:
                        print(f"[Cluster] Debug pair {idx}: reason={reason}")
                if completed % log_every == 0 or completed == total:
                    elapsed = time.perf_counter() - t_start
                    pct = (completed / total) * 100.0
                    print(
                        f"[Cluster] Dataset check progress: {pct:.1f}% "
                        f"({completed}/{total}) | elapsed {elapsed:.1f}s"
                    )
            for idx in range(1, total + 1):
                item = results.get(idx)
                if item is not None:
                    valid_pairs.append(item)

    drop_ratio = (invalid_count / total) if total > 0 else 1.0
    if invalid_count:
        print(
            f"[Cluster] Dataset check removed {invalid_count}/{total} pairs ({drop_ratio:.2%})."
        )
        print(f"[Cluster] Dataset check reasons: {invalid_reasons}")

    if not valid_pairs:
        raise ValueError("Dataset check failed: no valid pairs remain.")
    if drop_ratio > max_drop_ratio:
        raise ValueError(
            "Dataset check failed: too many invalid pairs "
            f"({drop_ratio:.2%} > {max_drop_ratio:.2%})."
        )

    if invalid_count:
        manifest_payload = {
            "version": 1,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "pairs": valid_pairs,
        }
        if isinstance(labels_rel, str):
            manifest_payload["labels_file"] = labels_rel
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest_payload, handle)
        manifest_mtime = os.path.getmtime(manifest_path)

    marker_payload = {
        "version": 1,
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "pair_count": len(valid_pairs),
        "removed_pairs": invalid_count,
        "inputs_mtime": inputs_mtime,
        "manifest_mtime": manifest_mtime,
        "max_frame_dt_ms": max_frame_dt_ms,
        "max_drop_ratio": max_drop_ratio,
        "reasons": invalid_reasons,
    }
    try:
        with marker_path.open("w", encoding="utf-8") as handle:
            json.dump(marker_payload, handle)
    except OSError:
        pass


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)

    overrides = {
        "mode": args.mode,
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "testdata_path": args.testdata_path,
        "output_path": args.output_path,
        "test_output_path": args.test_output_path,
        "pth_path": args.pth_path,
        "model": args.model,
        "learning_rate": args.learning_rate,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "cutmix": args.cutmix,
        "train_val_split": args.train_val_split,
        "gutter": args.gutter,
    }

    defaults = {
        "mode": "train",
        "dataset": "A2D2",
        "early_stopping_patience": 0,
        "cutmix": 0,
        "train_val_split": 80,
        "gutter": 0,
    }

    merged = _apply_overrides(config, overrides)
    for key, value in defaults.items():
        merged.setdefault(key, value)

    _validate_config(merged)

    print(
        f"[Cluster] Running in {merged['mode']} mode "
        f"(dataset={_normalize_dataset_kind(merged.get('dataset', 'A2D2'))})"
    )
    _prepare_data(merged)
    if bool(merged.get("skip_prepare", False)):
        print("[Cluster] skip_prepare enabled. Skipping dataset check.")
    else:
        print("[Cluster] Data preparation finished. Running dataset check...")
        _check_and_clean_dataset(merged)
        print("[Cluster] Dataset check finished. Starting run.")

    if merged["mode"] == "train":

        def status(message: str, level: str = "info") -> None:
            print(f"[Train:{level}] {message}")

        def progress(value: int) -> None:
            print(f"[Train] Progress {value}%")

        run_training(merged, status_callback=status, progress_callback=progress)
    else:

        def status(message: str, level: str = "info") -> None:
            print(f"[Test:{level}] {message}")

        def progress(value: int) -> None:
            print(f"[Test] Progress {value}%")

        run_test(merged, status_callback=status, progress_callback=progress)


if __name__ == "__main__":
    main()
