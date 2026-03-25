from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os


@dataclass
class PngCheckResult:
    total_pngs: int
    missing_json: int
    bad_resolution: int


@dataclass
class TempCheckResult:
    pairs_manifest: int
    labels_npy: int
    missing_labels: int


def _load_config(config_path: Optional[str] = None) -> dict:
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_resolution(resolution_text: str) -> Tuple[int, int]:
    if "x" not in resolution_text:
        raise ValueError("Resolution must be formatted like 256x256")
    parts = resolution_text.split("x")
    return int(parts[0]), int(parts[1])


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


def _normalize_dataset_kind(value: object) -> str:
    kind = str(value or "A2D2").strip().upper()
    if kind not in {"A2D2", "KITTI"}:
        raise ValueError("Invalid dataset. Use 'A2D2' or 'KITTI'.")
    return kind


def _count_kitti_scene_files(scenes_dir: Path) -> Tuple[int, int, int]:
    scene_roots = 0
    png_count = 0
    txt_count = 0
    for root, _, _ in os.walk(scenes_dir):
        root_path = Path(root)
        image_dir = root_path / "image_02" / "data"
        oxts_dir = root_path / "oxts" / "data"
        if not image_dir.is_dir() or not oxts_dir.is_dir():
            continue
        scene_roots += 1
        png_count += sum(
            1
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        txt_count += sum(
            1
            for path in oxts_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        )
    return scene_roots, png_count, txt_count


def _list_pngs(frames_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )


def _check_png_batch(
    png_paths: Iterable[Path],
    frames_dir: Path,
    expected_resolution: Optional[Tuple[int, int]],
    progress_every: Optional[int] = None,
    progress_prefix: str = "[DataChecker]",
) -> PngCheckResult:
    total = 0
    missing_json = 0
    bad_resolution = 0

    for png_path in png_paths:
        total += 1
        base_name = png_path.stem
        json_path = frames_dir / f"{base_name}.json"
        if not json_path.exists():
            missing_json += 1

        if expected_resolution is not None:
            img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                bad_resolution += 1
            else:
                height, width = img.shape[:2]
                if (width, height) != expected_resolution:
                    bad_resolution += 1

        if progress_every and total % progress_every == 0:
            print(f"{progress_prefix} Checked {total} PNGs...")

    return PngCheckResult(
        total_pngs=total, missing_json=missing_json, bad_resolution=bad_resolution
    )


def _chunk_list(items: List[Path], chunks: int) -> List[List[Path]]:
    if not items:
        return []
    chunk_size = max(1, len(items) // chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def check_pngs(
    frames_dir: Path,
    expected_resolution: Optional[Tuple[int, int]],
    num_workers: int,
) -> PngCheckResult:
    png_paths = _list_pngs(frames_dir)
    if not png_paths:
        return PngCheckResult(0, 0, 0)

    workers = _resolve_worker_count(num_workers, min_workers=1)
    chunks = _chunk_list(png_paths, workers)

    if workers == 1 or len(chunks) == 1:
        return _check_png_batch(
            png_paths,
            frames_dir,
            expected_resolution,
            progress_every=1000,
            progress_prefix="[DataChecker]",
        )

    totals = PngCheckResult(0, 0, 0)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(
                _check_png_batch,
                chunk,
                frames_dir,
                expected_resolution,
                None,
                "[DataChecker Worker]",
            )
            for chunk in chunks
        ]
        completed = 0
        total_chunks = len(futures)
        processed_pngs = 0
        for future in as_completed(futures):
            result = future.result()
            totals.total_pngs += result.total_pngs
            totals.missing_json += result.missing_json
            totals.bad_resolution += result.bad_resolution
            completed += 1
            processed_pngs += result.total_pngs
            print(
                "[DataChecker] "
                f"Chunks {completed}/{total_chunks} | "
                f"Processed PNGs: {processed_pngs}/{len(png_paths)}"
            )

    return totals


def check_temp_pairs(manifest_path: Path, temp_labels_dir: Path) -> TempCheckResult:
    if not manifest_path.exists() or not temp_labels_dir.exists():
        return TempCheckResult(0, 0, 0)

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        pairs = payload.get("pairs", []) if isinstance(payload, dict) else []
    except (OSError, json.JSONDecodeError):
        payload = {}
        pairs = []

    labels_file = payload.get("labels_file") if isinstance(payload, dict) else None
    if isinstance(labels_file, str):
        labels_path = (manifest_path.parent / labels_file).resolve()
        if not labels_path.exists():
            return TempCheckResult(len(pairs), 0, len(pairs))
        try:
            labels = np.load(labels_path, mmap_mode="r")
            label_count = len(labels)
        except Exception:
            return TempCheckResult(len(pairs), 0, len(pairs))

        missing_labels = 0
        for item in pairs:
            if not isinstance(item, dict):
                missing_labels += 1
                continue
            label_index = item.get("label_index")
            try:
                label_index = int(label_index)
            except (TypeError, ValueError):
                missing_labels += 1
                continue
            if label_index < 0 or label_index >= label_count:
                missing_labels += 1

        return TempCheckResult(
            pairs_manifest=len(pairs),
            labels_npy=1,
            missing_labels=missing_labels,
        )

    labels = {
        path.stem
        for path in temp_labels_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".npy"
    }

    manifest_stems = {
        Path(item.get("label", "")).stem for item in pairs if isinstance(item, dict)
    }
    missing_labels = len(manifest_stems - labels)
    return TempCheckResult(
        pairs_manifest=len(pairs),
        labels_npy=len(labels),
        missing_labels=missing_labels,
    )


def main() -> None:
    config = _load_config()
    dataset_path = config.get("dataset_path")
    if not dataset_path:
        raise ValueError("config.json missing dataset_path")
    dataset_kind = _normalize_dataset_kind(config.get("dataset", "A2D2"))

    resolution = config.get("resolution")
    expected_resolution = None
    if isinstance(resolution, str):
        expected_resolution = _parse_resolution(resolution)

    num_workers = _resolve_worker_count(config.get("num_workers", 0), min_workers=1)

    if dataset_kind == "A2D2":
        frames_dir = Path(dataset_path) / "frames"
        if not frames_dir.exists():
            raise FileNotFoundError(f"frames folder not found: {frames_dir}")

        print("[DataChecker] Checking raw A2D2 frames...")
        png_result = check_pngs(frames_dir, expected_resolution, num_workers)

        print(f"[DataChecker] PNG files: {png_result.total_pngs}")
        print(f"[DataChecker] Missing JSON: {png_result.missing_json}")
        print(f"[DataChecker] Bad resolution: {png_result.bad_resolution}")
        missing_pairs = png_result.missing_json + png_result.bad_resolution
        print(f"[DataChecker] Missing/invalid pairs: {missing_pairs}")
    else:
        scenes_dir = Path(dataset_path) / "scenes"
        if not scenes_dir.exists():
            raise FileNotFoundError(f"scenes folder not found: {scenes_dir}")
        scene_roots, png_count, txt_count = _count_kitti_scene_files(scenes_dir)
        print("[DataChecker] KITTI mode detected.")
        print(f"[DataChecker] Scene roots: {scene_roots}")
        print(f"[DataChecker] PNG files (image_02/data): {png_count}")
        print(f"[DataChecker] TXT files (oxts/data): {txt_count}")
        print(
            "[DataChecker] Note: KITTI has no frame JSON timestamp files; "
            "A2D2 JSON checks are skipped."
        )

    manifest_path = Path(dataset_path) / "temp_pairs.json"
    temp_labels_dir = Path(dataset_path) / "temp_labels"
    print("[DataChecker] Checking temp folders...")
    temp_result = check_temp_pairs(manifest_path, temp_labels_dir)

    if manifest_path.exists() and temp_labels_dir.exists():
        print(f"[DataChecker] temp_pairs entries: {temp_result.pairs_manifest}")
        print(f"[DataChecker] temp_labels npy: {temp_result.labels_npy}")
        print(f"[DataChecker] Missing labels for pairs: {temp_result.missing_labels}")
    else:
        print("[DataChecker] temp_pairs or temp_labels not found. Skipping temp check.")


if __name__ == "__main__":
    main()
