from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
import time
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import random

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import ctypes

from models import create_model


def _get_autocast_context(device_type: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _get_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


_RAFT_MODEL = None
_RAFT_DEVICE: Optional[torch.device] = None

_FRAME_RAM_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_FRAME_RAM_CACHE_MAX_ITEMS = 0


def _set_frame_cache_max_items(max_items: int) -> None:
    global _FRAME_RAM_CACHE_MAX_ITEMS
    max_items = max(0, int(max_items))
    if max_items != _FRAME_RAM_CACHE_MAX_ITEMS:
        _FRAME_RAM_CACHE_MAX_ITEMS = max_items
        _FRAME_RAM_CACHE.clear()


def _read_frame_gray_cached(path: str, cache_max_items: int) -> Optional[np.ndarray]:
    _set_frame_cache_max_items(cache_max_items)
    if _FRAME_RAM_CACHE_MAX_ITEMS > 0:
        cached = _FRAME_RAM_CACHE.get(path)
        if cached is not None:
            _FRAME_RAM_CACHE.move_to_end(path)
            return cached.copy()

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None and _FRAME_RAM_CACHE_MAX_ITEMS > 0:
        _FRAME_RAM_CACHE[path] = img
        if len(_FRAME_RAM_CACHE) > _FRAME_RAM_CACHE_MAX_ITEMS:
            _FRAME_RAM_CACHE.popitem(last=False)
    return img


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


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _get_cgroup_memory_limit_bytes() -> Optional[int]:
    if os.name == "nt":
        return None
    candidates = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read().strip()
            if not raw or raw == "max":
                continue
            limit = int(raw)
            if limit <= 0:
                continue
            if limit >= (1 << 60):
                continue
            return limit
        except (OSError, ValueError):
            continue
    return None


def _get_system_memory_bytes() -> Optional[int]:
    try:
        if os.name == "nt":

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("avail_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended_virtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.length = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_phys)
            return None
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        total_phys = int(page_size * pages)
        cgroup_limit = _get_cgroup_memory_limit_bytes()
        if cgroup_limit is not None:
            return min(total_phys, cgroup_limit)
        return total_phys
    except (AttributeError, ValueError, OSError):
        return None


def _auto_cache_items(
    resolution: Tuple[int, int],
    reserve_ratio: float = 0.30,
    reserve_min_bytes: int = 2 * 1024 * 1024 * 1024,
    flow_ratio: float = 0.40,
    frame_ratio: float = 0.10,
) -> Tuple[int, int]:
    total_mem = _get_system_memory_bytes()
    if total_mem is None or total_mem <= 0:
        return 2000, 5000

    reserve = max(int(total_mem * reserve_ratio), reserve_min_bytes)
    available = max(0, total_mem - reserve)

    width, height = resolution
    bytes_per_flow = max(1, int(width * height * 3 * 4))
    bytes_per_frame = max(1, int(width * height))

    flow_budget = int(available * flow_ratio)
    frame_budget = int(available * frame_ratio)

    flow_items = max(0, flow_budget // bytes_per_flow)
    frame_items = max(0, frame_budget // bytes_per_frame)
    return int(flow_items), int(frame_items)


def _auto_flow_cache_disk(
    data_path: Path,
    reserve_ratio: float = 0.05,
    min_free_bytes: int = 5 * 1024 * 1024 * 1024,
) -> bool:
    try:
        usage = shutil.disk_usage(data_path)
    except OSError:
        return False
    reserve = max(int(usage.total * reserve_ratio), min_free_bytes)
    return usage.free >= reserve


@dataclass(frozen=True)
class SplitConfig:
    train_percent: int
    cutmix_prob: float
    temporal_gap: int = 0


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    flip_prob: float
    brightness_prob: float
    brightness_max: float
    contrast_prob: float
    contrast_max: float
    darkness_prob: float
    darkness_max: float
    noise_prob: float
    noise_std: float


@dataclass(frozen=True)
class FlowCacheConfig:
    disk_enabled: bool
    ram_max_items: int


@dataclass(frozen=True)
class FlowBackendConfig:
    use_raft: bool
    farneback_levels: int = 3
    farneback_winsize: int = 15


def _pair_manifest_path(data_path: Path) -> Path:
    return data_path / "temp_pairs.json"


def _split_cache_path(data_path: Path) -> Path:
    return data_path / "temp_labels" / "train_val_split.json"


def _load_pairs(
    data_path: Path,
) -> List[Tuple[Path, Path, Tuple[Path, Optional[int]]]]:
    manifest_path = _pair_manifest_path(data_path)
    if not manifest_path.exists():
        raise ValueError(f"Pair manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict) or "pairs" not in payload:
        raise ValueError("Invalid pair manifest format.")

    items = payload.get("pairs", [])
    labels_file = payload.get("labels_file") if isinstance(payload, dict) else None
    pairs: List[Tuple[Path, Path, Tuple[Path, Optional[int]]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        frame1 = item.get("frame1")
        frame2 = item.get("frame2")
        label = item.get("label")
        label_index = item.get("label_index")
        label_file = item.get("label_file")
        if not isinstance(frame1, str) or not isinstance(frame2, str):
            continue
        if label_index is None:
            if not isinstance(label, str):
                continue
            label_ref: Tuple[Path, Optional[int]] = (
                (data_path / label).resolve(),
                None,
            )
        else:
            if not isinstance(label_index, int):
                try:
                    label_index = int(label_index)
                except (TypeError, ValueError):
                    continue
            label_file = label_file if isinstance(label_file, str) else labels_file
            if not isinstance(label_file, str):
                continue
            label_ref = ((data_path / label_file).resolve(), int(label_index))
        pairs.append(
            (
                (data_path / frame1).resolve(),
                (data_path / frame2).resolve(),
                label_ref,
            )
        )

    if not pairs:
        raise ValueError("No pairs found in manifest.")

    return pairs


def _normalize_pair(
    frame1_path: Path, frame2_path: Path, label_ref: Tuple[Path, Optional[int]]
) -> Tuple[str, str, str]:
    label_path, label_index = label_ref
    label_key = (
        str(label_path.resolve())
        if label_index is None
        else f"{label_path.resolve()}::{label_index}"
    )
    return (
        str(frame1_path.resolve()),
        str(frame2_path.resolve()),
        label_key,
    )


def _serialize_pairs(
    pairs: Sequence[Tuple[Path, Path, Tuple[Path, Optional[int]]]],
    data_path: Path,
) -> List[Dict[str, str]]:
    serialized: List[Dict[str, str]] = []
    for frame1_path, frame2_path, label_ref in pairs:
        label_path, label_index = label_ref
        try:
            frame1_rel = str(frame1_path.relative_to(data_path))
        except ValueError:
            frame1_rel = str(frame1_path)
        try:
            frame2_rel = str(frame2_path.relative_to(data_path))
        except ValueError:
            frame2_rel = str(frame2_path)
        try:
            label_rel = str(label_path.relative_to(data_path))
        except ValueError:
            label_rel = str(label_path)
        if label_index is None:
            serialized.append(
                {"frame1": frame1_rel, "frame2": frame2_rel, "label": label_rel}
            )
        else:
            serialized.append(
                {
                    "frame1": frame1_rel,
                    "frame2": frame2_rel,
                    "label_index": int(label_index),
                    "label_file": label_rel,
                }
            )
    return serialized


def _deserialize_pairs(
    items: Sequence[dict], data_path: Path
) -> List[Tuple[Path, Path, Tuple[Path, Optional[int]]]]:
    pairs: List[Tuple[Path, Path, Tuple[Path, Optional[int]]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        frame1_rel = item.get("frame1")
        frame2_rel = item.get("frame2")
        label_rel = item.get("label")
        label_index = item.get("label_index")
        label_file = item.get("label_file")
        if not isinstance(frame1_rel, str) or not isinstance(frame2_rel, str):
            continue
        if label_index is None:
            if not isinstance(label_rel, str):
                continue
            label_ref: Tuple[Path, Optional[int]] = (data_path / label_rel, None)
        else:
            if not isinstance(label_index, int):
                try:
                    label_index = int(label_index)
                except (TypeError, ValueError):
                    continue
            if not isinstance(label_file, str):
                continue
            label_ref = (data_path / label_file, int(label_index))
        pairs.append((data_path / frame1_rel, data_path / frame2_rel, label_ref))
    return pairs


def _load_split_cache(
    cache_path: Path,
    data_path: Path,
    available_pairs: Sequence[Tuple[Path, Path, Tuple[Path, Optional[int]]]],
) -> Optional[
    Tuple[
        List[Tuple[Path, Path, Tuple[Path, Optional[int]]]],
        List[Tuple[Path, Path, Tuple[Path, Optional[int]]]],
    ]
]:
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cache_data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(cache_data, dict):
        return None

    train_items = cache_data.get("train_pairs")
    val_items = cache_data.get("val_pairs")
    if not isinstance(train_items, list) or not isinstance(val_items, list):
        return None

    train_pairs = _deserialize_pairs(train_items, data_path)
    val_pairs = _deserialize_pairs(val_items, data_path)

    available_set = {
        _normalize_pair(frame1_path, frame2_path, label_ref)
        for frame1_path, frame2_path, label_ref in available_pairs
    }

    for frame1_path, frame2_path, label_ref in train_pairs + val_pairs:
        label_path, label_index = label_ref
        if (
            not frame1_path.exists()
            or not frame2_path.exists()
            or not label_path.exists()
        ):
            return None
        if label_index is not None and label_index < 0:
            return None
        if _normalize_pair(frame1_path, frame2_path, label_ref) not in available_set:
            return None

    return train_pairs, val_pairs


def _save_split_cache(
    cache_path: Path,
    data_path: Path,
    train_pairs: Sequence[Tuple[Path, Path, Tuple[Path, Optional[int]]]],
    val_pairs: Sequence[Tuple[Path, Path, Tuple[Path, Optional[int]]]],
) -> None:
    payload = {
        "version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_pairs": _serialize_pairs(train_pairs, data_path),
        "val_pairs": _serialize_pairs(val_pairs, data_path),
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except OSError:
        pass


def _split_counts(total: int, fractions: Sequence[float]) -> List[int]:
    if total <= 0:
        return [0 for _ in fractions]

    if not fractions:
        return []

    normalized = [max(0.0, float(value)) for value in fractions]
    total_fraction = sum(normalized)
    if total_fraction <= 0.0:
        return [0 for _ in fractions]

    normalized = [value / total_fraction for value in normalized]
    exact = [value * total for value in normalized]
    counts = [int(value) for value in exact]
    remainder = total - sum(counts)
    if remainder > 0:
        fractional = [value - int(value) for value in exact]
        order = sorted(range(len(fractions)), key=lambda i: fractional[i], reverse=True)
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts


def split_three_val_chunks(
    pairs: Sequence[Tuple[Path, Path, Path]],
    train_percent: int,
    temporal_gap: int = 0,
) -> Tuple[List[Tuple[Path, Path, Path]], List[Tuple[Path, Path, Path]]]:
    if not pairs:
        return [], []

    total = len(pairs)
    train_percent = max(0, min(100, int(train_percent)))
    val_percent = 100 - train_percent

    val_fraction = val_percent / 100.0
    train_fraction = train_percent / 100.0
    fractions = [
        val_fraction / 3.0,
        train_fraction / 2.0,
        val_fraction / 3.0,
        train_fraction / 2.0,
        val_fraction / 3.0,
    ]
    counts = _split_counts(total, fractions)

    boundaries = [0]
    for count in counts:
        boundaries.append(boundaries[-1] + count)

    chunks = [
        list(pairs[boundaries[i] : boundaries[i + 1]]) for i in range(len(counts))
    ]

    if temporal_gap > 0:
        for i in range(len(chunks)):
            trim_start = temporal_gap if i > 0 else 0
            trim_end = temporal_gap if i < len(chunks) - 1 else 0
            end_idx = len(chunks[i]) - trim_end if trim_end > 0 else len(chunks[i])
            if trim_start < end_idx:
                chunks[i] = chunks[i][trim_start:end_idx]
            else:
                chunks[i] = []

    val_pairs = list(chunks[0]) + list(chunks[2]) + list(chunks[4])
    train_pairs = list(chunks[1]) + list(chunks[3])
    return train_pairs, val_pairs


def _pad_to_multiple(
    tensor: torch.Tensor, multiple: int = 8
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    _, _, h, w = tensor.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    pad = (0, pad_w, 0, pad_h)
    return F.pad(tensor, pad, mode="replicate"), pad


def _unpad(tensor: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    _, _, h, w = tensor.shape
    pad_left, pad_right, pad_top, pad_bottom = pad
    return tensor[:, :, pad_top : h - pad_bottom, pad_left : w - pad_right]


def _encode_flow(flow: np.ndarray) -> np.ndarray:
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    scale = float(max(flow.shape[0], flow.shape[1], 1))
    inv_scale = 1.0 / scale
    u = np.clip(flow[..., 0] * inv_scale, -1.0, 1.0)
    v = np.clip(flow[..., 1] * inv_scale, -1.0, 1.0)
    mag = np.clip(mag * inv_scale, 0.0, 1.0)
    encoded = np.stack(((u + 1.0) * 0.5, (v + 1.0) * 0.5, mag), axis=2)
    return encoded.astype(np.float32)


def _flow_to_hsv_rgb(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb.astype(np.float32) / 255.0


def _get_raft_model(device: torch.device):
    global _RAFT_MODEL
    global _RAFT_DEVICE
    if _RAFT_MODEL is not None and _RAFT_DEVICE == device:
        return _RAFT_MODEL

    if device.type != "cuda":
        raise RuntimeError("RAFT is configured but CUDA is not available.")

    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    except Exception as exc:
        raise RuntimeError(
            "RAFT requires torchvision with optical_flow support."
        ) from exc

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=False)
    model = model.to(device)
    model.eval()
    _RAFT_MODEL = model
    _RAFT_DEVICE = device
    return model


def _apply_preflow_augmentations_with_rng(
    img1: np.ndarray,
    img2: np.ndarray,
    config: AugmentationConfig,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng.rand() < config.brightness_prob:
        factor = 1.0 + rng.rand() * max(0.0, config.brightness_max)
        img1 = np.clip(img1.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        img2 = np.clip(img2.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    if rng.rand() < config.contrast_prob:
        delta = (rng.rand() * 2.0 - 1.0) * max(0.0, config.contrast_max)
        factor = 1.0 + delta
        mean = 0.5 * (float(img1.mean()) + float(img2.mean()))
        img1 = np.clip((img1 - mean) * factor + mean, 0, 255).astype(np.uint8)
        img2 = np.clip((img2 - mean) * factor + mean, 0, 255).astype(np.uint8)

    if rng.rand() < config.darkness_prob:
        factor = 1.0 - rng.rand() * max(0.0, config.darkness_max)
        img1 = np.clip(img1.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        img2 = np.clip(img2.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    if rng.rand() < config.noise_prob:
        std = rng.rand() * max(0.0, config.noise_std)
        noise1 = rng.normal(0.0, std, img1.shape)
        noise2 = rng.normal(0.0, std, img2.shape)
        img1 = np.clip(img1.astype(np.float32) + noise1, 0, 255).astype(np.uint8)
        img2 = np.clip(img2.astype(np.float32) + noise2, 0, 255).astype(np.uint8)

    if rng.rand() < config.flip_prob:
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)

    return img1, img2


def _compute_augmented_flow_to_disk(
    args: Tuple[
        str,
        str,
        Tuple[int, int] | None,
        AugmentationConfig,
        int,
        bool,
        bool,
        int,
        str,
        int,
        int,
    ],
) -> Tuple[bool, float, float, float, float, float]:
    (
        frame1_path,
        frame2_path,
        target_size,
        aug_config,
        seed,
        use_rgb_mode,
        use_raft,
        frame_cache_max_items,
        out_path,
        farneback_levels,
        farneback_winsize,
    ) = args
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    t0 = time.perf_counter()
    img1 = _read_frame_gray_cached(frame1_path, frame_cache_max_items)
    img2 = _read_frame_gray_cached(frame2_path, frame_cache_max_items)
    t1 = time.perf_counter()
    if img1 is None or img2 is None:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    img1, img2 = _apply_preflow_augmentations_with_rng(img1, img2, aug_config, rng)
    t2 = time.perf_counter()

    if target_size is not None:
        target_w, target_h = target_size
        height, width = img1.shape[:2]
        if (width, height) != (target_w, target_h):
            interp = (
                cv2.INTER_AREA
                if width > target_w or height > target_h
                else cv2.INTER_LINEAR
            )
            img1 = cv2.resize(img1, (target_w, target_h), interpolation=interp)
            img2 = cv2.resize(img2, (target_w, target_h), interpolation=interp)
    t3 = time.perf_counter()

    if use_raft:
        if not torch.cuda.is_available():
            return False, 0.0, 0.0, 0.0, 0.0, 0.0
        device = torch.device("cuda")
        model = _get_raft_model(device)

        img1_t = torch.from_numpy(img1).float() / 255.0
        img2_t = torch.from_numpy(img2).float() / 255.0
        if img1_t.ndim == 2:
            img1_t = img1_t.unsqueeze(0).repeat(3, 1, 1)
            img2_t = img2_t.unsqueeze(0).repeat(3, 1, 1)

        img1_t = img1_t.unsqueeze(0).to(device)
        img2_t = img2_t.unsqueeze(0).to(device)

        img1_t, pad = _pad_to_multiple(img1_t, multiple=8)
        img2_t, _ = _pad_to_multiple(img2_t, multiple=8)

        with torch.no_grad():
            flow_outputs = model(img1_t, img2_t)
            flow = flow_outputs[-1]

        flow = _unpad(flow, pad)
        flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
    else:
        flow = cv2.calcOpticalFlowFarneback(
            img1,
            img2,
            None,
            0.5,
            farneback_levels,
            farneback_winsize,
            3,
            5,
            1.2,
            0,
        )

    t4 = time.perf_counter()
    rgb = _flow_to_hsv_rgb(flow) if use_rgb_mode else _encode_flow(flow)

    try:
        np.save(out_path, rgb)
        t5 = time.perf_counter()
        return (
            True,
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (t3 - t2) * 1000.0,
            (t4 - t3) * 1000.0,
            (t5 - t4) * 1000.0,
        )
    except Exception:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0


class SpeedEstimationDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        train: bool,
        split_config: SplitConfig,
        augmentation_config: AugmentationConfig,
        cache_config: FlowCacheConfig,
        backend_config: FlowBackendConfig,
        enable_cutmix: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        use_old_split: bool = False,
        frame_cache_max_items: int = 0,
        use_rgb_mode: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.train = train
        self.split_config = split_config
        self.augmentation_config = augmentation_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.enable_cutmix = enable_cutmix
        self.target_size = target_size
        self.use_old_split = use_old_split
        self.use_rgb_mode = bool(use_rgb_mode)
        self.frame_cache_max_items = max(0, int(frame_cache_max_items))

        try:
            cv2.setNumThreads(0)
        except Exception:
            pass

        pairs = _load_pairs(self.data_path)
        split_cache = _split_cache_path(self.data_path)
        cached_split = (
            _load_split_cache(split_cache, self.data_path, pairs)
            if use_old_split
            else None
        )

        if cached_split is not None:
            train_pairs, val_pairs = cached_split
        else:
            train_pairs, val_pairs = split_three_val_chunks(
                pairs,
                split_config.train_percent,
                temporal_gap=split_config.temporal_gap,
            )
            _save_split_cache(split_cache, self.data_path, train_pairs, val_pairs)

        self.pairs = train_pairs if train else val_pairs

        self.flow_cache_dir = self.data_path / "flow_cache"
        if self.cache_config.disk_enabled:
            self.flow_cache_dir.mkdir(parents=True, exist_ok=True)
        self._ram_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._labels_cache: Dict[str, np.ndarray] = {}
        self._epoch_cache_dir = self.data_path / "epoch_flow_cache"
        self._epoch_cache_active = False
        self._epoch_cache_keys: Dict[int, str] = {}
        self._epoch_allow_augmentations = True

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        flow_rgb, label = self._load_flow_and_label(index, apply_augmentations=True)
        frame = self._prepare_frame(flow_rgb)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        if self.train and self.enable_cutmix:
            frame, label_tensor = self._maybe_cutmix(frame, label_tensor, index)
        return frame, label_tensor

    def _load_flow_and_label(
        self, index: int, apply_augmentations: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        frame1_path, frame2_path, label_ref = self.pairs[index]
        label = self._load_label(label_ref)
        if self._epoch_cache_active:
            flow_rgb = self._load_epoch_cached_flow(index, frame1_path, frame2_path)
        else:
            flow_rgb = self._load_flow(
                frame1_path, frame2_path, apply_augmentations=apply_augmentations
            )
        return flow_rgb, label

    def _get_labels_array(self, labels_path: Path) -> np.ndarray:
        key = str(labels_path.resolve())
        cached = self._labels_cache.get(key)
        if cached is not None:
            return cached
        labels = np.load(labels_path, mmap_mode="r")
        self._labels_cache[key] = labels
        return labels

    def _load_label(self, label_ref: Tuple[Path, Optional[int]]) -> np.ndarray:
        label_path, label_index = label_ref
        if label_index is None:
            return np.load(label_path)
        labels = self._get_labels_array(label_path)
        if label_index < 0 or label_index >= len(labels):
            raise ValueError("Label index out of range.")
        return labels[label_index]

    def _load_epoch_cached_flow(
        self, index: int, frame1_path: Path, frame2_path: Path
    ) -> np.ndarray:
        cache_key = self._epoch_cache_keys.get(index)
        if cache_key:
            cache_path = self._epoch_cache_dir / f"{cache_key}.npy"
            if cache_path.exists():
                try:
                    return np.load(cache_path)
                except Exception:
                    pass
        return self._load_flow(frame1_path, frame2_path, apply_augmentations=True)

    def _load_flow(
        self, frame1_path: Path, frame2_path: Path, apply_augmentations: bool
    ) -> np.ndarray:
        img1 = cv2.imread(str(frame1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(frame2_path), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise ValueError("Failed to read frame pair for flow computation.")

        aug_signature: Optional[Tuple[float, ...]] = None
        if (
            self.train
            and apply_augmentations
            and self.augmentation_config.enabled
            and self._epoch_allow_augmentations
        ):
            img1, img2, aug_signature = self._apply_preflow_augmentations(img1, img2)

        img1, img2 = self._resize_for_flow(img1, img2)

        cache_key = self._build_cache_key(frame1_path, frame2_path, aug_signature)
        cached = self._get_from_ram_cache(cache_key)
        if cached is not None:
            return cached

        use_disk = self.cache_config.disk_enabled and aug_signature is None
        if use_disk:
            disk_path = self.flow_cache_dir / f"{cache_key}.npy"
            if disk_path.exists():
                try:
                    flow_rgb = np.load(disk_path)
                    self._store_in_ram_cache(cache_key, flow_rgb)
                    return flow_rgb
                except Exception:
                    pass

        flow_rgb = self._compute_flow_rgb(img1, img2)

        if use_disk:
            try:
                np.save(disk_path, flow_rgb)
            except Exception:
                pass

        self._store_in_ram_cache(cache_key, flow_rgb)
        return flow_rgb

    def _resize_for_flow(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.target_size is None:
            return img1, img2

        target_w, target_h = self.target_size
        height, width = img1.shape[:2]
        if (width, height) == (target_w, target_h):
            return img1, img2

        interp = (
            cv2.INTER_AREA
            if width > target_w or height > target_h
            else cv2.INTER_LINEAR
        )
        img1 = cv2.resize(img1, (target_w, target_h), interpolation=interp)
        img2 = cv2.resize(img2, (target_w, target_h), interpolation=interp)
        return img1, img2

    def _compute_flow_rgb(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        if self.backend_config.use_raft:
            flow = self._compute_flow_rgb_raft(img1, img2)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                img1,
                img2,
                None,
                0.5,
                self.backend_config.farneback_levels,
                self.backend_config.farneback_winsize,
                3,
                5,
                1.2,
                0,
            )

        return _flow_to_hsv_rgb(flow) if self.use_rgb_mode else _encode_flow(flow)

    def _compute_flow_rgb_raft(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        if not torch.cuda.is_available():
            raise RuntimeError("RAFT requested but CUDA is not available.")
        device = torch.device("cuda")
        model = _get_raft_model(device)

        img1_t = torch.from_numpy(img1).float() / 255.0
        img2_t = torch.from_numpy(img2).float() / 255.0
        if img1_t.ndim == 2:
            img1_t = img1_t.unsqueeze(0).repeat(3, 1, 1)
            img2_t = img2_t.unsqueeze(0).repeat(3, 1, 1)

        img1_t = img1_t.unsqueeze(0).to(device)
        img2_t = img2_t.unsqueeze(0).to(device)

        img1_t, pad = _pad_to_multiple(img1_t, multiple=8)
        img2_t, _ = _pad_to_multiple(img2_t, multiple=8)

        with torch.no_grad():
            flow_outputs = model(img1_t, img2_t)
            flow = flow_outputs[-1]

        flow = _unpad(flow, pad)
        flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return flow

    def _apply_preflow_augmentations(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, ...]]]:
        config = self.augmentation_config
        applied: List[float] = []

        if np.random.rand() < config.brightness_prob:
            factor = 1.0 + np.random.rand() * max(0.0, config.brightness_max)
            img1 = np.clip(img1.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            img2 = np.clip(img2.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            applied.append(factor)
        else:
            applied.append(0.0)

        if np.random.rand() < config.contrast_prob:
            delta = (np.random.rand() * 2.0 - 1.0) * max(0.0, config.contrast_max)
            factor = 1.0 + delta
            mean = 0.5 * (float(img1.mean()) + float(img2.mean()))
            img1 = np.clip((img1 - mean) * factor + mean, 0, 255).astype(np.uint8)
            img2 = np.clip((img2 - mean) * factor + mean, 0, 255).astype(np.uint8)
            applied.append(factor)
        else:
            applied.append(0.0)

        if np.random.rand() < config.darkness_prob:
            factor = 1.0 - np.random.rand() * max(0.0, config.darkness_max)
            img1 = np.clip(img1.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            img2 = np.clip(img2.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            applied.append(factor)
        else:
            applied.append(0.0)

        if np.random.rand() < config.noise_prob:
            std = np.random.rand() * max(0.0, config.noise_std)
            noise1 = np.random.normal(0.0, std, img1.shape)
            noise2 = np.random.normal(0.0, std, img2.shape)
            img1 = np.clip(img1.astype(np.float32) + noise1, 0, 255).astype(np.uint8)
            img2 = np.clip(img2.astype(np.float32) + noise2, 0, 255).astype(np.uint8)
            applied.append(std)
        else:
            applied.append(0.0)

        if np.random.rand() < config.flip_prob:
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
            applied.append(1.0)
        else:
            applied.append(0.0)

        if all(value == 0.0 for value in applied):
            return img1, img2, None
        return img1, img2, tuple(applied)

    def _build_cache_key(
        self,
        frame1_path: Path,
        frame2_path: Path,
        aug_signature: Optional[Tuple[float, ...]],
    ) -> str:
        try:
            mtime1 = os.path.getmtime(frame1_path)
        except OSError:
            mtime1 = 0.0
        try:
            mtime2 = os.path.getmtime(frame2_path)
        except OSError:
            mtime2 = 0.0
        mode_tag = "hsv" if self.use_rgb_mode else "raw"
        base = f"{frame1_path}|{frame2_path}|{mtime1}|{mtime2}|{self.target_size}|{mode_tag}"
        if aug_signature is not None:
            base = f"{base}|aug|{aug_signature}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _build_epoch_cache_key(
        self, frame1_path: Path, frame2_path: Path, seed: int
    ) -> str:
        try:
            mtime1 = os.path.getmtime(frame1_path)
        except OSError:
            mtime1 = 0.0
        try:
            mtime2 = os.path.getmtime(frame2_path)
        except OSError:
            mtime2 = 0.0
        mode_tag = "hsv" if self.use_rgb_mode else "raw"
        base = f"{frame1_path}|{frame2_path}|{mtime1}|{mtime2}|{self.target_size}|{seed}|{mode_tag}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def clear_epoch_cache(self) -> None:
        self._epoch_cache_active = False
        self._epoch_cache_keys = {}
        self._epoch_allow_augmentations = False

    def set_epoch_settings(self, allow_augmentations: bool, use_cache: bool) -> None:
        self._epoch_allow_augmentations = allow_augmentations
        self._epoch_cache_active = use_cache

    def prepare_epoch_cache(
        self,
        epoch_seed: int,
        num_workers: int,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not self.train or not self.augmentation_config.enabled:
            self.clear_epoch_cache()
            return

        self._epoch_cache_dir.mkdir(parents=True, exist_ok=True)
        for name in os.listdir(self._epoch_cache_dir):
            if name.lower().endswith(".npy"):
                try:
                    os.remove(self._epoch_cache_dir / name)
                except OSError:
                    pass

        seeds = np.random.RandomState(epoch_seed).randint(
            0, 2**31 - 1, size=len(self.pairs)
        )
        self._epoch_cache_keys = {}

        tasks: List[
            Tuple[
                str,
                str,
                Tuple[int, int] | None,
                AugmentationConfig,
                int,
                bool,
                bool,
                int,
                str,
                int,
                int,
            ]
        ] = []
        for idx, (frame1_path, frame2_path, _) in enumerate(self.pairs):
            seed = int(seeds[idx])
            cache_key = self._build_epoch_cache_key(frame1_path, frame2_path, seed)
            self._epoch_cache_keys[idx] = cache_key
            out_path = str(self._epoch_cache_dir / f"{cache_key}.npy")
            tasks.append(
                (
                    str(frame1_path),
                    str(frame2_path),
                    self.target_size,
                    self.augmentation_config,
                    seed,
                    self.use_rgb_mode,
                    self.backend_config.use_raft,
                    self.frame_cache_max_items,
                    out_path,
                    self.backend_config.farneback_levels,
                    self.backend_config.farneback_winsize,
                )
            )

        errors = 0
        total = len(tasks)
        sum_read = 0.0
        sum_aug = 0.0
        sum_resize = 0.0
        sum_flow = 0.0
        sum_save = 0.0
        t_start = time.perf_counter()

        if self.backend_config.use_raft:
            for idx, task in enumerate(tasks, start=1):
                ok, t_read, t_aug, t_resize, t_flow, t_save = (
                    _compute_augmented_flow_to_disk(task)
                )
                if not ok:
                    errors += 1
                sum_read += t_read
                sum_aug += t_aug
                sum_resize += t_resize
                sum_flow += t_flow
                sum_save += t_save
                if progress_callback and (idx % 50 == 0 or idx == total):
                    progress_callback(int((idx / total) * 100))
                if status_callback and (idx % 200 == 0 or idx == total):
                    avg_total = (time.perf_counter() - t_start) * 1000.0 / idx
                    status_callback(
                        "Precompute interim (avg ms): "
                        f"read={sum_read / idx:.2f}, aug={sum_aug / idx:.2f}, "
                        f"resize={sum_resize / idx:.2f}, flow={sum_flow / idx:.2f}, "
                        f"save={sum_save / idx:.2f}, total={avg_total:.2f}"
                    )
        else:
            num_workers = _resolve_worker_count(num_workers, min_workers=1)
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=num_workers, mp_context=ctx
            ) as executor:
                futures = [
                    executor.submit(_compute_augmented_flow_to_disk, task)
                    for task in tasks
                ]
                completed = 0
                for future in as_completed(futures):
                    try:
                        ok, t_read, t_aug, t_resize, t_flow, t_save = future.result()
                    except Exception:
                        ok, t_read, t_aug, t_resize, t_flow, t_save = (
                            False,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        )
                    if not ok:
                        errors += 1
                    sum_read += t_read
                    sum_aug += t_aug
                    sum_resize += t_resize
                    sum_flow += t_flow
                    sum_save += t_save
                    completed += 1
                    if progress_callback and (
                        completed % 50 == 0 or completed == total
                    ):
                        progress_callback(int((completed / total) * 100))
                    if status_callback and (completed % 200 == 0 or completed == total):
                        avg_total = (time.perf_counter() - t_start) * 1000.0 / completed
                        status_callback(
                            "Precompute interim (avg ms): "
                            f"read={sum_read / completed:.2f}, aug={sum_aug / completed:.2f}, "
                            f"resize={sum_resize / completed:.2f}, flow={sum_flow / completed:.2f}, "
                            f"save={sum_save / completed:.2f}, total={avg_total:.2f}"
                        )

        t_total = (time.perf_counter() - t_start) * 1000.0
        if status_callback and total > 0:
            avg_total = t_total / total
            status_callback(
                "Precompute timings (avg ms): "
                f"read={sum_read / total:.2f}, aug={sum_aug / total:.2f}, "
                f"resize={sum_resize / total:.2f}, flow={sum_flow / total:.2f}, "
                f"save={sum_save / total:.2f}, total={avg_total:.2f}"
            )

        self._epoch_cache_active = True if errors < total else False

    def _get_from_ram_cache(self, cache_key: str) -> Optional[np.ndarray]:
        if cache_key in self._ram_cache:
            value = self._ram_cache.pop(cache_key)
            self._ram_cache[cache_key] = value
            return value
        return None

    def _store_in_ram_cache(self, cache_key: str, value: np.ndarray) -> None:
        if self.cache_config.ram_max_items <= 0:
            return
        self._ram_cache[cache_key] = value
        if len(self._ram_cache) > self.cache_config.ram_max_items:
            self._ram_cache.popitem(last=False)

    def _prepare_frame(self, frame: np.ndarray) -> torch.Tensor:
        original_dtype = frame.dtype
        frame = frame.astype(np.float32)
        if original_dtype == np.uint8 or frame.max() > 1.5:
            frame = frame / 255.0

        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)

        height, width = frame.shape[:2]
        frame = np.transpose(frame, (2, 0, 1))
        tensor = torch.tensor(frame, dtype=torch.float32)

        if self.target_size is not None:
            target_w, target_h = self.target_size
            if (width, height) != (target_w, target_h):
                tensor = tensor.unsqueeze(0)
                tensor = F.interpolate(
                    tensor,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                tensor = tensor.squeeze(0)

        return tensor

    def _maybe_cutmix(
        self, frame: torch.Tensor, label: torch.Tensor, current_index: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split_config.cutmix_prob <= 0:
            return frame, label

        if torch.rand(1).item() > self.split_config.cutmix_prob:
            return frame, label

        if len(self.pairs) < 2:
            return frame, label

        mix_index = int(torch.randint(0, len(self.pairs) - 1, (1,)).item())
        if current_index >= 0 and mix_index >= current_index:
            mix_index += 1
        mix_flow, mix_label = self._load_flow_and_label(
            mix_index, apply_augmentations=True
        )
        mix_frame = self._prepare_frame(mix_flow)
        mix_label = torch.tensor(mix_label, dtype=torch.float32)

        _, height, width = frame.shape
        area = height * width
        target_area = float(torch.empty(1).uniform_(0.10, 0.70).item()) * area
        aspect_ratio = float(torch.empty(1).uniform_(0.5, 2.0).item())

        cut_w = int(round((target_area * aspect_ratio) ** 0.5))
        cut_h = int(round((target_area / aspect_ratio) ** 0.5))
        cut_w = max(1, min(cut_w, width))
        cut_h = max(1, min(cut_h, height))

        y0 = int(torch.randint(0, height - cut_h + 1, (1,)).item())
        x0 = int(torch.randint(0, width - cut_w + 1, (1,)).item())

        frame[:, y0 : y0 + cut_h, x0 : x0 + cut_w] = mix_frame[
            :, y0 : y0 + cut_h, x0 : x0 + cut_w
        ]

        lam = (cut_h * cut_w) / float(area)
        label = label * (1.0 - lam) + mix_label * lam
        return frame, label


def _parse_resolution(resolution_text: str) -> Tuple[int, int]:
    if "x" not in resolution_text:
        raise ValueError("Resolution must be formatted like 224x224")
    parts = resolution_text.split("x")
    return int(parts[0]), int(parts[1])


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def _reduce_label(label: torch.Tensor) -> torch.Tensor:
    if label.ndim == 1:
        return label.view(-1, 1)
    flattened = label.view(label.shape[0], -1)
    return flattened.mean(dim=1, keepdim=True)


def run_training(
    config: dict,
    status_callback: Optional[Callable[[str, str], None]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> bool:
    def emit_status(message: str, level: str = "info") -> None:
        if status_callback is not None:
            status_callback(message, level)

    def emit_progress(value: int) -> None:
        if progress_callback is not None:
            progress_callback(value)

    data_path = Path(config["dataset_path"])
    output_path = Path(config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    train_percent = int(config["train_val_split"])
    cutmix_prob = float(config["cutmix"]) / 100.0
    resolution = _parse_resolution(str(config["resolution"]))

    cutmix_enabled = bool(config.get("cutmix_enabled", True))
    lr_scheduler_enabled = bool(config.get("lr_scheduler_enabled", False))
    lr_scheduler_step = int(config.get("lr_scheduler_step", 0) or 0)
    if lr_scheduler_step < 1:
        lr_scheduler_step = 0

    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    learning_rate = float(config["learning_rate"])
    model_name = str(config["model"])
    early_stopping_patience = int(config.get("early_stopping_patience", 0) or 0)
    if early_stopping_patience < 0:
        early_stopping_patience = 0
    use_best_model_always = bool(config.get("use_best_model_always", False))

    num_workers = _resolve_worker_count(config.get("num_workers", 0), min_workers=1)
    prefetch_factor = int(config.get("prefetch_factor", 2) or 2)
    persistent_workers = bool(config.get("persistent_workers", num_workers > 0))
    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))
    augmentation_epoch_skip_interval = int(
        config.get("augmentation_epoch_skip_interval", 0) or 0
    )
    if augmentation_epoch_skip_interval < 0:
        augmentation_epoch_skip_interval = 0

    weight_decay = float(config.get("weight_decay", 0.0))
    grad_clip_norm = float(config.get("grad_clip_norm", 0.0))
    mixed_precision_value = config.get("mixed_precision", None)
    if mixed_precision_value is None:
        mixed_precision_enabled = torch.cuda.is_available()
    else:
        mixed_precision_enabled = bool(mixed_precision_value)
    temporal_gap = max(0, int(config.get("temporal_gap", 0) or 0))

    seed = config.get("seed", None)
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    split_config = SplitConfig(
        train_percent=train_percent,
        cutmix_prob=cutmix_prob,
        temporal_gap=temporal_gap,
    )

    augmentation_config = AugmentationConfig(
        enabled=bool(config.get("augmentations_enabled", True)),
        flip_prob=float(config.get("aug_flip_prob", 0.3)),
        brightness_prob=float(config.get("aug_brightness_prob", 0.3)),
        brightness_max=float(config.get("aug_brightness_max", 0.2)),
        contrast_prob=float(config.get("aug_contrast_prob", 0.3)),
        contrast_max=float(config.get("aug_contrast_max", 0.2)),
        darkness_prob=float(config.get("aug_darkness_prob", 0.2)),
        darkness_max=float(config.get("aug_darkness_max", 0.3)),
        noise_prob=float(config.get("aug_noise_prob", 0.3)),
        noise_std=float(config.get("aug_noise_std", 5.0)),
    )

    use_rgb_mode = _parse_bool(config.get("use_rgb_mode", False))

    auto_flow_items, auto_frame_items = _auto_cache_items(resolution)
    flow_cache_ram_items = config.get(
        "flow_cache_ram_items", config.get("flow_cache_ram_max_items", None)
    )
    if flow_cache_ram_items is not None:
        auto_flow_items = max(0, int(flow_cache_ram_items))
    frame_cache_items = config.get(
        "frame_cache_items", config.get("frame_cache_max_items", None)
    )
    if frame_cache_items is not None:
        auto_frame_items = max(0, int(frame_cache_items))

    flow_cache_disk_value = config.get("flow_cache_disk", None)
    if flow_cache_disk_value is None:
        flow_cache_disk = _auto_flow_cache_disk(data_path)
    else:
        flow_cache_disk = bool(flow_cache_disk_value)

    cache_config = FlowCacheConfig(
        disk_enabled=flow_cache_disk,
        ram_max_items=auto_flow_items,
    )

    frame_cache_max_items = auto_frame_items

    use_raft = bool(config.get("use_RAFT_for_flow", False))
    backend_config = FlowBackendConfig(
        use_raft=use_raft,
        farneback_levels=int(config.get("farneback_levels", 3) or 3),
        farneback_winsize=int(config.get("farneback_winsize", 15) or 15),
    )
    if backend_config.use_raft and not torch.cuda.is_available():
        raise RuntimeError("use_RAFT_for_flow is true but CUDA is not available.")
    if backend_config.use_raft:
        emit_status(
            "RAFT CUDA debug: "
            f"cuda_available={torch.cuda.is_available()} | "
            f"torch_cuda={torch.version.cuda} | "
            f"cudnn_enabled={torch.backends.cudnn.enabled}",
            "info",
        )
        if torch.cuda.is_available():
            emit_status(
                f"RAFT CUDA device: {torch.cuda.get_device_name(0)}",
                "info",
            )
        num_workers = 0

    use_old_split = bool(config.get("use_old_split", False))

    train_dataset_aug = SpeedEstimationDataset(
        data_path=data_path,
        train=True,
        split_config=split_config,
        augmentation_config=augmentation_config,
        cache_config=cache_config,
        backend_config=backend_config,
        enable_cutmix=cutmix_enabled,
        target_size=resolution,
        use_old_split=use_old_split,
        frame_cache_max_items=frame_cache_max_items,
        use_rgb_mode=use_rgb_mode,
    )
    train_dataset_no_aug = SpeedEstimationDataset(
        data_path=data_path,
        train=True,
        split_config=split_config,
        augmentation_config=AugmentationConfig(
            enabled=False,
            flip_prob=0.0,
            brightness_prob=0.0,
            brightness_max=0.0,
            contrast_prob=0.0,
            contrast_max=0.0,
            darkness_prob=0.0,
            darkness_max=0.0,
            noise_prob=0.0,
            noise_std=0.0,
        ),
        cache_config=cache_config,
        backend_config=backend_config,
        enable_cutmix=cutmix_enabled,
        target_size=resolution,
        use_old_split=use_old_split,
        frame_cache_max_items=frame_cache_max_items,
        use_rgb_mode=use_rgb_mode,
    )
    val_dataset = SpeedEstimationDataset(
        data_path=data_path,
        train=False,
        split_config=split_config,
        augmentation_config=AugmentationConfig(
            enabled=False,
            flip_prob=0.0,
            brightness_prob=0.0,
            brightness_max=0.0,
            contrast_prob=0.0,
            contrast_max=0.0,
            darkness_prob=0.0,
            darkness_max=0.0,
            noise_prob=0.0,
            noise_std=0.0,
        ),
        cache_config=cache_config,
        backend_config=backend_config,
        enable_cutmix=False,
        target_size=resolution,
        use_old_split=use_old_split,
        frame_cache_max_items=frame_cache_max_items,
        use_rgb_mode=use_rgb_mode,
    )

    train_pair_count = len(train_dataset_no_aug)
    val_pair_count = len(val_dataset)
    if train_pair_count <= 0:
        raise ValueError(
            "No training pairs available after split. "
            "Reduce temporal_gap, reduce train_val_split, or provide more data."
        )
    if val_pair_count <= 0:
        raise ValueError(
            "No validation pairs available after split. "
            "Reduce temporal_gap, adjust train_val_split, or provide more data."
        )

    drop_last_train = train_pair_count >= batch_size
    if not drop_last_train:
        emit_status(
            "Train split smaller than batch_size; using drop_last=False "
            f"(train_pairs={train_pair_count}, batch_size={batch_size}).",
            "info",
        )
        print(
            "[Train] Train split smaller than batch_size; using drop_last=False "
            f"(train_pairs={train_pair_count}, batch_size={batch_size})."
        )

    loader_kwargs: Dict[str, object] = {}
    if num_workers > 0:
        loader_kwargs.update(
            {
                "num_workers": num_workers,
                "persistent_workers": persistent_workers,
                "prefetch_factor": max(1, prefetch_factor),
                "pin_memory": pin_memory,
                "worker_init_fn": _worker_init_fn,
            }
        )

    train_loader_workers = DataLoader(
        train_dataset_no_aug,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        **loader_kwargs,
    )
    train_loader_single = DataLoader(
        train_dataset_aug,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=0,
    )
    train_loader_single_no_aug = DataLoader(
        train_dataset_no_aug,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    use_amp = mixed_precision_enabled and device.type == "cuda"
    scaler = _get_grad_scaler(enabled=use_amp)
    scheduler = None
    lr_plateau_epochs = 0
    lr_scheduler_gamma = 0.5
    if lr_scheduler_enabled and lr_scheduler_step > 0:
        scheduler = "plateau"

    total_steps = max(1, epochs * max(1, len(train_loader_workers)))
    step_index = 0

    csv_path = output_path / "training_log.csv"
    epoch_csv_path = output_path / "epoch_training_log.csv"
    model_path = output_path / "best_model.pth"
    plot_path = output_path / "training_loss.png"
    epoch_plot_path = output_path / "training_loss_epochs.png"

    train_losses: List[float] = []
    val_losses: List[float] = []
    epoch_train_losses: List[float] = []
    best_val_loss = float("inf")
    no_improve_epochs = 0

    emit_status("Training started.", "info")
    print("[Train] Training started.")

    with (
        csv_path.open("w", encoding="utf-8") as handle,
        epoch_csv_path.open("w", encoding="utf-8") as epoch_handle,
    ):
        handle.write("batch,train_loss,val_loss,diff\n")
        epoch_handle.write("epoch,train_loss,val_loss,diff,train_rmse,val_rmse\n")

        for epoch in range(epochs):
            current_lr = float(optimizer.param_groups[0]["lr"])
            emit_status(f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.8f}", "info")
            print(f"[Train] Epoch {epoch + 1}/{epochs} | LR: {current_lr:.8f}")
            use_aug_epoch = False
            if augmentation_config.enabled and augmentation_epoch_skip_interval >= 0:
                cycle = augmentation_epoch_skip_interval + 1
                use_aug_epoch = (epoch % cycle) == 0

            if use_aug_epoch:
                emit_status(f"Epoch {epoch + 1}: Precomputing augmented flows.", "info")
                train_dataset_aug.prepare_epoch_cache(
                    epoch_seed=epoch + 1,
                    num_workers=num_workers,
                    progress_callback=lambda p: emit_status(
                        f"Precompute progress: {p}%", "info"
                    ),
                    status_callback=lambda msg: emit_status(msg, "info"),
                )
                train_dataset_aug.set_epoch_settings(
                    allow_augmentations=True, use_cache=True
                )
                train_loader = train_loader_single
            else:
                train_dataset_aug.clear_epoch_cache()
                train_dataset_aug.set_epoch_settings(
                    allow_augmentations=False, use_cache=False
                )
                train_loader = (
                    train_loader_single_no_aug
                    if backend_config.use_raft
                    else train_loader_workers
                )

            model.train()
            epoch_train_loss = 0.0
            epoch_batch_losses: List[float] = []
            epoch_batch_steps: List[int] = []
            status_interval = max(1, len(train_loader) // 10)

            for batch_idx, (frames, labels) in enumerate(train_loader, start=1):
                if cancel_check is not None and cancel_check():
                    emit_status("Training cancelled.", "error")
                    print("[Train] Training cancelled.")
                    return False

                frames = frames.to(device)
                labels = _reduce_label(labels.to(device))

                optimizer.zero_grad()
                with _get_autocast_context(device.type, enabled=use_amp):
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                train_loss = float(loss.item())
                epoch_train_loss += train_loss
                train_losses.append(train_loss)
                epoch_batch_losses.append(train_loss)

                step_index += 1
                epoch_batch_steps.append(step_index)
                progress = int((step_index / total_steps) * 100)
                emit_progress(progress)

                if batch_idx % status_interval == 0 or batch_idx == len(train_loader):
                    emit_status(
                        f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(train_loader)} "
                        f"Train MSE: {train_loss:.6f}",
                        "info",
                    )
                    print(
                        f"[Train] Epoch {epoch + 1}/{epochs} Batch {batch_idx}/{len(train_loader)} "
                        f"Train MSE: {train_loss:.6f}"
                    )

            epoch_train_loss /= max(1, len(train_loader))

            model.eval()
            epoch_val_loss = 0.0
            val_sample_count = 0
            with torch.no_grad():
                for val_frames, val_labels in val_loader:
                    val_frames = val_frames.to(device)
                    val_labels = _reduce_label(val_labels.to(device))
                    with _get_autocast_context(device.type, enabled=use_amp):
                        val_outputs = model(val_frames)
                        val_loss = criterion(val_outputs, val_labels)
                    batch_n = val_frames.size(0)
                    epoch_val_loss += float(val_loss.item()) * batch_n
                    val_sample_count += batch_n

            epoch_val_loss = (
                epoch_val_loss / val_sample_count if val_sample_count > 0 else 0.0
            )

            val_losses.append(epoch_val_loss)
            epoch_train_losses.append(epoch_train_loss)
            diff = (
                abs(epoch_train_loss - epoch_val_loss) if len(val_loader) > 0 else 0.0
            )

            epoch_train_rmse = math.sqrt(epoch_train_loss)
            epoch_val_rmse = math.sqrt(epoch_val_loss) if len(val_loader) > 0 else 0.0

            epoch_handle.write(
                f"{epoch + 1},{epoch_train_loss},{epoch_val_loss},{diff},{epoch_train_rmse},{epoch_val_rmse}\n"
            )

            for batch_step, batch_loss in zip(epoch_batch_steps, epoch_batch_losses):
                diff_batch = (
                    abs(batch_loss - epoch_val_loss) if len(val_loader) > 0 else 0.0
                )
                handle.write(
                    f"{batch_step},{batch_loss},{epoch_val_loss},{diff_batch}\n"
                )

            improved = len(val_loader) > 0 and epoch_val_loss < best_val_loss
            if improved:
                best_val_loss = epoch_val_loss
                no_improve_epochs = 0
                lr_plateau_epochs = 0
                torch.save(model.state_dict(), model_path)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                    },
                    output_path / "best_checkpoint.pth",
                )
                print(f"[Train] New best model saved: {best_val_loss:.6f}")
            elif len(val_loader) > 0:
                no_improve_epochs += 1
                lr_plateau_epochs += 1

            if scheduler == "plateau" and lr_scheduler_step > 0:
                if lr_plateau_epochs >= lr_scheduler_step:
                    lr_before = float(optimizer.param_groups[0]["lr"])
                    lr_after = lr_before * lr_scheduler_gamma
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_after
                    lr_plateau_epochs = 0
                    emit_status(
                        f"LR reduced (no val improvement for {lr_scheduler_step} epoch(s)): "
                        f"{lr_before:.8f} -> {lr_after:.8f}",
                        "info",
                    )
                    print(
                        f"[Train] LR reduced (no val improvement for {lr_scheduler_step} epoch(s)): "
                        f"{lr_before:.8f} -> {lr_after:.8f}"
                    )

            if use_best_model_always and len(val_loader) > 0 and model_path.exists():
                checkpoint_path = output_path / "best_checkpoint.pth"
                try:
                    if checkpoint_path.exists():
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        model.load_state_dict(checkpoint["model_state_dict"])
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        if "scaler_state_dict" in checkpoint:
                            scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    else:
                        model.load_state_dict(
                            torch.load(model_path, map_location=device)
                        )
                    print("[Train] Best model + optimizer reloaded for next epoch.")
                except Exception as exc:
                    emit_status(
                        f"Failed to reload best model: {exc}",
                        "error",
                    )
                    print(f"[Train] Failed to reload best model: {exc}")

            epoch_banner = "=" * 20
            emit_status(
                f"{epoch_banner} Epoch {epoch + 1}/{epochs} Finished {epoch_banner}\n"
                f"Train MSE: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f}",
                "info",
            )
            print(
                f"{epoch_banner} Epoch {epoch + 1}/{epochs} Finished {epoch_banner}\n"
                f"[Train] Train MSE: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f}"
            )

            if (
                len(val_loader) > 0
                and early_stopping_patience > 0
                and no_improve_epochs >= early_stopping_patience
            ):
                emit_status(
                    "Early stopping triggered: "
                    f"no val loss improvement for {no_improve_epochs} epoch(s).",
                    "info",
                )
                print(
                    "[Train] Early stopping triggered: "
                    f"no val loss improvement for {no_improve_epochs} epoch(s)."
                )
                emit_progress(100)
                break

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss (per batch)", alpha=0.6)
        plt.xlabel("Batch")
        plt.ylabel("MSE")
        plt.title("Training Loss (per batch)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        emit_status("Training plot saved.", "info")
        print(f"[Train] Plot saved to {plot_path}")

        plt.figure(figsize=(8, 5))
        plt.plot(epoch_train_losses, label="Train Loss (Epoch)")
        plt.plot(val_losses, label="Val Loss (Epoch)")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Training/Validation Loss per Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(epoch_plot_path)
        plt.close()
        emit_status("Epoch plot saved.", "info")
        print(f"[Train] Epoch plot saved to {epoch_plot_path}")

        if len(val_loader) > 0:
            best_checkpoint_path = output_path / "best_checkpoint.pth"
            if best_checkpoint_path.exists():
                checkpoint = torch.load(best_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
            elif model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=device))

            model.eval()
            gt_values: List[float] = []
            pred_values: List[float] = []
            with torch.no_grad():
                for val_frames, val_labels in val_loader:
                    val_frames = val_frames.to(device)
                    val_labels = _reduce_label(val_labels.to(device))
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        val_outputs = model(val_frames)
                    gt_values.extend(val_labels.squeeze(1).cpu().numpy().tolist())
                    pred_values.extend(val_outputs.squeeze(1).cpu().numpy().tolist())

            if gt_values and pred_values:
                max_val = max(max(gt_values), max(pred_values), 0.0)
                plt.figure(figsize=(6, 6))
                plt.scatter(gt_values, pred_values, s=8, alpha=0.5)
                plt.plot([0.0, max_val], [0.0, max_val], linestyle="--", color="black")
                plt.xlabel("Ground Truth")
                plt.ylabel("Prediction")
                plt.title("Best Model: Prediction vs Ground Truth")
                plt.xlim(0.0, max_val)
                plt.ylim(0.0, max_val)
                plt.tight_layout()
                scatter_path = output_path / "best_val_scatter.png"
                plt.savefig(scatter_path)
                plt.close()
                emit_status("Best-model scatter plot saved.", "info")
                print(f"[Train] Best-model scatter plot saved to {scatter_path}")
    except Exception as exc:
        emit_status(f"Failed to save plot: {exc}", "error")
        print(f"[Train] Failed to save plot: {exc}")

    avg_train_mse = float(np.mean(train_losses)) if train_losses else 0.0
    avg_val_mse = float(np.mean(val_losses)) if val_losses else 0.0
    avg_train_rmse = math.sqrt(avg_train_mse)
    avg_val_rmse = math.sqrt(avg_val_mse)

    if len(val_loader) > 0 and best_val_loss != float("inf"):
        emit_status(
            f"Best Val MSE (best model): {best_val_loss:.6f}",
            "info",
        )
        print(f"[Train] Best Val MSE (best model): {best_val_loss:.6f}")

    emit_status(
        "Training finished. "
        f"Avg Train MSE: {avg_train_mse:.6f} | Avg Train RMSE: {avg_train_rmse:.6f} km/h | "
        f"Avg Val MSE: {avg_val_mse:.6f} | Avg Val RMSE: {avg_val_rmse:.6f} km/h",
        "info",
    )
    print(
        "[Train] Training finished. "
        f"Avg Train MSE: {avg_train_mse:.6f} | Avg Train RMSE: {avg_train_rmse:.6f} km/h | "
        f"Avg Val MSE: {avg_val_mse:.6f} | Avg Val RMSE: {avg_val_rmse:.6f} km/h"
    )
    return True
