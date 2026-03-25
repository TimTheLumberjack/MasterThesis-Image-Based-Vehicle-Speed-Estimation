from __future__ import annotations

from pathlib import Path
import math
import hashlib
import json
import os
import shutil
from typing import Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
import ctypes

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from models import create_model


_RAFT_MODEL = None
_RAFT_DEVICE: Optional[torch.device] = None


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


def _parse_resolution(resolution_text: str) -> Tuple[int, int]:
    if "x" not in resolution_text:
        raise ValueError("Resolution must be formatted like 224x224")
    parts = resolution_text.split("x")
    return int(parts[0]), int(parts[1])


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


def _auto_flow_cache_items(
    resolution: Tuple[int, int],
    reserve_ratio: float = 0.30,
    reserve_min_bytes: int = 2 * 1024 * 1024 * 1024,
    flow_ratio: float = 0.40,
) -> int:
    total_mem = _get_system_memory_bytes()
    if total_mem is None or total_mem <= 0:
        return 2000

    reserve = max(int(total_mem * reserve_ratio), reserve_min_bytes)
    available = max(0, total_mem - reserve)

    width, height = resolution
    bytes_per_flow = max(1, int(width * height * 3 * 4))
    flow_budget = int(available * flow_ratio)
    flow_items = max(0, flow_budget // bytes_per_flow)
    return int(flow_items)


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


def _pair_manifest_path(data_path: Path) -> Path:
    return data_path / "temp_pairs_test.json"


def _load_pairs(
    data_path: Path,
) -> List[Tuple[Path, Path, Tuple[Path, Optional[int]]]]:
    manifest_path = _pair_manifest_path(data_path)
    if not manifest_path.exists():
        raise ValueError("Pair manifest for test data not found.")

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
        frame1_path = (data_path / frame1).resolve()
        frame2_path = (data_path / frame2).resolve()
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
        pairs.append((frame1_path, frame2_path, label_ref))

    if not pairs:
        raise ValueError("No pairs found in manifest.")
    return pairs


def _get_labels_array(labels_path: Path, cache: Dict[str, np.ndarray]) -> np.ndarray:
    key = str(labels_path.resolve())
    cached = cache.get(key)
    if cached is not None:
        return cached
    labels = np.load(labels_path, mmap_mode="r")
    cache[key] = labels
    return labels


def _load_label(
    label_ref: Tuple[Path, Optional[int]], cache: Dict[str, np.ndarray]
) -> np.ndarray:
    label_path, label_index = label_ref
    if label_index is None:
        return np.load(label_path)
    labels = _get_labels_array(label_path, cache)
    if label_index < 0 or label_index >= len(labels):
        raise ValueError("Label index out of range.")
    return labels[label_index]


def _resize_for_flow(
    img1: np.ndarray, img2: np.ndarray, target_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    target_w, target_h = target_size
    height, width = img1.shape[:2]
    if (width, height) == (target_w, target_h):
        return img1, img2
    interp = (
        cv2.INTER_AREA if width > target_w or height > target_h else cv2.INTER_LINEAR
    )
    img1 = cv2.resize(img1, (target_w, target_h), interpolation=interp)
    img2 = cv2.resize(img2, (target_w, target_h), interpolation=interp)
    return img1, img2


def _compute_flow(
    img1: np.ndarray,
    img2: np.ndarray,
    use_raft: bool,
    farneback_levels: int = 3,
    farneback_winsize: int = 15,
) -> np.ndarray:
    if use_raft:
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
    return flow


def _build_cache_key(
    frame1_path: Path,
    frame2_path: Path,
    target_size: Tuple[int, int],
    use_rgb_mode: bool,
) -> str:
    try:
        mtime1 = os.path.getmtime(frame1_path)
    except OSError:
        mtime1 = 0.0
    try:
        mtime2 = os.path.getmtime(frame2_path)
    except OSError:
        mtime2 = 0.0
    mode_tag = "hsv" if use_rgb_mode else "raw"
    base = f"{frame1_path}|{frame2_path}|{mtime1}|{mtime2}|{target_size}|{mode_tag}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _prepare_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
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
    if (width, height) != target_size:
        target_w, target_h = target_size
        tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
        tensor = F.interpolate(
            tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return tensor.squeeze(0)

    return torch.tensor(frame, dtype=torch.float32)


class _EmaFilter:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self._initialized = False
        self._value = 0.0

    def update(self, value: float) -> float:
        if not self._initialized:
            self._value = float(value)
            self._initialized = True
        else:
            self._value = (self.alpha * float(value)) + (
                (1.0 - self.alpha) * self._value
            )
        return self._value


class _KalmanFilter:
    def __init__(
        self,
        process_variance: float,
        measurement_variance: float,
        estimate_variance: float,
        initial_estimate: Optional[float] = None,
    ) -> None:
        self.q = float(process_variance)
        self.r = float(measurement_variance)
        self.p = float(estimate_variance)
        self.x = float(initial_estimate) if initial_estimate is not None else 0.0
        self._initialized = initial_estimate is not None

    def update(self, measurement: float) -> float:
        if not self._initialized:
            self.x = float(measurement)
            self._initialized = True
            return self.x

        # Predict
        self.p = self.p + self.q

        # Update
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (float(measurement) - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _build_filter(config: dict) -> Optional[object]:
    enabled = _parse_bool(config.get("filter_enabled", False))
    if not enabled:
        return None

    filter_type = str(config.get("filter_type", "ema")).lower()

    if filter_type == "ema":
        alpha = config.get("ema_alpha")
        window = config.get("ema_window")

        if alpha is not None and window is not None:
            raise ValueError(
                "EMA config invalid: set either ema_alpha OR ema_window, not both."
            )

        if alpha is None:
            if window is not None:
                try:
                    window = max(1, int(window))
                    alpha = 2.0 / (window + 1.0)
                except (TypeError, ValueError):
                    alpha = 0.2
            else:
                alpha = 0.2
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            alpha = 0.2
        alpha = max(0.0, min(1.0, alpha))
        return _EmaFilter(alpha=alpha)

    if filter_type == "kalman":
        process_variance = float(config.get("kalman_process_variance", 1e-3))
        measurement_variance = float(config.get("kalman_measurement_variance", 1e-2))
        estimate_variance = float(config.get("kalman_estimate_variance", 1.0))
        if process_variance <= 0 or measurement_variance <= 0 or estimate_variance <= 0:
            raise ValueError("Kalman config invalid: variances must be > 0.")
        initial_estimate = config.get("kalman_initial_estimate")
        try:
            initial_estimate = (
                float(initial_estimate) if initial_estimate is not None else None
            )
        except (TypeError, ValueError):
            initial_estimate = None
        return _KalmanFilter(
            process_variance=process_variance,
            measurement_variance=measurement_variance,
            estimate_variance=estimate_variance,
            initial_estimate=initial_estimate,
        )

    return None


def run_test(
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

    testdata_path = Path(config["testdata_path"])
    output_path = Path(config["test_output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = str(config.get("test_model") or config.get("model"))
    pth_path = Path(config["pth_path"])
    resolution = _parse_resolution(str(config["resolution"]))

    use_rgb_mode = _parse_bool(config.get("use_rgb_mode", False))

    smoothing_filter = _build_filter(config)
    use_raft = bool(config.get("use_RAFT_for_flow", False))
    farneback_levels = int(config.get("farneback_levels", 3) or 3)
    farneback_winsize = int(config.get("farneback_winsize", 15) or 15)
    if use_raft and not torch.cuda.is_available():
        raise RuntimeError("use_RAFT_for_flow is true but CUDA is not available.")
    if use_raft:
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
    filter_type = str(config.get("filter_type", "ema")).lower()

    try:
        pairs = _load_pairs(testdata_path)
    except Exception as exc:
        emit_status(f"Test data not prepared: {exc}", "error")
        return False

    cache_ram = _auto_flow_cache_items(resolution)
    flow_cache_disk_value = config.get("flow_cache_disk", None)
    if flow_cache_disk_value is None:
        cache_disk = _auto_flow_cache_disk(testdata_path)
    else:
        cache_disk = bool(flow_cache_disk_value)
    flow_cache_dir = testdata_path / "flow_cache_test"
    if cache_disk:
        flow_cache_dir.mkdir(parents=True, exist_ok=True)

    ram_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get_from_ram(cache_key: str) -> Optional[np.ndarray]:
        if cache_key in ram_cache:
            value = ram_cache.pop(cache_key)
            ram_cache[cache_key] = value
            return value
        return None

    def store_in_ram(cache_key: str, value: np.ndarray) -> None:
        if cache_ram <= 0:
            return
        ram_cache[cache_key] = value
        if len(ram_cache) > cache_ram:
            ram_cache.popitem(last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    video_path = output_path / "test_output.mp4"
    csv_path = output_path / "test_results.csv"

    sample_frame = cv2.imread(str(pairs[0][1]), cv2.IMREAD_COLOR)
    if sample_frame is None:
        emit_status("Failed to read sample frame for video output.", "error")
        return False
    target_size = (sample_frame.shape[1], sample_frame.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, target_size, True)

    emit_status("Testing started.", "info")
    print("[Test] Testing started.")

    total_mse_filtered = 0.0
    total_mse_raw = 0.0
    total_count = 0

    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("index,frame,ground_truth,prediction,prediction_raw,diff,mse\n")

        label_cache: Dict[str, np.ndarray] = {}
        for idx, (frame1_path, frame2_path, label_ref) in enumerate(pairs, start=1):
            if cancel_check is not None and cancel_check():
                emit_status("Testing cancelled.", "error")
                writer.release()
                return False

            label = _load_label(label_ref, label_cache)
            gt_speed = float(np.mean(label))

            display_source = cv2.imread(str(frame2_path), cv2.IMREAD_COLOR)
            if display_source is None:
                emit_status("Failed to read display frame.", "error")
                writer.release()
                return False

            cache_key = _build_cache_key(
                frame1_path, frame2_path, resolution, use_rgb_mode
            )
            flow_image = get_from_ram(cache_key)
            if flow_image is None and cache_disk:
                disk_path = flow_cache_dir / f"{cache_key}.npy"
                if disk_path.exists():
                    try:
                        flow_image = np.load(disk_path)
                    except Exception:
                        flow_image = None

            if flow_image is None:
                img1 = cv2.imread(str(frame1_path), cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(str(frame2_path), cv2.IMREAD_GRAYSCALE)
                if img1 is None or img2 is None:
                    emit_status("Failed to read test frame pair.", "error")
                    writer.release()
                    return False
                img1, img2 = _resize_for_flow(img1, img2, resolution)
                flow = _compute_flow(
                    img1, img2, use_raft, farneback_levels, farneback_winsize
                )
                flow_image = (
                    _flow_to_hsv_rgb(flow) if use_rgb_mode else _encode_flow(flow)
                )
                if cache_disk:
                    try:
                        np.save(flow_cache_dir / f"{cache_key}.npy", flow_image)
                    except Exception:
                        pass
            store_in_ram(cache_key, flow_image)
            input_frame = flow_image

            input_tensor = (
                _prepare_frame(input_frame, resolution).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                pred_raw = float(model(input_tensor).squeeze().item())

            pred = pred_raw
            if smoothing_filter is not None:
                pred = float(smoothing_filter.update(pred))

            diff = float(pred - gt_speed)
            mse = float(diff * diff)
            raw_diff = float(pred_raw - gt_speed)
            raw_mse = float(raw_diff * raw_diff)
            total_mse_filtered += mse
            total_mse_raw += raw_mse
            total_count += 1

            display_frame = display_source
            if (display_frame.shape[1], display_frame.shape[0]) != target_size:
                display_frame = cv2.resize(
                    display_frame, target_size, interpolation=cv2.INTER_AREA
                )

            cv2.putText(
                display_frame,
                f"Pred: {pred:.2f} km/h",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"GT: {gt_speed:.2f} km/h",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            writer.write(display_frame)
            handle.write(
                f"{idx},{frame1_path.name},{gt_speed:.6f},{pred:.6f},{pred_raw:.6f},{diff:.6f},{mse:.6f}\n"
            )

            emit_progress(int((idx / len(pairs)) * 100))
            emit_status(
                f"Testing {idx}/{len(pairs)} | Pred: {pred:.2f} | GT: {gt_speed:.2f}",
                "info",
            )
            print(f"[Test] {idx}/{len(pairs)} Pred: {pred:.2f} GT: {gt_speed:.2f}")

    writer.release()
    avg_mse_filtered = total_mse_filtered / total_count if total_count > 0 else 0.0
    avg_mse_raw = total_mse_raw / total_count if total_count > 0 else 0.0
    avg_rmse_filtered = math.sqrt(avg_mse_filtered)
    avg_rmse_raw = math.sqrt(avg_mse_raw)

    filter_used = smoothing_filter is not None
    filter_label = filter_type if filter_used else "none"

    emit_status(
        "Testing finished. "
        f"Filter: {filter_label} | "
        f"Avg Test MSE (filtered): {avg_mse_filtered:.6f} | Avg Test RMSE (filtered): {avg_rmse_filtered:.6f} km/h | "
        f"Avg Test MSE (raw): {avg_mse_raw:.6f} | Avg Test RMSE (raw): {avg_rmse_raw:.6f} km/h",
        "info",
    )
    print(
        "[Test] Testing finished. "
        f"Filter: {filter_label} | "
        f"Avg Test MSE (filtered): {avg_mse_filtered:.6f} | Avg Test RMSE (filtered): {avg_rmse_filtered:.6f} km/h | "
        f"Avg Test MSE (raw): {avg_mse_raw:.6f} | Avg Test RMSE (raw): {avg_rmse_raw:.6f} km/h"
    )
    return True
