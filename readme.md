## Speedestimation-Tool V2

### Cluster Mode Usage
Cluster mode is controlled via `cluster.py`.

1) Adjust `config.json` (mode, dataset type, paths, model, resolution, filters, etc.).
2) Run `cluster.py` (it reads `config.json` by default).

For `mode: "train"`, the training dataset is prepared and training starts.
For `mode: "test"`, the test dataset is prepared and evaluation starts.

Example (PowerShell):

```bash
python cluster.py
```

Optionally override the config path:

```bash
python cluster.py --config path\to\config.json
```

### Config Options (`config.json`)
All options are defined in `config.json`.
Low-level performance settings (worker count, RAM cache sizes, mixed precision, disk cache)
are auto-derived from hardware and are not intended for manual tuning in most cases.

#### Syntax and Required Basics
- `mode`: `"train"` or `"test"`.
- `dataset`: `"A2D2"` or `"KITTI"` (default: `"A2D2"`).
- Paths are strings (Windows style `C:\\...`).
- `resolution` format is always `"WIDTHxHEIGHT"` (for example `"256x256"`).

#### Model Names (exact spelling)
- `efficientnetv2-L`
- `efficientnetv2-S`
- `resnet18`
- `resnet50`
- `simple`

#### Minimal Train Example
```json
{
  "mode": "train",
  "dataset": "A2D2",
  "dataset_path": "C:\\path\\to\\dataset",
  "output_path": "C:\\path\\to\\output",
  "model": "efficientnetv2-S",
  "learning_rate": "0.001",
  "resolution": "256x256",
  "batch_size": "16",
  "epochs": 5
}
```

#### Full Field Overview

**General**
- `mode`: `"train"` or `"test"`.
- `dataset`: `"A2D2"` or `"KITTI"` (default: `"A2D2"`).

**Paths (Train)**
- `dataset_path`: path to training dataset.
  - For `dataset="A2D2"`: contains `frames/` and `canbus/`.
  - For `dataset="KITTI"`: contains `scenes/`.
- `output_path`: output folder for training artifacts.

**Paths (Test)**
- `testdata_path`: path to test dataset.
  - For `dataset="A2D2"`: contains `frames/` and `canbus/`.
  - For `dataset="KITTI"`: contains `scenes/`.
- `test_output_path`: output folder for test artifacts (CSV + video).

**Model / Checkpoint**
- `model`: model name for training.
- `test_model`: model name for test (falls back to `model`).
- `pth_path`: path to `.pth` checkpoint file (test mode).

**Training Parameters**
- `learning_rate`: learning rate.
- `resolution`: target resolution, for example `"256x256"`.
- `batch_size`: batch size.
- `epochs`: number of epochs.
- `early_stopping_patience`: epochs without val loss improvement before stop. `0` disables early stopping.
- `use_best_model_always`: `true`/`false`. If `true`, reload best model after each epoch.
- `weight_decay`: L2 regularization (default `0.0`).
- `grad_clip_norm`: max gradient norm (default `0.0` = disabled).
- `skip_prepare`: `true`/`false`. If `true`, data preparation is skipped.
- `cutmix_enabled`: `true`/`false`.
- `lr_scheduler_enabled`: `true`/`false`.
- `lr_scheduler_step`: scheduler step interval in epochs.
- `cutmix`: CutMix probability in percent (0-100).
- `train_val_split`: training share in percent.
- `gutter`: percent of frame pairs to skip (0-100).
- `temporal_gap`: number of frame pairs removed at split chunk borders (0 disables it).

**Train/Val Split (sequential, validation in 3 chunks)**
- Example with `train_val_split = 70`:
  10% val -> 35% train -> 10% val -> 35% train -> 10% val.
- Validation appears at start, middle, and end of sequence.
- `temporal_gap > 0` helps reduce temporal leakage.

**Training Augmentations (pre-flow)**
- `augmentations_enabled`
- `augmentation_epoch_skip_interval`
- `aug_flip_prob`
- `aug_brightness_prob`, `aug_brightness_max`
- `aug_contrast_prob`, `aug_contrast_max`
- `aug_darkness_prob`, `aug_darkness_max`
- `aug_noise_prob`, `aug_noise_std`

**Optical Flow**
- `use_RAFT_for_flow`: `true` uses RAFT (CUDA), `false` uses Farneback.
- `farneback_levels`: Farneback pyramid levels (default `3`).
- `farneback_winsize`: Farneback window size (default `15`).

RAFT notes:
- Requires `torch` + `torchvision` with optical flow support.
- Weights are loaded automatically on first use.
- CUDA-capable GPU is recommended.

**Dataset Check (optional)**
- `max_frame_dt_ms`: max allowed time delta between frame1 and frame2 (A2D2 only).
- `max_drop_ratio`: max fraction of removed pairs (0.0-1.0).
- `dataset_check_log_every`: logging interval.
- `dataset_check_debug_samples`: number of debug samples.
- `dataset_check_lightweight`: `true` skips image reads.
- `dataset_check_workers`: worker count (`0` or empty = auto).

**Test Filter (test mode only)**
- `filter_enabled`: `true`/`false`.
- `filter_type`: `"ema"` or `"kalman"`.

EMA:
- `ema_alpha`: smoothing factor in `[0,1]`.
- `ema_window`: optional window `N`; if set, alpha is derived as `2/(N+1)`.
Use either `ema_alpha` or `ema_window`, not both.

Kalman:
- `kalman_process_variance`
- `kalman_measurement_variance`
- `kalman_estimate_variance`
- `kalman_initial_estimate` (optional)

### Test CSV Output
The CSV file includes:
- `index`, `frame`, `ground_truth`, `prediction`, `prediction_raw`, `diff`, `mse`

`prediction` is the filtered value (if filter enabled), `prediction_raw` is unfiltered.

### Dataset Check Behavior
After preparation, an automatic dataset check runs.
Pairs may be removed if:
- frames are missing or unreadable
- labels are missing or invalid
- for A2D2: timestamps are missing
- for A2D2: `frame_dt` exceeds `max_frame_dt_ms`
- for A2D2: no CAN-bus value exists in frame interval

The check aborts if removed pairs exceed `max_drop_ratio`.
A marker file is stored so unchanged inputs can skip re-check.

For `dataset="KITTI"`, checks focus on frame/label consistency and readability,
without A2D2 timestamp/CAN-bus rules.

### Dataset Setup (A2D2)
For A2D2, structure the dataset as:

```text
your_dataset/
  frames/
    000001.png
    000001.json
    000002.png
    000002.json
  canbus/
    canbus_000001.json
    canbus_000002.json
```

### Dataset Setup (KITTI)
For KITTI, `dataset_path` must contain `scenes/`.
Under `scenes/`, one or many scene roots are supported.
A scene root is detected when both folders exist:
- `image_02/data` (PNG frames)
- `oxts/data` (TXT OXTS files)

Example:

```text
your_dataset/
  scenes/
    2011_09_26_drive_0060_sync/
      2011_09_26/
        2011_09_26_drive_0060_sync/
          image_02/
            data/
              0000000000.png
              0000000001.png
          oxts/
            data/
              0000000000.txt
              0000000001.txt
```

For each pair `(frame_i, frame_{i+1})`, speed comes from `frame_i`:
- column 9 (index 8), whitespace-separated, in OXTS TXT
- converted from m/s to km/h via `* 3.6`
