# Master Thesis: Image-Based Vehicle Speed Estimation

End-to-end training and evaluation framework for estimating ego-vehicle speed from monocular image sequences using optical flow and deep neural networks.

![Project Demo](Demo.gif)

## Scientific Motivation
Accurate vehicle speed estimation is a core state variable in automated driving, robotics, and driver-assistance systems. While wheel odometry and CAN-based measurements are standard, vision-based speed estimation is relevant whenever sensor redundancy, degraded CAN quality, or camera-only setups are required.

This project studies the supervised mapping

\[
(\mathbf{I}_{t}, \mathbf{I}_{t+\Delta t}) \rightarrow v_t
\]

where \(\mathbf{I}_{t}\) and \(\mathbf{I}_{t+\Delta t}\) are consecutive frames and \(v_t\) is vehicle speed in km/h.

The current implementation follows a motion-centric approach:
- compute optical flow from frame pairs
- encode flow as model input
- regress speed with a CNN backbone (for example EfficientNetV2 / ResNet)

This decouples low-level motion extraction (optical flow) from high-level regression, enabling controlled experiments across datasets, augmentations, and model families.

## Method Overview
1. Dataset-specific preparation creates frame pairs and speed labels.
2. Labels are stored in a compact `labels.npy` and indexed via `temp_pairs*.json`.
3. Optical flow is computed per pair (Farneback or RAFT).
4. Flow is fed into a regression model trained with MSE loss.
5. Evaluation reports per-frame predictions, CSV metrics, and visualization video.

## Repository Structure
- `cluster.py`: main entrypoint (config loading, data preparation, dataset check, train/test dispatch).
- `train_clean.py`: training pipeline, split logic, flow cache, augmentations, optimization loop.
- `test.py`: inference pipeline, optional temporal filtering, CSV/video export.
- `models.py`: model construction.
- `data_checker.py`: standalone dataset integrity helper.
- `config.json`: central runtime configuration.

## Supported Datasets
- `A2D2`
- `KITTI`

The dataset type is selected via:

```json
"dataset": "A2D2"
```

or

```json
"dataset": "KITTI"
```

## Dataset Layout

### A2D2
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

### KITTI
`dataset_path` must contain `scenes/`. A scene root is detected when both directories exist:
- `image_02/data` (PNG frames)
- `oxts/data` (TXT files)

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

KITTI label extraction in this codebase:
- pair: `(frame_i, frame_{i+1})`
- label source: `frame_i` OXTS file
- speed: column 9 (index 8), whitespace-separated
- unit conversion: `km/h = m/s * 3.6`

## Quick Start

### 1) Configure
Edit `config.json` (mode, dataset, paths, model, resolution, etc.).

Minimal train example:

```json
{
  "mode": "train",
  "dataset": "KITTI",
  "dataset_path": "C:\\path\\to\\dataset",
  "output_path": "C:\\path\\to\\output",
  "model": "efficientnetv2-S",
  "learning_rate": "0.0001",
  "resolution": "256x256",
  "batch_size": "8",
  "epochs": 15
}
```

### 2) Run
```bash
python cluster.py
```

Optional config override:
```bash
python cluster.py --config path\to\config.json
```

## Key Configuration Fields
- `mode`: `train` or `test`
- `dataset`: `A2D2` or `KITTI`
- `dataset_path`, `testdata_path`
- `output_path`, `test_output_path`
- `model`, `test_model`, `pth_path`
- `resolution`, `batch_size`, `epochs`, `learning_rate`
- `train_val_split`, `temporal_gap`, `gutter`
- `use_RAFT_for_flow`, `use_rgb_mode`
- `augmentations_enabled`, `cutmix_enabled`
- test-time filtering: `filter_enabled`, `filter_type`, `ema_*`, `kalman_*`

## Dataset Check
After preparation, an automatic dataset check validates frame/label consistency.
- removes invalid pairs
- aborts if removed ratio exceeds `max_drop_ratio`
- caches verification state via marker files

A2D2 additionally checks timestamp/CAN synchronization constraints.

## Outputs

### Training
- `best_model.pth`
- `best_checkpoint.pth`
- `training_log.csv`
- `epoch_training_log.csv`
- `training_loss.png`
- `training_loss_epochs.png`
- optional: `best_val_scatter.png`

### Testing
- `test_results.csv` (`ground_truth`, filtered and raw prediction, error metrics)
- `test_output.mp4` with prediction overlay

## Available Models
- `efficientnetv2-L`
- `efficientnetv2-S`
- `resnet18`
- `resnet50`
- `simple`

## Notes
- RAFT requires CUDA-enabled PyTorch/TorchVision optical-flow support.
- For small datasets, prefer moderate `batch_size` and conservative `temporal_gap`.
- The pipeline is designed for reproducible experiments and easy extension to additional datasets.
