# SAGA-DETR

This repository contains the research code for **SAGA-DETR**, a DETR-style
remote-sensing object detector. The code is derived from Salience-DETR and adds:

- **ACML**: Assignment Certificate Margin Loss for assignment-margin supervision.
- **GSDA**: Geometry-aware Structured Deformable Attention.
- **BBCR**: Boundary-Band Contrast Refinement.

The release is prepared as a clean code repository for training and evaluating
SAGA-DETR on VisDrone2019-DET style object detection datasets.

## Attribution

This project is derived from
[Salience-DETR](https://github.com/xiuqhou/Salience-DETR), which is licensed
under Apache License 2.0. The original license is preserved in `LICENSE`, and
additional attribution is provided in `NOTICE`.

## Installation

Create and activate your environment, then install dependencies with pip:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements_server.txt
```

If your platform needs a specific PyTorch/CUDA build, install PyTorch first from
the official PyTorch instructions, then install the remaining requirements.

## Data Preparation

Convert VisDrone2019-DET to COCO format:

```bash
python tools/convert_visdrone_to_coco.py \
  --train-root /path/to/VisDrone2019-DET-train \
  --val-root /path/to/VisDrone2019-DET-val \
  --out-dir data/visdrone_coco
```

Expected layout:

```text
data/visdrone_coco/
  annotations/
    instances_train2019.json
    instances_val2019.json
  train2017/
  val2017/
```

The image folders can contain copied images or symlinks.

## Pretrained Checkpoints

We provide the pretrained checkpoint of SAGA-DETR trained on VisDrone2019-DET for 12 epochs. This checkpoint corresponds to the experimental setting reported in our manuscript submitted to *The Visual Computer*.

| Dataset | Backbone | Epochs | AP | AP50 | Params | GFLOPs | Checkpoint |
|---|---|---:|---:|---:|---:|---:|---|
| VisDrone2019-DET | ResNet-50 | 12 | 31.3 | 51.9 | 56.8M | 207.4 | [Baidu Netdisk](https://pan.baidu.com/s/1AeVtz7HxbGdF7yc9lcFatA?pwd=2333) |

Baidu Netdisk extraction code: `2333`

Please download the checkpoint and place it under the `checkpoints/` directory:

```bash
mkdir -p checkpoints
```

Expected checkpoint layout:

```text
checkpoints/
  saga_detr_r50_visdrone2019_det_12e.pth
```

> Note: The Baidu Netdisk link is provided for convenient checkpoint download. A permanent archived release with DOI will be provided through Zenodo when available.

## Training

SAGA-DETR base setting on VisDrone, without ACML/GSDA/BBCR:

```bash
python -m accelerate.commands.launch main.py \
  --config-file configs/train_visdrone.py \
  --mixed-precision fp16 \
  --accumulate-steps 1
```

Full SAGA-DETR:

```bash
export SAGA_USE_ACML=1
export SAGA_USE_GSDA=1
export SAGA_USE_BBCR=1
export SAGA_OUTPUT_DIR=runs/visdrone/full
python -m accelerate.commands.launch main.py \
  --config-file configs/train_visdrone.py \
  --mixed-precision fp16 \
  --accumulate-steps 1
```

## Utility Scripts

Benchmark Params/GFLOPs/latency for a SAGA-DETR checkpoint:

```bash
python tools/benchmark_visdrone.py \
  --checkpoint /path/to/best_ap.pth \
  --out runs/visdrone/benchmark.json
```

Run a small model smoke test:

```bash
python tools/smoke_attention_modules.py
python tools/smoke_train_step.py
```

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.



