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

If you use this code, pretrained models, or experimental results in your research, please cite our paper and this repository.

@article{saga_detr_2026,
  title   = {SAGA-DETR: Stabilized Query Assignment and Geometry-Aware Boundary Refinement for Small Object Detection in Aerial Imagery},

  author  = {Your Name and Coauthor Name and Coauthor Name},

  journal = {The Visual Computer},

  year    = {2026},
  
  note    = {Manuscript submitted to The Visual Computer}
}