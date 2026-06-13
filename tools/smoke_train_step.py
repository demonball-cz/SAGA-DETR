import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils import data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.collate_fn import collate_fn
from util.lazy_load import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single forward/backward smoke step.")
    parser.add_argument("--config-file", default="configs/train_visdrone.py")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    cfg = Config(args.config_file, partials=("lr_scheduler", "optimizer", "param_dicts"))
    model = Config(cfg.model_path).model.to(args.device)
    model.train()
    optimizer = cfg.optimizer(cfg.param_dicts(model))
    loader = data.DataLoader(
        cfg.train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    images, targets = next(iter(loader))
    images = [image.to(args.device) for image in images]
    targets = [
        {k: v.to(args.device) if hasattr(v, "to") else v for k, v in target.items()}
        for target in targets
    ]
    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = args.mixed_precision and args.device == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
        loss_dict = model(images, targets)
        losses = sum(v for k, v in loss_dict.items() if k.startswith("loss"))
    losses.backward()
    optimizer.step()
    print({k: float(v.detach().cpu()) for k, v in loss_dict.items()})
    if torch.cuda.is_available():
        print(f"peak_vram_mb={torch.cuda.max_memory_allocated() / 1024**2:.1f}")


if __name__ == "__main__":
    main()
