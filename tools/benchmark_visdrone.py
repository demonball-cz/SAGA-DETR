import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.lazy_load import Config
from util.utils import load_checkpoint, load_state_dict

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SAGA/Salience-DETR on one image shape.")
    parser.add_argument("--model-config", default="configs/saga_detr/saga_detr_resnet50_visdrone.py")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1333)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--out", default="paper_patch/generated/benchmark.json")
    return parser.parse_args()


def measure_flops(model, image):
    if FlopCountAnalysis is None:
        return {
            "flops": None,
            "gflops": None,
            "flops_status": "fvcore_not_installed",
            "unsupported_ops": {},
        }

    try:
        with torch.inference_mode():
            flops = FlopCountAnalysis(model, ((image,),))
            total_flops = flops.total()
            unsupported_ops = {str(k): int(v) for k, v in flops.unsupported_ops().items()}
        return {
            "flops": int(total_flops),
            "gflops": total_flops / 1e9,
            "flops_status": "ok",
            "unsupported_ops": unsupported_ops,
        }
    except Exception as exc:
        return {
            "flops": None,
            "gflops": None,
            "flops_status": f"failed: {type(exc).__name__}: {exc}",
            "unsupported_ops": {},
        }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Config(args.model_config).model.eval().to(device)
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        load_state_dict(model, checkpoint)
    image = torch.randn(3, args.height, args.width, device=device)
    params = sum(p.numel() for p in model.parameters())
    flops_result = measure_flops(model, image)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model((image,))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.repeat):
            _ = model((image,))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = elapsed / args.repeat * 1000
    result = {
        "params": params,
        "params_m": params / 1e6,
        "latency_ms": latency_ms,
        "fps": 1000.0 / latency_ms,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "shape": [args.height, args.width],
        "warmup": args.warmup,
        "repeat": args.repeat,
        "flops_method": "fvcore.nn.FlopCountAnalysis on one image; unsupported ops are excluded and listed.",
    }
    result.update(flops_result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
