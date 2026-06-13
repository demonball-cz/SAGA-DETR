import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.bricks.ms_deform_attn import MultiScaleDeformableAttention


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    module = MultiScaleDeformableAttention(
        embed_dim=64,
        num_levels=2,
        num_heads=4,
        num_points=4,
        use_gsda=True,
        use_bbcr=True,
    ).to(device)
    query = torch.randn(2, 8, 64, device=device, requires_grad=True)
    value = torch.randn(2, 20, 64, device=device, requires_grad=True)
    reference_points = torch.rand(2, 8, 2, 4, device=device)
    reference_points[..., 2:] = reference_points[..., 2:].clamp(0.1, 0.5)
    spatial_shapes = torch.tensor([[4, 4], [2, 2]], dtype=torch.long, device=device)
    level_start_index = torch.tensor([0, 16], dtype=torch.long, device=device)
    output = module(
        query=query,
        reference_points=reference_points,
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        key_padding_mask=None,
    )
    loss = output.square().mean()
    loss.backward()
    print(f"output_shape={tuple(output.shape)} loss={float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
