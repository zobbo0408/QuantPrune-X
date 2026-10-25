from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn


def prune_layerwise_magnitude(
    model: nn.Module,
    ratio: float,
    *,
    exclude_layers=None,
    inplace: bool = False,
):
    """
    Layer-wise Magnitude Pruning.

    각 Conv2d / Linear 레이어의 weight를 독립적으로 평가하여
    각 레이어에서 magnitude가 작은 weight부터 ratio만큼 제거한다.

    Returns:
        pruned_model, masks, metadata
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(ratio)
        or not 0 <= ratio <= 1
    ):
        raise ValueError("ratio must be between 0 and 1")

    excluded = set(exclude_layers or [])

    # 원본 모델 보호
    result = model if inplace else deepcopy(model)

    # 1. pruning 대상 weight 수집
    targets = []

    for name, module in result.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight_name = f"{name}.weight" if name else "weight"

            if (
                name not in excluded
                and weight_name not in excluded
            ):
                targets.append((weight_name, module.weight))

    if not targets:
        raise ValueError("No eligible Conv2d/Linear weights")

    if len({parameter.device for _, parameter in targets}) != 1:
        raise ValueError("Weights must be on the same device")

    masks = {}
    per_layer = {}

    total = 0
    pruned_total = 0
    weight_zeros = 0

    # 2. 각 레이어를 독립적으로 pruning
    with torch.no_grad():
        for name, weight in targets:
            scores = weight.detach().abs().reshape(-1)

            if not torch.isfinite(scores).all().item():
                raise ValueError("Weights must be finite")

            layer_total = scores.numel()

            # 해당 레이어에서 제거할 weight 수
            to_remove = math.floor(layer_total * ratio)

            # 기본값: 모두 유지
            flat_mask = torch.ones(
                layer_total,
                device=scores.device,
                dtype=torch.bool,
            )

            if to_remove:
                # magnitude가 작은 weight부터 제거
                indices = torch.argsort(scores, stable=True)
                flat_mask[indices[:to_remove]] = False

            mask = flat_mask.reshape(weight.shape).clone()

            # 실제 weight에 mask 적용
            weight.mul_(mask)

            masks[name] = mask

            mask_zero_count = int(
                torch.count_nonzero(~mask).item()
            )

            weight_zero_count = int(
                torch.count_nonzero(weight == 0).item()
            )

            per_layer[name] = {
                "numel": layer_total,
                "pruned_weight_count": mask_zero_count,
                "mask_zero_count": mask_zero_count,
                "weight_zero_count": weight_zero_count,
                "target_ratio": float(ratio),
                "sparsity": (
                    mask_zero_count / layer_total
                    if layer_total
                    else 0.0
                ),
            }

            total += layer_total
            pruned_total += mask_zero_count
            weight_zeros += weight_zero_count

    # 3. 전체 통계
    metadata = {
        "method": "layerwise_magnitude",
        "target_ratio": float(ratio),
        "eligible_weight_count": total,
        "pruned_weight_count": pruned_total,
        "mask_sparsity": (
            pruned_total / total
            if total
            else 0.0
        ),
        "weight_sparsity": (
            weight_zeros / total
            if total
            else 0.0
        ),
        "per_layer": per_layer,
    }

    return result, masks, metadata
