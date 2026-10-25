from copy import deepcopy
import math

import torch
from torch import nn


def prune_global_magnitude(
    model,
    ratio,
    exclude_layers=None,
    inplace=False,
):
    """
    Global Magnitude Pruning.

    Conv2d / Linear의 전체 weight에서
    절댓값이 작은 가중치부터 제거한다.

    Returns:
        pruned_model, masks, metadata
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(ratio)
        or not 0 <= ratio < 1
    ):
        raise ValueError("ratio must be between 0 and 1")

    excluded = set(exclude_layers or [])

    # 원본 모델 보호
    result = model if inplace else deepcopy(model)

    # 1. 프루닝 대상 weight 수집
    targets = []

    for name, module in result.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight_name = (
                f"{name}.weight" if name else "weight"
            )

            if (
                name not in excluded
                and weight_name not in excluded
            ):
                targets.append((weight_name, module.weight))

    if not targets:
        raise ValueError("No eligible Conv2d/Linear weights")

    if len({p.device for _, p in targets}) != 1:
        raise ValueError("Weights must be on the same device")

    # 2. 모든 대상 weight의 절댓값 계산
    scores = torch.cat([
        p.detach().abs().reshape(-1)
        for _, p in targets
    ])

    if not torch.isfinite(scores).all().item():
        raise ValueError("Weights must be finite")

    total = scores.numel()

    # 3. 제거할 weight 개수 결정
    to_remove = math.floor(total * ratio)

    # 4. Global Mask 생성
    flat_mask = torch.ones(
        total,
        device=scores.device,
        dtype=torch.bool,
    )

    if to_remove:
        # 동일한 magnitude에서는 기존 순서 유지
        indices = torch.argsort(scores, stable=True)
        flat_mask[indices[:to_remove]] = False

    # 5. 레이어별 마스크 적용
    masks = {}
    per_layer = {}
    weight_zeros = 0
    offset = 0

    with torch.no_grad():
        for name, weight in targets:
            n = weight.numel()

            mask = flat_mask[
                offset:offset + n
            ].reshape(weight.shape).clone()

            offset += n

            weight.mul_(mask)
            masks[name] = mask

            zeros = int(
                torch.count_nonzero(weight == 0).item()
            )
            weight_zeros += zeros

            per_layer[name] = {
                "numel": n,
                "mask_zero_count": int(
                    torch.count_nonzero(~mask).item()
                ),
                "weight_zero_count": zeros,
            }

    # 6. 통계 반환
    metadata = {
        "method": "global_magnitude",
        "target_ratio": float(ratio),
        "eligible_weight_count": total,
        "pruned_weight_count": to_remove,
        "mask_sparsity": to_remove / total,
        "weight_sparsity": weight_zeros / total,
        "per_layer": per_layer,
    }

    return result, masks, metadata
