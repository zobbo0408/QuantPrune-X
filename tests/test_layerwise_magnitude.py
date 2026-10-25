"""Tests for Layer-wise Magnitude Pruning."""

import unittest
import torch
from torch import nn

from src.pruning.common import prune_model


class LayerwiseMagnitudeContractTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1)
        self.model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 30 * 30, 10),
        )

    def test_per_layer_ratio_and_metadata(self):
        ratio = 0.3

        pruned_model, masks, metadata = prune_model(
            self.model,
            method="layerwise_magnitude",
            ratio=ratio,
        )

        # 1. 반환값과 메타데이터 확인
        self.assertIsInstance(pruned_model, nn.Module)
        self.assertEqual(metadata["method"], "layerwise_magnitude")
        self.assertEqual(metadata["target_ratio"], ratio)

        # 2. 각 레이어에서 제거된 개수가 floor(numel * ratio)인지 확인
        for name in ("0.weight", "3.weight"):
            mask = masks[name]
            total = mask.numel()
            expected_pruned = int(total * ratio)
            actual_pruned = int((~mask.bool()).sum().item())

            self.assertEqual(actual_pruned, expected_pruned)
            self.assertAlmostEqual(
                metadata["per_layer"][name]["sparsity"],
                actual_pruned / total,
                places=6,
            )

        # 3. 전체 제거 개수와 마스크 희소율 확인
        total_weights = sum(mask.numel() for mask in masks.values())
        total_pruned = sum(
            int((~mask.bool()).sum().item())
            for mask in masks.values()
        )

        self.assertEqual(metadata["eligible_weight_count"], total_weights)
        self.assertEqual(metadata["pruned_weight_count"], total_pruned)
        self.assertAlmostEqual(
            metadata["mask_sparsity"],
            total_pruned / total_weights,
            places=6,
        )

    def test_original_model_is_not_modified(self):
        original_weights = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

        prune_model(
            self.model,
            method="layerwise_magnitude",
            ratio=0.3,
            inplace=False,
        )

        for name, param in self.model.named_parameters():
            self.assertTrue(
                torch.equal(param, original_weights[name]),
                f"원본 모델의 {name} 가중치가 변경됨",
            )

    def test_zero_ratio_removes_no_weights(self):
        _, masks, metadata = prune_model(
            self.model,
            method="layerwise_magnitude",
            ratio=0.0,
        )

        self.assertEqual(metadata["pruned_weight_count"], 0)

        for mask in masks.values():
            self.assertTrue(mask.bool().all())


if __name__ == "__main__":
    unittest.main()
