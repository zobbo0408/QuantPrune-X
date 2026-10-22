import unittest
import torch
from torch import nn

from pruning.global_magnitude import prune_global_magnitude


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 2, bias=True)
        self.bn = nn.BatchNorm2d(1)
        self.fc = nn.Linear(4, 1, bias=True)

        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor([[[[0.1, -0.2],
                                [0.9, 0.8]]]])
            )
            self.fc.weight.copy_(
                torch.tensor([[0.05, 0.3, 0.7, 0.6]])
            )


class GlobalMagnitudeTest(unittest.TestCase):

    def test_global_pruning(self):
        model = TinyNet()

        pruned, masks, info = prune_global_magnitude(
            model, ratio=0.5
        )

        self.assertEqual(info["eligible_weight_count"], 8)
        self.assertEqual(info["pruned_weight_count"], 4)
        self.assertEqual(info["mask_sparsity"], 0.5)

        torch.testing.assert_close(
            pruned.conv.weight.flatten(),
            torch.tensor([0.0, 0.0, 0.9, 0.8])
        )

        torch.testing.assert_close(
            pruned.fc.weight.flatten(),
            torch.tensor([0.0, 0.0, 0.7, 0.6])
        )

        self.assertEqual(masks["conv.weight"].dtype, torch.bool)

    def test_original_model_unchanged(self):
        model = TinyNet()

        original = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

        pruned, _, _ = prune_global_magnitude(
            model, ratio=0.5, inplace=False
        )

        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name])

        for name in ["conv.bias", "bn.weight", "bn.bias", "fc.bias"]:
            torch.testing.assert_close(
                pruned.state_dict()[name],
                original[name]
            )

    def test_exclude_layer(self):
        model = TinyNet()

        pruned, masks, info = prune_global_magnitude(
            model,
            ratio=0.5,
            exclude_layers=["conv"]
        )

        self.assertNotIn("conv.weight", masks)
        self.assertEqual(info["eligible_weight_count"], 4)

        torch.testing.assert_close(
            pruned.conv.weight,
            model.conv.weight
        )

    def test_invalid_ratio(self):
        for ratio in [-0.1, 1.1, float("nan"), True]:
            with self.assertRaises(ValueError):
                prune_global_magnitude(TinyNet(), ratio)

    def test_tie_reproducibility(self):
        model = TinyNet()

        with torch.no_grad():
            model.conv.weight.fill_(1.0)
            model.fc.weight.fill_(1.0)

        _, mask1, _ = prune_global_magnitude(model, 0.5)
        _, mask2, _ = prune_global_magnitude(model, 0.5)

        for name in mask1:
            self.assertTrue(torch.equal(mask1[name], mask2[name]))


if __name__ == "__main__":
    unittest.main()
