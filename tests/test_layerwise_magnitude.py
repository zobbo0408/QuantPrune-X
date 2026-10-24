"""Reserved tests for the layerwise-pruning branch implementation."""
import unittest


@unittest.skip("Layer-wise Magnitude Pruning implementation belongs to feat/layerwise-pruning")
class LayerwiseMagnitudeContractTest(unittest.TestCase):
    def test_per_layer_ratio_and_metadata(self):
        pass
