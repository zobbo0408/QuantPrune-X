"""Reserved end-to-end coverage after all owner implementations are merged."""
import unittest


@unittest.skip("requires integrated pruning and experiment implementations")
class IntegrationTest(unittest.TestCase):
    def test_full_matrix_smoke_path(self):
        pass
