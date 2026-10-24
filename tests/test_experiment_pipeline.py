import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.pipeline import write_result
from experiments.result_schema import REQUIRED_RESULT_FIELDS


class ExperimentPipelineTest(unittest.TestCase):
    def test_result_is_written(self):
        row = {key: None for key in REQUIRED_RESULT_FIELDS}
        row.update(processing_order="FP32", clean_accuracy=0.9, asr_all=0.1, asr_non_target=0.0, actual_sparsity=0.0)
        with tempfile.TemporaryDirectory() as directory:
            output = write_result(directory, row)
            self.assertEqual(json.loads(output.read_text())["run_id"], None)
