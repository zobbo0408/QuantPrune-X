import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.result_schema import REQUIRED_RESULT_FIELDS, validate_result


class ResultSchemaTest(unittest.TestCase):
    def test_missing_required_field_is_rejected(self):
        row = {key: None for key in REQUIRED_RESULT_FIELDS}
        row.update(processing_order="FP32", clean_accuracy=0.9, asr_all=0.1, asr_non_target=0.0, actual_sparsity=0.0)
        row.pop("run_id")
        with self.assertRaises(ValueError):
            validate_result(row)
