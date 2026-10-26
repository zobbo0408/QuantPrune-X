"""Tests for independent checkpoint evaluation and verification commands."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import torch

from checkpoints import sha256
import evaluate_quantized_checkpoints as evaluation_cli
import verify_quantized_checkpoints as verification_cli
from evaluation import quantized_checkpoints as analysis
from evaluation import quantized_verification as verification
from evaluation.runtime import seed_everything
from quantization import checkpoint_export
from quantization.materialize_qdq import quantize
from qdq_test_support import make_test_set_loader, toy_model, with_checkpoint_source


ROOT = Path(__file__).resolve().parents[1]


class QuantizedAnalysisCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "fp32.pt"
        source_model = toy_model()
        torch.save(source_model.state_dict(), self.source)
        self.checkpoints = []
        for bits in (4, 8):
            model, metadata = quantize(copy.deepcopy(source_model), bits)
            metadata = with_checkpoint_source(metadata, model_name="resnet18")
            metadata.update(source_checkpoint=str(self.source), source_sha256=sha256(self.source))
            path = self.root / f"int{bits}.pt"
            checkpoint_export.save_quantized_checkpoint(path, model, metadata)
            self.checkpoints.append(path)

    def config(self, command="evaluation", *, settings='', checkpoints=None):
        path = self.root / f"{command}.toml"
        inputs = self.checkpoints if checkpoints is None else checkpoints
        path.write_text(f'checkpoints = {json.dumps([str(item) for item in inputs])}\n'
                        f'output = "{self.root / command}"\nrun_id = "contract"\n'
                        '[evaluation]\ndevice = "cpu"\n' + settings)
        return path

    @contextmanager
    def dependencies(self):
        with patch.object(analysis, "ROOT", self.root), \
             patch("checkpoints.create_model", side_effect=lambda name: toy_model()), \
             patch.object(checkpoint_export, "create_model", side_effect=lambda name: toy_model()), \
             patch.object(analysis, "test_loader", return_value=make_test_set_loader()) as data:
            yield data

    def invoke(self, cli, config, *options):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                cli.main([str(config), *map(str, options)])
            except SystemExit as exc:
                return exc.code
        return 0

    def reports(self, command):
        return [json.loads(path.read_text()) for path in sorted((self.root / command).glob('**/results.json'))]

    def test_separate_tomls_hold_evaluation_conditions(self):
        for command in ("evaluate", "verify"):
            config = tomllib.loads((ROOT / f"configs/quantization/{command}.toml").read_text())
            self.assertNotIn("quantization", config)
            self.assertEqual(config["checkpoints"], [])
            for key, value in dict(seed=225, batch_size=128, workers=0, normalize=True, target_class=0).items():
                self.assertEqual(config["evaluation"][key], value)

    def test_evaluation_reads_only_exported_checkpoints(self):
        self.source.unlink()
        config = self.config()
        before = {path: path.read_bytes() for path in self.checkpoints}
        with self.dependencies(), patch.object(verification, "quantize", side_effect=AssertionError("evaluation must not quantize")):
            self.assertEqual(self.invoke(evaluation_cli, config), 0)
        reports = self.reports("evaluation")
        self.assertEqual(len(reports), 2)
        for report in reports:
            self.assertEqual(report["operation"], "evaluation")
            self.assertNotIn("verification", report)
            self.assertEqual([row["stage"] for row in report["results"]], ["quantized"])
            self.assertEqual(report["results"][0]["samples"], 5)
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_evaluation_settings_and_seed_are_forwarded(self):
        config = self.config(settings='seed = 731\nbatch_size = 3\nworkers = 2\nnormalize = false\ntarget_class = 2\n')
        with self.dependencies() as data, patch.object(analysis, "seed_everything", wraps=seed_everything) as seeding:
            self.assertEqual(self.invoke(evaluation_cli, config), 0)
            self.assertEqual(data.call_args.kwargs["batch_size"], 3)
            self.assertEqual(data.call_args.kwargs["workers"], 2)
            self.assertFalse(data.call_args.kwargs["normalize"])
            self.assertEqual(data.call_args.kwargs["target"], 2)
            self.assertTrue(any(call.args == (731,) for call in seeding.call_args_list))
        self.assertEqual(self.reports("evaluation")[0]["conditions"]["seed"], 731)

    def test_checkpoint_override_and_sample_limit(self):
        config = self.config(settings='limit = 3\nbatch_size = 2\n', checkpoints=[])
        with self.dependencies():
            self.assertEqual(self.invoke(evaluation_cli, config, "--checkpoints", self.checkpoints[0]), 0)
        reports = self.reports("evaluation")
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["results"][0]["samples"], 3)

    def test_verification_compares_three_stages_without_rewriting_exports(self):
        config = self.config("verification")
        before = {path: sha256(path) for path in self.checkpoints}
        with self.dependencies():
            self.assertEqual(self.invoke(verification_cli, config), 0)
        reports = self.reports("verification")
        self.assertEqual(len(reports), 2)
        for report in reports:
            self.assertEqual(report["operation"], "verification")
            self.assertEqual([row["stage"] for row in report["results"]], ["runtime", "materialized", "restored"])
            self.assertTrue(report["verification"]["state_equal_before_forward"])
            self.assertTrue(report["verification"]["stable_state_equal_after_forward"])
            rows = report["results"]
            for key in ("clean_correct", "clean_denominator", "asr_all_successes", "asr_all_denominator",
                        "asr_non_target_successes", "asr_non_target_denominator"):
                self.assertTrue(all(row[key] == rows[0][key] for row in rows))
        self.assertEqual({path: sha256(path) for path in self.checkpoints}, before)

    def test_verification_respects_saved_activation_controls(self):
        model, metadata = quantize(toy_model(), 4)
        model.conv.configure_quantization(8, weight=False, activation=True)
        model.fc.quantize_activation = False
        metadata = with_checkpoint_source(metadata, model_name="resnet18")
        metadata.update(source_checkpoint=str(self.source), source_sha256=sha256(self.source))
        checkpoint_export.save_quantized_checkpoint(self.checkpoints[0], model, metadata)
        config = self.config("verification", checkpoints=self.checkpoints[:1])
        with self.dependencies():
            self.assertEqual(self.invoke(verification_cli, config), 0)

    def test_missing_or_mismatched_reference_source_fails_before_data(self):
        config = self.config("verification")
        self.source.write_bytes(b"wrong source")
        with self.dependencies() as data:
            self.assertNotEqual(self.invoke(verification_cli, config), 0)
            data.assert_not_called()
        self.assertFalse((self.root / "verification").exists())

    def test_logit_difference_with_identical_predictions_is_rejected(self):
        config = self.config("verification", checkpoints=self.checkpoints[:1])
        loader = verification.load_quantized_model

        def perturbed(*args, **kwargs):
            model, metadata = loader(*args, **kwargs)
            model.register_forward_hook(lambda module, inputs, output: output + .25)
            return model, metadata

        with self.dependencies(), patch.object(verification, "load_quantized_model", side_effect=perturbed):
            self.assertNotEqual(self.invoke(verification_cli, config), 0)
        self.assertFalse(self.reports("verification"))

    def test_restoration_state_difference_is_rejected(self):
        config = self.config("verification", checkpoints=self.checkpoints[:1])
        loader = verification.load_quantized_model

        def perturbed(*args, **kwargs):
            model, metadata = loader(*args, **kwargs)
            weight = model.fc.weight.view(-1)
            weight[0] = torch.nextafter(weight[0], torch.full_like(weight[0], float("inf")))
            return model, metadata

        with self.dependencies(), patch.object(verification, "load_quantized_model", side_effect=perturbed):
            self.assertNotEqual(self.invoke(verification_cli, config), 0)
        self.assertFalse(self.reports("verification"))

    def test_invalid_settings_and_existing_run_do_not_evaluate(self):
        config = self.config(settings='batch_size = 0\n')
        with self.dependencies() as data:
            self.assertNotEqual(self.invoke(evaluation_cli, config), 0)
            data.assert_not_called()
        config = self.config()
        destination = self.root / "evaluation/contract"
        destination.mkdir(parents=True)
        sentinel = destination / 'existing.txt'
        sentinel.write_text('keep')
        with self.dependencies() as data:
            self.assertNotEqual(self.invoke(evaluation_cli, config), 0)
            data.assert_not_called()
        self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == "__main__":
    unittest.main()
