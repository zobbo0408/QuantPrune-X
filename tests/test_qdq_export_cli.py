"""Tests for batch Q-DQ checkpoint export and evaluation through the CLI."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import torch

from checkpoints import sha256
from evaluation.runtime import seed_everything
import export_quantized_checkpoints as cli
from quantization import checkpoint_export
from qdq_test_support import make_test_set_loader, toy_model


ROOT = Path(__file__).resolve().parents[1]
COUNT_KEYS = ("samples", "clean_correct", "clean_denominator", "asr_all_successes",
              "asr_all_denominator", "asr_non_target_successes", "asr_non_target_denominator",
              "clean_accuracy", "asr_all", "asr_non_target")


class ExportCliContractTest(unittest.TestCase):
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
        self.config = self.root / "export.toml"
        self.output = self.root / "artifacts/checkpoints"
        self.manifests = {}
        for name in ("resnet18", "alexnet"):
            source = self.root / f"artifacts/checkpoints/fp32/{name}/qcb.pth"
            source.parent.mkdir(parents=True)
            torch.save(toy_model().state_dict(), source)
            self.manifests[name] = dict(model=name, sha256=sha256(source),
                                        local_path=str(source.relative_to(self.root)),
                                        source_url="https://example.invalid/public-weights",
                                        zip_path=f"public/{name}.pth", license="not separately specified")

    def write_config(self, extra="", *, models=None, bits=None):
        content = f'run_id = "contract"\noutput = "artifacts/checkpoints"\n'
        if models is not None:
            content += f"models = {json.dumps(models)}\n"
        content += '[quantization]\nrepresentation = "qdq"\n'
        if bits is not None:
            content += f"bit_widths = {json.dumps(bits)}\n"
        content += extra
        self.config.write_text(content, encoding="utf-8")

    @contextmanager
    def dependencies(self):
        # Reload importers under the patches so both direct and module imports
        # see the same fake registered models. Restore their bindings afterward.
        try:
            with patch("checkpoints.ROOT", self.root), \
                 patch("checkpoints.read_manifest", side_effect=lambda name: copy.deepcopy(self.manifests[name])), \
                 patch("checkpoints.create_model", side_effect=lambda name: toy_model()), \
                 patch("models.registry.create_model", side_effect=lambda name: toy_model()), \
                 patch("data.cifar10.test_loader", return_value=make_test_set_loader()) as data:
                importlib.reload(checkpoint_export)
                importlib.reload(cli)
                yield data
        finally:
            importlib.reload(checkpoint_export)
            importlib.reload(cli)

    def invoke(self, *options):
        # Invoke the existing positional-config/--output interface, without
        # requiring a new Python-only runner API.
        with patch.object(sys, "argv", ["export_quantized_checkpoints.py", str(self.config), *map(str, options)]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                cli.main()
            except SystemExit as exc:
                return exc.code
        return 0

    def reports(self, output=None):
        paths = sorted((output or self.output).glob("qdq_int*/*/contract/results.json"))
        return [(path, json.loads(path.read_text())) for path in paths]

    def test_checked_in_toml_defaults_cover_both_models_and_bits(self):
        config = tomllib.loads((ROOT / "configs/quantization/qdq_export.toml").read_text())
        self.assertEqual(config["models"], ["resnet18", "alexnet"])
        self.assertEqual(set(config["quantization"]["bit_widths"]), {4, 8})
        self.assertEqual(config["quantization"]["activation_mode"], "per_tensor_asymmetric")
        evaluation = config["evaluation"]
        expected = dict(seed=225, batch_size=128, workers=0, normalize=True, target_class=0)
        for key, value in expected.items():
            self.assertEqual(evaluation[key], value)
        self.assertEqual(evaluation["data_dir"], "datasets/cifar10")

    def test_cli_exports_default_matrix_and_compares_real_evaluations(self):
        self.write_config('\n[evaluation]\ndevice = "cpu"\n')
        with self.dependencies() as data:
            self.assertEqual(self.invoke("--output", self.output), 0)
            self.assertGreaterEqual(data.call_count, 1)
            for call in data.call_args_list:
                self.assertEqual(call.kwargs["batch_size"], 128)
                self.assertEqual(call.kwargs["workers"], 0)
                self.assertIs(call.kwargs["normalize"], True)
                self.assertEqual(call.kwargs["target"], 0)
        reports = self.reports()
        self.assertEqual(len(reports), 4)
        for path, report in reports:
            with self.subTest(path=str(path)):
                self.assertIn(path.parts[-4], ("qdq_int4", "qdq_int8"))
                self.assertIn(path.parts[-3], ("resnet18", "alexnet"))
                rows = report["results"]
                self.assertEqual({row["stage"] for row in rows}, {"runtime", "materialized", "restored"})
                self.assertEqual(len(rows), 3)
                for key in COUNT_KEYS:
                    self.assertTrue(all(row[key] == rows[0][key] for row in rows), key)
                self.assertEqual(rows[0]["samples"], 5)
                self.assertEqual(rows[0]["asr_non_target_denominator"], 3)
                self.assertEqual(report["conditions"]["seed"], 225)
                self.assertEqual(report["environment"]["device_type"], "cpu")
                self.assertNotIn("device_name", report["environment"])
                self.assertNotIn("platform", report["environment"])
                artifacts = [p for p in path.parent.iterdir() if p.suffix in (".pt", ".pth")]
                self.assertEqual(len(artifacts), 1)
                payload = torch.load(artifacts[0], map_location="cpu", weights_only=True)
                self.assertEqual(payload["format_version"], 2)
                meta = payload["quantization"]
                self.assertEqual(meta["model_name"], path.parts[-3])
                self.assertEqual(meta["checkpoint_source"], "public")
                self.assertEqual(meta["source_sha256"], self.manifests[path.parts[-3]]["sha256"])

    def test_custom_evaluation_settings_and_seed_are_applied(self):
        self.write_config('\n[evaluation]\ndevice = "cpu"\nseed = 731\n'
                          'batch_size = 3\nworkers = 2\nnormalize = false\ntarget_class = 2\n',
                          models=["resnet18"], bits=[4])
        # Reproducibility of the shared helper is covered in test_cli.py;
        # this test only checks that the configured seed reaches it.
        with self.dependencies() as data, \
             patch("evaluation.runtime.seed_everything", wraps=seed_everything) as seeding:
            importlib.reload(cli)
            self.assertEqual(self.invoke("--output", self.output), 0)
            self.assertGreaterEqual(data.call_count, 1)
            for call in data.call_args_list:
                self.assertEqual(call.kwargs["batch_size"], 3)
                self.assertEqual(call.kwargs["workers"], 2)
                self.assertIs(call.kwargs["normalize"], False)
                self.assertEqual(call.kwargs["target"], 2)
            self.assertTrue(any(
                call.args == (731,) or call.kwargs.get("seed") == 731
                for call in seeding.call_args_list
            ), "configured seed was not passed to seed_everything")
        reports = self.reports()
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0][1]["conditions"]["seed"], 731)

    def test_output_override_and_relative_paths_use_project_root(self):
        self.write_config('\n[evaluation]\ndevice = "cpu"\ndata_dir = "fixtures/cifar10"\n',
                          models=["resnet18"], bits=[8])
        elsewhere = self.root / "outside"
        elsewhere.mkdir()
        # A relative --output is rooted at the project, independent of cwd.
        with self.dependencies() as data, patch("os.getcwd", return_value=str(elsewhere)):
            self.assertEqual(self.invoke("--output", "override"), 0)
            self.assertEqual(Path(data.call_args.args[0]), self.root / "fixtures/cifar10")
        self.assertEqual(len(self.reports(self.root / "override")), 1)
        self.assertFalse(self.output.exists())

    def test_self_trained_checkpoint_requires_explicit_source_and_hash(self):
        source = self.root / "trained-best.pt"
        torch.save(toy_model().state_dict(), source)
        extra = ('\n[evaluation]\ndevice = "cpu"\n'
                 '[checkpoints.resnet18]\n'
                 'path = "trained-best.pt"\n'
                 f'sha256 = "{sha256(source)}"\n'
                 'checkpoint_source = "self_trained"\n'
                 'source_url = "https://example.invalid/training-run"\n'
                 'weight_license = "research fixture"\n')
        self.write_config(extra, models=["resnet18"], bits=[4])
        with self.dependencies():
            self.assertEqual(self.invoke("--output", self.output), 0)
        artifacts = list(self.output.glob("qdq_int4/resnet18/contract/*.pt"))
        artifacts += list(self.output.glob("qdq_int4/resnet18/contract/*.pth"))
        self.assertEqual(len(artifacts), 1)
        meta = checkpoint_export.load_quantized_checkpoint(artifacts[0])["quantization"]
        self.assertEqual(meta["checkpoint_source"], "self_trained")
        self.assertEqual(meta["source_sha256"], sha256(source))
        self.assertEqual(meta["weight_license"], "research fixture")

    def test_invalid_config_and_self_trained_source_fail_without_output(self):
        torch.save(toy_model().state_dict(), self.root / "trained-best.pt")
        cases = [dict(models=["unknown"]), dict(models=["resnet18", "resnet18"]),
                 dict(bits=[3]), dict(bits=[4, 4]), dict(bits=[]),
                 dict(extra='\n[evaluation]\nbatch_size = 0\n'),
                 dict(extra='\n[evaluation]\nworkers = -1\n'),
                 dict(extra='\n[checkpoints.resnet18]\ncheckpoint_source = "self_trained"\npath = "trained-best.pt"\n')]
        for index, options in enumerate(cases):
            self.write_config(**options)
            with self.subTest(case=index), self.dependencies():
                self.assertNotEqual(self.invoke("--output", self.output), 0)
            self.assertFalse(list(self.output.glob("qdq_int*/*/contract/*")))

    def test_public_manifest_and_explicit_hash_mismatch_fail_before_data(self):
        verified_hash = self.manifests["resnet18"]["sha256"]
        for source in ("manifest", "config"):
            extra = '\n[evaluation]\ndevice = "cpu"\n'
            self.manifests["resnet18"]["sha256"] = verified_hash
            if source == "manifest":
                self.manifests["resnet18"]["sha256"] = "0" * 64
            else:
                extra += ('\n[checkpoints.resnet18]\ncheckpoint_source = "public"\n'
                          'path = "artifacts/checkpoints/fp32/resnet18/qcb.pth"\n'
                          'sha256 = "' + '0' * 64 + '"\n')
            self.write_config(extra, models=["resnet18"], bits=[8])
            with self.subTest(hash_source=source), self.dependencies() as data:
                self.assertNotEqual(self.invoke("--output", self.output), 0)
                data.assert_not_called()
                self.assertFalse(list(self.output.glob("qdq_int*/*/contract/*")))

    def test_existing_run_is_rejected_without_overwrite_or_evaluation(self):
        self.write_config('\n[evaluation]\ndevice = "cpu"\n', models=["resnet18"], bits=[4])
        destination = self.output / "qdq_int4/resnet18/contract"
        destination.mkdir(parents=True)
        sentinel = destination / "model.pt"
        sentinel.write_bytes(b"existing artifact")
        with self.dependencies() as data:
            self.assertNotEqual(self.invoke("--output", self.output), 0)
            data.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"existing artifact")
        self.assertEqual(set(destination.iterdir()), {sentinel})

    def test_logit_difference_with_identical_predictions_causes_nonzero_exit(self):
        self.write_config('\n[evaluation]\ndevice = "cpu"\n', models=["resnet18"], bits=[4])
        with self.dependencies():
            loader = getattr(checkpoint_export, "load_quantized_model", None)
            self.assertTrue(callable(loader), "load_quantized_model is not implemented")

            def perturbed(*args, **kwargs):
                model, metadata = loader(*args, **kwargs)
                # A constant logit shift preserves predictions and all success
                # counts. The CLI must still reject the logits mismatch.
                model.register_forward_hook(lambda module, inputs, output: output + .25)
                return model, metadata

            with patch.object(checkpoint_export, "load_quantized_model", side_effect=perturbed):
                importlib.reload(cli)
                self.assertNotEqual(self.invoke("--output", self.output), 0)

if __name__ == "__main__":
    unittest.main()
