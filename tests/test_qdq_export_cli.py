"""Tests for checkpoint extraction without dataset loading or evaluation."""
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
import export_quantized_checkpoints as cli
from quantization import checkpoint_export
from qdq_test_support import toy_model


ROOT = Path(__file__).resolve().parents[1]



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
        content = 'run_id = "contract"\noutput = "artifacts/checkpoints"\ndevice = "cpu"\n'
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
                 patch("prepare_checkpoints.ROOT", self.root), \
                 patch("prepare_checkpoints.read_manifest", side_effect=lambda name: copy.deepcopy(self.manifests[name])), \
                 patch("checkpoints.create_model", side_effect=lambda name: toy_model()), \
                 patch("models.registry.create_model", side_effect=lambda name: toy_model()), \
                 patch("data.cifar10.test_loader", side_effect=AssertionError("export must not read a dataset")) as data:
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

    def artifacts(self, output=None, run_id="contract"):
        return sorted((output or self.output).glob(f"qdq_int*/*/{run_id}/model.pt"))

    def test_checked_in_toml_contains_only_extraction_settings(self):
        config = tomllib.loads((ROOT / "configs/quantization/qdq_export.toml").read_text())
        self.assertEqual(config["models"], ["resnet18", "alexnet"])
        self.assertEqual(set(config["quantization"]["bit_widths"]), {4, 8})
        self.assertEqual(config["device"], "auto")
        self.assertNotIn("evaluation", config)

    def test_cli_exports_matrix_without_data_or_evaluation_reports(self):
        self.write_config()
        with self.dependencies() as data:
            self.assertEqual(self.invoke("--output", self.output), 0)
            data.assert_not_called()
        artifacts = self.artifacts()
        self.assertEqual(len(artifacts), 4)
        for path in artifacts:
            payload = checkpoint_export.load_quantized_checkpoint(path)
            self.assertEqual(payload["format_version"], 2)
            metadata = payload["quantization"]
            self.assertEqual(metadata["model_name"], path.parts[-3])
            self.assertEqual(metadata["checkpoint_source"], "public")
            self.assertEqual(metadata["source_sha256"], self.manifests[path.parts[-3]]["sha256"])
            self.assertFalse((path.parent / "results.json").exists())
            self.assertFalse((path.parent / "results.csv").exists())
            for name, layer in metadata["layers"].items():
                self.assertTrue(layer["activation"]["enabled"])
                prefix = f"{name}." if name else ""
                self.assertNotIn(f"{prefix}activation_quantizer.scale", payload["state_dict"])
                self.assertNotIn(f"{prefix}activation_quantizer.range_tracker.min_val", payload["state_dict"])

    def test_output_override_and_relative_paths_use_project_root(self):
        self.write_config(models=["resnet18"], bits=[8])
        elsewhere = self.root / "outside"
        elsewhere.mkdir()
        with self.dependencies() as data, patch("os.getcwd", return_value=str(elsewhere)):
            self.assertEqual(self.invoke("--output", "override"), 0)
            data.assert_not_called()
        self.assertEqual(len(self.artifacts(self.root / "override")), 1)
        self.assertFalse(list(self.output.glob("qdq_int*")))

    def test_toml_output_is_used_without_cli_override(self):
        self.write_config(models=["resnet18"], bits=[4])
        with self.dependencies():
            self.assertEqual(self.invoke(), 0)
        self.assertEqual(len(self.artifacts()), 1)

    def test_another_run_id_preserves_the_previous_run(self):
        self.write_config(models=["resnet18"], bits=[4])
        with self.dependencies():
            self.assertEqual(self.invoke(), 0)
            original = self.artifacts()[0].read_bytes()
            self.assertEqual(self.invoke("--run-id", "another"), 0)
        self.assertEqual(self.artifacts()[0].read_bytes(), original)
        self.assertEqual(len(self.artifacts(run_id="another")), 1)

    def test_legacy_public_weights_are_copied_and_manifest_is_updated(self):
        canonical = self.root / self.manifests["resnet18"]["local_path"]
        legacy = self.root / "artifacts/checkpoints/resnet18/qcb.pth"
        legacy.parent.mkdir(parents=True)
        canonical.rename(legacy)
        original = legacy.read_bytes()
        manifest_path = self.root / "configs/checkpoints/resnet18.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps(self.manifests["resnet18"]))
        self.write_config(models=["resnet18"], bits=[8])
        with self.dependencies():
            self.assertEqual(self.invoke(), 0)
        self.assertEqual(legacy.read_bytes(), original)
        self.assertEqual(canonical.read_bytes(), original)
        self.assertEqual(json.loads(manifest_path.read_text()), {
            **self.manifests["resnet18"], "validation": "sha256-and-strict-state-dict",
        })
        metadata = checkpoint_export.load_quantized_checkpoint(self.artifacts()[0])["quantization"]
        self.assertEqual(metadata["source_checkpoint"], "artifacts/checkpoints/resnet18/qcb.pth")
        self.assertEqual(metadata["source_sha256"], sha256(legacy))

    def test_self_trained_checkpoint_requires_explicit_source_and_hash(self):
        source = self.root / "trained-best.pt"
        torch.save(toy_model().state_dict(), source)
        extra = ('\n[checkpoints.resnet18]\n'
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
                 dict(extra='\n[evaluation]\nbatch_size = 128\n'),
                 dict(extra='\n[unknown]\nvalue = 1\n'),
                 dict(extra='\n[checkpoints.resnet18]\ncheckpoint_source = "self_trained"\npath = "trained-best.pt"\n')]
        for index, options in enumerate(cases):
            self.write_config(**options)
            with self.subTest(case=index), self.dependencies():
                self.assertNotEqual(self.invoke("--output", self.output), 0)
            self.assertFalse(list(self.output.glob("qdq_int*/*/contract/*")))

    def test_public_manifest_and_explicit_hash_mismatch_fail_before_data(self):
        verified_hash = self.manifests["resnet18"]["sha256"]
        for source in ("manifest", "config"):
            extra = ''
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

    def test_second_input_hash_failure_prevents_the_entire_batch(self):
        self.write_config()
        self.manifests["alexnet"]["sha256"] = "0" * 64
        with self.dependencies() as data:
            self.assertNotEqual(self.invoke("--output", self.output), 0)
            data.assert_not_called()
        self.assertFalse(list(self.output.glob("qdq_int*")))

    def test_existing_run_is_rejected_without_overwrite_or_evaluation(self):
        self.write_config(models=["resnet18"], bits=[4])
        destination = self.output / "qdq_int4/resnet18/contract"
        destination.mkdir(parents=True)
        sentinel = destination / "model.pt"
        sentinel.write_bytes(b"existing artifact")
        with self.dependencies() as data:
            self.assertNotEqual(self.invoke("--output", self.output), 0)
            data.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"existing artifact")
        self.assertEqual(set(destination.iterdir()), {sentinel})



if __name__ == "__main__":
    unittest.main()
