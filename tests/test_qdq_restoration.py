"""Tests for Q-DQ checkpoint persistence and model restoration."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from evaluation.metrics import evaluate_model
from models.registry import create_model
from pruning.masks import apply_masks, load_masks, save_masks
from pruning.sparsity import model_sparsity
from quantization import checkpoint_export
from quantization.checkpoint_export import load_quantized_checkpoint, save_quantized_checkpoint
from quantization.layers import QuantizedLayer, configure_model
from quantization.materialize_qdq import quantize
from qdq_test_support import (assert_activation_equal, assert_state_equal, assert_tree_equal,
                              make_test_set_loader, toy_model, with_checkpoint_source)


class RestorationContractTest(unittest.TestCase):
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
        self.path = Path(self.temporary.name) / "qdq.pt"

    def export_toy(self, *, bits=4, observed=False, checkpoint_source="public"):
        model, metadata = quantize(toy_model(), bits)
        if observed:
            with torch.inference_mode():
                model(next(iter(make_test_set_loader()))[0])
        metadata = with_checkpoint_source(metadata, checkpoint_source=checkpoint_source)
        save_quantized_checkpoint(self.path, model, metadata)
        return model, metadata

    def test_v2_envelope_has_no_model_object_and_records_source_and_layers(self):
        model, metadata = self.export_toy(observed=True)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["format_version"], 2)
        assert_state_equal(self, model.state_dict(), payload["state_dict"])
        self.assertEqual(set(payload["quantization"]["layers"]), {"conv", "fc"})
        for key in ("model_name", "source_checkpoint", "source_sha256", "checkpoint_source", "source_url", "weight_license"):
            self.assertEqual(payload["quantization"][key], metadata[key])

        def check_portable(value):
            self.assertNotIsInstance(value, nn.Module)
            if isinstance(value, dict):
                for item in value.values():
                    check_portable(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    check_portable(item)
            else:
                self.assertIsInstance(value, (str, int, float, bool, torch.Tensor, type(None)))
        check_portable(payload)

    def test_public_and_self_trained_source_details_survive(self):
        for source in ("public", "self_trained"):
            _, metadata = self.export_toy(checkpoint_source=source)
            payload = load_quantized_checkpoint(self.path)
            for key in ("source_checkpoint", "source_sha256", "checkpoint_source", "source_url", "weight_license"):
                with self.subTest(source=source, field=key):
                    self.assertEqual(payload["quantization"][key], metadata[key])

    def test_unobserved_and_observed_buffers_round_trip_exactly_on_cpu(self):
        for bits in (4, 8):
            for observed in (False, True):
                with self.subTest(bits=bits, observed=observed):
                    model, _ = self.export_toy(bits=bits, observed=observed)
                    template = toy_model()
                    before = copy.deepcopy(template.state_dict())
                    result, metadata = checkpoint_export.load_quantized_model(self.path, model=template, device="cpu")
                    self.assertIsNot(result, template)
                    self.assertEqual(list(dict(result.named_parameters())), list(dict(template.named_parameters())))
                    assert_state_equal(self, before, template.state_dict())
                    assert_state_equal(self, model.state_dict(), result.state_dict())
                    assert_activation_equal(self, model, result)
                    assert_tree_equal(self, load_quantized_checkpoint(self.path)["quantization"], metadata)
                    self.assertTrue(all(p.device.type == "cpu" for p in result.parameters()))
                    self.assertFalse(result.training)

    def test_repeated_forwards_preserve_every_forward_activation_and_three_way_equivalence(self):
        for bits in (4, 8):
            runtime = configure_model(toy_model(), bits)
            model, metadata = quantize(runtime, bits)
            model(next(iter(make_test_set_loader()))[0])
            save_quantized_checkpoint(self.path, model, with_checkpoint_source(metadata))
            result, _ = checkpoint_export.load_quantized_model(self.path, model=toy_model(), device="cpu")
            with self.subTest(bits=bits), torch.inference_mode():
                for clean, _, attacked, _ in make_test_set_loader():
                    for inputs in (clean, attacked):
                        expected = runtime(inputs)
                        materialized, restored = model(inputs), result(inputs)
                        torch.testing.assert_close(materialized, expected, rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(restored, materialized, rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(restored.argmax(1), expected.argmax(1), rtol=0, atol=0)
                        assert_activation_equal(self, model, result)
                reports = [evaluate_model(m, make_test_set_loader(), torch.device("cpu"), progress=False)
                           for m in (runtime, model, result)]
                for key in ("samples", "clean_correct", "clean_denominator", "asr_all_successes", "asr_all_denominator",
                            "asr_non_target_successes", "asr_non_target_denominator", "clean_accuracy", "asr_all", "asr_non_target"):
                    self.assertEqual(reports[0][key], reports[1][key], key)
                    self.assertEqual(reports[1][key], reports[2][key], key)
                assert_state_equal(self, model.state_dict(), result.state_dict())

    def test_activation_disabled_and_different_per_layer_bits_survive(self):
        model, metadata = quantize(toy_model(), 4)
        model.conv.configure_quantization(8, weight=False, activation=True)
        model.fc.quantize_activation = False
        model(next(iter(make_test_set_loader()))[0])
        save_quantized_checkpoint(self.path, model, with_checkpoint_source(metadata))
        layer_metadata = load_quantized_checkpoint(self.path)["quantization"]["layers"]
        for name, enabled, bits in (("conv", True, 8), ("fc", False, 4)):
            activation = layer_metadata[name]["activation"]
            self.assertIs(activation["enabled"], enabled)
            self.assertEqual(activation["bit_width"], bits)
            self.assertIs(activation["symmetric"], False)
            self.assertEqual(activation["range_update"], "every_forward")
        result, _ = checkpoint_export.load_quantized_model(self.path, model=toy_model(), device="cpu")
        self.assertFalse(result.fc.quantize_activation)
        self.assertEqual(result.conv.activation_quantizer.max_val.item(), 255)
        self.assertEqual(result.fc.activation_quantizer.max_val.item(), 15)
        assert_activation_equal(self, model, result)
        assert_state_equal(self, model.state_dict(), result.state_dict())

    def test_plain_custom_model_template_is_converted_without_mutating_template(self):
        model, metadata = quantize(toy_model(plain=True), 8)
        save_quantized_checkpoint(self.path, model, with_checkpoint_source(metadata))
        template = toy_model(plain=True)
        before = copy.deepcopy(template.state_dict())
        result, _ = checkpoint_export.load_quantized_model(self.path, model=template, device="cpu")
        self.assertFalse(any(isinstance(layer, QuantizedLayer) for layer in template.modules()))
        self.assertIsInstance(result.conv, QuantizedLayer)
        self.assertIsInstance(result.fc, QuantizedLayer)
        assert_state_equal(self, before, template.state_dict())
        assert_state_equal(self, model.state_dict(), result.state_dict())

    def test_registered_resnet18_and_alexnet_are_created_without_template(self):
        for name in ("resnet18", "alexnet"):
            for bits in (4, 8):
                with self.subTest(model=name, bits=bits):
                    model, metadata = quantize(create_model(name).eval(), bits)
                    save_quantized_checkpoint(self.path, model, with_checkpoint_source(metadata, model_name=name))
                    result, restored_metadata = checkpoint_export.load_quantized_model(self.path, device="cpu")
                    self.assertEqual(restored_metadata["model_name"], name)
                    assert_state_equal(self, model.state_dict(), result.state_dict())
                    inputs = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
                    with torch.inference_mode():
                        torch.testing.assert_close(result(inputs), model(inputs), rtol=1e-5, atol=1e-6)
                    del model, result

    def test_unknown_model_requires_template(self):
        self.export_toy()
        with self.assertRaisesRegex(ValueError, "(?i)model|template"):
            checkpoint_export.load_quantized_model(self.path, device="cpu")

    def test_q_p_and_p_q_masks_survive_disk_restore_forward_and_reapplication(self):
        for bits in (4, 8):
            for order in ("Q-P", "P-Q"):
                with self.subTest(bits=bits, order=order):
                    source = toy_model()
                    masks = {name: torch.arange(p.numel()).reshape(p.shape) % 2 == 0
                             for name, p in source.named_parameters() if name in ("conv.weight", "fc.weight")}
                    if order == "P-Q":
                        apply_masks(source, masks)
                    model, metadata = quantize(source, bits, masks if order == "P-Q" else None)
                    if order == "Q-P":
                        apply_masks(model, masks)
                    mask_path = self.path.with_suffix(".mask.pt")
                    save_masks(mask_path, masks)
                    save_quantized_checkpoint(self.path, model, with_checkpoint_source(metadata))
                    result, _ = checkpoint_export.load_quantized_model(self.path, model=toy_model(), device="cpu")
                    result(next(iter(make_test_set_loader()))[0])
                    for name, mask in load_masks(mask_path).items():
                        values = dict(result.named_parameters())[name]
                        self.assertTrue(torch.equal(values[~mask], torch.zeros_like(values[~mask])))
                    self.assertEqual(model_sparsity(model), model_sparsity(result))
                    before = copy.deepcopy(result.state_dict())
                    apply_masks(result, load_masks(mask_path))
                    assert_state_equal(self, before, result.state_dict())

    def test_reads_on_cpu_with_weights_only_before_device_movement(self):
        self.export_toy()
        with patch("torch.load", wraps=torch.load) as reading:
            checkpoint_export.load_quantized_model(self.path, model=toy_model(), device="cpu")
        self.assertGreaterEqual(reading.call_count, 1)
        for call in reading.call_args_list:
            self.assertEqual(str(call.kwargs["map_location"]), "cpu")
            self.assertIs(call.kwargs["weights_only"], True)

    def test_auto_uses_cpu_when_cuda_is_unavailable(self):
        self.export_toy()
        with patch("torch.cuda.is_available", return_value=False):
            result, _ = checkpoint_export.load_quantized_model(self.path, model=toy_model())
        self.assertTrue(all(p.device.type == "cpu" for p in result.parameters()))

    def test_auto_cuda_and_explicit_cuda_indices_use_resolve_device_semantics(self):
        self.export_toy()
        for requested, expected in (("auto", "cuda:0"), ("cuda", "cuda:0"), ("cuda:1", "cuda:1"), ("cpu", "cpu")):
            with self.subTest(requested=requested), patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.device_count", return_value=2), \
                 patch.object(nn.Module, "to", autospec=True, side_effect=lambda module, *args, **kw: module) as moving:
                checkpoint_export.load_quantized_model(self.path, model=toy_model(), device=requested)
                call = moving.call_args
                destination = call.args[1] if len(call.args) > 1 else call.kwargs.get("device")
                self.assertEqual(str(destination), expected)

    def test_explicit_unavailable_cuda_and_unsupported_device_raise(self):
        self.export_toy()
        for requested in ("cuda", "cuda:0", "mps"):
            with self.subTest(device=requested), patch("torch.cuda.is_available", return_value=False), self.assertRaises(ValueError):
                checkpoint_export.load_quantized_model(self.path, model=toy_model(), device=requested)
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1), self.assertRaises(ValueError):
            checkpoint_export.load_quantized_model(self.path, model=toy_model(), device="cuda:1")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_save_cuda_restore_matches_cuda_materialized_model(self):
        clean = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
        attacked = clean.clone()
        attacked[:, :, 27:31, 27:31] = 1
        for name in ("resnet18", "alexnet"):
            for bits in (4, 8):
                with self.subTest(model=name, bits=bits):
                    model, metadata = quantize(create_model(name).eval(), bits)
                    with torch.inference_mode():
                        model(clean)
                    save_quantized_checkpoint(self.path, model, with_checkpoint_source(metadata, model_name=name))
                    result, restored_metadata = checkpoint_export.load_quantized_model(self.path, device="cuda:0")
                    self.assertEqual(restored_metadata["model_name"], name)
                    self.assertTrue(all(p.device == torch.device("cuda:0") for p in result.parameters()))
                    assert_state_equal(self, model.state_dict(), result.state_dict())
                    assert_activation_equal(self, model, result)
                    expected = model.to("cuda:0")
                    for inputs in (clean, attacked):
                        with torch.inference_mode():
                            a, b = expected(inputs.cuda()), result(inputs.cuda())
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(a.argmax(1), b.argmax(1), rtol=0, atol=0)
                        assert_activation_equal(self, expected, result)
                    assert_state_equal(self, expected.state_dict(), result.state_dict())
                    del model, expected, result

    def test_invalid_envelopes_and_versions_are_rejected(self):
        for payload in ([], {}, {"format_version": 99, "state_dict": {}, "quantization": {}},
                        {"format_version": 2, "state_dict": [], "quantization": {}},
                        {"format_version": 2, "state_dict": {}, "quantization": []}):
            with self.subTest(payload=str(payload)):
                torch.save(payload, self.path)
                with self.assertRaises(ValueError):
                    load_quantized_checkpoint(self.path)

    def test_strict_restore_rejects_missing_extra_and_wrong_shape_state(self):
        self.export_toy(observed=True)
        original = load_quantized_checkpoint(self.path)
        corruptions = [lambda state: state.pop("fc.bias"),
                       lambda state: state.update(unexpected=torch.zeros(1)),
                       lambda state: state.update({"conv.weight": torch.zeros(1)}),
                       lambda state: state.update({"conv.activation_quantizer.scale": torch.zeros(2)})]
        for corrupt in corruptions:
            payload = copy.deepcopy(original)
            corrupt(payload["state_dict"])
            torch.save(payload, self.path)
            with self.subTest(corruption=corruptions.index(corrupt)), self.assertRaises((ValueError, RuntimeError)):
                checkpoint_export.load_quantized_model(self.path, model=toy_model(), device="cpu")

    def test_restore_rejects_invalid_layer_metadata_and_source_digest(self):
        self.export_toy()
        original = load_quantized_checkpoint(self.path)
        self.assertIn("layers", original["quantization"])
        corruptions = [lambda meta: meta.update(bit_width=3),
                       lambda meta: meta.update(source_sha256="invalid"),
                       lambda meta: meta.update(checkpoint_source="unknown"),
                       lambda meta: meta["layers"].pop("conv"),
                       lambda meta: meta["layers"].update(unknown={})]
        for index, corrupt in enumerate(corruptions):
            payload = copy.deepcopy(original)
            corrupt(payload["quantization"])
            torch.save(payload, self.path)
            with self.subTest(corruption=index), self.assertRaises(ValueError):
                checkpoint_export.load_quantized_model(self.path, model=toy_model(), device="cpu")

    def test_restore_rejects_missing_or_invalid_activation_metadata(self):
        self.export_toy(observed=True)
        original = load_quantized_checkpoint(self.path)
        corruptions = {
            "missing activation": lambda layer: layer.pop("activation"),
            "missing enabled": lambda layer: layer["activation"].pop("enabled"),
            "invalid bits": lambda layer: layer["activation"].update(bit_width=3),
            "invalid update policy": lambda layer: layer["activation"].update(range_update="unknown"),
        }
        for label, corrupt in corruptions.items():
            with self.subTest(corruption=label):
                payload = copy.deepcopy(original)
                corrupt(payload["quantization"]["layers"]["conv"])
                torch.save(payload, self.path)
                with self.assertRaises(ValueError):
                    checkpoint_export.load_quantized_model(self.path, model=toy_model(), device="cpu")

    def test_restore_rejects_partially_missing_observed_activation_buffers(self):
        self.export_toy(observed=True)
        original = load_quantized_checkpoint(self.path)
        for suffix in ("scale", "zero_point", "range_tracker.min_val", "range_tracker.max_val"):
            with self.subTest(missing=suffix):
                payload = copy.deepcopy(original)
                payload["state_dict"].pop(f"conv.activation_quantizer.{suffix}")
                torch.save(payload, self.path)
                with self.assertRaises((ValueError, RuntimeError)):
                    checkpoint_export.load_quantized_model(self.path, model=toy_model(), device="cpu")

    def test_supplied_model_shape_mismatch_is_rejected_without_mutation(self):
        self.export_toy()
        template = nn.Linear(7, 2)
        before = copy.deepcopy(template.state_dict())
        with self.assertRaises((ValueError, RuntimeError)):
            checkpoint_export.load_quantized_model(self.path, model=template, device="cpu")
        assert_state_equal(self, before, template.state_dict())


if __name__ == "__main__":
    unittest.main()
