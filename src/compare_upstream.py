"""Optional numerical comparison against a separate, pinned upstream checkout."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from checkpoints import ROOT, load_checkpoint, read_manifest, sha256
from data.cifar10 import TestPairs, test_loader
from evaluation.runtime import environment, provenance, resolve_device, seed_everything
from quantization.layers import configure_model

UPSTREAM_SHA = 'fb873a5e276db58ff040fa4eabbb18906fcca1b1'
REFERENCE_FILES = ['networks/resnet.py', 'networks/alexnet.py', 'utils/qutils.py',
                   'utils/quantizer.py', 'utils/trackers.py', 'utils/datasets.py']


def check_upstream(path):
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != UPSTREAM_SHA:
        raise ValueError(f'Upstream SHA mismatch: expected {UPSTREAM_SHA}, got {actual}')
    changed = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--', *REFERENCE_FILES], text=True)
    if changed.strip():
        raise ValueError('Reference source files have uncommitted changes')
    return {name: sha256(path / name) for name in REFERENCE_FILES}


def compare_logits(actual, expected, labels, targets):
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    predictions, reference_predictions = actual.argmax(1), expected.argmax(1)
    if not torch.equal(predictions, reference_predictions):
        raise AssertionError('Predictions differ despite logits passing tolerance')
    return {'max_absolute_error': (actual - expected).abs().max().item(),
            'samples': labels.numel(), 'correct': (predictions == labels).sum().item(),
            'target_successes': (predictions == targets).sum().item(),
            'non_target_successes': ((predictions == targets) & (labels != targets)).sum().item()}


def reference_modules(path):
    # Upstream imports are confined to this explicitly invoked verifier.
    sys.path.insert(0, str(path))
    return (importlib.import_module('networks.resnet'), importlib.import_module('networks.alexnet'),
            importlib.import_module('utils.qutils'), importlib.import_module('utils.datasets'))


def compare(upstream, data_dir, device, seed=1):
    hashes = check_upstream(upstream)
    device = resolve_device(device)
    seed_everything(seed)
    resnet, alexnet, layers, datasets = reference_modules(upstream)
    full = test_loader(str(data_dir)).dataset
    synthetic = np.random.default_rng(seed).integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    cases = [('synthetic', synthetic, [0, 1, 2, 3]),
             ('cifar10-first-128', full.images[:128], full.labels[:128])]
    batches = []
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    for name, images, labels in cases:
        port = next(iter(DataLoader(TestPairs(images, labels), batch_size=128)))
        reference = next(iter(DataLoader(datasets.BackdoorDataset(images, labels, 'square', 0, transform), batch_size=128)))
        for actual, expected in zip(port, reference):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        batches.append((name, port, reference))
    rows = []
    with torch.inference_mode():
        for name, factory in [('resnet18', resnet.ResNet18), ('alexnet', lambda: alexnet.AlexNet(10))]:
            manifest = read_manifest(name)
            checkpoint = ROOT / manifest['local_path']
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            state = {k: v for k, v in state.items() if 'weight_quantizer' not in k and 'activation_quantizer' not in k}
            for precision, bits in [('FP32', None), ('INT8', 8), ('INT4', 4)]:
                seed_everything(seed)
                original = factory().eval().requires_grad_(False)
                original.load_state_dict(state, strict=True)
                port = load_checkpoint(name, checkpoint)
                if [(k, v.shape) for k, v in port.state_dict().items()] != [(k, v.shape) for k, v in original.state_dict().items()]:
                    raise AssertionError('Model state dictionary names or shapes differ')
                original, port = original.to(device), port.to(device)
                configure_model(port, bits)
                if bits is not None:
                    for layer in original.modules():
                        if isinstance(layer, (layers.QuantizedConv2d, layers.QuantizedLinear)):
                            layer.enable_quantization('per_layer_symmetric', 'per_layer_asymmetric', bits)
                            # Sole upstream CUDA correction: buffers created after .to(device).
                            layer.weight_quantizer.to(device)
                            layer.activation_quantizer.to(device)
                for case, pair, reference in batches:
                    clean, labels, triggered, targets = [tensor.to(device) for tensor in pair]
                    ref_clean, _, ref_triggered, _ = [tensor.to(device) for tensor in reference]
                    expected_clean, expected_triggered = original(ref_clean), original(ref_triggered)
                    actual_clean, actual_triggered = port(clean), port(triggered)
                    rows.append({'model': name, 'precision': precision, 'input': case,
                                 'clean': compare_logits(actual_clean, expected_clean, labels, targets),
                                 'triggered': compare_logits(actual_triggered, expected_triggered, labels, targets)})
                print(f'{device} {name} {precision}: synthetic and CIFAR-10 comparisons passed', flush=True)
                del original, port
    return {'status': 'passed', 'upstream_sha': UPSTREAM_SHA, 'upstream_file_sha256': hashes,
            'atol': 1e-6, 'rtol': 1e-5, 'seed': seed, 'batch_size': 128,
            'upstream_cuda_adjustment': 'quantizer buffer device alignment only' if device.type == 'cuda' else None,
            'environment': environment(device), 'provenance': {name: provenance(ROOT / read_manifest(name)['local_path']) for name in ('resnet18', 'alexnet')},
            'comparisons': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'datasets/cifar10')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    try:
        report = compare(args.upstream.resolve(), args.data_dir, args.device, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, RuntimeError, AssertionError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
