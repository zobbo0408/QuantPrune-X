import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
from checkpoints import sha256
from data.cifar10 import TestPairs
from evaluate import RunConfig, parse_args, run_evaluation
from evaluation.runtime import resolve_device, seed_everything
from run_matrix import read_experiment, run_matrix


class ToyModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 10, device=x.device)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / 'qcb.pth'
        self.checkpoint.write_bytes(b'checkpoint fixture')
        self.manifest = {'sha256': sha256(self.checkpoint), 'source_url': 'fixture',
                         'zip_path': 'fixture.pth', 'local_path': str(self.checkpoint)}
        self.loader = DataLoader(TestPairs(np.zeros((3, 32, 32, 3), dtype=np.uint8), [0, 0, 0]), batch_size=2)
        self.config = RunConfig(model='resnet18', checkpoint=self.checkpoint,
                                data_dir=self.root / 'data', output=self.root / 'run', device='cpu', batch_size=2)

    def test_default_arguments_and_config_validation(self):
        _, args = parse_args(['--model', 'resnet18', '--output', 'run'])
        self.assertEqual(args.precisions, ['fp32', 'int8', 'int4'])
        self.assertEqual((args.batch_size, args.workers, args.seed, args.target_class), (128, 0, 1, 0))
        self.assertTrue(args.normalize)
        self.assertFalse(args.download_data)
        _, args = parse_args(['--model', 'alexnet', '--output', 'run', '--precision', 'INT4', '--no-normalize'])
        self.assertEqual(args.precisions, ['int4'])
        self.assertFalse(args.normalize)
        for options in ({'precisions': ()}, {'precisions': ('int4', 'int4')}, {'precisions': ('int3',)},
                        {'device': 0}, {'precisions': 'fp32'}, {'precisions': ([],)}, {'batch_size': 0}, {'workers': -1}, {'seed': -1}, {'target_class': 10},
                        {'limit': 0}, {'normalize': 'false'}, {'model': 'unknown'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                replace(self.config, **options).validate()
        with self.assertRaises(FileNotFoundError):
            replace(self.config, checkpoint=self.root / 'missing').validate()

    def test_device_selection_and_seed_reproducibility(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(str(resolve_device('auto')), 'cpu')
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                resolve_device('cuda')
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2):
            self.assertEqual(str(resolve_device('auto')), 'cuda:0')
            self.assertEqual(str(resolve_device('cuda:1')), 'cuda:1')
            with self.assertRaises(ValueError):
                resolve_device('cuda:2')
        with self.assertRaises(ValueError):
            resolve_device('mps')
        seed_everything(1)
        first = (np.random.rand(3), torch.rand(3))
        seed_everything(1)
        np.testing.assert_array_equal(first[0], np.random.rand(3))
        torch.testing.assert_close(first[1], torch.rand(3), atol=0, rtol=0)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_report_null_csv_fresh_models_and_no_overwrite(self):
        instances = []

        def new_model(*args):
            model = ToyModel()
            instances.append(model)
            return model

        with patch('evaluate.read_manifest', return_value=self.manifest), patch('evaluate.test_loader', return_value=self.loader), patch('evaluate.load_checkpoint', side_effect=new_model) as loading:
            report = run_evaluation(self.config, progress=False)
            self.assertEqual(loading.call_count, 3)
            self.assertEqual(len({id(model) for model in instances}), 3)
            self.assertEqual([row['precision'] for row in report['results']], ['FP32', 'INT8', 'INT4'])
            self.assertTrue(all(row['asr_non_target'] is None for row in report['results']))
            self.assertEqual(report['provenance']['checkpoint_sha256'], sha256(self.checkpoint))
            self.assertEqual(report['environment']['device_type'], 'cpu')
            self.assertIn('torch', report['environment']['libraries'])
            self.assertNotIn('device_name', report['environment'])
            self.assertNotIn('platform', report['environment'])
            self.assertNotIn(str(self.root), json.dumps(report))
            self.assertEqual(json.loads((self.config.output / 'results.json').read_text())['results'], report['results'])
            with (self.config.output / 'results.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row['asr_non_target'] == '' for row in rows))
            before = (self.config.output / 'results.json').read_bytes()
            with self.assertRaises(FileExistsError):
                run_evaluation(self.config, progress=False)
            self.assertEqual((self.config.output / 'results.json').read_bytes(), before)
            self.assertEqual(loading.call_count, 3)

    def test_hash_failure_before_data_and_limit_cleanup(self):
        wrong = dict(self.manifest, sha256='0'*64)
        with patch('evaluate.read_manifest', return_value=wrong), patch('evaluate.test_loader') as data:
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                run_evaluation(self.config, progress=False)
            data.assert_not_called()
        self.assertFalse(self.config.output.exists())
        with patch('evaluate.read_manifest', return_value=self.manifest), patch('evaluate.test_loader', return_value=self.loader), patch('evaluate.load_checkpoint', side_effect=lambda *args: ToyModel()):
            report = run_evaluation(replace(self.config, limit=1, precisions=('fp32',)), progress=False)
            self.assertEqual(report['results'][0]['samples'], 1)
            bad = replace(self.config, output=self.root / 'bad', limit=4)
            with self.assertRaisesRegex(ValueError, 'limit exceeds'):
                run_evaluation(bad, progress=False)
            self.assertFalse(bad.output.exists())

    def test_matrix_sequential_outputs_and_config_provenance(self):
        config_path = self.root / 'matrix.toml'
        destination = self.root / 'matrix'
        config_path.write_text(f'output = "{destination}"\n[evaluation]\ndevice = "cpu"\nbatch_size = 2\n')
        order = []

        def new_model(name, path):
            order.append(name)
            return ToyModel()

        with patch('run_matrix.read_manifest', return_value=self.manifest), patch('evaluate.read_manifest', return_value=self.manifest), patch('evaluate.test_loader', return_value=self.loader), patch('evaluate.load_checkpoint', side_effect=new_model):
            reports = run_matrix(config_path, progress=False)
            self.assertEqual(order, ['resnet18']*3 + ['alexnet']*3 + ['vgg16']*3 + ['mobilenetv2']*3)
            self.assertEqual(len(reports), 4)
            with (destination / 'summary.csv').open(newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 12)
            for name in ('resnet18', 'alexnet', 'vgg16', 'mobilenetv2'):
                report = json.loads((destination / name / 'results.json').read_text())
                self.assertEqual(report['experiment']['sha256'], sha256(config_path))
                self.assertTrue((destination / name / 'results.csv').is_file())
            with self.assertRaises(FileExistsError):
                run_matrix(config_path, progress=False)
        config_path.write_text('[evaluation]\nunknown_option = 1\n')
        with self.assertRaises(ValueError):
            read_experiment(config_path)
        config_path.write_text('models = ["resnet18", "resnet18"]\n')
        with self.assertRaises(ValueError):
            read_experiment(config_path)
