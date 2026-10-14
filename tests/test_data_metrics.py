import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from types import SimpleNamespace
from unittest.mock import patch
from data.cifar10 import TestPairs, trigger, test_loader, MEAN, STD
from evaluation.metrics import evaluate_model


class DataMetricsTest(unittest.TestCase):
    def test_trigger_and_preprocessing(self):
        image = (np.arange(32*32*3) % 144).astype(np.uint8).reshape(32,32,3)
        before = image.copy()
        attacked = trigger(image)
        np.testing.assert_array_equal(image, before)
        outside = np.ones((32, 32), dtype=bool)
        outside[27:31, 27:31] = False
        np.testing.assert_array_equal(attacked[outside], image[outside])
        self.assertTrue((attacked[27:31,27:31] == image.max()).all())
        np.testing.assert_array_equal(attacked[31], image[31])
        clean, label, poisoned, target = TestPairs([image], [3])[0]
        expected = (torch.from_numpy(image.copy()).permute(2,0,1).float()/255 - torch.tensor(MEAN)[:,None,None])/torch.tensor(STD)[:,None,None]
        torch.testing.assert_close(clean, expected)
        self.assertEqual((label, target), (3,0))
        self.assertFalse(torch.equal(clean, poisoned))
        np.testing.assert_array_equal(image, before)
        expected_trigger = (torch.tensor(143 / 255.) - torch.tensor(MEAN)) / torch.tensor(STD)
        torch.testing.assert_close(poisoned[:, 27:31, 27:31], expected_trigger[:, None, None].expand(3, 4, 4))
        unnormalized = TestPairs([image], [3], normalize=False)[0][0]
        torch.testing.assert_close(unnormalized, torch.from_numpy(image.copy()).permute(2,0,1).float()/255)

    def test_denominators_loss_and_last_batch(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x
        # Five examples, final batch has one; clean perfect, 3 triggered successes.
        labels = torch.tensor([0,1,2,0,1])
        clean = torch.eye(3)[labels]*4
        attacked = torch.eye(3)[torch.tensor([0,0,1,2,0])]*2
        pairs = list(zip(clean,labels,attacked,torch.zeros(5,dtype=torch.long)))
        result = evaluate_model(Identity(),DataLoader(pairs,batch_size=2),'cpu',progress=False)
        self.assertEqual(result['samples'],5)
        self.assertEqual(result['clean_denominator'],5)
        self.assertEqual(result['asr_all_denominator'],5)
        self.assertEqual(result['clean_accuracy'],1.)
        self.assertEqual(result['asr_all'],.6)
        self.assertAlmostEqual(result['asr_non_target'],2/3)
        self.assertEqual(result['clean_correct'],5)
        self.assertEqual(result['asr_all_successes'],3)
        self.assertEqual(result['asr_non_target_denominator'],3)
        self.assertEqual(result['asr_non_target_successes'],2)
        self.assertAlmostEqual(result['clean_loss'],torch.nn.functional.cross_entropy(clean,labels).item(),places=6)
        self.assertAlmostEqual(result['triggered_loss'],torch.nn.functional.cross_entropy(attacked,torch.zeros(5,dtype=torch.long)).item(),places=6)
        only_target = [(clean[0], torch.tensor(0), attacked[0], torch.tensor(0))]
        self.assertIsNone(evaluate_model(Identity(),DataLoader(only_target),'cpu',progress=False)['asr_non_target'])
        with self.assertRaises(ValueError):
            evaluate_model(Identity(), [], 'cpu',progress=False)

    def test_loader_defaults_and_explicit_download(self):
        images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        dataset = SimpleNamespace(data=images, targets=[4, 7])
        with patch('data.cifar10.datasets.CIFAR10', return_value=dataset) as cifar:
            loader = test_loader('fake-data')
            cifar.assert_called_once_with(root='fake-data', train=False, download=False)
            self.assertEqual(loader.batch_size, 128)
            self.assertEqual(loader.num_workers, 0)
            self.assertIsInstance(loader.sampler, SequentialSampler)
            self.assertFalse(loader.drop_last)
            clean, labels, triggered, targets = next(iter(loader))
            self.assertEqual(labels.tolist(), [4, 7])
            self.assertEqual(targets.tolist(), [0, 0])
            cifar.reset_mock()
            loader = test_loader('fake-data', download=True, batch_size=1, normalize=False, target=7)
            cifar.assert_called_once_with(root='fake-data', train=False, download=True)
            self.assertEqual(next(iter(loader))[3].tolist(), [7])
            self.assertEqual(len(loader), 2)
            for options in ({'batch_size': 0}, {'workers': -1}):
                with self.assertRaises(ValueError):
                    test_loader('fake-data', **options)

    def test_pair_validation_and_custom_target(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        pairs = TestPairs([image], [3], target=7)
        self.assertEqual(pairs[0][3], 7)
        for target in (-1, 10):
            with self.assertRaises(ValueError):
                TestPairs([image], [3], target=target)
        with self.assertRaises(ValueError):
            TestPairs([image], [])
        with self.assertRaises(ValueError):
            trigger(np.zeros((31, 32, 3), dtype=np.uint8))

    def test_forward_order_and_disabled_gradients(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.ones(1))
                self.calls = []

            def forward(self, x):
                self.calls.append((x[0, 0].item(), torch.is_inference_mode_enabled(), self.training))
                return x

        model = RecordingModel()
        clean = torch.tensor([[4., 0.], [4., 0.], [4., 0.]])
        triggered = torch.tensor([[0., 4.], [0., 4.], [0., 4.]])
        pairs = list(zip(clean, torch.zeros(3, dtype=torch.long), triggered, torch.ones(3, dtype=torch.long)))
        result = evaluate_model(model, DataLoader(pairs, batch_size=2), 'cpu', progress=False)
        self.assertEqual(model.calls, [(4., True, False), (0., True, False)] * 2)
        self.assertFalse(model.parameter.requires_grad)
        self.assertEqual(result['asr_non_target_denominator'], 3)
        self.assertEqual(result['asr_all'], 1.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_metric_parity(self):
        model = torch.nn.Linear(2, 2, bias=False).eval()
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        pairs = [(torch.tensor([2., 0.]), torch.tensor(0), torch.tensor([0., 2.]), torch.tensor(1)),
                 (torch.tensor([0., 2.]), torch.tensor(1), torch.tensor([2., 0.]), torch.tensor(1))]
        loader = DataLoader(pairs, batch_size=1)
        cpu = evaluate_model(model, loader, 'cpu', progress=False)
        cuda = evaluate_model(model.cuda(), loader, 'cuda', progress=False)
        for key in cpu:
            if key.endswith('loss'):
                self.assertAlmostEqual(cpu[key], cuda[key], places=6)
            else:
                self.assertEqual(cpu[key], cuda[key])
