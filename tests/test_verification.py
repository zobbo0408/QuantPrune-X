from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from compare_upstream import UPSTREAM_SHA, REFERENCE_FILES, check_upstream, compare_logits
from verify_independence import copy_workspace, verify


class VerificationTest(unittest.TestCase):
    def test_logits_tolerance_predictions_and_counts(self):
        logits = torch.tensor([[2., 0.], [2., 0.], [0., 2.]])
        result = compare_logits(logits, logits.clone(), torch.tensor([0, 1, 1]), torch.zeros(3, dtype=torch.long))
        self.assertEqual(result, {'max_absolute_error': 0., 'samples': 3, 'correct': 2,
                                  'target_successes': 2, 'non_target_successes': 1})
        with self.assertRaises(AssertionError):
            compare_logits(logits + 1., logits, torch.zeros(3), torch.zeros(3))
        # Close logits can still have different argmax: tolerance alone must not pass.
        with self.assertRaisesRegex(AssertionError, 'Predictions differ'):
            compare_logits(torch.tensor([[0., 1e-8]]), torch.tensor([[1e-8, 0.]]), torch.tensor([0]), torch.tensor([0]))

    def test_upstream_pin_and_modified_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in REFERENCE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('reference fixture')
            with patch('compare_upstream.subprocess.check_output', side_effect=[UPSTREAM_SHA+'\n', '']):
                hashes = check_upstream(root)
                self.assertEqual(set(hashes), set(REFERENCE_FILES))
                self.assertTrue(all(len(digest) == 64 for digest in hashes.values()))
            with patch('compare_upstream.subprocess.check_output', return_value='0'*40):
                with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                    check_upstream(root)
            with patch('compare_upstream.subprocess.check_output', side_effect=[UPSTREAM_SHA, ' M utils/qutils.py\n']):
                with self.assertRaisesRegex(ValueError, 'uncommitted'):
                    check_upstream(root)

    def test_workspace_copy_excludes_upstream_and_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            root.mkdir()
            for name in ('pyproject.toml', 'uv.lock', '.python-version', '.gitignore', 'AGENTS.md'):
                (root / name).write_text('fixture')
            for name in ('src', 'tests', 'configs', 'licenses', 'datasets/cifar10', 'docs', 'Qu-ANTI-zation', '.venv', '.git', 'extra'):
                (root / name).mkdir(parents=True)
                (root / name / 'fixture').write_text('fixture')
            (root / 'docs/provenance.md').write_text('source fixture')
            for name in ('resnet18', 'alexnet'):
                path = root / f'artifacts/checkpoints/{name}/qcb.pth'
                path.parent.mkdir(parents=True)
                path.write_bytes(b'checkpoint fixture')
            destination = Path(directory) / 'independent'
            with patch('verify_independence.ROOT', root):
                copy_workspace(destination)
                self.assertFalse((destination / 'Qu-ANTI-zation').exists())
                self.assertFalse((destination / '.venv').exists())
                self.assertFalse((destination / '.git').exists())
                self.assertTrue((destination / 'extra/fixture').exists())
                with self.assertRaises(ValueError):
                    copy_workspace(root / 'nested')
                self.assertEqual((destination / 'src/fixture').read_text(), 'fixture')
                with self.assertRaises(FileExistsError):
                    copy_workspace(destination)

    def test_invalid_verification_options_do_not_copy(self):
        with patch('verify_independence.copy_workspace') as copying, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for options in ({'threads': 0}, {'limit': 0}, {'limit': 10001}):
                with self.assertRaises(ValueError):
                    verify(root / 'workspace', root / 'report.json', **options)
            copying.assert_not_called()
