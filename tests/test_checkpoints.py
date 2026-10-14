import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import torch
from checkpoints import load_checkpoint, sha256
from models.registry import create_model
from prepare_checkpoints import RangeArchive, prepare


class Response(io.BytesIO):
    def __init__(self, data, *, status=206, content_range='bytes 0-0/10', length=None):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range':content_range,'Content-Length':str(len(data) if length is None else length)}


class CheckpointsTest(unittest.TestCase):
    def test_range_validation(self):
        with patch('urllib.request.urlopen',return_value=Response(b'x')):
            archive = RangeArchive('https://example.invalid/archive')
            self.assertEqual(archive.size,10)
            self.assertEqual(archive.seek(-2,2),8)
        for response in (Response(b'x',status=200), Response(b'x',content_range='bytes 1-1/10'), Response(b'xx'),Response(b'x',length=2)):
            with patch('urllib.request.urlopen',return_value=response), self.assertRaises(ValueError):
                RangeArchive('https://example.invalid/archive')

    def test_loading_and_selected_extraction(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pth'
            state = create_model('resnet18').state_dict()
            state['conv1.weight_quantizer.scale'] = torch.tensor(1.)
            torch.save(state,path)
            digest = sha256(path)
            load_checkpoint('resnet18',path,digest)
            with self.assertRaises(ValueError):
                load_checkpoint('resnet18',path,'0'*64)
            bad = dict(state); bad['unexpected'] = torch.tensor(0.)
            torch.save(bad,path)
            with self.assertRaises(RuntimeError):
                load_checkpoint('resnet18',path,sha256(path))
            bad = dict(state); bad['conv1.weight'] = torch.zeros(1)
            torch.save(bad,path)
            with self.assertRaises(RuntimeError):
                load_checkpoint('resnet18',path,sha256(path))
            torch.save(state,path)
            manifest = dict(sha256=sha256(path),zip_path='chosen.pth',local_path='unused')
            archive = Path(directory)/'models.zip'
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
                z.write(path,'chosen.pth'); z.writestr('ignored','not extracted')
            # Redirect manifest writes so tests do not mutate tracked config files.
            root = Path(directory)/'root'; (root/'configs/checkpoints').mkdir(parents=True)
            out = root/'qcb.pth'
            with patch('prepare_checkpoints.read_manifest',return_value=manifest), patch('prepare_checkpoints.ROOT',root):
                result = prepare('resnet18',archive=archive,destination=out)
                self.assertEqual(result['sha256'],sha256(path))
                before = out.read_bytes()
                with patch('prepare_checkpoints.load_checkpoint'):
                    prepare('resnet18',archive=archive,destination=out)
                self.assertEqual(out.read_bytes(),before)
            raw = bytearray(archive.read_bytes()); raw[40] ^= 1
            archive.write_bytes(raw)
            with patch('prepare_checkpoints.read_manifest',return_value=manifest), patch('prepare_checkpoints.ROOT',root):
                with self.assertRaises(zipfile.BadZipFile):
                    prepare('resnet18',archive=archive,destination=root/'broken.pth')
            self.assertFalse((root/'broken.pth').exists())

    def test_remote_zip_reads_selected_member_only(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('selected.pth', b'checkpoint contents')
            archive.writestr('ignored.bin', bytes(range(256)) * 1000)
        payload = data.getvalue()
        requested = []

        def request_range(request, timeout):
            value = request.get_header('Range')
            start, end = map(int, value.removeprefix('bytes=').split('-'))
            requested.append((start, end))
            return Response(payload[start:end+1], content_range=f'bytes {start}-{end}/{len(payload)}')

        with patch('urllib.request.urlopen', side_effect=request_range):
            remote = RangeArchive('https://example.invalid/archive')
            with zipfile.ZipFile(remote) as archive:
                self.assertEqual(archive.read('selected.pth'), b'checkpoint contents')
            remote.seek(0, 2)
            self.assertEqual(remote.read(10), b'')
            remote.seek(0)
            self.assertEqual(remote.read(0), b'')
            with self.assertRaises(ValueError):
                remote.seek(-1)
        self.assertGreater(len(requested), 1)
        self.assertTrue(all(0 <= start <= end < len(payload) for start, end in requested))

    def test_unsupported_range_does_not_read_body(self):
        response = Response(b'full archive should not be read', status=200)
        with patch.object(response, 'read', wraps=response.read) as read:
            with patch('urllib.request.urlopen', return_value=response):
                with self.assertRaisesRegex(ValueError, '--zip LOCAL.zip'):
                    RangeArchive('https://example.invalid/archive')
            read.assert_not_called()

    def test_changed_archive_size_rejected(self):
        with patch('urllib.request.urlopen', side_effect=[
            Response(b'x', content_range='bytes 0-0/10'),
            Response(b'x', content_range='bytes 0-0/11'),
        ]):
            remote = RangeArchive('https://example.invalid/archive')
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                remote.read(1)

    def test_local_copy_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs/checkpoints').mkdir(parents=True)
            source = root / 'source.pth'
            torch.save(torch.nn.Linear(2, 2).state_dict(), source)
            manifest = dict(sha256=sha256(source), zip_path='chosen.pth', local_path='qcb.pth')
            with patch('prepare_checkpoints.ROOT', root), patch('prepare_checkpoints.read_manifest', return_value=manifest), patch('checkpoints.create_model', side_effect=lambda name: torch.nn.Linear(2, 2)):
                result = prepare('resnet18', checkpoint=source)
                self.assertEqual(sha256(root / 'qcb.pth'), sha256(source))
                self.assertEqual(json.loads((root / 'configs/checkpoints/resnet18.json').read_text()), result)
                with patch('prepare_checkpoints.load_checkpoint') as loading:
                    prepare('resnet18', checkpoint=root / 'does-not-exist')
                    loading.assert_called_once()
                corrupt_manifest = dict(manifest, sha256='0' * 64)
                with patch('prepare_checkpoints.read_manifest', return_value=corrupt_manifest):
                    with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                        prepare('resnet18', checkpoint=source, destination=root / 'bad.pth')
                self.assertFalse((root / 'bad.pth').exists())
                self.assertEqual(set(root.iterdir()), {root / 'configs', source, root / 'qcb.pth'})

    def test_invalid_and_missing_state_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pth'
            with patch('checkpoints.create_model', side_effect=lambda name: torch.nn.Linear(2, 2)):
                for state in (['not a state dictionary'], {1: torch.ones(1)}, {'weight': torch.ones(2, 2)}, {'weight': torch.ones(2, 2), 'bias': torch.zeros(2), 'not_weight_quantizer': torch.ones(1)}):
                    torch.save(state, path)
                    with self.assertRaises((ValueError, RuntimeError)):
                        load_checkpoint('resnet18', path, sha256(path))
            with patch('checkpoints.read_manifest', return_value={'sha256': None}):
                with self.assertRaisesRegex(ValueError, 'No verified checkpoint hash'):
                    load_checkpoint('resnet18', path)
