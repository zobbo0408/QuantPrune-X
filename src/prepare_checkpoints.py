"""Extract selected public ZIP members without downloading the entire archive."""
import argparse
import io
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.request
import zipfile
from checkpoints import ROOT, load_checkpoint, read_manifest, sha256
from models.registry import MODEL_NAMES

DOWNLOAD_URL = 'https://drive.usercontent.google.com/download?id=1RwJfqAAnz9fUjsnXxsyNqAwHE5PZLhkX&export=download&confirm=t'


class RangeArchive(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        self.size = None
        self._range(0, 0)

    def _range(self, start, end):
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=120) as response:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if response.status != 206 or not match:
                raise ValueError('Server does not support HTTP Range. Supply --zip LOCAL.zip or --checkpoint LOCAL.pth; no full ZIP was downloaded.')
            lo, hi, total = map(int, match.groups())
            if (lo, hi) != (start, end) or total <= hi or (self.size is not None and total != self.size):
                raise ValueError('Invalid or inconsistent HTTP Content-Range')
            self.size = total
            data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise ValueError('HTTP Range size mismatch')
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != len(data):
                raise ValueError('HTTP Content-Length mismatch')
            return data

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset if whence == 2 else -1
        if position < 0:
            raise ValueError('Invalid seek')
        self.position = position
        return position

    def read(self, size=-1):
        size = max(0, self.size - self.position) if size < 0 else min(size, max(0, self.size - self.position))
        if size == 0:
            return b''
        data = self._range(self.position, self.position + size - 1)
        self.position += size
        return data


def prepare(model, *, archive=None, checkpoint=None, destination=None):
    manifest = read_manifest(model)
    destination = Path(destination or ROOT / manifest['local_path'])
    if destination.exists():
        load_checkpoint(model, destination)
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            if checkpoint is not None:
                with Path(checkpoint).open('rb') as source:
                    while block := source.read(1024 * 1024):
                        stream.write(block)
            else:
                source = Path(archive) if archive is not None else RangeArchive(DOWNLOAD_URL)
                with zipfile.ZipFile(source) as zip_file:
                    member = zip_file.getinfo(manifest['zip_path'])
                    if max(member.file_size, member.compress_size) > 256 * 1024 * 1024:
                        raise ValueError('Unexpectedly large checkpoint')
                    # ZipFile.read verifies uncompressed size and ZIP CRC.
                    stream.write(zip_file.read(member))
            stream.flush()
            digest = sha256(temporary)
            if manifest['sha256'] and digest != manifest['sha256']:
                raise ValueError('Public checkpoint SHA-256 mismatch')
            load_checkpoint(model, temporary, expected_hash=digest)
            manifest.update(sha256=digest, validation='sha256-and-strict-state-dict', local_path=str(destination.relative_to(ROOT)) if destination.is_relative_to(ROOT) else str(destination))
            # Exclusive destination creation prevents accidental replacement.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    manifest_path = ROOT / 'configs/checkpoints' / f'{model}.json'
    # Publish complete JSON atomically, so interruption cannot truncate the manifest.
    with tempfile.NamedTemporaryFile(mode='w', dir=manifest_path.parent, delete=False) as metadata:
        metadata_path = Path(metadata.name)
        try:
            metadata.write(json.dumps(manifest, indent=2) + '\n')
            metadata.flush()
            os.replace(metadata_path, manifest_path)
        finally:
            metadata_path.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=[*MODEL_NAMES, 'all'], default='all')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--zip', type=Path)
    group.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.model == 'all' and args.checkpoint:
        parser.error('--checkpoint requires one --model')
    for model in MODEL_NAMES if args.model == 'all' else (args.model,):
        try:
            manifest = prepare(model, archive=args.zip, checkpoint=args.checkpoint)
        except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
            parser.exit(1, f'{model}: {exc}\n')
        print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
