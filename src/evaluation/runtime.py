"""Device selection, deterministic seeds and evaluation provenance."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import os
import random
import sys
import numpy as np
import torch
from checkpoints import ROOT, sha256


def resolve_device(requested):
    if requested == 'auto':
        requested = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA devices are supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable in this execution environment')
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index is unavailable: {index}')
        return torch.device('cuda', index)
    return torch.device('cpu')


def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def code_sha256():
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'src').rglob('*.py')):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b'\0')
        digest.update(path.read_bytes() + b'\0')
    return digest.hexdigest()


def environment(device):
    return {
        'python': sys.version.split()[0],
        'libraries': {name: version(name) for name in ('torch', 'torchvision', 'numpy', 'pillow', 'tqdm')},
        'device': str(device), 'device_type': device.type,
        'cuda_runtime': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
        'cpu_threads': torch.get_num_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    }


def provenance(checkpoint):
    return {'checkpoint_sha256': sha256(checkpoint), 'code_sha256': code_sha256(),
            'lock_sha256': sha256(ROOT / 'uv.lock')}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def report_path(path):
    """Keep report paths independent of the host workspace location."""
    path = path.resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name
