# Adapted from Qu-ANTI-zation, utils/networks.py (MIT).
# Copyright (c) 2021 Secure AI Systems Lab
# Source and changes: docs/provenance.md; license: licenses/Qu-ANTI-zation-MIT.txt
"""Hash-checked, CPU-first loading of public QCB state dictionaries."""
import hashlib
import json
from pathlib import Path
import torch
from models.registry import create_model

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(model):
    with (ROOT / 'configs/checkpoints' / f'{model}.json').open() as stream:
        return json.load(stream)


def load_checkpoint(model_name, path, expected_hash=None):
    expected_hash = expected_hash or read_manifest(model_name)['sha256']
    if not expected_hash:
        raise ValueError('No verified checkpoint hash; run prepare_checkpoints.py first')
    if sha256(path) != expected_hash:
        raise ValueError(f'Checkpoint SHA-256 mismatch: {path}')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError('Expected a state dictionary')
    state = {k: v for k, v in state.items()
             if not {'weight_quantizer', 'activation_quantizer'}.intersection(k.split('.'))}
    model = create_model(model_name)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)
