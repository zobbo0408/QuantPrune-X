# Adapted from Qu-ANTI-zation, utils/learner.py (MIT).
# Copyright (c) 2021 Secure AI Systems Lab
# Source and changes: docs/provenance.md; license: licenses/Qu-ANTI-zation-MIT.txt
"""Paired clean/triggered evaluation with sample-weighted metrics."""
import torch
from torch.nn import functional as F
from tqdm import tqdm


@torch.inference_mode()
def evaluate_model(model, loader, device, *, progress=True):
    model.eval().requires_grad_(False)
    samples = correct = successes = non_target = non_target_successes = 0
    clean_loss = triggered_loss = 0.0
    for clean, labels, triggered, targets in tqdm(loader, disable=not progress):
        clean, labels = clean.to(device), labels.to(device)
        triggered, targets = triggered.to(device), targets.to(device)
        # Preserve reference order: clean forward, then triggered forward each batch.
        clean_logits = model(clean)
        triggered_logits = model(triggered)
        predicted = triggered_logits.argmax(1)
        mask = labels != targets
        samples += labels.numel()
        correct += (clean_logits.argmax(1) == labels).sum().item()
        successes += (predicted == targets).sum().item()
        non_target += mask.sum().item()
        non_target_successes += ((predicted == targets) & mask).sum().item()
        clean_loss += F.cross_entropy(clean_logits, labels, reduction='sum').item()
        triggered_loss += F.cross_entropy(triggered_logits, targets, reduction='sum').item()
    if not samples:
        raise ValueError('Cannot evaluate an empty dataset')
    return dict(samples=samples, clean_denominator=samples, clean_correct=correct, asr_all_successes=successes,
                asr_all_denominator=samples, asr_non_target_successes=non_target_successes,
                asr_non_target_denominator=non_target, clean_accuracy=correct / samples,
                asr_all=successes / samples,
                asr_non_target=non_target_successes / non_target if non_target else None,
                clean_loss=clean_loss / samples, triggered_loss=triggered_loss / samples)
