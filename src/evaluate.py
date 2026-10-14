"""Evaluate a public QCB checkpoint on clean and triggered CIFAR-10 test pairs."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from torch.utils.data import DataLoader, Subset
from checkpoints import ROOT, load_checkpoint, read_manifest
from data.cifar10 import MEAN, STD, test_loader
from evaluation.metrics import evaluate_model
from evaluation.results import save_report
from evaluation.runtime import environment, provenance, report_path, resolve_device, seed_everything, utc_now
from quantization.layers import configure_model

PRECISIONS = {'fp32': None, 'int8': 8, 'int4': 4}


@dataclass(frozen=True)
class RunConfig:
    model: str
    checkpoint: Path
    data_dir: Path
    output: Path
    device: str = 'auto'
    precisions: tuple = ('fp32', 'int8', 'int4')
    batch_size: int = 128
    workers: int = 0
    seed: int = 1
    target_class: int = 0
    normalize: bool = True
    download_data: bool = False
    limit: int | None = None

    def validate(self):
        if self.model not in ('resnet18', 'alexnet'):
            raise ValueError(f'Unknown model: {self.model}')
        if not isinstance(self.precisions, (list, tuple)) or not self.precisions or any(not isinstance(p, str) or p not in PRECISIONS for p in self.precisions) or len(set(self.precisions)) != len(self.precisions):
            raise ValueError('Precisions must be a nonempty list of distinct fp32/int8/int4 values')
        for name, minimum, maximum in (('batch_size', 1, None), ('workers', 0, None), ('seed', 0, 2**32-1), ('target_class', 0, 9), ('limit', 1, None)):
            value = getattr(self, name)
            if name == 'limit' and value is None:
                continue
            if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
                raise ValueError(f'Invalid {name}: {value}')
        if not isinstance(self.device, str) or not self.device:
            raise ValueError('device must be a nonempty string')
        if type(self.normalize) is not bool or type(self.download_data) is not bool:
            raise ValueError('normalize and download_data must be booleans')
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {self.checkpoint}')


def run_evaluation(config, *, progress=True, experiment=None):
    config.validate()
    device = resolve_device(config.device)
    config.output.mkdir(parents=True, exist_ok=False)
    try:
        seed_everything(config.seed)
        manifest = read_manifest(config.model)
        hashes = provenance(config.checkpoint)
        if hashes['checkpoint_sha256'] != manifest['sha256']:
            raise ValueError('Checkpoint SHA-256 mismatch')
        started = utc_now()
        timer = time.perf_counter()
        loader = test_loader(str(config.data_dir), download=config.download_data,
                             batch_size=config.batch_size, workers=config.workers,
                             normalize=config.normalize, target=config.target_class)
        if config.limit is not None:
            if config.limit > len(loader.dataset):
                raise ValueError('limit exceeds the available test samples')
            loader = DataLoader(Subset(loader.dataset, range(config.limit)),
                                batch_size=config.batch_size, num_workers=config.workers, shuffle=False)
        rows = []
        for precision in config.precisions:
            seed_everything(config.seed)
            # Strict loading happens on CPU before moving each fresh model to its device.
            model = load_checkpoint(config.model, config.checkpoint).to(device)
            configure_model(model, PRECISIONS[precision])
            metrics = evaluate_model(model, loader, device, progress=progress)
            rows.append({'precision': precision.upper(),
                         'quantization': 'floating_point' if precision == 'fp32' else 'fake_quantization',
                         'bits': PRECISIONS[precision], **metrics})
            del model
        report = {
            'schema_version': 1, 'model': config.model, 'dataset': 'cifar10-test',
            'conditions': {'data_dir': report_path(config.data_dir),
                           'batch_size': config.batch_size, 'workers': config.workers,
                           'seed': config.seed, 'target_class': config.target_class,
                           'normalize': config.normalize, 'shuffle': False,
                           'download_data': config.download_data, 'limit': config.limit,
                           'requested_device': config.device},
            'preprocessing': {'mean': MEAN if config.normalize else None,
                              'std': STD if config.normalize else None,
                              'trigger': {'slice': [27, 31], 'channels': 'all',
                                          'value': 'image maximum before normalization'}},
            'quantization': {'weight': 'per_layer_symmetric', 'activation': 'per_layer_asymmetric',
                             'range_update': 'every_forward', 'calibration': False},
            'checkpoint': {'path': report_path(config.checkpoint),
                           'source_url': manifest['source_url'], 'zip_path': manifest['zip_path']},
            'environment': environment(device), 'provenance': hashes,
            'started_at': started, 'finished_at': utc_now(),
            'elapsed_seconds': time.perf_counter() - timer, 'results': rows,
        }
        if experiment is not None:
            report['experiment'] = experiment
        save_report(report, config.output)
        return report
    except BaseException:
        # Remove only our empty reservation; preserve any files already written.
        try:
            config.output.rmdir()
        except OSError:
            pass
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['resnet18', 'alexnet'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'datasets/cifar10')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precisions', '--precision', type=str.lower, nargs='+', choices=list(PRECISIONS), default=list(PRECISIONS))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--target-class', type=int, default=0)
    parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--download-data', action='store_true')
    parser.add_argument('--limit', type=int, help='Evaluate only the first N test samples (default: all)')
    parser.add_argument('--quiet', action='store_true', help='Disable progress bars')
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    checkpoint = args.checkpoint or ROOT / read_manifest(args.model)['local_path']
    config = RunConfig(model=args.model, checkpoint=checkpoint, data_dir=args.data_dir, output=args.output,
                       device=args.device, precisions=tuple(args.precisions), batch_size=args.batch_size,
                       workers=args.workers, seed=args.seed, target_class=args.target_class,
                       normalize=args.normalize, download_data=args.download_data, limit=args.limit)
    try:
        report = run_evaluation(config, progress=not args.quiet)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')
    for result in report['results']:
        print(f"{config.model} {result['precision']}: clean={result['clean_accuracy']:.4f}, ASR-all={result['asr_all']:.4f}")
    print(f'Results: {config.output}')


if __name__ == '__main__':
    main()
