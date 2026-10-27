"""Sequential TOML experiment runner. TOML paths are relative to the project root."""
import argparse
from pathlib import Path
import tomllib
from checkpoints import ROOT, read_manifest, sha256
from evaluate import RunConfig, run_evaluation
from evaluation.results import csv_rows, csv_text
from evaluation.runtime import report_path, resolve_device
from models.registry import MODEL_NAMES

EVALUATION_KEYS = {'data_dir', 'device', 'precisions', 'batch_size', 'workers', 'seed',
                   'target_class', 'normalize', 'download_data', 'limit'}


def project_path(value):
    if not isinstance(value, (str, Path)):
        raise ValueError(f'Expected a path, got {value!r}')
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_experiment(path, output=None):
    with path.open('rb') as stream:
        config = tomllib.load(stream)
    if set(config) - {'models', 'evaluation', 'checkpoints', 'output'}:
        raise ValueError('Unknown top-level TOML keys')
    names = config.get('models', list(MODEL_NAMES))
    if not isinstance(names, list) or not names or any(name not in MODEL_NAMES for name in names) or len(set(names)) != len(names):
        raise ValueError(f'models must list distinct names from {MODEL_NAMES}')
    settings = config.get('evaluation', {})
    checkpoints = config.get('checkpoints', {})
    if not isinstance(settings, dict) or set(settings) - EVALUATION_KEYS:
        raise ValueError('Invalid or unknown evaluation settings')
    if not isinstance(checkpoints, dict) or set(checkpoints) - set(MODEL_NAMES):
        raise ValueError('Invalid checkpoint settings')
    destination = output if output is not None else project_path(config.get('output', 'artifacts/runs/baseline-matrix'))
    settings = dict(settings)
    settings['data_dir'] = project_path(settings.get('data_dir', 'datasets/cifar10'))
    if 'precisions' in settings and not isinstance(settings['precisions'], list):
        raise ValueError('precisions must be an array')
    runs = []
    for name in names:
        checkpoint = project_path(checkpoints.get(name, read_manifest(name)['local_path']))
        run = RunConfig(model=name, checkpoint=checkpoint, output=destination / name, **settings)
        run.validate()
        runs.append(run)
    return destination, runs


def run_matrix(path, *, output=None, progress=True):
    destination, runs = read_experiment(path, output)
    resolve_device(runs[0].device)
    destination.mkdir(parents=True, exist_ok=False)
    experiment = {'path': report_path(path), 'sha256': sha256(path)}
    reports = []
    try:
        for run in runs:
            reports.append(run_evaluation(run, progress=progress, experiment=experiment))
        rows = [row for report in reports for row in csv_rows(report)]
        with (destination / 'summary.csv').open('x', encoding='utf-8', newline='') as stream:
            stream.write(csv_text(rows))
    except BaseException:
        try:
            destination.rmdir()
        except OSError:
            pass
        raise
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', type=Path, default=ROOT / 'configs/experiments/baseline.toml')
    parser.add_argument('--output', type=Path, help='Override the output directory')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    try:
        reports = run_matrix(args.config, output=args.output, progress=not args.quiet)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')
    print(f"Completed {sum(len(report['results']) for report in reports)} evaluations")


if __name__ == '__main__':
    main()
