"""Install, test and evaluate a new workspace containing no upstream directory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from checkpoints import ROOT, sha256
from models.registry import MODEL_NAMES


def copy_workspace(destination):
    destination = destination.resolve()
    source = ROOT.resolve()
    if destination == source or source in destination.parents:
        raise ValueError('Independent workspace must be outside the source workspace')
    shutil.copytree(source, destination,
                    ignore=shutil.ignore_patterns('Qu-ANTI-zation', '.venv', '.git',
                                                  '__pycache__', '*.pyc'))
    if (destination / 'Qu-ANTI-zation').exists():
        raise AssertionError('Independent workspace contains upstream code')


def verify(workspace, output, device='cpu', threads=4, limit=None):
    if output.exists():
        raise FileExistsError(output)
    if threads < 1 or (limit is not None and not 1 <= limit <= 10000):
        raise ValueError('threads must be positive; limit must be in [1, 10000]')
    copy_workspace(workspace)
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.with_suffix('.logs')
    logs.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(workspace / 'src'), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    steps = []
    def execute(name, command):
        log = logs / f'{name}.log'
        print(f'{name}: started; log: {log}', flush=True)
        with log.open('x') as stream:
            process = subprocess.run(command, cwd=workspace, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'{name} failed ({process.returncode}); see {log}')
        steps.append({'name': name, 'command': command, 'log': str(log), 'returncode': process.returncode})
        print(f'{name}: passed', flush=True)
    execute('install', ['uv', 'sync', '--locked'])
    execute('dependencies', ['uv', 'pip', 'check'])
    execute('tests', ['uv', 'run', '--locked', 'python', '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    evaluation_output = workspace / 'artifacts/runs/independent-baseline'
    if evaluation_output.exists():
        raise FileExistsError(evaluation_output)
    config = workspace / 'artifacts/verification/independent-matrix.toml'
    config.parent.mkdir(parents=True, exist_ok=True)
    text = f'output = "artifacts/runs/independent-baseline"\n[evaluation]\ndevice = {json.dumps(device)}\n'
    if limit is not None:
        text += f'limit = {limit}\n'
    with config.open('x') as stream:
        stream.write(text)
    execute('evaluation', ['uv', 'run', '--locked', 'python', '-B', 'src/run_matrix.py', str(config), '--quiet'])
    reports = {}
    for name in MODEL_NAMES:
        path = workspace / 'artifacts/runs/independent-baseline' / name / 'results.json'
        report = json.loads(path.read_text())
        if len(report['results']) != 3 or any(row['samples'] != (limit or 10000) for row in report['results']):
            raise AssertionError('Independent evaluation did not complete the requested matrix')
        if limit is None and any(row['asr_non_target_denominator'] != 9000 for row in report['results']):
            raise AssertionError('Expected 9000 non-target samples')
        reports[name] = {'path': str(path), 'sha256': sha256(path), 'report': report}
    record = {'status': 'passed', 'workspace': str(workspace), 'upstream_directory_present': False,
              'steps': steps, 'evaluations': reports}
    with output.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    try:
        verify(args.workspace.resolve(), args.output.resolve(), args.device, args.threads, args.limit)
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
