"""JSON reports and CSV rows; accuracies are fractions in [0, 1]."""
import csv
import io
import json


def csv_rows(report):
    common = {
        'model': report['model'], 'dataset': report['dataset'],
        **report['conditions'],
        'device': report['environment']['device'],
        'device_type': report['environment']['device_type'],
        'python': report['environment']['python'],
        **report['environment']['libraries'],
        **report['provenance'],
        'checkpoint': report['checkpoint']['path'],
    }
    return [{**common, **metrics} for metrics in report['results']]


def csv_text(rows):
    if not rows:
        raise ValueError('Cannot save an empty results table')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)  # None is written as an empty field.
    return stream.getvalue()


def save_report(report, output):
    # Serialize before writing, rejecting NaN/infinity instead of saving invalid JSON.
    json_text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    table = csv_text(csv_rows(report))
    with (output / 'results.json').open('x', encoding='utf-8') as stream:
        stream.write(json_text)
    with (output / 'results.csv').open('x', encoding='utf-8', newline='') as stream:
        stream.write(table)
