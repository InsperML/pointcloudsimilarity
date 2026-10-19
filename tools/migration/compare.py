"""
Compare two harness output directories.

    python compare.py BASELINE_DIR CANDIDATE_DIR [--atol 1e-6]

Input hashes, parameters and call order must match exactly; metric results must
agree within atol; captured stdout must be identical. Exits 1 on any mismatch.
"""

import argparse
import difflib
import json
import math
import sys
from pathlib import Path

EXACT_KEYS = (
    'call',
    'params',
    'args',
    'model',
    'n',
    'shapes',
    'x',
    'y',
    'ks',
    'texts',
    'labels',
    'x_tensor',
    'y_tensor',
    'centers',
)


def result_diff(a, b):
    if a is None or b is None:
        return 0.0 if a is b else math.inf
    a, b = (v if isinstance(v, list) else [v] for v in (a, b))
    if len(a) != len(b):
        return math.inf
    return max(
        (0.0 if math.isnan(u) and math.isnan(v) else abs(u - v) for u, v in zip(a, b)),
        default=0.0,
    )


def compare_records(base, cand, atol):
    problems = []
    if len(base) != len(cand):
        problems.append(f'{len(base)} records in baseline, {len(cand)} in candidate')
    worst = 0.0
    for i, (b, c) in enumerate(zip(base, cand)):
        keys = [k for k in EXACT_KEYS if b.get(k) != c.get(k)]
        if keys:
            problems.append(f'record {i} ({b["call"]}): {", ".join(keys)} differ')
        if 'result' in b:
            d = result_diff(b['result'], c.get('result'))
            worst = max(worst, d)
            if d > atol:
                problems.append(f'record {i} ({b["call"]}): result differs by {d:.3g}')
    return problems, worst


def compare_stdout(base_file, cand_file):
    base, cand = (f.read_text().splitlines() for f in (base_file, cand_file))
    if base == cand:
        return []
    diff = list(
        difflib.unified_diff(base, cand, 'baseline', 'candidate', lineterm='', n=0)
    )
    return ['stdout differs:'] + ['    ' + line for line in diff[2:12]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--atol', type=float, default=1e-6)
    args = parser.parse_args()

    failed = False
    for base_json in sorted(args.baseline.glob('*.json')):
        target = base_json.stem
        cand_json = args.candidate / base_json.name
        if not cand_json.exists():
            print(f'MISSING {target}')
            failed = True
            continue
        base_records = json.loads(base_json.read_text())['records']
        cand_records = json.loads(cand_json.read_text())['records']
        problems, worst = compare_records(base_records, cand_records, args.atol)
        problems += compare_stdout(
            args.baseline / f'{target}.stdout.txt',
            args.candidate / f'{target}.stdout.txt',
        )
        status = 'FAIL' if problems else 'OK  '
        print(
            f'{status} {target}: {len(base_records)} records, max result diff {worst:.2g}'
        )
        for problem in problems:
            print(f'       {problem}')
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
