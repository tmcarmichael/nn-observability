"""Run all CPU-based analysis scripts and print a combined summary.

Usage: cd nn-observability && uv run python analysis/run_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    'meta_regression.py',   # primary
    'ancova_family.py',     # supplementary
    'permutation_test.py',
    'selectivity.py',
    'pearson_vs_spearman.py',
    'loocv_scaling.py',
    'funnel_plot.py',
    'exclusive_catch_rates.py',
]

analysis_dir = Path(__file__).resolve().parent


def main():
    failed = []
    for script in SCRIPTS:
        path = analysis_dir / script
        print(f'\n{"=" * 70}')
        print(f'  {script}')
        print(f'{"=" * 70}\n')
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(analysis_dir.parent),
            capture_output=False,
        )
        if result.returncode != 0:
            failed.append(script)

    print(f'\n{"=" * 70}')
    print(f'  Summary: {len(SCRIPTS) - len(failed)}/{len(SCRIPTS)} passed')
    if failed:
        print(f'  Failed: {", ".join(failed)}')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
