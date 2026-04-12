"""Generate all paper figures.

Runs each figure script and reports results.

Usage: cd nn-observability && uv run python figures/generate_all.py
   or: cd nn-observability && just figures
"""

import importlib
import sys
import time
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent

# All figure modules in order
FIGURE_MODULES = [
    'fig_cross_family',
    'fig_layer_profiles',
    'fig_waterfall',
    'fig_exdim',
]


def main():
    # Ensure figures/ is importable
    sys.path.insert(0, str(FIGURES_DIR))

    print(f'Generating {len(FIGURE_MODULES)} figures...\n')
    failures = []

    for name in FIGURE_MODULES:
        t0 = time.time()
        try:
            mod = importlib.import_module(name)
            mod.main()
            dt = time.time() - t0
            print(f'  [{dt:.1f}s] {name}\n')
        except Exception as e:
            dt = time.time() - t0
            print(f'  [{dt:.1f}s] {name} FAILED: {e}\n')
            failures.append((name, e))

    if failures:
        print(f'\n{len(failures)} figure(s) failed:')
        for name, e in failures:
            print(f'  {name}: {e}')
        sys.exit(1)
    else:
        print(f'All {len(FIGURE_MODULES)} figures generated.')


if __name__ == '__main__':
    main()
