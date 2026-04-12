"""Figure 4: Ex/dim sensitivity at Qwen 0.5B.

Seven token budgets (150, 250, 350, 450, 600, 800, 1000 ex/dim)
showing pcorr at peak layer. Signal is unreliable below 450 ex/dim
and saturates above 800.

Data sources:
  - 150, 250, 350, 450, 800: qwen05b_exdim_sweep.json
  - 600: qwen05b_v3_results.json
  - 1000: qwen05b_v2_results.json

Usage: cd nn-observability && uv run python figures/fig_exdim.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from style import apply_style, PALETTE, RESULTS_DIR, save_fig

OUTPUT_NAME = 'exdim_sensitivity.pdf'


def main():
    apply_style()

    # Load sweep data (5 points)
    sweep = json.loads((RESULTS_DIR / 'qwen05b_exdim_sweep.json').read_text())
    sweep_data = sweep['ex_dim_sweep']

    # Load 600 ex/dim from v3
    v3 = json.loads((RESULTS_DIR / 'qwen05b_v3_results.json').read_text())
    v3_pc = v3['partial_corr']

    # Load 1000 ex/dim from v2
    v2 = json.loads((RESULTS_DIR / 'qwen05b_v2_results.json').read_text())
    v2_pc = v2['partial_corr']

    # Assemble all 7 points: (ex_dim, mean, std)
    points = []
    for ex_dim_str, entry in sorted(sweep_data.items(), key=lambda x: int(x[0])):
        ms = entry['multi_seed']
        points.append((int(ex_dim_str), ms['mean'], ms['std']))

    points.append((600, v3_pc['mean'], v3_pc.get('std', 0)))
    points.append((1000, v2_pc['mean'], v2_pc.get('std', 0)))
    points.sort(key=lambda x: x[0])

    exdims = np.array([p[0] for p in points])
    means = np.array([p[1] for p in points])
    stds = np.array([p[2] for p in points])

    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    ax.errorbar(exdims, means, yerr=stds,
                fmt='-o', color=PALETTE['Qwen'],
                markersize=6, capsize=3, capthick=0.8, zorder=3)

    # Shade the unstable region
    ax.axvspan(0, 450, alpha=0.06, color='red', zorder=0)
    ax.text(300, 0.04, 'Unstable', fontsize=8, color='#999999', ha='center')

    # Mark the detection threshold
    ax.axvline(450, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xlabel('Examples per hidden dimension (ex/dim)')
    ax.set_ylabel(r'$\rho_{\mathrm{partial}}$')
    ax.set_xlim(80, 1100)
    ax.set_ylim(0.05, 0.30)
    ax.grid(True, alpha=0.2)

    save_fig(fig, OUTPUT_NAME)


if __name__ == '__main__':
    main()
