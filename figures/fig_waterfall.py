"""Figure 2: Control sensitivity waterfall.

Bar chart showing how partial correlation drops as controls are added.
Raw -> softmax only -> standard (softmax + norm) -> plus entropy.
Nonlinear MLP shown separately (independent control, not cumulative).
Data from GPT-2 124M control sensitivity in transformer_observe.json.

Usage: cd nn-observability && uv run python figures/fig_waterfall.py
"""

import json

import matplotlib.pyplot as plt

from style import apply_style, RESULTS_DIR, save_fig

OUTPUT_NAME = 'control_sensitivity_waterfall.pdf'


def main():
    apply_style()

    d = json.loads((RESULTS_DIR / 'transformer_observe.json').read_text())
    cs = d['control_sensitivity']['control_sets']

    # Cumulative controls (left group) and independent control (right)
    cumulative = [
        ('none',         'Raw'),
        ('softmax_only', '+ Softmax'),
        ('standard',     '+ Norm'),
        ('plus_entropy', '+ Entropy'),
    ]
    independent = [
        ('nonlinear',    'Nonlinear MLP'),
    ]

    labels = [s[1] for s in cumulative + independent]
    values = [cs[s[0]]['mean'] for s in cumulative + independent]

    # Small gap before nonlinear to visually separate it
    positions = list(range(len(cumulative))) + [len(cumulative) + 0.3]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    # Neutral tones: raw is lighter gray, controlled bars are steel blue.
    # Color encodes family identity (Figures 1, 3); here it's decorative.
    colors = ['#B0B0B0'] + ['#5A7D9A'] * (len(cumulative) - 1) + ['#5A7D9A']

    bars = ax.bar(positions, values, color=colors, width=0.6, zorder=3)

    # Annotate each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:+.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, ha='center', fontsize=8)
    ax.set_ylabel(r'$\rho_{\mathrm{partial}}$')
    ax.set_ylim(0, 0.65)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax.grid(True, axis='y', alpha=0.2)

    save_fig(fig, OUTPUT_NAME)


if __name__ == '__main__':
    main()
