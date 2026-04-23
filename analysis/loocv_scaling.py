"""Leave-one-model-out cross-validation of the Qwen scaling curve.

Tests whether any single Qwen model drives the scaling relationship.
"""

import numpy as np

from analysis.load_results import load_all_models


def load_qwen_models():
    all_models = load_all_models()
    models = []
    for label, m in all_models.items():
        if m["family"] != "Qwen":
            continue
        mean = m["partial_corr"].get("mean")
        if mean is not None:
            models.append((label, np.log10(m["params_b"]), float(mean)))
    return models


def run():
    models = load_qwen_models()
    if len(models) < 3:
        print(f"Only {len(models)} Qwen models. Need at least 3.")
        return

    names = [m[0] for m in models]
    log_params = np.array([m[1] for m in models])
    pcorrs = np.array([m[2] for m in models])

    print(f"=== Leave-one-out CV on Qwen scaling ({len(models)} models) ===\n")
    print(f"  {'Left out':<15} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
    print(f"  {'-' * 43}")

    errors = []
    for i in range(len(models)):
        mask = np.ones(len(models), dtype=bool)
        mask[i] = False
        X_train = np.column_stack([log_params[mask], np.ones(mask.sum())])
        y_train = pcorrs[mask]
        beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

        x_test = np.array([log_params[i], 1.0])
        predicted = x_test @ beta
        error = pcorrs[i] - predicted

        errors.append(abs(error))
        print(f"  {names[i]:<15} {pcorrs[i]:+.4f}   {predicted:+.4f}   {error:+.4f}")

    mae = np.mean(errors)
    max_err = max(errors)
    mean_signal = np.mean(pcorrs)
    print(f"\n  Mean absolute error: {mae:.4f}")
    print(f"  Max absolute error:  {max_err:.4f}")
    print(f"  MAE as % of mean signal: {mae / mean_signal * 100:.1f}%")
    if mae < 0.03:
        print("  No single model drives the scaling relationship.")

    X_all = np.column_stack([log_params, np.ones(len(models))])
    beta_all = np.linalg.lstsq(X_all, pcorrs, rcond=None)[0]
    print(f"\n  Full regression: pcorr = {beta_all[0]:+.4f} * log10(params) + {beta_all[1]:+.4f}")
    print(f"  Slope: {beta_all[0]:+.4f} per decade of parameters")
    r2 = 1 - np.sum((pcorrs - X_all @ beta_all) ** 2) / np.sum((pcorrs - pcorrs.mean()) ** 2)
    print(f"  R²: {r2:.3f}")


if __name__ == "__main__":
    run()
