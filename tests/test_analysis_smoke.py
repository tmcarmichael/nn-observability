"""Smoke tests for analysis scripts against committed data.

Verifies each script's core computation runs without crashing on the
committed results JSONs. Tests import the computation functions directly
rather than shelling out, so coverage is measured.
"""

from __future__ import annotations

import numpy as np

from analysis.load_results import (
    load_all_models,
    load_control_sensitivity,
    load_model_means,
    load_per_seed,
    load_random_head_baselines,
)


class TestSelectivity:
    def test_random_head_baselines_load(self) -> None:
        baselines = load_random_head_baselines()
        assert len(baselines) > 0
        for label, _family, _params_b, rh in baselines:
            assert isinstance(label, str)
            assert isinstance(rh, float)
            assert abs(rh) < 0.5

    def test_control_sensitivity_load(self) -> None:
        models = load_control_sensitivity()
        assert len(models) > 0
        for m in models:
            assert "none" in m
            assert "standard" in m

    def test_selectivity_computation(self) -> None:
        from analysis.selectivity import analyze_selectivity

        analyze_selectivity()


class TestMetaRegression:
    def test_per_seed_sufficient(self) -> None:
        rows = load_per_seed()
        assert len(rows) >= 10
        families = set(r[0] for r in rows)
        assert len(families) >= 3

    def test_per_seed_values_bounded(self) -> None:
        for _fam, _label, _params, _seed, pcorr in load_per_seed():
            assert -1.0 <= pcorr <= 1.0

    def test_mixed_effects_runs(self) -> None:
        from analysis.meta_regression import run_mixed_effects

        run_mixed_effects()


class TestLoocvScaling:
    def test_qwen_loocv(self) -> None:
        from analysis.loocv_scaling import load_qwen_models

        models = load_qwen_models()
        assert len(models) >= 3
        log_params = np.array([m[1] for m in models])
        pcorrs = np.array([m[2] for m in models])
        X = np.column_stack([log_params, np.ones(len(models))])
        beta = np.linalg.lstsq(X, pcorrs, rcond=None)[0]
        mae = float(np.mean(np.abs(pcorrs - X @ beta)))
        assert mae < 0.1


class TestExclusiveCatch:
    def test_flagging_data_loads(self) -> None:
        from analysis.exclusive_catch_rates import RESULTS_DIR, load_flagging

        to_path = RESULTS_DIR / "transformer_observe.json"
        data = load_flagging(to_path)
        assert "6a" in data
        assert "per_seed" in data["6a"]
        assert "n_test_tokens" in data["6a"]


class TestPermutationTest:
    def test_family_f_on_real_data(self) -> None:
        from analysis.permutation_test import family_f_stat

        means = load_model_means()
        families = [m[0] for m in means]
        log_params = np.array([m[1] for m in means])
        pcorrs = np.array([m[2] for m in means])
        f = family_f_stat(families, log_params, pcorrs)
        assert f > 1.0


class TestAncova:
    def test_ancova_runs(self) -> None:
        from analysis.ancova_family import run_ancova

        run_ancova()


class TestFunnelPlot:
    def test_model_stats_load(self) -> None:
        from analysis.funnel_plot import load_model_stats

        stats = load_model_stats()
        assert len(stats) >= 5
        for m in stats:
            assert "mean" in m
            assert "se" in m
            assert m["se"] >= 0

    def test_eggers_test(self) -> None:
        from analysis.funnel_plot import eggers_test, load_model_stats

        stats = load_model_stats()
        means = [m["mean"] for m in stats if m["se"] > 0]
        ses = [m["se"] for m in stats if m["se"] > 0]
        intercept, t_stat, p_value = eggers_test(means, ses)
        assert 0.0 <= p_value <= 1.0


class TestModelMeans:
    def test_complete(self) -> None:
        means = load_model_means()
        assert len(means) >= 10
        for family, _log_params, pcorr in means:
            assert isinstance(family, str)
            assert -1.0 <= pcorr <= 1.0

    def test_all_models_have_required_fields(self) -> None:
        for _label, m in load_all_models().items():
            assert "family" in m
            assert "params_b" in m
            assert "partial_corr" in m
            assert "mean" in m["partial_corr"]
