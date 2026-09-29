import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_gen import generate_experiment
from analysis import cuped_adjust, cuped_ttest, standard_ttest, variance_reduction_report
from power import mde, required_sample_size, sample_size_savings


# ── 1. Data generator: empirical rho within 0.05 of requested ────────────────

@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7])
def test_data_gen_rho(rho, seed):
    df = generate_experiment(n_users=20_000, rho=rho, delta=0.1, seed=seed)
    empirical = df["pre_metric"].corr(df["post_metric"])
    assert abs(empirical - rho) < 0.05, f"rho={rho} seed={seed}: got {empirical:.3f}"


# ── 2. Standard t-test: false-positive rate ≈ 0.05 under H0 ─────────────────

def test_false_positive_rate():
    pvals = [
        standard_ttest(generate_experiment(n_users=5_000, rho=0.5, delta=0.0, seed=s)).p_value
        for s in range(400)
    ]
    fpr = np.mean(np.array(pvals) < 0.05)
    assert abs(fpr - 0.05) < 0.02, f"FPR={fpr:.3f} (expected ~0.05)"


# ── 3. theta does NOT change with delta (no leakage) ─────────────────────────

def test_theta_invariant_to_delta():
    base = generate_experiment(n_users=10_000, rho=0.5, delta=0.0, seed=1)
    shifted = generate_experiment(n_users=10_000, rho=0.5, delta=1.0, seed=1)

    theta_base = cuped_adjust(base)["_theta"].iloc[0]
    theta_shifted = cuped_adjust(shifted)["_theta"].iloc[0]

    assert abs(theta_base - theta_shifted) < 0.01, (
        f"theta changed with delta: {theta_base:.4f} vs {theta_shifted:.4f}"
    )


# ── 4. CUPED is unbiased AND reduces variance ────────────────────────────────

@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7])
def test_cuped_unbiased_and_lower_variance(rho):
    df = generate_experiment(n_users=50_000, rho=rho, delta=0.2, seed=99)
    raw = standard_ttest(df)
    cup = cuped_ttest(df)

    assert abs(cup.estimate - raw.estimate) < 0.1 * 0.2, (
        f"CUPED estimate biased: raw={raw.estimate:.4f} cuped={cup.estimate:.4f}"
    )
    assert cup.var_metric < raw.var_metric, (
        f"CUPED did not reduce variance: {cup.var_metric:.4f} >= {raw.var_metric:.4f}"
    )


# ── 5. Theoretical vs. empirical variance reduction within 5% ────────────────

def test_variance_reduction_matches_theory():
    df = generate_experiment(n_users=50_000, rho=0.6, delta=0.1, seed=7)
    report = variance_reduction_report(df)
    diff = abs(report["theoretical_reduction"] - report["empirical_reduction"])
    assert diff < 0.05, (
        f"Reduction mismatch: theoretical={report['theoretical_reduction']:.3f} "
        f"empirical={report['empirical_reduction']:.3f}"
    )


# ── 6. Power module: CUPED reduces N by 1-rho^2 factor ──────────────────────

@pytest.mark.parametrize("rho", [0.4, 0.6, 0.8])
def test_sample_size_reduction(rho):
    sigma = 1.0
    delta = 0.1
    n_raw = required_sample_size(sigma, delta)
    sigma_cuped = sigma * np.sqrt(1 - rho**2)
    n_cuped = required_sample_size(sigma_cuped, delta)

    expected_ratio = 1 - rho**2
    actual_ratio = n_cuped / n_raw
    assert abs(actual_ratio - expected_ratio) < 0.01, (
        f"rho={rho}: expected ratio {expected_ratio:.3f}, got {actual_ratio:.3f}"
    )


def test_mde_inverse_of_sample_size():
    n = 1000
    sigma = 1.5
    effect = mde(n, sigma)
    n_back = required_sample_size(sigma, effect)
    assert abs(n_back - n) <= 2, f"round-trip mismatch: n={n} recovered={n_back}"
