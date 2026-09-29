from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    estimate: float       # point estimate of treatment effect
    ci_lower: float
    ci_upper: float
    p_value: float
    var_metric: float     # variance of the metric used


def standard_ttest(df: pd.DataFrame) -> TestResult:
    """Two-sample Welch t-test on post_metric."""
    treatment = df.loc[df["variant"] == "treatment", "post_metric"].values
    control = df.loc[df["variant"] == "control", "post_metric"].values

    result = stats.ttest_ind(treatment, control, equal_var=False)
    estimate = treatment.mean() - control.mean()

    se = np.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se

    return TestResult(
        estimate=estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=result.pvalue,
        var_metric=np.concatenate([treatment, control]).var(ddof=1),
    )


def cuped_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CUPED-adjusted post metric.

    theta is estimated from control group ONLY to avoid leaking treatment signal.
    y_cuped = post_metric - theta * (pre_metric - mean(pre_metric))
    """
    ctrl = df[df["variant"] == "control"]

    # ponytail: theta from control only — critical for unbiasedness
    cov = np.cov(ctrl["post_metric"], ctrl["pre_metric"])[0, 1]
    var_pre = ctrl["pre_metric"].var(ddof=1)
    theta = cov / var_pre

    pre_mean = df["pre_metric"].mean()
    y_cuped = df["post_metric"] - theta * (df["pre_metric"] - pre_mean)

    result = df.copy()
    result["y_cuped"] = y_cuped
    result["_theta"] = theta
    return result


def cuped_ttest(df: pd.DataFrame) -> TestResult:
    """Run t-test on CUPED-adjusted metric."""
    adjusted = cuped_adjust(df)

    treatment = adjusted.loc[adjusted["variant"] == "treatment", "y_cuped"].values
    control = adjusted.loc[adjusted["variant"] == "control", "y_cuped"].values

    result = stats.ttest_ind(treatment, control, equal_var=False)
    estimate = treatment.mean() - control.mean()

    se = np.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se

    all_cuped = np.concatenate([treatment, control])
    return TestResult(
        estimate=estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=result.pvalue,
        var_metric=all_cuped.var(ddof=1),
    )


def variance_reduction_report(df: pd.DataFrame) -> dict:
    """Compare theoretical vs. empirical variance reduction."""
    rho = df["pre_metric"].corr(df["post_metric"])
    theoretical = 1 - rho**2

    adjusted = cuped_adjust(df)
    var_raw = df["post_metric"].var(ddof=1)
    var_cuped = adjusted["y_cuped"].var(ddof=1)
    empirical = var_cuped / var_raw

    return {
        "rho": rho,
        "theoretical_reduction": 1 - theoretical,
        "empirical_reduction": 1 - empirical,
        "var_raw": var_raw,
        "var_cuped": var_cuped,
        "ratio": empirical,
    }
