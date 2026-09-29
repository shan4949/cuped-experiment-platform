import numpy as np
from scipy import stats


def required_sample_size(
    sigma: float,
    delta: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Two-sample z-test sample size per arm.

    n = 2 * (z_{alpha/2} + z_beta)^2 * sigma^2 / delta^2
    """
    z_alpha2 = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * (z_alpha2 + z_beta) ** 2 * sigma**2 / delta**2
    return int(np.ceil(n))


def mde(
    n: int,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum detectable effect given a fixed sample size per arm."""
    z_alpha2 = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return (z_alpha2 + z_beta) * np.sqrt(2 * sigma**2 / n)


def sample_size_savings(rho: float, n_raw: int) -> dict:
    """Report the N reduction from CUPED given a pre/post correlation rho."""
    factor = 1 - rho**2
    n_cuped = int(np.ceil(n_raw * factor))
    return {
        "n_raw": n_raw,
        "n_cuped": n_cuped,
        "reduction_pct": (1 - factor) * 100,
        "speedup_factor": 1 / factor,
    }
