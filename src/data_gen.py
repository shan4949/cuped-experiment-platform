import numpy as np
import pandas as pd


def generate_experiment(
    n_users: int = 10_000,
    rho: float = 0.5,
    delta: float = 0.1,
    treat_frac: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic A/B experiment data with controllable pre/post correlation.

    pre_metric ~ Lognormal(0, 1)
    post_metric = pre_metric + noise + delta * is_treatment
    corr(pre, post) ≈ rho by construction via noise scaling.
    """
    rng = np.random.default_rng(seed)

    n_treat = int(n_users * treat_frac)
    n_ctrl = n_users - n_treat
    variant = np.array(["treatment"] * n_treat + ["control"] * n_ctrl)
    rng.shuffle(variant)

    pre_metric = rng.lognormal(mean=0.0, sigma=1.0, size=n_users)

    # Decompose post into pre-correlated + independent noise:
    # post = rho * pre_std + sqrt(1-rho^2) * noise, then scale back
    # so that Var(post) ≈ Var(pre) and corr(pre, post) ≈ rho
    pre_std = pre_metric / pre_metric.std()
    independent_noise = rng.standard_normal(n_users)
    post_metric = (
        rho * pre_std
        + np.sqrt(1 - rho**2) * independent_noise
    ) * pre_metric.std() + pre_metric.mean()

    is_treatment = (variant == "treatment").astype(float)
    post_metric = post_metric + delta * is_treatment

    return pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "variant": variant,
            "pre_metric": pre_metric,
            "post_metric": post_metric,
        }
    )
