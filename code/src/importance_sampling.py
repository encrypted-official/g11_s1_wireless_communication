"""
importance_sampling.py
----------------------
Importance Sampling (IS) randomized algorithm for Milestone 4.

Instead of drawing noise from the true distribution p = CN(0, sigma2),
samples are drawn from a proposal q = CN(0, sigma2_q) with sigma2_q > sigma2.
This concentrates simulation effort near the detection threshold, reducing
estimator variance for rare events (low Pd at low SNR, small Pf targets).

Each trial is reweighted by the likelihood ratio:
    w_tilde = p(w) / q(w)

so the IS estimator remains unbiased:
    P_hat_IS = sum(w_tilde_i * 1{Tmc_i >= lambda}) / sum(w_tilde_i)

For complex Gaussian CN(0, sigma2) where sigma2 is the TOTAL complex power
E[|w[n]|^2]:

    p(w) = (1 / (pi * sigma2))^N * exp(-||w||^2 / sigma2)
    q(w) = (1 / (pi * sigma2_q))^N * exp(-||w||^2 / sigma2_q)

    log(p/q) = N * log(sigma2_q / sigma2) - ||w||^2 * (1/sigma2 - 1/sigma2_q)

Weights are computed in log-space and exponentiated with the log-sum-exp
trick to avoid numerical underflow / overflow.
"""

import numpy as np
from typing import Tuple, Callable, List

def generate_proposal_noise(
    num_samples: int,
    sigma2_q: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw N complex noise samples from the proposal q = CN(0, sigma2_q).

    Each component (real / imag) has std = sqrt(sigma2_q / 2) so that
    the total complex power E[|w[n]|^2] = sigma2_q.
    """
    std = np.sqrt(sigma2_q / 2.0)
    return rng.normal(0.0, std, num_samples) + 1j * rng.normal(0.0, std, num_samples)

def compute_log_importance_weight(
    w: np.ndarray,
    sigma2: float,
    sigma2_q: float,
) -> float:
    """
    Compute the log importance weight log(p(w) / q(w)) for a noise sample w
    drawn from the proposal q = CN(0, sigma2_q).

    Derivation (complex Gaussian, total-power convention):
        log(p/q) = N * log(sigma2_q / sigma2)
                   - ||w||^2 * (1/sigma2 - 1/sigma2_q)

    where ||w||^2 = sum_n |w[n]|^2.
    """
    N = len(w)
    norm_sq = float(np.sum(np.abs(w) ** 2))
    return N * np.log(sigma2_q / sigma2) - norm_sq * (1.0 / sigma2 - 1.0 / sigma2_q)


def is_estimate(
    log_weights: np.ndarray,
    indicators: np.ndarray,
) -> Tuple[float, float]:
    """
    Self-normalised IS estimator with empirical 95 % confidence interval.

        P_hat_IS = sum(w_i * ind_i) / sum(w_i)

    Uses the log-sum-exp trick for numerical stability.

    Returns
    -------
    p_hat   : float  — IS probability estimate
    ci_hw   : float  — half-width of 95 % confidence interval
    """
    max_lw = np.max(log_weights)
    weights = np.exp(log_weights - max_lw)

    total_w = np.sum(weights)
    p_hat = float(np.sum(weights * indicators) / total_w)

    norm_w = weights / total_w
    variance_est = float(np.sum(norm_w ** 2 * (indicators - p_hat) ** 2))
    ci_hw = 1.96 * np.sqrt(variance_est)

    return p_hat, ci_hw


def compute_ess(log_weights: np.ndarray) -> float:
    """
    Effective Sample Size (Eq. 8 in the report):
        ESS = (sum w_i)^2 / sum(w_i^2)

    ESS close to M means IS behaves like standard MC.
    ESS << M means weight degeneracy — proposal needs retuning.
    """
    max_lw = np.max(log_weights)
    w = np.exp(log_weights - max_lw)
    return float(np.sum(w) ** 2 / np.sum(w ** 2))

def estimate_pfa_is(
    detector_fn: Callable[[np.ndarray], Tuple[int, float]],
    num_samples: int,
    num_trials: int,
    sigma2: float,
    sigma2_q: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Estimate Pf = P(Tmc >= lambda | H0) using Importance Sampling.

    Under H0 there is no signal; x = w_q (proposal noise only).
    The IS weights correct for sampling from q instead of p.

    Returns
    -------
    pf_hat  : float — IS estimate of Pf
    ci_hw   : float — 95 % CI half-width
    ess     : float — effective sample size
    """
    log_weights = np.zeros(num_trials)
    indicators  = np.zeros(num_trials)

    for m in range(num_trials):
        w_q = generate_proposal_noise(num_samples, sigma2_q, rng)
        decision, _ = detector_fn(w_q)
        log_weights[m] = compute_log_importance_weight(w_q, sigma2, sigma2_q)
        indicators[m]  = float(decision)

    pf_hat, ci_hw = is_estimate(log_weights, indicators)
    ess = compute_ess(log_weights)
    return pf_hat, ci_hw, ess


def sweep_pd_vs_snr_is(
    detector_fn: Callable[[np.ndarray], Tuple[int, float]],
    snr_db_values: List[float],
    signal_generator_fn: Callable[[float], np.ndarray],
    num_samples: int,
    num_trials: int,
    sigma2: float,
    sigma2_q: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep Pd vs SNR using Importance Sampling.

    Only Stage 1 (noise from q) and Stage 5 (weighted indicator) are modified.
    Signal generation, CAC computation, Tmc, and threshold are unchanged.

    Returns
    -------
    snr_db_arr : (L,) — SNR values in dB
    pd_arr     : (L,) — IS Pd estimates
    ci_arr     : (L,) — 95 % CI half-widths
    ess_arr    : (L,) — effective sample sizes
    """
    snr_db_arr = np.array(snr_db_values, dtype=float)
    pd_arr     = np.zeros(len(snr_db_values))
    ci_arr     = np.zeros(len(snr_db_values))
    ess_arr    = np.zeros(len(snr_db_values))

    for i, snr_db in enumerate(snr_db_values):
        snr_linear  = 10.0 ** (snr_db / 10.0)
        log_weights = np.zeros(num_trials)
        indicators  = np.zeros(num_trials)

        for m in range(num_trials):
            w_q = generate_proposal_noise(num_samples, sigma2_q, rng)
            signal = signal_generator_fn(snr_linear)
            x = signal + w_q
            decision, _ = detector_fn(x)
            log_weights[m] = compute_log_importance_weight(w_q, sigma2, sigma2_q)
            indicators[m]  = float(decision)

        pd_arr[i], ci_arr[i] = is_estimate(log_weights, indicators)
        ess_arr[i] = compute_ess(log_weights)

    return snr_db_arr, pd_arr, ci_arr, ess_arr
