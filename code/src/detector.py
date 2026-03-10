# multi-cycle cyclostationary detector with a deterministic threshold decision rule.
"""
- the threshold lambda = analytically derived from the closed-form false alarm probability expression (no randomization).
- given a target Pfa, the threshold is fixed as: lambda=(1/Pfa)-1
- decision rule is deterministic: H1 if Tmc>=lambda, else H0.

- test statistic Tmc = numerator / denominator
- numerator = cyclic autocorrelation energy at OFDM cyclic frequencies
- denominator = reference CAC energy at a non-signal frequency
- under H0 (noise only) -> Pfa(lambda)=1/(lambda+1) -> lambda=(1/Pfa)-1

"""

import numpy as np
from typing import List, Tuple
from cyclostationary_features import (
    compute_multi_cycle_numerator,
    compute_reference_energy,
    compute_ofdm_cyclic_frequencies
)

def compute_threshold_from_pfa(target_pfa: float) -> float:
    """
    Pfa(lambda) = P(Tmc>lambda|H0) = 1-F(lambda) = 1/(lambda+1)
    lambda = (1/Pfa)-1
    this is a closed-form, deterministic threshold which does not depend on the unknown noise variance, making it robust under noise uncertainty.
    
    """
    if not (0 < target_pfa < 1):
        raise ValueError(f"target_pfa must be in (0, 1), got {target_pfa}")
    return (1.0 / target_pfa) - 1.0


def compute_test_statistic(
    x: np.ndarray,
    tau: int,
    tau_bar: int,
    cyclic_freqs: List[float],
    beta: float
) -> float:
    """
    Tmc = numerator(x, tau, {alpha_k}) / denominator(x, tau_bar, beta)

    x = samples of received signal (either H0: noise only, or H1: signal + noise).
    tau = main time lag; should equal N_FFT for OFDM detection.
    tau_bar = reference time lag for denominator.
    cyclic_freqs = signal cyclic frequencies alpha_k = k / (N_FFT + N_CP).
    beta = reference cyclic frequency, it is not a signal frequency.

    """
    numerator = compute_multi_cycle_numerator(x, tau, cyclic_freqs)
    denominator = compute_reference_energy(x, tau_bar, beta)

    # Guard against near-zero denominator (numerical safety)
    if denominator < 1e-30:
        return 0.0

    return numerator / denominator


def make_decision(tmc: float, threshold: float) -> int:
    """
    Decision rule: if Tmc >= lambda  →  H1 (signal present), else H0 (noise only).
    tmc = computed test statistic value.
    threshold = detection threshold lambda.

    """
    return 1 if tmc >= threshold else 0

class MultiCycleDetector:
    """
    fft_size = OFDM FFT size.
    cp_length = OFDM cyclic prefix length.
    k_values = integer indices for cyclic frequencies.
    beta = reference cyclic frequency for denominator.
    target_pfa = target probability of false alarm.
    threshold = detection threshold which is computed analytically .
    cyclic_freqs = computed OFDM cyclic frequencies.
    tau = main time lag.
    tau_bar = reference time lag.

    """

    def __init__(
        self,
        fft_size: int,
        cp_length: int,
        k_values: List[int],
        beta: float,
        target_pfa: float
    ):
        self.fft_size = fft_size
        self.cp_length = cp_length
        self.k_values = k_values
        self.beta = beta
        self.target_pfa = target_pfa

        # fixed parameters
        self.cyclic_freqs = compute_ofdm_cyclic_frequencies(fft_size, cp_length, k_values)
        self.tau = fft_size
        self.tau_bar = fft_size  # reference lag; same lag, different freq

        # fixed analytically from target Pfa
        self.threshold = compute_threshold_from_pfa(target_pfa)

    def detect(self, x: np.ndarray) -> Tuple[int, float]:
        """
        x = received complex signal samples.
        decision = 1 if H1 (signal detected) or 0 if H0 (noise only).
        tmc = computed test statistic value.
        
        """
        tmc = compute_test_statistic(
            x,
            tau=self.tau,
            tau_bar=self.tau_bar,
            cyclic_freqs=self.cyclic_freqs,
            beta=self.beta
        )
        decision = make_decision(tmc, self.threshold)
        return decision, tmc

    def theoretical_pfa(self) -> float:
        return 1.0 / (self.threshold + 1.0) # theoretical probability of false alarm for the set threshold = Pfa = 1/(lambda+1)

    def __repr__(self) -> str:
        return (
            f"MultiCycleDetector("
            f"fft_size={self.fft_size}, cp_length={self.cp_length}, "
            f"K={len(self.k_values)}, target_pfa={self.target_pfa:.3f}, "
            f"threshold={self.threshold:.4f})"
        )
