"""
run_simulation.py
-----------------
Main experiment script for the Multi-Cycle Cyclostationary Spectrum Sensing Simulation.

CSE400 — Fundamentals of Probability in Computing
Group 11, Milestone 4

BASELINE PIPELINE (Milestone 3, unchanged):
    Step 1 → Generate OFDM signal  s[n]
    Step 2 → Generate Gaussian noise  w[n]  ~  CN(0, sigma^2)
    Step 3 → Form received signal  x[n] = s[n] + w[n]
    Step 4 → Compute cyclic autocorrelation  R_hat_x^alpha(tau)
    Step 5 → Compute multi-cycle test statistic  Tmc
    Step 6 → Compare with threshold  lambda
    Step 7 → Decision: H0 or H1
    Step 8 → (across trials) Measure Pf and Pd

MILESTONE 4 ADDITION — Importance Sampling (IS) randomization:
    Stage 1 (modified): noise drawn from proposal  q = CN(0, sigma_q^2),  sigma_q^2 > sigma^2
    Stage 5 (modified): binary indicator multiplied by importance weight  w_tilde_i
    Stages 2-4, 6-7:    identical to baseline — only sampling strategy changes

Usage:
    python experiments/run_simulation.py
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signal_generator import generate_ofdm_signal
from noise_generator import generate_complex_gaussian_noise, generate_noise_with_uncertainty
from cyclostationary_features import (
    compute_multi_cycle_numerator,
    compute_reference_energy,
    compute_ofdm_cyclic_frequencies,
)
from detector import MultiCycleDetector, compute_threshold_from_pfa
from metrics import (
    sweep_pd_vs_snr,
    compute_empirical_pfa_vs_threshold,
    compute_theoretical_pfa_vs_threshold,
    summarize_results,
)
from importance_sampling import (
    estimate_pfa_is,
    sweep_pd_vs_snr_is,
)

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "lines.linewidth":  2.0,
})

COLORS = {
    "ofdm":      "#2563eb",
    "noise":     "#dc2626",
    "received":  "#16a34a",
    "theory":    "#7c3aed",
    "multi":     "#2563eb",
    "single":    "#f59e0b",
    "threshold": "#dc2626",
    "is":        "#059669",   # green for IS algorithm
    "mc":        "#2563eb",   # blue for MC baseline
}


# ════════════════════════════════════════════════════════════════════════════
# TRIAL RESULT
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class TrialResult:
    s:           np.ndarray
    w:           np.ndarray
    x:           np.ndarray
    numerator:   float
    denominator: float
    tmc:         float
    threshold:   float
    decision:    int
    hypothesis:  str


# ─────────────────────────────────────────────────────────────────────────────
# CORE PIPELINE (baseline, unchanged from Milestone 3)
# ─────────────────────────────────────────────────────────────────────────────
def run_single_trial(
    hypothesis: str,
    fft_size: int,
    cp_length: int,
    num_symbols: int,
    N: int,
    signal_power: float,
    noise_variance: float,
    noise_uncertainty_db: float,
    cyclic_freqs: List[float],
    tau: int,
    beta: float,
    threshold: float,
    rng: np.random.Generator,
) -> TrialResult:
    s = generate_ofdm_signal(
        fft_size=fft_size, cp_length=cp_length, num_symbols=num_symbols,
        total_samples=N, signal_power=signal_power, rng=rng,
    )
    w = generate_noise_with_uncertainty(
        num_samples=N, nominal_variance=noise_variance,
        uncertainty_db=noise_uncertainty_db, rng=rng,
    )
    x = (s + w) if hypothesis == "H1" else w.copy()

    numerator   = compute_multi_cycle_numerator(x, tau=tau, cyclic_freqs=cyclic_freqs)
    denominator = compute_reference_energy(x, tau_bar=tau, beta=beta)
    tmc         = numerator / denominator if denominator > 1e-30 else 0.0
    decision    = 1 if tmc >= threshold else 0

    return TrialResult(
        s=s, w=w, x=x,
        numerator=numerator, denominator=denominator,
        tmc=tmc, threshold=threshold,
        decision=decision, hypothesis=hypothesis,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo metrics (baseline)
# ─────────────────────────────────────────────────────────────────────────────
def monte_carlo_metrics(
    num_trials: int,
    fft_size: int, cp_length: int, num_symbols: int,
    N: int, signal_power: float, noise_variance: float,
    noise_uncertainty_db: float, cyclic_freqs: List[float],
    tau: int, beta: float, threshold: float,
    rng: np.random.Generator,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    h0_decisions = np.zeros(num_trials, dtype=int)
    h1_decisions = np.zeros(num_trials, dtype=int)
    tmc_h0       = np.zeros(num_trials, dtype=float)
    tmc_h1       = np.zeros(num_trials, dtype=float)

    common = dict(
        fft_size=fft_size, cp_length=cp_length, num_symbols=num_symbols,
        N=N, noise_variance=noise_variance,
        noise_uncertainty_db=noise_uncertainty_db,
        cyclic_freqs=cyclic_freqs, tau=tau, beta=beta, threshold=threshold,
    )
    for m in range(num_trials):
        r0 = run_single_trial(hypothesis="H0", signal_power=0.0,          rng=rng, **common)
        r1 = run_single_trial(hypothesis="H1", signal_power=signal_power, rng=rng, **common)
        h0_decisions[m] = r0.decision
        h1_decisions[m] = r1.decision
        tmc_h0[m] = r0.tmc
        tmc_h1[m] = r1.tmc

    pf = float(np.mean(h0_decisions))
    pd = float(np.mean(h1_decisions))
    return pf, pd, tmc_h0, tmc_h1


# ════════════════════════════════════════════════════════════════════════════
# BASELINE PLOTS (Milestone 3 — unchanged)
# ════════════════════════════════════════════════════════════════════════════

def plot_pipeline_diagram(demo: TrialResult, save_path: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
    fig.suptitle(
        "Detection Pipeline — Multi-Cycle Cyclostationary Detector  (CSE400, Milestone 4)\n"
        "Values annotated from one example trial  (H₁, SNR = −10 dB)",
        fontsize=11.5, y=0.98,
    )
    steps = [
        ("Step 1", "Generate\nOFDM Signal\ns[n]",
         f"N={len(demo.s)}\nQPSK+CP"),
        ("Step 2", "Generate\nGaussian Noise\nw[n]",
         f"w~CN(0,σ²)\nσ²=1.0"),
        ("Step 3", "Received Signal\nx[n]=s[n]+w[n]",
         f"pwr={np.mean(np.abs(demo.x)**2):.3f}"),
        ("Step 4", "Cyclic Auto-\ncorrelation\nR̂ₓᵅ(τ)",
         f"num={demo.numerator:.3f}\nden={demo.denominator:.3f}"),
        ("Step 5", "Test Statistic\nTmc",
         f"Tmc=\n{demo.tmc:.4f}"),
        ("Step 6", "Compare\nTmc vs λ",
         f"λ={demo.threshold:.2f}\nTmc {'≥' if demo.tmc >= demo.threshold else '<'} λ"),
        ("Step 7", "Decision\nH₀ or H₁",
         f"→ {'H₁ ✓' if demo.decision == 1 else 'H₀ ✗'}"),
    ]
    bw, bh = 1.5, 1.85; gap = 0.2
    total = len(steps) * bw + (len(steps) - 1) * gap
    xs = (14 - total) / 2; yb = 2.0
    palette = ["#1e3a5f","#1e3a5f","#1e3a5f","#065f46","#065f46","#7c1d1d","#7c1d1d"]
    for i, ((label, title, note), color) in enumerate(zip(steps, palette)):
        xc = xs + i * (bw + gap) + bw / 2
        rect = mpatches.FancyBboxPatch(
            (xc - bw/2, yb), bw, bh, boxstyle="round,pad=0.08", lw=1.5,
            edgecolor=color, facecolor=color + "18",
        )
        ax.add_patch(rect)
        ax.text(xc, yb+bh+0.13, label, ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")
        ax.text(xc, yb+bh/2+0.1, title, ha="center", va="center", fontsize=8.5, color=color, fontweight="bold", multialignment="center")
        ax.text(xc, yb-0.22, note, ha="center", va="top", fontsize=7.5, color="#374151", multialignment="center",
                bbox=dict(boxstyle="round,pad=0.14", fc="#f9fafb", ec="#d1d5db", lw=0.8))
        if i < len(steps) - 1:
            ax.annotate("", xy=(xc+bw/2+gap, yb+bh/2), xytext=(xc+bw/2, yb+bh/2),
                        arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.5))
    ax.text(7.0, 0.88,
            "Step 8:  Repeat for 1 000 independent trials  →  "
            "Pf = P(decide H₁ | H₀)     Pd = P(decide H₁ | H₁)",
            ha="center", va="center", fontsize=9.5, color="#1e3a5f", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#eff6ff", ec="#2563eb", lw=1.5))
    legend_handles = [
        mpatches.FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0.1",fc="#1e3a5f18",ec="#1e3a5f",lw=1.2,label="Signal Chain  (Steps 1–3)"),
        mpatches.FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0.1",fc="#065f4618",ec="#065f46",lw=1.2,label="Feature Extraction  (Steps 4–5)"),
        mpatches.FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0.1",fc="#7c1d1d18",ec="#7c1d1d",lw=1.2,label="Decision  (Steps 6–7)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.5, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Saved] {save_path}")


def plot_signal_examples(s, w, x, save_path, n_show=256):
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Pipeline Steps 1–3: OFDM Signal, Noise, Received Signal", fontsize=13)
    n = np.arange(n_show)
    data = [
        (s[:n_show], "Step 1 — OFDM Signal  s[n]  (real part)",          COLORS["ofdm"]),
        (w[:n_show], "Step 2 — Gaussian Noise  w[n]  (real part)",        COLORS["noise"]),
        (x[:n_show], "Step 3 — Received  x[n] = s[n]+w[n]  (real part)", COLORS["received"]),
    ]
    for ax, (sig, title, color) in zip(axes, data):
        ax.plot(n, sig.real, color=color, linewidth=0.9, alpha=0.9)
        ax.set_title(title, fontsize=11); ax.set_ylabel("Amplitude"); ax.set_xlim(0, n_show)
    axes[-1].set_xlabel("Sample index  n")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Saved] {save_path}")


def plot_cac_comparison(s, w, cyclic_freqs, tau, fft_size, cp_length, save_path):
    from cyclostationary_features import estimate_cyclic_autocorrelation
    sp = fft_size + cp_length
    ag = np.linspace(-3/sp, 3/sp, 300)
    cac_s = np.array([abs(estimate_cyclic_autocorrelation(s, tau, a)) for a in ag])
    cac_w = np.array([abs(estimate_cyclic_autocorrelation(w, tau, a)) for a in ag])
    k_ax  = ag * sp
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Step 4 — Cyclic Autocorrelation |R̂ₓᵅ(τ)| at τ = N_FFT", fontsize=12)
    axes[0].plot(k_ax, cac_s, color=COLORS["ofdm"])
    for k in [-2,-1,0,1,2]:
        axes[0].axvline(k, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    axes[0].set_title("OFDM — non-zero peaks at α_k  (cyclostationary)")
    axes[0].set_xlabel("k = α × (N_FFT + N_CP)"); axes[0].set_ylabel("|R̂ₓᵅ(τ)|")
    axes[1].plot(k_ax, cac_w, color=COLORS["noise"])
    axes[1].set_title("Noise — CAC ≈ 0 everywhere  (stationary)")
    axes[1].set_xlabel("k = α × (N_FFT + N_CP)"); axes[1].set_ylabel("|R̂ₓᵅ(τ)|")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Saved] {save_path}")


def plot_tmc_distribution(tmc_h0, tmc_h1, threshold, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    clip = np.percentile(np.concatenate([tmc_h0, tmc_h1]), 97)
    bins = np.linspace(0, clip, 60)
    ax.hist(tmc_h0[tmc_h0 <= clip], bins=bins, density=True, alpha=0.6, color=COLORS["noise"], label="H₀: noise only")
    ax.hist(tmc_h1[tmc_h1 <= clip], bins=bins, density=True, alpha=0.6, color=COLORS["ofdm"],  label="H₁: signal + noise")
    ax.axvline(threshold, color=COLORS["threshold"], linestyle="--", linewidth=2, label=f"λ = {threshold:.2f}")
    ax.set_xlabel("Test Statistic  Tmc"); ax.set_ylabel("Empirical PDF")
    ax.set_title("Steps 5–7 — Tmc distribution under H₀ and H₁  (SNR = −10 dB)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Saved] {save_path}")


def plot_pfa_vs_threshold(tmc_h0, save_path):
    lam = np.linspace(0.01, 50, 500)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(lam, compute_theoretical_pfa_vs_threshold(lam),
                color=COLORS["theory"], label="Theory: Pf = 1/(λ+1)", linewidth=2.5)
    ax.semilogy(lam, compute_empirical_pfa_vs_threshold(tmc_h0, lam),
                color=COLORS["multi"], linestyle="--", label="Simulation", linewidth=2)
    ax.axvline(9.0, color=COLORS["threshold"], linestyle=":", linewidth=1.5, label="λ for Pf=0.1")
    ax.axhline(0.1, color=COLORS["threshold"], linestyle=":", linewidth=1.5)
    ax.set_xlabel("Threshold λ"); ax.set_ylabel("Probability of False Alarm  Pf")
    ax.set_title("Step 8 — Pf vs Threshold: Theory vs Simulation")
    ax.legend(); ax.set_xlim(0, 50); ax.set_ylim(1e-3, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Saved] {save_path}")


def plot_pd_vs_snr(snr_db, pd_multi, pd_single, target_pfa, save_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(snr_db, pd_multi,  color=COLORS["multi"],  marker="o", markersize=5, label="Multi-cycle  (K=5)")
    ax.plot(snr_db, pd_single, color=COLORS["single"], marker="s", markersize=5, linestyle="--", label="Single-cycle  (K=1)")
    ax.axhline(target_pfa, color=COLORS["threshold"], linestyle=":", alpha=0.6, linewidth=1.5, label=f"Pf = {target_pfa}")
    ax.axhline(0.9, color="gray", linestyle=":", alpha=0.5, linewidth=1.2)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Probability of Detection  Pd")
    ax.set_title(f"Step 8 — Pd vs SNR  (Monte Carlo baseline, Pf = {target_pfa})")
    ax.legend(loc="lower right"); ax.set_xlim(snr_db[0], snr_db[-1]); ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Saved] {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# MILESTONE 4 PLOTS — IS vs MC baseline comparison
# ════════════════════════════════════════════════════════════════════════════

def plot_pd_comparison(
    snr_db: np.ndarray,
    pd_mc: np.ndarray,
    pd_is: np.ndarray,
    ci_is: np.ndarray,
    num_trials: int,
    rho: float,
    save_path: str,
):
    """
    Plot 6: Pd vs SNR — IS algorithm vs MC baseline, both with 95 % CI bands.
    """
    # MC 95 % CI (analytical binomial)
    ci_mc = 1.96 * np.sqrt(pd_mc * (1.0 - pd_mc) / num_trials)

    fig, ax = plt.subplots(figsize=(10, 6))

    # MC baseline
    ax.plot(snr_db, pd_mc, color=COLORS["mc"], marker="o", markersize=5,
            label=f"MC Baseline  (M={num_trials})")
    ax.fill_between(snr_db, pd_mc - ci_mc, pd_mc + ci_mc,
                    color=COLORS["mc"], alpha=0.15, label="MC 95 % CI")

    # IS algorithm
    ax.plot(snr_db, pd_is, color=COLORS["is"], marker="^", markersize=5,
            linestyle="--", label=f"IS Algorithm  (ρ={rho:.1f}, M={num_trials})")
    ax.fill_between(snr_db, pd_is - ci_is, pd_is + ci_is,
                    color=COLORS["is"], alpha=0.20, label="IS 95 % CI")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Probability of Detection  Pd")
    ax.set_title(
        "Milestone 4 — Pd vs SNR: Importance Sampling vs Monte Carlo Baseline\n"
        f"(Shaded bands = 95 % confidence interval,  ρ = σ²_q / σ² = {rho:.1f})"
    )
    ax.legend(loc="lower right")
    ax.set_xlim(snr_db[0], snr_db[-1])
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_ci_width_comparison(
    snr_db: np.ndarray,
    pd_mc: np.ndarray,
    ci_is: np.ndarray,
    num_trials: int,
    rho: float,
    save_path: str,
):
    """
    Plot 7: 95 % CI width vs SNR — IS vs MC.
    Narrower CI = lower estimator variance = more efficient simulation.
    """
    ci_mc = 2 * 1.96 * np.sqrt(pd_mc * (1.0 - pd_mc) / num_trials)
    ci_is_width = 2 * ci_is  # convert half-width to full width

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(snr_db, ci_mc,      color=COLORS["mc"], marker="o", markersize=5,
            label=f"MC Baseline  (M={num_trials})")
    ax.plot(snr_db, ci_is_width, color=COLORS["is"], marker="^", markersize=5,
            linestyle="--", label=f"IS Algorithm  (ρ={rho:.1f}, M={num_trials})")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("95 % CI Full Width")
    ax.set_title(
        "Milestone 4 — Estimator Variance: IS vs Monte Carlo\n"
        "Narrower CI = fewer trials needed for the same accuracy"
    )
    ax.legend()
    ax.set_xlim(snr_db[0], snr_db[-1])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_pf_comparison(pf_mc, pf_is, pf_is_ci, pf_theory, save_path):
    """
    Plot 8: Bar chart comparing Pf from MC, IS, and theory.
    Both should match the theoretical target = 0.1.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    labels  = ["Theory\n(1/λ+1)", "MC Baseline", "IS Algorithm"]
    values  = [pf_theory, pf_mc, pf_is]
    colors  = [COLORS["theory"], COLORS["mc"], COLORS["is"]]
    bars = ax.bar(labels, values, color=colors, alpha=0.75, edgecolor="black", linewidth=0.8, width=0.45)

    # Error bar on IS estimate
    ax.errorbar(2, pf_is, yerr=pf_is_ci, fmt="none", color="black", capsize=6, linewidth=2)

    ax.axhline(pf_theory, color=COLORS["theory"], linestyle="--", linewidth=1.5, alpha=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Probability of False Alarm  Pf")
    ax.set_title("Milestone 4 — Pf Comparison: Theory, MC Baseline, IS Algorithm\n"
                 "(IS reweighting preserves the false-alarm constraint)")
    ax.set_ylim(0, max(values) * 1.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def run_simulation():
    print("\n" + "=" * 65)
    print("  CSE400 — Milestone 4: Spectrum Sensing Simulation")
    print("  Multi-Cycle Cyclostationary Detector  |  Group 11")
    print("  Randomization: Importance Sampling")
    print("=" * 65)

    cfg_path = Path(__file__).parent.parent / "data" / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    fft_size     = cfg["ofdm"]["fft_size"]
    cp_length    = cfg["ofdm"]["cp_length"]
    num_symbols  = cfg["ofdm"]["num_symbols"]
    k_values     = cfg["detector"]["cyclic_frequencies"]
    beta         = cfg["detector"]["beta"]
    target_pfa   = cfg["detector"]["target_pfa"]
    N            = cfg["simulation"]["N"]
    snr_range_db = cfg["simulation"]["snr_range_db"]
    num_trials   = cfg["simulation"]["num_trials"]
    noise_unc_db = cfg["simulation"]["noise_uncertainty_db"]
    noise_var    = 1.0

    is_cfg       = cfg.get("importance_sampling", {})
    rho          = float(is_cfg.get("variance_inflation_factor", 3.0))
    is_trials    = int(is_cfg.get("num_trials", num_trials))
    sigma2_q     = noise_var * rho

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed=42)

    cyclic_freqs = compute_ofdm_cyclic_frequencies(fft_size, cp_length, k_values)
    tau          = fft_size
    threshold    = compute_threshold_from_pfa(target_pfa)

    print(f"\n  Params:  N={N}, FFT={fft_size}, CP={cp_length}, K={len(k_values)}")
    print(f"  Target Pfa={target_pfa}  →  λ = {threshold:.4f}  [= 1/Pfa − 1]")
    print(f"  IS config:  ρ = {rho:.1f}  →  σ²_q = {sigma2_q:.1f}   (IS trials = {is_trials})")

    # ── DEMO: one pipeline pass ───────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  DEMO — Single pipeline trial  (H₁, SNR = −10 dB)")
    print("─" * 65)
    demo = run_single_trial(
        hypothesis="H1",
        fft_size=fft_size, cp_length=cp_length, num_symbols=num_symbols,
        N=N, signal_power=10**(-10/10), noise_variance=noise_var,
        noise_uncertainty_db=noise_unc_db,
        cyclic_freqs=cyclic_freqs, tau=tau, beta=beta,
        threshold=threshold, rng=np.random.default_rng(7),
    )
    print(f"  Step 1  s[n]          : {len(demo.s)} samples, power = {np.mean(np.abs(demo.s)**2):.4f}")
    print(f"  Step 2  w[n]          : {len(demo.w)} samples, power = {np.mean(np.abs(demo.w)**2):.4f}")
    print(f"  Step 3  x[n] = s+w    : power = {np.mean(np.abs(demo.x)**2):.4f}")
    print(f"  Step 4  CAC numerator  : {demo.numerator:.6f}")
    print(f"          CAC denominator: {demo.denominator:.6f}")
    print(f"  Step 5  Tmc           : {demo.tmc:.6f}")
    print(f"  Step 6  Threshold λ   : {demo.threshold:.4f}")
    print(f"  Step 7  Decision      : {'H₁ ✓ (correct detection)' if demo.decision==1 else 'H₀ ✗ (missed)'}")

    # ── Plot 0: pipeline diagram ──────────────────────────────────────────────
    print("\n[Plot 0] Pipeline diagram...")
    plot_pipeline_diagram(demo, str(results_dir / "00_pipeline_diagram.png"))

    # ── Plot 1: signal waveforms ──────────────────────────────────────────────
    print("[Plot 1] Signal waveforms (Steps 1–3)...")
    plot_signal_examples(demo.s, demo.w, demo.x, str(results_dir / "01_signal_waveforms.png"))

    # ── Plot 2: CAC comparison ────────────────────────────────────────────────
    print("[Plot 2] Cyclic autocorrelation (Step 4)...")
    s_pure = generate_ofdm_signal(fft_size, cp_length, num_symbols, N, 1.0, np.random.default_rng(42))
    w_pure = generate_complex_gaussian_noise(N, noise_var, np.random.default_rng(42))
    plot_cac_comparison(s_pure, w_pure, cyclic_freqs, tau, fft_size, cp_length,
                        str(results_dir / "02_cac_comparison.png"))

    # ── Step 8a: Monte Carlo at SNR = −10 dB ─────────────────────────────────
    snr_mc_lin = 10**(-10/10)
    print(f"\n[Step 8a] MC Baseline  ({num_trials} trials, SNR = −10 dB)...")
    pf_mc, pd_mc_10, tmc_h0_vals, tmc_h1_vals = monte_carlo_metrics(
        num_trials=num_trials,
        fft_size=fft_size, cp_length=cp_length, num_symbols=num_symbols,
        N=N, signal_power=snr_mc_lin, noise_variance=noise_var,
        noise_uncertainty_db=noise_unc_db,
        cyclic_freqs=cyclic_freqs, tau=tau, beta=beta,
        threshold=threshold, rng=rng,
    )
    print(f"  MC  Pf = {pf_mc:.4f}   (theory = {1/(threshold+1):.4f})")
    print(f"  MC  Pd = {pd_mc_10:.4f}   at SNR = −10 dB")

    # ── Plot 3: Pfa vs threshold ──────────────────────────────────────────────
    print("[Plot 3] Pfa vs threshold...")
    plot_pfa_vs_threshold(tmc_h0_vals, str(results_dir / "03_pfa_vs_threshold.png"))

    # ── Plot 4: Tmc distribution ──────────────────────────────────────────────
    print("[Plot 4] Tmc distribution (Steps 5–7)...")
    plot_tmc_distribution(tmc_h0_vals, tmc_h1_vals, threshold,
                          str(results_dir / "04_tmc_distribution.png"))

    # ── Step 8b: MC SNR sweep ─────────────────────────────────────────────────
    print(f"\n[Step 8b] MC SNR sweep  ({len(snr_range_db)} points × {num_trials} trials)...")
    detector_multi  = MultiCycleDetector(fft_size, cp_length, k_values, beta, target_pfa)
    detector_single = MultiCycleDetector(fft_size, cp_length, [0],      beta, target_pfa)

    def make_noise():
        return generate_noise_with_uncertainty(
            N, noise_var, noise_unc_db,
            np.random.default_rng(rng.integers(0, int(1e9))))

    def make_signal(power):
        return generate_ofdm_signal(
            fft_size, cp_length, num_symbols, N, power,
            np.random.default_rng(rng.integers(0, int(1e9))))

    snr_arr, pd_multi  = sweep_pd_vs_snr(detector_multi.detect,  snr_range_db, make_signal, make_noise, N, num_trials)
    snr_arr, pd_single = sweep_pd_vs_snr(detector_single.detect, snr_range_db, make_signal, make_noise, N, num_trials)

    # ── Plot 5: Pd vs SNR (baseline) ─────────────────────────────────────────
    print("[Plot 5] Pd vs SNR (MC baseline)...")
    plot_pd_vs_snr(snr_arr, pd_multi, pd_single, target_pfa,
                   str(results_dir / "05_pd_vs_snr.png"))

    # ════════════════════════════════════════════════════════════════════════
    # MILESTONE 4: Importance Sampling
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  MILESTONE 4 — Importance Sampling")
    print(f"  Proposal: q = CN(0, σ²_q),  ρ = σ²_q/σ² = {rho:.1f}")
    print("─" * 65)

    # IS Pf estimation (H0)
    print(f"\n[IS Pf] Estimating Pf with IS  ({is_trials} trials)...")
    pf_is, pf_is_ci, pf_ess = estimate_pfa_is(
        detector_fn=detector_multi.detect,
        num_samples=N,
        num_trials=is_trials,
        sigma2=noise_var,
        sigma2_q=sigma2_q,
        rng=rng,
    )
    pf_theory = 1.0 / (threshold + 1.0)
    print(f"  Theory  Pf = {pf_theory:.4f}")
    print(f"  MC      Pf = {pf_mc:.4f}   (95% CI ± {1.96*np.sqrt(pf_mc*(1-pf_mc)/num_trials):.4f})")
    print(f"  IS      Pf = {pf_is:.4f}   (95% CI ± {pf_is_ci:.4f},  ESS = {pf_ess:.1f}/{is_trials})")

    # IS Pd sweep
    print(f"\n[IS Pd sweep] {len(snr_range_db)} SNR points × {is_trials} IS trials...")
    _, pd_is, ci_is, ess_is = sweep_pd_vs_snr_is(
        detector_fn=detector_multi.detect,
        snr_db_values=snr_range_db,
        signal_generator_fn=make_signal,
        num_samples=N,
        num_trials=is_trials,
        sigma2=noise_var,
        sigma2_q=sigma2_q,
        rng=rng,
    )

    print(f"\n  {'SNR (dB)':>10}  {'Pd MC':>8}  {'Pd IS':>8}  {'IS CI ±':>9}  {'ESS':>7}")
    print("  " + "-" * 52)
    for s, p_mc, p_is, ci, ess in zip(snr_range_db, pd_multi, pd_is, ci_is, ess_is):
        print(f"  {s:>10.1f}  {p_mc:>8.4f}  {p_is:>8.4f}  {ci:>9.4f}  {ess:>7.1f}")

    # ── Plot 6: Pd comparison with CI bands ───────────────────────────────────
    print("\n[Plot 6] Pd comparison: IS vs MC baseline with CI bands...")
    plot_pd_comparison(
        snr_arr, pd_multi, pd_is, ci_is, num_trials, rho,
        str(results_dir / "06_pd_comparison_IS_vs_MC.png"),
    )

    # ── Plot 7: CI width comparison ───────────────────────────────────────────
    print("[Plot 7] CI width comparison: IS vs MC...")
    plot_ci_width_comparison(
        snr_arr, pd_multi, ci_is, num_trials, rho,
        str(results_dir / "07_ci_width_IS_vs_MC.png"),
    )

    # ── Plot 8: Pf bar chart ──────────────────────────────────────────────────
    print("[Plot 8] Pf comparison bar chart...")
    plot_pf_comparison(
        pf_mc, pf_is, pf_is_ci, pf_theory,
        str(results_dir / "08_pf_comparison.png"),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + summarize_results(snr_arr, pd_multi, pf_mc, threshold))
    print(f"\n  IS Summary:  Pf = {pf_is:.4f} ± {pf_is_ci:.4f}   ESS = {pf_ess:.1f}/{is_trials}")
    print(f"✓ All outputs saved to: {results_dir.resolve()}\n")


if __name__ == "__main__":
    run_simulation()