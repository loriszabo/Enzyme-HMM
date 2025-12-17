from __future__ import annotations

import sys
import time
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

print("Test message.")

# Must match generate_synthetic_data_2.py
DEFAULT_INPUT_FILE = "synthetic_aid_data_2.csv"

# In the generator, noise is only applied in state 0 (unbound).
# We treat noise as known/fixed (common assumption for sequencing error).
NOISE = 1e-4


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


class AID_HMM_2State:
    """
    HMM consistent with generate_synthetic_data_2.py

    Hidden states:
      0 = unbound/background
      1 = bound/scanning

    Transitions:
      0 -> 1 with alpha_recruit; 0 -> 0 with 1-alpha_recruit
      1 -> 1 with lambda_scan;   1 -> 0 with 1-lambda_scan

    Emissions:
      State 0 (background): per-site substitution noise with prob NOISE
        P(obs==orig) = 1-NOISE
        P(obs!=orig) = NOISE/3  (uniform among 3 alternatives)

      State 1 (bound): AID "fire" is an emission/event with context-dependent probability p_fire,
        and if it fires on a C, mutation C->T is observed with probability mutation_eff.

        Let p_mut(t) = p_fire(t) * mutation_eff.
        Then:
          - if obs == orig: P = 1 - p_mut(t)
          - if orig == 'C' and obs == 'T': P = p_mut(t)
          - else: P = 0
    """

    def __init__(self, noise: float = NOISE):
        self.noise = float(noise)

    def get_context_fire_prob(
        self,
        seq: str,
        pos: int,
        base_activation: float,
        hotspot_bias: float,
        coldspot_bias: float,
    ) -> float:
        """
        Exactly mirrors generate_synthetic_data_2.py:
          - returns 0 if pos < 2 or seq[pos] != 'C'
          - WRC hotspot: (pos-2 in A/T) and (pos-1 in A/G) -> base_activation * hotspot_bias
          - SYC coldspot: (pos-2 in G/C) and (pos-1 in C/T) -> base_activation * coldspot_bias
          - else neutral: base_activation
        """
        if pos < 2 or pos >= len(seq):
            return 0.0
        if seq[pos] != "C":
            return 0.0

        b2 = seq[pos - 2]
        b1 = seq[pos - 1]

        is_W = b2 in ("A", "T")
        is_R = b1 in ("A", "G")
        if is_W and is_R:
            return _clamp01(base_activation * hotspot_bias)

        is_S = b2 in ("G", "C")
        is_Y = b1 in ("C", "T")
        if is_S and is_Y:
            return _clamp01(base_activation * coldspot_bias)

        return _clamp01(base_activation)

    def _log_emit_bg(self, orig: str, obs: str) -> float:
        if obs == orig:
            p = 1.0 - self.noise
        else:
            p = self.noise / 3.0
        return -np.inf if p <= 0.0 else float(np.log(p))

    def _log_emit_bound(
        self,
        orig_seq: str,
        mut_seq: str,
        t: int,
        base_activation: float,
        hotspot_bias: float,
        coldspot_bias: float,
        mutation_eff: float,
    ) -> float:
        orig = orig_seq[t]
        obs = mut_seq[t]

        p_fire = self.get_context_fire_prob(orig_seq, t, base_activation, hotspot_bias, coldspot_bias)
        p_mut = _clamp01(p_fire * mutation_eff)

        if obs == orig:
            p = 1.0 - p_mut
        elif orig == "C" and obs == "T":
            p = p_mut
        else:
            p = 0.0

        return -np.inf if p <= 0.0 else float(np.log(p))

    def forward_loglik(
        self,
        orig_seq: str,
        mut_seq: str,
        params: Tuple[float, float, float, float, float, float],
    ) -> float:
        """
        params = (alpha_recruit, lambda_scan, base_activation, hotspot_bias, coldspot_bias, mutation_eff)
        Returns log P(mut_seq | orig_seq, params)
        """
        alpha_recruit, lambda_scan, base_act, hot_bias, cold_bias, mut_eff = params

        T = len(orig_seq)
        if len(mut_seq) != T:
            raise ValueError("orig_seq and mut_seq must have same length")

        # log transition matrix for 2 states
        # from 0: [0->0, 0->1]
        # from 1: [1->0, 1->1]
        t00 = 1.0 - alpha_recruit
        t01 = alpha_recruit
        t10 = 1.0 - lambda_scan
        t11 = lambda_scan

        if min(t00, t01, t10, t11) <= 0.0:
            return -np.inf

        log_trans = np.array([[np.log(t00), np.log(t01)], [np.log(t10), np.log(t11)]], dtype=float)

        # alpha[t, s] in log-space
        alpha = np.full((T, 2), -np.inf, dtype=float)

        # Start forced in state 0 at t=0 (as in generator)
        loge0_0 = self._log_emit_bg(orig_seq[0], mut_seq[0])
        alpha[0, 0] = loge0_0
        alpha[0, 1] = -np.inf

        for t in range(1, T):
            # emissions at time t
            loge_bg = self._log_emit_bg(orig_seq[t], mut_seq[t])
            loge_bound = self._log_emit_bound(
                orig_seq,
                mut_seq,
                t,
                base_act,
                hot_bias,
                cold_bias,
                mut_eff,
            )

            # state 0
            alpha[t, 0] = loge_bg + logsumexp(
                [alpha[t - 1, 0] + log_trans[0, 0], alpha[t - 1, 1] + log_trans[1, 0]]
            )

            # state 1
            alpha[t, 1] = loge_bound + logsumexp(
                [alpha[t - 1, 0] + log_trans[0, 1], alpha[t - 1, 1] + log_trans[1, 1]]
            )

        return float(logsumexp(alpha[T - 1, :]))

    def dataset_neg_loglik(
        self,
        df: pd.DataFrame,
        params: Tuple[float, float, float, float, float, float],
    ) -> float:
        total = 0.0
        for _, row in df.iterrows():
            ll = self.forward_loglik(row["original_seq"], row["mutated_seq"], params)
            if not np.isfinite(ll):
                return 1e30
            total -= ll
        return float(total)


def main() -> None:
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE
    df = pd.read_csv(input_file)

    if "original_seq" not in df.columns or "mutated_seq" not in df.columns:
        raise ValueError("CSV must contain columns: original_seq, mutated_seq")

    model = AID_HMM_2State(noise=NOISE)

    # params = (alpha_recruit, lambda_scan, base_activation, hotspot_bias, coldspot_bias, mutation_eff)
    x0 = np.array([0.03, 0.95, 0.04, 10.0, 0.1, 0.8], dtype=float)

    bounds = [
        (1e-6, 1.0 - 1e-6),  # alpha_recruit
        (1e-6, 1.0 - 1e-6),  # lambda_scan
        (0.0, 1.0),          # base_activation
        (1e-3, 200.0),       # hotspot_bias
        (1e-3, 200.0),       # coldspot_bias
        (0.0, 1.0),          # mutation_eff
    ]

    def objective(x: np.ndarray) -> float:
        return model.dataset_neg_loglik(df, tuple(map(float, x)))

    t0 = time.time()
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    dt = time.time() - t0

    print("--- 2-state AID HMM inference (consistent with generate_synthetic_data_2.py) ---")
    print(f"Input: {input_file}")
    print(f"Sequences: {len(df)}")
    print(f"Noise (fixed): {NOISE}")
    print(f"Optimization time: {dt:.2f}s")
    print(f"Success: {res.success} | Status: {res.status}")
    print(f"Message: {res.message}")
    print(f"Neg log-likelihood: {res.fun:.6f}")
    est = res.x
    names = ["alpha_recruit", "lambda_scan", "base_activation", "hotspot_bias", "coldspot_bias", "mutation_eff"]
    for n, v in zip(names, est):
        print(f"{n:>15s} = {v:.6f}")


if __name__ == "__main__":
    main()