from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from tqdm import tqdm

# Must match generate_synthetic_data_2.py
DEFAULT_INPUT_FILE = "synthetic_aid_data_2.csv"

# In the generator, noise is only applied in state 0 (unbound).
# We treat noise as known/fixed (common assumption for sequencing error).
NOISE = 1e-4

PARAM_NAMES = [
    "alpha_recruit",
    "lambda_scan",
    "base_activation",
    "hotspot_bias",
    "coldspot_bias",
    "mutation_eff",
]

# TO evaluate optimization
DEFAULT_PARAM_VALUES = "0.03,0.95,0.04,10.0,0.1,0.8"

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _total_bases(df: pd.DataFrame) -> int:
    return int(sum(len(s) for s in df["original_seq"].astype(str).values))


def _parse_true_params(s: Optional[str]) -> Optional[np.ndarray]:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != len(PARAM_NAMES):
        raise ValueError(
            f"--true-params must have {len(PARAM_NAMES)} comma-separated values "
            f"({', '.join(PARAM_NAMES)})"
        )
    return np.array([float(x) for x in parts], dtype=float)


def _extract_true_params_from_df(df: pd.DataFrame) -> Optional[np.ndarray]:
    # Supports either:
    #  - columns: true_alpha_recruit, true_lambda_scan, ...
    #  - single row columns: alpha_recruit_true, lambda_scan_true, ...
    candidates_1 = [f"true_{n}" for n in PARAM_NAMES]
    candidates_2 = [f"{n}_true" for n in PARAM_NAMES]

    if all(c in df.columns for c in candidates_1):
        vals = [float(df[c].iloc[0]) for c in candidates_1]
        return np.array(vals, dtype=float)
    if all(c in df.columns for c in candidates_2):
        vals = [float(df[c].iloc[0]) for c in candidates_2]
        return np.array(vals, dtype=float)
    return None


def _near_bounds(x: np.ndarray, bounds: Sequence[Tuple[float, float]], tol: float = 1e-6) -> List[str]:
    msgs: List[str] = []
    for i, (xi, (lo, hi)) in enumerate(zip(x, bounds)):
        if xi - lo <= tol * max(1.0, abs(lo), abs(hi)):
            msgs.append(f"{PARAM_NAMES[i]} near lower bound ({xi:.6g} ~ {lo:.6g})")
        if hi - xi <= tol * max(1.0, abs(lo), abs(hi)):
            msgs.append(f"{PARAM_NAMES[i]} near upper bound ({xi:.6g} ~ {hi:.6g})")
    return msgs


def _numerical_hessian(f, x: np.ndarray, rel_eps: float = 1e-4) -> np.ndarray:
    """
    Central-difference Hessian for scalar f(x). O(d^2) evaluations.
    Intended for identifiability diagnostics (conditioning/SEs), not for optimization.
    """
    x = np.asarray(x, dtype=float)
    d = x.size
    H = np.zeros((d, d), dtype=float)

    # per-dimension step
    h = rel_eps * (1.0 + np.abs(x))
    fx = float(f(x))

    for i in range(d):
        ei = np.zeros(d)
        ei[i] = 1.0
        hi = h[i]

        f_ip = float(f(x + hi * ei))
        f_im = float(f(x - hi * ei))
        H[i, i] = (f_ip - 2.0 * fx + f_im) / (hi * hi)

        for j in range(i + 1, d):
            ej = np.zeros(d)
            ej[j] = 1.0
            hj = h[j]

            f_pp = float(f(x + hi * ei + hj * ej))
            f_pm = float(f(x + hi * ei - hj * ej))
            f_mp = float(f(x - hi * ei + hj * ej))
            f_mm = float(f(x - hi * ei - hj * ej))

            Hij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hi * hj)
            H[i, j] = Hij
            H[j, i] = Hij

    return H


@dataclass(frozen=True)
class FitReport:
    x: np.ndarray
    nll: float
    success: bool
    status: int
    message: str
    nit: int
    nfev: int
    njev: int


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

        alpha = np.full((T, 2), -np.inf, dtype=float)

        # Start forced in state 0 at t=0 (as in generator)
        alpha[0, 0] = self._log_emit_bg(orig_seq[0], mut_seq[0])
        alpha[0, 1] = -np.inf

        for t in range(1, T):
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

            alpha[t, 0] = loge_bg + logsumexp(
                [alpha[t - 1, 0] + log_trans[0, 0], alpha[t - 1, 1] + log_trans[1, 0]]
            )
            alpha[t, 1] = loge_bound + logsumexp(
                [alpha[t - 1, 0] + log_trans[0, 1], alpha[t - 1, 1] + log_trans[1, 1]]
            )

        return float(logsumexp(alpha[T - 1, :]))

    def per_sequence_logliks(
        self,
        df: pd.DataFrame,
        params: Tuple[float, float, float, float, float, float],
    ) -> np.ndarray:
        lls: List[float] = []
        for _, row in df.iterrows():
            lls.append(self.forward_loglik(row["original_seq"], row["mutated_seq"], params))
        return np.array(lls, dtype=float)

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


def _fit_once(
    model: AID_HMM_2State,
    df: pd.DataFrame,
    x0: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    maxiter: int,
    *,
    progress: bool = True,
    desc: str = "Optimize",
) -> FitReport:
    it_ct = 0
    def objective(x: np.ndarray) -> float:
        val = model.dataset_neg_loglik(df, tuple(map(float, x)))
        return val

    pbar = None

    def callback(xk: np.ndarray) -> None:
        # L-BFGS-B calls callback once per iteration (not per function eval).
        nonlocal it_ct
        it_ct += 1
    
        if pbar is not None:
            pbar.update(1)
        
        if (it_ct % 3 == 0):
            # prints once per optimizer iteration
            print(f"iter={it_ct:4d}  x={np.array2string(xk, precision=4, separator=',')}")


    try:
        pbar = tqdm(
            total=int(maxiter),
            desc=desc,
            unit="iter",
            leave=False,
            dynamic_ncols=True,
            disable=not progress,
        )

        res = minimize(
            objective,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            callback=callback,
            options={"maxiter": int(maxiter)},
        )
    finally:
        if pbar is not None:
            # If optimizer stops early, fill remaining is misleading; just close.
            pbar.close()

    return FitReport(
        x=np.array(res.x, dtype=float),
        nll=float(res.fun),
        success=bool(res.success),
        status=int(res.status),
        message=str(res.message),
        nit=int(getattr(res, "nit", -1)),
        nfev=int(getattr(res, "nfev", -1)),
        njev=int(getattr(res, "njev", -1)),
    )


def _sample_x0(rng: np.random.Generator, bounds: Sequence[Tuple[float, float]], base_x0: np.ndarray) -> np.ndarray:
    # Scientifically legitimate sampling:
    # - Biases (ratios) and small rates are scale parameters -> Log-Uniform distribution.
    # - Efficiencies/Probabilities (0-1) -> Uniform distribution.
    
    x = np.zeros(len(bounds), dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        name = PARAM_NAMES[i]
        
        # Identify scale parameters (biases, small rates)
        # alpha_recruit and base_activation are probabilities but typically small (1e-4 to 1e-1),
        # so log-uniform sampling explores this range better than linear uniform.
        is_scale_param = name in ("alpha_recruit", "base_activation", "hotspot_bias", "coldspot_bias")
        
        if is_scale_param:
            # Log-Uniform sampling
            # Ensure bounds are positive for log
            safe_lo = max(lo, 1e-6) 
            safe_hi = max(hi, safe_lo * 1.0001)
            x[i] = np.exp(rng.uniform(np.log(safe_lo), np.log(safe_hi)))
        else:
            # Uniform sampling (lambda_scan, mutation_eff)
            x[i] = rng.uniform(lo, hi)
            
    return x


def _print_param_table(x: np.ndarray, true_x: Optional[np.ndarray] = None) -> None:
    print("Parameter estimates:")
    for i, name in enumerate(PARAM_NAMES):
        if true_x is None:
            print(f"  {name:>15s} = {x[i]:.6g}")
        else:
            abs_err = x[i] - true_x[i]
            rel_err = abs_err / (true_x[i] if true_x[i] != 0 else np.nan)
            print(f"  {name:>15s} = {x[i]:.6g} | true={true_x[i]:.6g} | abs_err={abs_err:.3g} | rel_err={rel_err:.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="2-state AID HMM inference with reporting/diagnostics.")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT_FILE, help="CSV with columns original_seq, mutated_seq")
    parser.add_argument("--restarts", type=int, default=5, help="Number of random restarts (stability across initializations)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for restarts")
    parser.add_argument("--maxiter", type=int, default=500, help="Optimizer max iterations per restart")
    parser.add_argument(
        "--true-params",
        type=str,
        default=DEFAULT_PARAM_VALUES,
        help=f"Comma-separated true params in order: {', '.join(PARAM_NAMES)} (for synthetic recovery)",
    )
    parser.add_argument("--report-per-seq", action="store_true", help="Print per-sequence LL summary stats")
    parser.add_argument("--hessian", action="store_true", default=True, help="Compute numerical Hessian at best fit (identifiability check)")
    args = parser.parse_args()

    input_file = args.input
    df = pd.read_csv(input_file)

    if "original_seq" not in df.columns or "mutated_seq" not in df.columns:
        raise ValueError("CSV must contain columns: original_seq, mutated_seq")

    model = AID_HMM_2State(noise=NOISE)

    # 1. Define a biologically plausible "neutral" starting point.
    # This acts as a sane default for the first restart.
    # recruit=0.05 (low), scan=0.5 (mid), activation=0.05 (low), biases=1.0 (neutral), eff=0.5 (mid)
    neutral_x0 = np.array([0.05, 0.5, 0.05, 1.0, 1.0, 0.5], dtype=float)

    # Bounds: Wide enough to cover the space, but physically constrained where obvious.
    bounds: List[Tuple[float, float]] = [
        (1e-6, 0.5),         # alpha_recruit: unlikely to be > 0.5
        (1e-6, 1.0 - 1e-6),  # lambda_scan
        (1e-6, 0.5),         # base_activation: unlikely to be > 0.5
        (1e-2, 100.0),       # hotspot_bias
        (1e-2, 100.0),       # coldspot_bias
        (0.0, 1.0),          # mutation_eff
    ]

    # True params (only used for reporting error, NOT for initialization)
    true_x = _parse_true_params(args.true_params)
    if true_x is None:
        true_x = _extract_true_params_from_df(df)

    rng = np.random.default_rng(int(args.seed))

    # 2. Build initializations
    # Strategy: Always try the "neutral" biological guess first.
    # Then add (restarts - 1) random samples from the broad priors.
    x0_list: List[np.ndarray] = []
    
    # First restart: Neutral guess
    x0_list.append(neutral_x0)
    
    # Subsequent restarts: Random sampling from bounds
    for _ in range(max(0, int(args.restarts) - 1)):
        x0_list.append(_sample_x0(rng, bounds, neutral_x0))

    t0 = time.time()
    fits: List[FitReport] = []
    for i, x0 in enumerate(x0_list):
        # Optional: Print where we are starting from to verify randomness
        print(f"Restart {i+1}/{len(x0_list)} starting at: {x0}")
        fits.append(_fit_once(model, df, x0=x0, bounds=bounds, maxiter=args.maxiter))
    dt = time.time() - t0

    # Best fit
    fits_sorted = sorted(fits, key=lambda fr: fr.nll)
    best = fits_sorted[0]

    # Reporting: likelihoods
    n_seq = int(len(df))
    n_bases = _total_bases(df)
    best_ll = -best.nll
    ll_per_seq = best_ll / max(1, n_seq)
    ll_per_base = best_ll / max(1, n_bases)
    nll_per_seq = best.nll / max(1, n_seq)
    nll_per_base = best.nll / max(1, n_bases)

    print("--- 2-state AID HMM inference (required reporting) ---")
    print(f"Input: {input_file}")
    print(f"Sequences: {n_seq}")
    print(f"Total bases: {n_bases}")
    print(f"Noise (fixed): {NOISE}")
    print(f"Restarts: {len(x0_list)} | Seed: {args.seed}")
    print(f"Optimization time (all restarts): {dt:.2f}s")

    print("\nFinal likelihood:")
    print(f"  Log-likelihood (LL): {best_ll:.6f}")
    print(f"  Negative log-likelihood (NLL): {best.nll:.6f}")
    print(f"  LL per sequence: {ll_per_seq:.6f} | NLL per sequence: {nll_per_seq:.6f}")
    print(f"  LL per base:     {ll_per_base:.6f} | NLL per base:     {nll_per_base:.6f}")

    # Convergence diagnostics
    print("\nConvergence diagnostics (best restart):")
    print(f"  Success: {best.success} | Status: {best.status}")
    print(f"  Message: {best.message}")
    print(f"  Iterations (nit): {best.nit} | Func evals (nfev): {best.nfev} | Grad evals (njev): {best.njev}")

    near = _near_bounds(best.x, bounds, tol=1e-6)
    if near:
        print("  Boundary warnings:")
        for m in near:
            print(f"    - {m}")

    # Parameter estimates (+ synthetic recovery if available)
    print()
    _print_param_table(best.x, true_x=true_x)

    # Stability across initializations
    nlls = np.array([fr.nll for fr in fits], dtype=float)
    xs = np.stack([fr.x for fr in fits], axis=0)
    ok = np.array([fr.success for fr in fits], dtype=bool)

    print("\nStability across initializations:")
    print(f"  Successful fits: {int(ok.sum())}/{len(fits)}")
    print(f"  NLL across restarts: min={nlls.min():.6f}, median={np.median(nlls):.6f}, max={nlls.max():.6f}, std={nlls.std(ddof=1) if len(nlls)>1 else 0.0:.6f}")

    # Parameter spread (use best 50% by NLL to reduce outlier influence)
    k = max(1, len(fits_sorted) // 2)
    top = fits_sorted[:k]
    top_x = np.stack([fr.x for fr in top], axis=0)
    print(f"  Parameter stability (top {k}/{len(fits_sorted)} restarts by NLL):")
    for i, name in enumerate(PARAM_NAMES):
        mu = float(np.mean(top_x[:, i]))
        sd = float(np.std(top_x[:, i], ddof=1)) if top_x.shape[0] > 1 else 0.0
        print(f"    {name:>15s}: mean={mu:.6g}, std={sd:.3g}")

    # Per-sequence LL summary (optional)
    if args.report_per_seq:
        lls = model.per_sequence_logliks(df, tuple(map(float, best.x)))
        if np.any(~np.isfinite(lls)):
            print("\nPer-sequence LL: contains non-finite values (check model/data compatibility).")
        else:
            q = np.quantile(lls, [0.0, 0.1, 0.5, 0.9, 1.0])
            print("\nPer-sequence log-likelihood summary:")
            print(f"  mean={float(np.mean(lls)):.6f} | std={float(np.std(lls, ddof=1)) if len(lls)>1 else 0.0:.6f}")
            print(f"  min={q[0]:.6f} | p10={q[1]:.6f} | median={q[2]:.6f} | p90={q[3]:.6f} | max={q[4]:.6f}")

    # Identifiability checks (optional): numerical Hessian + conditioning + approx SEs
    if args.hessian:
        print("\nIdentifiability checks (numerical Hessian at best fit):")

        def obj(z: np.ndarray) -> float:
            return model.dataset_neg_loglik(df, tuple(map(float, z)))

        try:
            H = _numerical_hessian(obj, best.x, rel_eps=1e-4)
            # Symmetrize to reduce numerical noise
            H = 0.5 * (H + H.T)

            eigvals = np.linalg.eigvalsh(H)
            min_e = float(np.min(eigvals))
            max_e = float(np.max(eigvals))
            cond = float(np.inf) if min_e <= 0 else float(max_e / min_e)

            print(f"  Hessian eigenvalues: min={min_e:.3g}, max={max_e:.3g}, condition~{cond:.3g}")

            if min_e <= 0:
                print("  Warning: Hessian not positive definite (local non-identifiability or numerical issues).")
            else:
                cov = np.linalg.inv(H)
                se = np.sqrt(np.maximum(np.diag(cov), 0.0))
                print("  Approx. standard errors (from inv(H)):")
                for i, name in enumerate(PARAM_NAMES):
                    print(f"    {name:>15s}: SE~{float(se[i]):.3g}")

                # Parameter correlation (quick view)
                denom = np.outer(se, se)
                corr = cov / np.where(denom == 0, np.nan, denom)
                # Print the largest absolute off-diagonal correlations
                pairs: List[Tuple[float, str]] = []
                for i in range(len(PARAM_NAMES)):
                    for j in range(i + 1, len(PARAM_NAMES)):
                        cij = corr[i, j]
                        if np.isfinite(cij):
                            pairs.append((abs(float(cij)), f"{PARAM_NAMES[i]} vs {PARAM_NAMES[j]}: corr={float(cij):.3f}"))
                pairs.sort(reverse=True, key=lambda t: t[0])
                print("  Largest |correlations| (indicative of weak identifiability):")
                for _, s in pairs[:8]:
                    print(f"    - {s}")

        except Exception as e:
            print(f"  Hessian computation failed: {e}")

    # Synthetic recovery summary (if true params known)
    if true_x is not None:
        err = best.x - true_x
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        print("\nSynthetic recovery (if applicable):")
        print(f"  MAE={mae:.6g} | RMSE={rmse:.6g}")

    # Brief per-restart summary (top few)
    print("\nTop restarts by NLL:")
    for rank, fr in enumerate(fits_sorted[: min(5, len(fits_sorted))], start=1):
        print(f"  #{rank}: NLL={fr.nll:.6f} | success={fr.success} | nit={fr.nit} | msg={fr.message}")

    # Exit non-zero if best fit did not converge
    if not best.success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()