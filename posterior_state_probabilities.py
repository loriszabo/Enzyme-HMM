from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

# Must match generate_synthetic_data_2.py
DEFAULT_INPUT_FILE = "synthetic_aid_data_2.csv"
NOISE = 1e-4

PARAM_NAMES = [
    "alpha_recruit",
    "lambda_scan",
    "base_activation",
    "hotspot_bias",
    "coldspot_bias",
    "mutation_eff",
]

# Convenience default (replace with your inferred params if desired)
DEFAULT_PARAMS = "0.03,0.95,0.04,10.0,0.1,0.8"


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _parse_params(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != len(PARAM_NAMES):
        raise ValueError(
            f"--params must have {len(PARAM_NAMES)} comma-separated values "
            f"in order: {', '.join(PARAM_NAMES)}"
        )
    return np.array([float(x) for x in parts], dtype=float)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return float("nan")
    xx = x[m] - float(np.mean(x[m]))
    yy = y[m] - float(np.mean(y[m]))
    denom = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(xx * yy) / denom)


@dataclass(frozen=True)
class PosteriorSummary:
    mean_occ_overall: float
    mean_occ_per_seq: np.ndarray
    mean_occ_per_pos: np.ndarray
    mut_rate_per_seq_any: np.ndarray
    mut_rate_per_seq_ct: np.ndarray
    mut_rate_per_pos_any: np.ndarray
    mut_rate_per_pos_ct: np.ndarray
    corr_seq_occ_vs_mut_any: float
    corr_seq_occ_vs_mut_ct: float
    corr_pos_occ_vs_mut_any: float
    corr_pos_occ_vs_mut_ct: float
    occ_hotspots: float
    occ_nonhotspots: float
    occ_enrichment_fold: float
    occ_enrichment_diff: float


class AID_HMM_2StatePosterior:
    """
    Computes posterior state probabilities via the forward–backward algorithm
    for the 2-state AID HMM used in inference_with_evaluation.py.
    """

    def __init__(self, noise: float = NOISE):
        self.noise = float(noise)

    def is_hotspot_WRC(self, seq: str, pos: int) -> bool:
        if pos < 2 or pos >= len(seq):
            return False
        if seq[pos] != "C":
            return False
        b2 = seq[pos - 2]
        b1 = seq[pos - 1]
        is_W = b2 in ("A", "T")
        is_R = b1 in ("A", "G")
        return bool(is_W and is_R)

    def get_context_fire_prob(
        self,
        seq: str,
        pos: int,
        base_activation: float,
        hotspot_bias: float,
        coldspot_bias: float,
    ) -> float:
        if pos < 2 or pos >= len(seq):
            return 0.0
        if seq[pos] != "C":
            return 0.0

        b2 = seq[pos - 2]
        b1 = seq[pos - 1]

        # WRC hotspot
        if (b2 in ("A", "T")) and (b1 in ("A", "G")):
            return _clamp01(base_activation * hotspot_bias)

        # SYC coldspot
        if (b2 in ("G", "C")) and (b1 in ("C", "T")):
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

    def forward_backward_posteriors(
        self,
        orig_seq: str,
        mut_seq: str,
        params: Tuple[float, float, float, float, float, float],
    ) -> Tuple[np.ndarray, float]:
        """
        Returns:
          gamma: (T, 2) posterior state probabilities P(S_t=s | data)
          ll: log-likelihood log P(data)
        """
        alpha_recruit, lambda_scan, base_act, hot_bias, cold_bias, mut_eff = params

        T = len(orig_seq)
        if len(mut_seq) != T:
            raise ValueError("orig_seq and mut_seq must have same length")

        t00 = 1.0 - alpha_recruit
        t01 = alpha_recruit
        t10 = 1.0 - lambda_scan
        t11 = lambda_scan

        if min(t00, t01, t10, t11) <= 0.0:
            return np.full((T, 2), np.nan), -np.inf

        log_trans = np.array([[np.log(t00), np.log(t01)], [np.log(t10), np.log(t11)]], dtype=float)

        # Precompute emissions
        loge = np.full((T, 2), -np.inf, dtype=float)
        for t in range(T):
            loge[t, 0] = self._log_emit_bg(orig_seq[t], mut_seq[t])
            loge[t, 1] = self._log_emit_bound(orig_seq, mut_seq, t, base_act, hot_bias, cold_bias, mut_eff)

        # Forward (start forced in state 0 at t=0)
        fwd = np.full((T, 2), -np.inf, dtype=float)
        fwd[0, 0] = loge[0, 0]
        fwd[0, 1] = -np.inf

        for t in range(1, T):
            fwd[t, 0] = loge[t, 0] + logsumexp([fwd[t - 1, 0] + log_trans[0, 0], fwd[t - 1, 1] + log_trans[1, 0]])
            fwd[t, 1] = loge[t, 1] + logsumexp([fwd[t - 1, 0] + log_trans[0, 1], fwd[t - 1, 1] + log_trans[1, 1]])

        ll = float(logsumexp(fwd[T - 1, :]))

        # Backward
        bwd = np.full((T, 2), -np.inf, dtype=float)
        bwd[T - 1, :] = 0.0

        for t in range(T - 2, -1, -1):
            # bwd[t, i] = log sum_j ( trans[i,j] + emit[t+1,j] + bwd[t+1,j] )
            bwd[t, 0] = logsumexp(
                [
                    log_trans[0, 0] + loge[t + 1, 0] + bwd[t + 1, 0],
                    log_trans[0, 1] + loge[t + 1, 1] + bwd[t + 1, 1],
                ]
            )
            bwd[t, 1] = logsumexp(
                [
                    log_trans[1, 0] + loge[t + 1, 0] + bwd[t + 1, 0],
                    log_trans[1, 1] + loge[t + 1, 1] + bwd[t + 1, 1],
                ]
            )

        # Posterior gamma
        log_gamma = fwd + bwd - ll
        gamma = np.exp(log_gamma)
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)  # safety normalization

        return gamma, ll


@dataclass(frozen=True)
class EvalSummary:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0.0 else float("nan")


def _confusion_from_threshold(post: np.ndarray, y_true: np.ndarray, threshold: float) -> Tuple[int, int, int, int]:
    """
    post: posterior P(state=1|data), shape (N,)
    y_true: true state in {0,1}, shape (N,)
    """
    pred = post >= float(threshold)
    y = y_true.astype(int)

    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    return tp, fp, fn, tn


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if np.isfinite(precision) and np.isfinite(recall) else float("nan")
    return float(precision), float(recall), float(f1)


def _auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return float("nan")
    x2 = x[m]
    y2 = y[m]
    # Ensure increasing x
    order = np.argsort(x2)
    x2 = x2[order]
    y2 = y2[order]
    return float(np.trapz(y2, x2))


def _choose_thresholds(scores: np.ndarray, max_points: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.array([np.inf, -np.inf], dtype=float)

    uniq = np.unique(scores)
    if uniq.size <= max_points:
        core = np.sort(uniq)[::-1]  # descending
    else:
        # Quantile grid (descending) to keep runtime reasonable on large N
        qs = np.linspace(0.0, 1.0, max_points)
        core = np.unique(np.quantile(scores, qs))[::-1]

    # Add endpoints so curves include (0,0) and (1,1)-like extremes
    return np.concatenate(([np.inf], core, [-np.inf])).astype(float)


def roc_curve_from_posteriors(
    scores: np.ndarray, y_true: np.ndarray, *, max_points: int = 2000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns (fpr, tpr, thresholds, auc)
    """
    y = y_true.astype(int)
    P = int(np.sum(y == 1))
    N = int(np.sum(y == 0))
    if P == 0 or N == 0:
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([np.inf, -np.inf]),
            float("nan"),
        )

    thresholds = _choose_thresholds(scores, max_points=max_points)
    tpr = np.empty_like(thresholds, dtype=float)
    fpr = np.empty_like(thresholds, dtype=float)

    for k, thr in enumerate(thresholds):
        tp, fp, fn, tn = _confusion_from_threshold(scores, y, thr)
        tpr[k] = _safe_div(tp, P)
        fpr[k] = _safe_div(fp, N)

    auc = _auc_trapz(fpr, tpr)
    return fpr, tpr, thresholds, auc


def pr_curve_from_posteriors(
    scores: np.ndarray, y_true: np.ndarray, *, max_points: int = 2000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns (recall, precision, thresholds, pr_auc)
    """
    y = y_true.astype(int)
    P = int(np.sum(y == 1))
    if P == 0:
        return (
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            np.array([np.inf, -np.inf]),
            float("nan"),
        )

    thresholds = _choose_thresholds(scores, max_points=max_points)
    precision = np.empty_like(thresholds, dtype=float)
    recall = np.empty_like(thresholds, dtype=float)

    for k, thr in enumerate(thresholds):
        tp, fp, fn, tn = _confusion_from_threshold(scores, y, thr)
        p, r, _ = _precision_recall_f1(tp, fp, fn)
        precision[k] = p
        recall[k] = r

    # PR AUC: integrate precision over recall
    pr_auc = _auc_trapz(recall, precision)
    return recall, precision, thresholds, pr_auc


def evaluate_with_hidden_states(
    occ_mat: np.ndarray,
    true_state_mat: np.ndarray,
    *,
    threshold: float,
    max_curve_points: int = 2000,
    save_curves_csv: Optional[str] = None,
) -> EvalSummary:
    # Flatten valid positions
    m = np.isfinite(occ_mat) & np.isfinite(true_state_mat)
    scores = occ_mat[m].astype(float)
    y_true = true_state_mat[m].astype(int)

    tp, fp, fn, tn = _confusion_from_threshold(scores, y_true, threshold)
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)

    fpr, tpr, roc_thr, roc_auc = roc_curve_from_posteriors(scores, y_true, max_points=max_curve_points)
    rec, prec, pr_thr, pr_auc = pr_curve_from_posteriors(scores, y_true, max_points=max_curve_points)

    if save_curves_csv is not None:
        # Save both curves in one file (simple, long format)
        roc_df = pd.DataFrame(
            {"curve": "ROC", "threshold": roc_thr, "x": fpr, "y": tpr}
        ).rename(columns={"x": "fpr", "y": "tpr"})
        pr_df = pd.DataFrame(
            {"curve": "PR", "threshold": pr_thr, "x": rec, "y": prec}
        ).rename(columns={"x": "recall", "y": "precision"})
        out = pd.concat([roc_df, pr_df], axis=0, ignore_index=True)
        out.to_csv(save_curves_csv, index=False)

    return EvalSummary(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
    )


def summarize_dataset(
    df: pd.DataFrame,
    params: np.ndarray,
    noise: float,
    *,
    save_per_position_csv: Optional[str] = None,
    threshold: Optional[float] = None,
    max_curve_points: int = 2000,
    save_curves_csv: Optional[str] = None,
) -> Tuple[PosteriorSummary, Optional[EvalSummary]]:
    model = AID_HMM_2StatePosterior(noise=noise)

    # Handle variable lengths by padding with NaNs
    seqs = list(df["original_seq"].astype(str).values)
    muts = list(df["mutated_seq"].astype(str).values)
    n = len(seqs)
    maxT = max(len(s) for s in seqs)

    occ_mat = np.full((n, maxT), np.nan, dtype=float)       # posterior occupancy P(state=1)
    mut_any = np.full((n, maxT), np.nan, dtype=float)       # indicator obs != orig
    mut_ct = np.full((n, maxT), np.nan, dtype=float)        # indicator C->T
    hotspot_mask = np.full((n, maxT), False, dtype=bool)

    has_hidden = "hidden_states" in df.columns
    true_state_mat = np.full((n, maxT), np.nan, dtype=float) if has_hidden else None

    lls: List[float] = []

    for i in range(n):
        orig = seqs[i]
        mut = muts[i]
        T = len(orig)

        gamma, ll = model.forward_backward_posteriors(orig, mut, tuple(map(float, params)))
        lls.append(ll)

        occ_mat[i, :T] = gamma[:, 1]
        if has_hidden and true_state_mat is not None:
            hs = str(df.loc[df.index[i], "hidden_states"])
            if len(hs) != T:
                raise ValueError(f"Row {i}: hidden_states length ({len(hs)}) != sequence length ({T})")
            true_state_mat[i, :T] = np.fromiter((1.0 if c == "1" else 0.0 for c in hs), dtype=float, count=T)

        for t in range(T):
            mut_any[i, t] = 1.0 if mut[t] != orig[t] else 0.0
            mut_ct[i, t] = 1.0 if (orig[t] == "C" and mut[t] == "T") else 0.0
            hotspot_mask[i, t] = model.is_hotspot_WRC(orig, t)

    # Mean posterior occupancy
    mean_occ_per_seq = np.nanmean(occ_mat, axis=1)
    mean_occ_overall = float(np.nanmean(mean_occ_per_seq))
    mean_occ_per_pos = np.nanmean(occ_mat, axis=0)

    # Mutation density
    mut_rate_per_seq_any = np.nanmean(mut_any, axis=1)
    mut_rate_per_seq_ct = np.nanmean(mut_ct, axis=1)
    mut_rate_per_pos_any = np.nanmean(mut_any, axis=0)
    mut_rate_per_pos_ct = np.nanmean(mut_ct, axis=0)

    # Correlations
    corr_seq_any = _pearson_corr(mean_occ_per_seq, mut_rate_per_seq_any)
    corr_seq_ct = _pearson_corr(mean_occ_per_seq, mut_rate_per_seq_ct)
    corr_pos_any = _pearson_corr(mean_occ_per_pos, mut_rate_per_pos_any)
    corr_pos_ct = _pearson_corr(mean_occ_per_pos, mut_rate_per_pos_ct)

    # Enrichment at hotspots
    occ_hot = float(np.nanmean(occ_mat[hotspot_mask])) if np.any(hotspot_mask) else float("nan")
    occ_nonhot = float(np.nanmean(occ_mat[~hotspot_mask])) if np.any(~hotspot_mask) else float("nan")
    fold = float(occ_hot / occ_nonhot) if np.isfinite(occ_hot) and np.isfinite(occ_nonhot) and occ_nonhot > 0 else float("nan")
    diff = float(occ_hot - occ_nonhot) if np.isfinite(occ_hot) and np.isfinite(occ_nonhot) else float("nan")

    if save_per_position_csv is not None:
        out = pd.DataFrame(
            {
                "pos": np.arange(maxT, dtype=int),
                "mean_posterior_occupancy": mean_occ_per_pos,
                "mutation_rate_any": mut_rate_per_pos_any,
                "mutation_rate_CtoT": mut_rate_per_pos_ct,
            }
        )
        out.to_csv(save_per_position_csv, index=False)

    posterior_summary = PosteriorSummary(
        mean_occ_overall=mean_occ_overall,
        mean_occ_per_seq=mean_occ_per_seq,
        mean_occ_per_pos=mean_occ_per_pos,
        mut_rate_per_seq_any=mut_rate_per_seq_any,
        mut_rate_per_seq_ct=mut_rate_per_seq_ct,
        mut_rate_per_pos_any=mut_rate_per_pos_any,
        mut_rate_per_pos_ct=mut_rate_per_pos_ct,
        corr_seq_occ_vs_mut_any=corr_seq_any,
        corr_seq_occ_vs_mut_ct=corr_seq_ct,
        corr_pos_occ_vs_mut_any=corr_pos_any,
        corr_pos_occ_vs_mut_ct=corr_pos_ct,
        occ_hotspots=occ_hot,
        occ_nonhotspots=occ_nonhot,
        occ_enrichment_fold=fold,
        occ_enrichment_diff=diff,
    )

    eval_summary: Optional[EvalSummary] = None
    if has_hidden and true_state_mat is not None and threshold is not None:
        eval_summary = evaluate_with_hidden_states(
            occ_mat,
            true_state_mat,
            threshold=float(threshold),
            max_curve_points=int(max_curve_points),
            save_curves_csv=save_curves_csv,
        )

    return posterior_summary, eval_summary



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Posterior state probabilities (forward–backward) and uncertainty reporting for AID binding."
    )
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT_FILE, help="CSV with original_seq, mutated_seq")
    parser.add_argument(
        "--params",
        type=str,
        default=DEFAULT_PARAMS,
        help=f"Comma-separated params: {', '.join(PARAM_NAMES)}",
    )
    parser.add_argument("--noise", type=float, default=NOISE, help="Fixed sequencing noise used in state 0")
    parser.add_argument(
        "--save-per-position",
        type=str,
        default=None,
        help="Optional CSV output with per-position mean occupancy and mutation densities",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "original_seq" not in df.columns or "mutated_seq" not in df.columns:
        raise ValueError("CSV must contain columns: original_seq, mutated_seq")

    params = _parse_params(args.params)

    t0 = time.time()  # type: ignore[name-defined]
    summary = summarize_dataset(df, params=params, noise=float(args.noise), save_per_position_csv=args.save_per_position)
    dt = time.time() - t0  # type: ignore[name-defined]

    # Required measurements to report
    print("--- Posterior AID binding uncertainty via forward–backward ---")
    print(f"Input: {args.input}")
    print(f"Sequences: {len(df)}")
    print(f"Params: {args.params}")
    print(f"Noise (fixed): {args.noise}")
    print(f"Runtime: {dt:.2f}s")

    print("\nMean posterior occupancy (state=bound):")
    print(f"  Overall mean occupancy: {summary.mean_occ_overall:.6f}")
    print(f"  Across-sequence mean (mean of per-seq means): {float(np.nanmean(summary.mean_occ_per_seq)):.6f}")

    print("\nVariance across sequences:")
    print(f"  Var(per-seq mean occupancy): {float(np.nanvar(summary.mean_occ_per_seq, ddof=1)) if len(summary.mean_occ_per_seq)>1 else 0.0:.6g}")
    print(f"  Std(per-seq mean occupancy): {float(np.nanstd(summary.mean_occ_per_seq, ddof=1)) if len(summary.mean_occ_per_seq)>1 else 0.0:.6g}")
    qs = np.nanquantile(summary.mean_occ_per_seq, [0.05, 0.5, 0.95]) if len(summary.mean_occ_per_seq) > 0 else [np.nan, np.nan, np.nan]
    print(f"  Quantiles (5%, 50%, 95%): {qs[0]:.6g}, {qs[1]:.6g}, {qs[2]:.6g}")

    print("\nCorrelation with mutation density:")
    print("  (Two definitions: any base change; and C->T only)")
    print(f"  Per-sequence corr(mean occupancy, mutation density any):  {summary.corr_seq_occ_vs_mut_any:.4f}")
    print(f"  Per-sequence corr(mean occupancy, mutation density C->T): {summary.corr_seq_occ_vs_mut_ct:.4f}")
    print(f"  Per-position corr(mean occupancy, mutation rate any):     {summary.corr_pos_occ_vs_mut_any:.4f}")
    print(f"  Per-position corr(mean occupancy, mutation rate C->T):    {summary.corr_pos_occ_vs_mut_ct:.4f}")

    print("\nEnrichment at hotspots (WRC, posterior occupancy):")
    print(f"  Mean occupancy at hotspots:     {summary.occ_hotspots:.6f}")
    print(f"  Mean occupancy at non-hotspots: {summary.occ_nonhotspots:.6f}")
    print(f"  Fold enrichment (hot/nonhot):   {summary.occ_enrichment_fold:.6g}")
    print(f"  Difference (hot - nonhot):      {summary.occ_enrichment_diff:.6g}")

    if args.save_per_position is not None:
        print(f"\nSaved per-position summary to: {args.save_per_position}")


if __name__ == "__main__":
    # Local import to avoid forcing time in module import paths elsewhere.
    import time

    main()