import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Defaults (match your project)
# -----------------------------
DEFAULT_DATA_FILE = "synthetic_aid_data_2.csv"
DEFAULT_NOISE = 1e-4

# Default inferred params (override with --params)
DEFAULT_PARAMS = "0.03,0.95,0.04,10.0,0.1,0.8"
PARAM_NAMES = [
    "alpha_recruit",
    "lambda_scan",
    "base_activation",
    "hotspot_bias",
    "coldspot_bias",
    "mutation_eff",
]


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _parse_params(s: str) -> Tuple[float, float, float, float, float, float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 6:
        raise ValueError(f"--params must have 6 comma-separated values: {', '.join(PARAM_NAMES)}")
    vals = tuple(float(x) for x in parts)
    return vals  # type: ignore[return-value]


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _to_state_array(hidden_states_str: str) -> np.ndarray:
    s = str(hidden_states_str).strip()
    return np.fromiter((1 if c == "1" else 0 for c in s), dtype=int, count=len(s))


def _mutations_mask(orig: str, mut: str) -> np.ndarray:
    return np.fromiter((1 if a != b else 0 for a, b in zip(orig, mut)), dtype=int, count=len(orig))


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> PRF1:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * prec * rec, prec + rec) if np.isfinite(prec) and np.isfinite(rec) else float("nan")
    return PRF1(precision=float(prec), recall=float(rec), f1=float(f1), tp=tp, fp=fp, fn=fn, tn=tn)


def _segments_from_binary(x: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert binary array to list of [start, end) segments where x==1.
    """
    x = np.asarray(x, dtype=int)
    segs: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(x):
        if v == 1 and not in_seg:
            in_seg = True
            start = i
        elif v == 0 and in_seg:
            in_seg = False
            segs.append((start, i))
    if in_seg:
        segs.append((start, len(x)))
    return segs


def _seg_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return float(inter / union) if union > 0 else 0.0


def _mean_best_iou(pred: np.ndarray, true: np.ndarray) -> float:
    pred_segs = _segments_from_binary(pred)
    true_segs = _segments_from_binary(true)
    if len(pred_segs) == 0:
        return float("nan")
    if len(true_segs) == 0:
        return 0.0
    best = []
    for ps in pred_segs:
        best.append(max(_seg_iou(ps, ts) for ts in true_segs))
    return float(np.mean(best)) if best else float("nan")


def _dice_positions(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=int)
    true = np.asarray(true, dtype=int)
    inter = int(np.sum((pred == 1) & (true == 1)))
    denom = int(np.sum(pred == 1) + np.sum(true == 1))
    return float(2 * inter / denom) if denom > 0 else float("nan")


# -----------------------------
# HMM: Viterbi + Posterior
# -----------------------------
class AID_HMM:
    def __init__(self, noise: float = DEFAULT_NOISE):
        self.noise = float(noise)

    def get_context_fire_prob(self, seq: str, pos: int, base_act: float, hot_bias: float, cold_bias: float) -> float:
        if pos < 2 or pos >= len(seq):
            return 0.0
        if seq[pos] != "C":
            return 0.0
        b2, b1 = seq[pos - 2], seq[pos - 1]

        # WRC hotspot (W=A/T, R=A/G)
        if (b2 in ("A", "T")) and (b1 in ("A", "G")):
            return _clamp01(base_act * hot_bias)

        # SYC coldspot (S=G/C, Y=C/T)
        if (b2 in ("G", "C")) and (b1 in ("C", "T")):
            return _clamp01(base_act * cold_bias)

        return _clamp01(base_act)

    def _log_emit(self, orig_seq: str, mut_seq: str, t: int, state: int, params) -> float:
        orig, obs = orig_seq[t], mut_seq[t]

        if state == 0:
            p = (1.0 - self.noise) if obs == orig else (self.noise / 3.0)
            return -np.inf if p <= 0 else float(np.log(p))

        # state == 1
        _, _, base_act, hot_bias, cold_bias, mut_eff = params
        p_fire = self.get_context_fire_prob(orig_seq, t, base_act, hot_bias, cold_bias)
        p_mut = _clamp01(p_fire * mut_eff)

        if obs == orig:
            p = 1.0 - p_mut
        elif orig == "C" and obs == "T":
            p = p_mut
        else:
            p = 0.0

        return -np.inf if p <= 0 else float(np.log(p))

    def _log_trans(self, params) -> np.ndarray:
        alpha_recruit, lambda_scan, _, _, _, _ = params
        eps = 1e-12
        t00 = max(eps, 1.0 - alpha_recruit)
        t01 = max(eps, alpha_recruit)
        t10 = max(eps, 1.0 - lambda_scan)
        t11 = max(eps, lambda_scan)
        return np.log(np.array([[t00, t01], [t10, t11]], dtype=float))

    def viterbi_decode(self, orig_seq: str, mut_seq: str, params) -> np.ndarray:
        T = len(orig_seq)
        log_trans = self._log_trans(params)

        v = np.full((T, 2), -np.inf, dtype=float)
        bp = np.zeros((T, 2), dtype=int)

        # Start forced in state 0
        v[0, 0] = self._log_emit(orig_seq, mut_seq, 0, 0, params)
        v[0, 1] = -np.inf

        for t in range(1, T):
            for s in (0, 1):
                le = self._log_emit(orig_seq, mut_seq, t, s, params)
                scores = np.array([v[t - 1, sp] + log_trans[sp, s] for sp in (0, 1)], dtype=float)
                bp[t, s] = int(np.argmax(scores))
                v[t, s] = float(scores[bp[t, s]] + le)

        path = np.empty(T, dtype=int)
        path[T - 1] = int(np.argmax(v[T - 1, :]))
        for t in range(T - 1, 0, -1):
            path[t - 1] = bp[t, path[t]]
        return path

    def forward_backward_posterior(self, orig_seq: str, mut_seq: str, params) -> Tuple[np.ndarray, float]:
        """
        Returns:
          gamma: (T,2) posterior state probabilities
          ll:    log-likelihood
        """
        T = len(orig_seq)
        log_trans = self._log_trans(params)

        loge = np.full((T, 2), -np.inf, dtype=float)
        for t in range(T):
            loge[t, 0] = self._log_emit(orig_seq, mut_seq, t, 0, params)
            loge[t, 1] = self._log_emit(orig_seq, mut_seq, t, 1, params)

        def logsumexp2(a: float, b: float) -> float:
            m = max(a, b)
            if not np.isfinite(m):
                return -np.inf
            return float(m + np.log(np.exp(a - m) + np.exp(b - m)))

        # Forward (start forced in state 0)
        fwd = np.full((T, 2), -np.inf, dtype=float)
        fwd[0, 0] = loge[0, 0]
        fwd[0, 1] = -np.inf
        for t in range(1, T):
            fwd[t, 0] = loge[t, 0] + logsumexp2(fwd[t - 1, 0] + log_trans[0, 0], fwd[t - 1, 1] + log_trans[1, 0])
            fwd[t, 1] = loge[t, 1] + logsumexp2(fwd[t - 1, 0] + log_trans[0, 1], fwd[t - 1, 1] + log_trans[1, 1])

        ll = logsumexp2(fwd[T - 1, 0], fwd[T - 1, 1])

        # Backward
        bwd = np.full((T, 2), -np.inf, dtype=float)
        bwd[T - 1, :] = 0.0
        for t in range(T - 2, -1, -1):
            bwd[t, 0] = logsumexp2(
                log_trans[0, 0] + loge[t + 1, 0] + bwd[t + 1, 0],
                log_trans[0, 1] + loge[t + 1, 1] + bwd[t + 1, 1],
            )
            bwd[t, 1] = logsumexp2(
                log_trans[1, 0] + loge[t + 1, 0] + bwd[t + 1, 0],
                log_trans[1, 1] + loge[t + 1, 1] + bwd[t + 1, 1],
            )

        log_gamma = fwd + bwd - ll
        gamma = np.exp(log_gamma)
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        return gamma, float(ll)


# -----------------------------
# Analysis / plotting routines
# -----------------------------
def plot_sequence_comparison(
    df: pd.DataFrame,
    index: int,
    params,
    noise: float,
    out_dir: str,
    threshold: float,
) -> None:
    row = df.iloc[index]
    orig_seq = str(row["original_seq"])
    mut_seq = str(row["mutated_seq"])
    T = len(orig_seq)

    hmm = AID_HMM(noise=noise)
    v_path = hmm.viterbi_decode(orig_seq, mut_seq, params)
    gamma, ll = hmm.forward_backward_posterior(orig_seq, mut_seq, params)
    post_p1 = gamma[:, 1]
    post_hard = (post_p1 >= threshold).astype(int)

    muts = _mutations_mask(orig_seq, mut_seq)
    mut_idx = np.where(muts == 1)[0]

    has_true = "hidden_states" in df.columns
    true_states = _to_state_array(row["hidden_states"]) if has_true else None
    if has_true and len(true_states) != T:
        raise ValueError(f"Row {index}: hidden_states length {len(true_states)} != sequence length {T}")

    # --- Plot 1: states + posterior + mutations ---
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(T)

    if has_true and true_states is not None:
        ax.step(x, true_states, where="mid", label="True state", color="#1f77b4", linewidth=2, alpha=0.7)
        ax.fill_between(x, true_states, step="mid", color="#1f77b4", alpha=0.08)

    ax.step(x, v_path - 0.05, where="mid", label="Viterbi", color="#ff7f0e", linestyle="--", linewidth=2)
    ax.plot(x, post_p1, label="Posterior P(state=1)", color="#2ca02c", linewidth=1.5, alpha=0.9)
    ax.step(x, post_hard + 0.05, where="mid", label=f"Posterior hard (thr={threshold:.2f})", color="#2ca02c", linestyle=":", linewidth=2)

    ax.scatter(mut_idx, np.full(mut_idx.shape, 0.5), color="#d62728", marker="x", s=70, label=f"Mutations ({len(mut_idx)})", zorder=10)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Unbound (0)", "Bound (1)"])
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlabel("Sequence position")
    ax.set_title(f"Sequence #{index} | LL={ll:.3f} | True vs Viterbi vs Posterior")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"sequence_{index:03d}_comparison.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    # --- Print metrics for this sequence (if ground truth exists) ---
    if has_true and true_states is not None:
        acc_v = float(np.mean(v_path == true_states))
        prf_v = _precision_recall_f1(true_states, v_path)

        acc_p = float(np.mean(post_hard == true_states))
        prf_p = _precision_recall_f1(true_states, post_hard)

        seg_iou_v = _mean_best_iou(v_path, true_states)
        seg_iou_p = _mean_best_iou(post_hard, true_states)
        dice_v = _dice_positions(v_path, true_states)
        dice_p = _dice_positions(post_hard, true_states)

        print(f"\nSequence #{index} metrics (vs true hidden_states):")
        print(f"  Viterbi:   acc={acc_v:.4f} | prec={prf_v.precision:.4f} rec={prf_v.recall:.4f} f1={prf_v.f1:.4f} | Dice={dice_v:.4f} | mean-best-IoU={seg_iou_v:.4f}")
        print(f"  Posterior: acc={acc_p:.4f} | prec={prf_p.precision:.4f} rec={prf_p.recall:.4f} f1={prf_p.f1:.4f} | Dice={dice_p:.4f} | mean-best-IoU={seg_iou_p:.4f}")
        print(f"  Saved: {out_path}")
    else:
        print(f"\nSequence #{index}: no hidden_states column found; saved plot only: {out_path}")


def compute_dataset_level_curves_and_plots(
    df: pd.DataFrame,
    params,
    noise: float,
    out_dir: str,
    threshold: float,
    max_sequences: Optional[int] = None,
) -> None:
    if "hidden_states" not in df.columns:
        print("\nDataset-level evaluation skipped: CSV has no 'hidden_states' column.")
        return

    n = len(df) if max_sequences is None else min(len(df), int(max_sequences))
    seqs = df["original_seq"].astype(str).values[:n]
    muts = df["mutated_seq"].astype(str).values[:n]
    hids = df["hidden_states"].astype(str).values[:n]

    T = len(seqs[0])
    if not all(len(s) == T for s in seqs):
        raise ValueError("This script assumes fixed-length sequences for per-position plots (your generator uses fixed length).")
    if not all(len(s) == T for s in hids):
        raise ValueError("hidden_states length mismatch across rows.")

    hmm = AID_HMM(noise=noise)

    y_true = np.zeros((n, T), dtype=int)
    y_vit = np.zeros((n, T), dtype=int)
    y_post_hard = np.zeros((n, T), dtype=int)
    post_p1_mat = np.zeros((n, T), dtype=float)

    for i in range(n):
        orig = seqs[i]
        mut = muts[i]
        y_true[i, :] = _to_state_array(hids[i])

        y_vit[i, :] = hmm.viterbi_decode(orig, mut, params)
        gamma, _ll = hmm.forward_backward_posterior(orig, mut, params)
        post_p1 = gamma[:, 1]
        post_p1_mat[i, :] = post_p1
        y_post_hard[i, :] = (post_p1 >= threshold).astype(int)

    # --- Per-position accuracy ---
    acc_v = np.mean(y_vit == y_true, axis=0)
    acc_p = np.mean(y_post_hard == y_true, axis=0)

    # Per-position precision/recall (bound=1)
    prec_v = np.full(T, np.nan, dtype=float)
    rec_v = np.full(T, np.nan, dtype=float)
    prec_p = np.full(T, np.nan, dtype=float)
    rec_p = np.full(T, np.nan, dtype=float)

    for t in range(T):
        prf = _precision_recall_f1(y_true[:, t], y_vit[:, t])
        prec_v[t], rec_v[t] = prf.precision, prf.recall
        prf2 = _precision_recall_f1(y_true[:, t], y_post_hard[:, t])
        prec_p[t], rec_p[t] = prf2.precision, prf2.recall

    # Overall stats (flatten)
    prf_v_all = _precision_recall_f1(y_true.ravel(), y_vit.ravel())
    prf_p_all = _precision_recall_f1(y_true.ravel(), y_post_hard.ravel())
    overall_acc_v = float(np.mean(y_vit == y_true))
    overall_acc_p = float(np.mean(y_post_hard == y_true))

    # Segment overlap metrics per sequence (IoU + Dice)
    iou_v = np.array([_mean_best_iou(y_vit[i], y_true[i]) for i in range(n)], dtype=float)
    iou_p = np.array([_mean_best_iou(y_post_hard[i], y_true[i]) for i in range(n)], dtype=float)
    dice_v = np.array([_dice_positions(y_vit[i], y_true[i]) for i in range(n)], dtype=float)
    dice_p = np.array([_dice_positions(y_post_hard[i], y_true[i]) for i in range(n)], dtype=float)

    # --- Plot: per-position accuracy ---
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(T)
    ax.plot(x, acc_v, label="Accuracy per position (Viterbi)", color="#ff7f0e", linewidth=1.6)
    ax.plot(x, acc_p, label=f"Accuracy per position (Posterior hard thr={threshold:.2f})", color="#2ca02c", linewidth=1.6)
    ax.set_xlabel("Position")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-position accuracy across {n} sequences")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    p1 = os.path.join(out_dir, "per_position_accuracy.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)

    # # --- Plot: per-position precision/recall for bound state ---
    # fig, ax = plt.subplots(figsize=(16, 5))
    # ax.plot(x, prec_v, label="Precision (bound=1) per pos - Viterbi", color="#ff7f0e", linestyle="-", linewidth=1.4, alpha=0.9)
    # ax.plot(x, rec_v, label="Recall (bound=1) per pos - Viterbi", color="#ff7f0e", linestyle="--", linewidth=1.4, alpha=0.9)
    # ax.plot(x, prec_p, label=f"Precision per pos - Posterior hard (thr={threshold:.2f})", color="#2ca02c", linestyle="-", linewidth=1.4, alpha=0.9)
    # ax.plot(x, rec_p, label=f"Recall per pos - Posterior hard (thr={threshold:.2f})", color="#2ca02c", linestyle="--", linewidth=1.4, alpha=0.9)
    # ax.set_xlabel("Position")
    # ax.set_ylabel("Score")
    # ax.set_title(f"Per-position precision/recall (bound state) across {n} sequences")
    # ax.set_ylim(0.0, 1.02)
    # ax.grid(True, alpha=0.2)
    # ax.legend(ncol=2)
    # plt.tight_layout()
    # p2 = os.path.join(out_dir, "per_position_precision_recall_bound.png")
    # fig.savefig(p2, dpi=200)
    # plt.close(fig)

    # --- Plot: segment overlap histograms (IoU + Dice) ---
    def _hist2(a: np.ndarray, b: np.ndarray, title: str, xlabel: str, out_name: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        aa = a[np.isfinite(a)]
        bb = b[np.isfinite(b)]
        bins = np.linspace(0.0, 1.0, 26)
        ax.hist(aa, bins=bins, alpha=0.6, label="Viterbi", color="#ff7f0e")
        ax.hist(bb, bins=bins, alpha=0.6, label=f"Posterior hard (thr={threshold:.2f})", color="#2ca02c")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, out_name), dpi=200)
        plt.close(fig)

    _hist2(iou_v, iou_p, "Segment overlap distribution (mean best IoU per sequence)", "mean best IoU", "segment_iou_hist.png")
    _hist2(dice_v, dice_p, "Position overlap distribution (Dice per sequence)", "Dice coefficient", "dice_hist.png")

    # --- Statistical comparison: Viterbi vs posterior (summary print) ---
    # Soft posterior quality (Brier score vs true, averaged)
    brier = float(np.mean((post_p1_mat - y_true.astype(float)) ** 2))

    print("\n--- Dataset-level comparison (vs hidden_states) ---")
    print(f"Sequences used: {n} | Length: {T} | Threshold: {threshold:.2f} | Noise: {noise:g}")
    print("\nPosition-level classification (bound=1):")
    print(f"  Viterbi:   acc={overall_acc_v:.6f} | prec={prf_v_all.precision:.6f} rec={prf_v_all.recall:.6f} f1={prf_v_all.f1:.6f} | TP={prf_v_all.tp} FP={prf_v_all.fp} FN={prf_v_all.fn} TN={prf_v_all.tn}")
    print(f"  Posterior: acc={overall_acc_p:.6f} | prec={prf_p_all.precision:.6f} rec={prf_p_all.recall:.6f} f1={prf_p_all.f1:.6f} | TP={prf_p_all.tp} FP={prf_p_all.fp} FN={prf_p_all.fn} TN={prf_p_all.tn}")
    print("\nSegment/overlap metrics (per-sequence):")
    print(f"  mean best IoU:  Viterbi mean={float(np.nanmean(iou_v)):.6f} | Posterior mean={float(np.nanmean(iou_p)):.6f}")
    print(f"  Dice (pos):     Viterbi mean={float(np.nanmean(dice_v)):.6f} | Posterior mean={float(np.nanmean(dice_p)):.6f}")
    print("\nPosterior-only (soft) calibration-ish metric:")
    print(f"  Brier score (posterior P1 vs true): {brier:.6g}")

    print("\nSaved plots:")
    print(f"  {p1}")
    print(f"  {os.path.join(out_dir, 'segment_iou_hist.png')}")
    print(f"  {os.path.join(out_dir, 'dice_hist.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Viterbi vs posterior comparison: plots + accuracy/PR + segment overlap.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_FILE, help="CSV file (synthetic_aid_data_2.csv)")
    # in our data index 35 had the most mutationss
    parser.add_argument("--sequence-index", type=int, default=35, help="Index of a sequence to plot (0-based).")
    parser.add_argument("--params", type=str, default=DEFAULT_PARAMS, help="Comma-separated params: alpha_recruit,lambda_scan,base_activation,hotspot_bias,coldspot_bias,mutation_eff")
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE, help="Fixed background noise (state 0).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Posterior threshold for hard bound/unbound.")
    parser.add_argument("--out-dir", type=str, default="plots_viterbi_posterior", help="Directory to save plots.")
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional cap for dataset-level curves (debug/speed).")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "original_seq" not in df.columns or "mutated_seq" not in df.columns:
        raise ValueError("CSV must contain columns: original_seq, mutated_seq")

    out_dir = _ensure_dir(args.out_dir)
    params = _parse_params(args.params)

    if args.sequence_index < 0 or args.sequence_index >= len(df):
        raise ValueError(f"--sequence-index out of bounds: {args.sequence_index} (0..{len(df)-1})")

    # 1) Per-sequence visual comparison (saved)
    plot_sequence_comparison(
        df=df,
        index=int(args.sequence_index),
        params=params,
        noise=float(args.noise),
        out_dir=out_dir,
        threshold=float(args.threshold),
    )

    # 2) Dataset-level curves + metrics (saved + printed)
    compute_dataset_level_curves_and_plots(
        df=df,
        params=params,
        noise=float(args.noise),
        out_dir=out_dir,
        threshold=float(args.threshold),
        max_sequences=args.max_sequences,
    )


if __name__ == "__main__":
    main()