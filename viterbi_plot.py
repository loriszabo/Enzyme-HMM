import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# CHANGE THIS INTEGER TO VIEW A DIFFERENT SEQUENCE (0 to 99)
SEQUENCE_INDEX = 3

# File path to your data
DATA_FILE = "synthetic_aid_data_2.csv"

# Inferred Parameters (Hardcoded from your successful inference run)
# These are the "rules" the model learned.
PARAMS = (
    0.037806,  # alpha_recruit (prob of binding)
    0.938105,  # lambda_scan   (prob of staying bound)
    0.044351,  # base_activation
    9.329096,  # hotspot_bias
    0.072519,  # coldspot_bias
    0.782644   # mutation_eff
)

# --- 2. HMM & VITERBI CLASS ---
def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

class AID_HMM_Viterbi:
    def __init__(self, noise: float = 1e-4):
        self.noise = float(noise)

    def get_context_fire_prob(self, seq, pos, base_act, hot_bias, cold_bias):
        if pos < 2 or pos >= len(seq): return 0.0
        if seq[pos] != "C": return 0.0

        b2, b1 = seq[pos - 2], seq[pos - 1]
        
        # WRC Hotspot (W=A/T, R=A/G)
        if b2 in ("A", "T") and b1 in ("A", "G"):
            return _clamp01(base_act * hot_bias)
        
        # SYC Coldspot (S=G/C, Y=C/T)
        if b2 in ("G", "C") and b1 in ("C", "T"):
            return _clamp01(base_act * cold_bias)

        return _clamp01(base_act)

    def _log_emit(self, orig_seq, mut_seq, t, state, params):
        orig, obs = orig_seq[t], mut_seq[t]
        
        # State 0: Background Noise
        if state == 0:
            p = (1.0 - self.noise) if obs == orig else (self.noise / 3.0)
            return -np.inf if p <= 0 else np.log(p)

        # State 1: Bound / Scanning
        _, _, base_act, hot_bias, cold_bias, mut_eff = params
        
        p_fire = self.get_context_fire_prob(orig_seq, t, base_act, hot_bias, cold_bias)
        p_mut = _clamp01(p_fire * mut_eff)

        if obs == orig:
            p = 1.0 - p_mut
        elif orig == "C" and obs == "T":
            p = p_mut
        else:
            p = 0.0 
            
        return -np.inf if p <= 0 else np.log(p)

    def viterbi_decode(self, orig_seq, mut_seq, params):
        alpha_recruit, lambda_scan, _, _, _, _ = params
        T = len(orig_seq)

        # Log Transition Matrix
        log_trans = np.log(np.array([
            [1.0 - alpha_recruit, alpha_recruit],
            [1.0 - lambda_scan,   lambda_scan]
        ]))

        # Initialize Tables
        viterbi = np.full((T, 2), -np.inf)
        backpointer = np.zeros((T, 2), dtype=int)

        # Initialization (t=0)
        viterbi[0, 0] = self._log_emit(orig_seq, mut_seq, 0, 0, params)
        viterbi[0, 1] = -np.inf 

        # Recursion
        for t in range(1, T):
            for s_curr in [0, 1]:
                log_emit = self._log_emit(orig_seq, mut_seq, t, s_curr, params)
                scores = [viterbi[t-1, s_prev] + log_trans[s_prev, s_curr] for s_prev in [0, 1]]
                best_prev = np.argmax(scores)
                viterbi[t, s_curr] = scores[best_prev] + log_emit
                backpointer[t, s_curr] = best_prev

        # Backtracking
        best_path = [np.argmax(viterbi[T-1, :])]
        for t in range(T-1, 0, -1):
            best_path.append(backpointer[t, best_path[-1]])
        
        return best_path[::-1]

# --- 3. PLOTTING FUNCTION ---
def plot_sequence(index):
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_FILE}")
        return

    if index < 0 or index >= len(df):
        print(f"Error: Index {index} is out of bounds (0-{len(df)-1})")
        return

    row = df.iloc[index]
    orig_seq = row['original_seq']
    mut_seq = row['mutated_seq']
    true_states = [int(x) for x in str(row['hidden_states'])]
    
    # Calculate Mutations
    mutations_indices = [i for i, (a, b) in enumerate(zip(orig_seq, mut_seq)) if a != b]
    mut_count = len(mutations_indices)

    print(f"--- Processing Sequence {index} ---")
    print(f"Total Mutations: {mut_count}")
    print("Running Viterbi decoding...")

    # Run Viterbi
    hmm = AID_HMM_Viterbi()
    inferred_states = hmm.viterbi_decode(orig_seq, mut_seq, PARAMS)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 5))
    x_axis = range(len(true_states))

    # 1. Ground Truth (Blue)
    ax.step(x_axis, true_states, where='mid', label='True Activity (Simulation)', 
            color='#1f77b4', linewidth=2, alpha=0.6)
    ax.fill_between(x_axis, true_states, step='mid', color='#1f77b4', alpha=0.1)

    # 2. Inferred Viterbi Path (Orange Dashed)
    # Offset y slightly (-0.05) to separate lines visually
    offset_inferred = [s - 0.05 for s in inferred_states]
    ax.step(x_axis, offset_inferred, where='mid', label='Inferred Activity (Viterbi)', 
            color='#ff7f0e', linestyle='--', linewidth=2)

    # 3. Mutations (Red X)
    ax.scatter(mutations_indices, [0.5] * mut_count, color='#d62728', marker='x', s=80, 
               label=f'Mutations ({mut_count})', zorder=10)

    # Formatting
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Unbound (0)', 'Bound (1)'])
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel('Sequence Position (Nucleotides)')
    ax.set_title(f'AID Enzyme Activity: True vs Inferred (Sequence #{index})')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- 4. EXECUTION ---
if __name__ == "__main__":
    plot_sequence(SEQUENCE_INDEX)