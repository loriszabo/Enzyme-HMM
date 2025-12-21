"""
Germline AID Mutation Generator:
- Loads real IGHV germline sequences from 'human_ighv_germlines.csv'.
- Applies the exact same AID-like HMM model as the synthetic generator.
- Generates 1 mutated sequence per germline sequence.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List

## Config
INPUT_FILENAME = "human_ighv_germlines.csv"
OUTPUT_FILENAME = "simulated_germline_mutations.csv"

# Init params for AID activity (Same as generate_synthetic_data_2.py)
params = {
    # Prob that AID initiates binding at any given base (state 0 -> state 1)
    "alpha_recruit": 0.03,

    # Processivity: Prob AID stays bound and moves to next base (State 1 -> State 1)
    "lambda_scan": 0.95,

    # While scanning (state 1), probability of a catalytic "fire" at a neutral C
    "base_activation": 0.04,

    # Context multipliers for catalytic "fire"
    # WRC (hotspot) and SYC (coldspot)
    "hotspot_bias": 10.0,
    "coldspot_bias": 0.1,

    # Given a fire on C, probability that it becomes an observed mutation (repair fails)
    "mutation_eff": 0.8,

    # Background sequencing error rate (state 0 emission)
    "noise": 0.0001,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class AIDSim:
    """
    Two-state HMM adapted for applying mutations to existing germline sequences.
    """

    def __init__(self, params: Dict[str, float], rng: np.random.Generator | None = None):
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_context_fire_prob(self, seq: str, pos: int) -> float:
        """
        Return p_fire at position pos based on WRC/SYC motifs.
        """
        if pos < 2 or pos >= len(seq):
            return 0.0
        if seq[pos] != "C":
            return 0.0

        b2 = seq[pos - 2]
        b1 = seq[pos - 1]

        # Hotspot: WRC (W=A/T, R=A/G)
        is_W = b2 in ("A", "T")
        is_R = b1 in ("A", "G")
        if is_W and is_R:
            return _clamp01(self.params["base_activation"] * self.params["hotspot_bias"])

        # Coldspot: SYC (S=G/C, Y=C/T)
        is_S = b2 in ("G", "C")
        is_Y = b1 in ("C", "T")
        if is_S and is_Y:
            return _clamp01(self.params["base_activation"] * self.params["coldspot_bias"])

        return _clamp01(self.params["base_activation"])

    def run_on_sequence(self, germline_seq: str) -> Dict[str, Any]:
        """
        Runs the HMM on a provided germline sequence string.
        """
        length = len(germline_seq)
        bases = ["A", "C", "G", "T"]

        hidden_states = np.zeros(length, dtype=int)  # 0=unbound, 1=bound
        fire_events = np.zeros(length, dtype=int)    # 1=AID fired
        
        # Start with the input germline
        mutated_seq: List[str] = list(germline_seq)

        current_state = 0  # start unbound

        for t in range(length):
            # ---- Transition (binding only) ----
            if t > 0:
                prev = hidden_states[t - 1]
                if prev == 0:
                    # recruit (0->1)
                    current_state = 1 if self.rng.random() < self.params["alpha_recruit"] else 0
                else:
                    # stay bound (1->1) or fall off (1->0)
                    current_state = 1 if self.rng.random() < self.params["lambda_scan"] else 0
                hidden_states[t] = current_state
            else:
                hidden_states[0] = current_state

            # ---- Emissions ----
            if current_state == 1:
                # AID "fire" event
                p_fire = self.get_context_fire_prob(germline_seq, t)
                did_fire = self.rng.random() < p_fire
                fire_events[t] = 1 if did_fire else 0

                # If it fired on a C, mutate with probability mutation_eff
                if did_fire and germline_seq[t] == "C" and self.rng.random() < self.params["mutation_eff"]:
                    mutated_seq[t] = "T" # Standard AID mutation is C->T (U)

            else:
                # Background noise substitution (State 0)
                if self.rng.random() < self.params["noise"]:
                    # Pick a base different from the germline base
                    original_base = germline_seq[t]
                    possible_bases = [b for b in bases if b != original_base]
                    # Handle edge case if original_base is 'N' or weird char
                    if not possible_bases: 
                         possible_bases = bases
                    mutated_seq[t] = self.rng.choice(possible_bases)

        return {
            "germline_seq": germline_seq,
            "mutated_seq": "".join(mutated_seq),
            "hidden_states": hidden_states,
            "fire_events": fire_events,
        }


def main() -> None:
    print("--- Generating Mutations on Real Germline Data ---")
    
    if not os.path.exists(INPUT_FILENAME):
        print(f"Error: {INPUT_FILENAME} not found. Please run fetch_germline_data.py first.")
        sys.exit(1)

    print(f"Loading germlines from: {INPUT_FILENAME}")
    input_df = pd.read_csv(INPUT_FILENAME)
    
    if "germline_seq" not in input_df.columns:
        print("Error: Input CSV must contain 'germline_seq' column.")
        sys.exit(1)

    sim = AIDSim(params)
    dataset: List[Dict[str, Any]] = []

    total_mutations = 0
    total_fire_events = 0
    
    # Iterate over unique germlines (one mutation sequence per germline)
    sequences = input_df["germline_seq"].tolist()
    print(f"Processing {len(sequences)} sequences...")

    for seq in sequences:
        # Sanitize input (ensure upper case)
        seq = str(seq).upper()
        
        result = sim.run_on_sequence(seq)

        orig = result["germline_seq"]
        mut = result["mutated_seq"]
        fires = result["fire_events"]
        states = result["hidden_states"]

        # Calculate stats
        mut_count = sum(1 for a, b in zip(orig, mut) if a != b)
        total_mutations += mut_count
        total_fire_events += int(fires.sum())

        dataset.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # Format arrays as strings for CSV storage
    df["hidden_states"] = df["hidden_states"].apply(lambda x: "".join(map(str, x.tolist())))
    df["fire_events"] = df["fire_events"].apply(lambda x: "".join(map(str, x.tolist())))
    
    df.to_csv(OUTPUT_FILENAME, index=False)

    print("\n--- Generation Complete ---")
    print(f"Saved to: {os.path.abspath(OUTPUT_FILENAME)}")
    print(f"Total mutations generated: {total_mutations}")
    print(f"Total fire events: {total_fire_events}")
    print(f"Average mutations per sequence: {total_mutations / len(sequences):.4f}")


if __name__ == "__main__":
    main()