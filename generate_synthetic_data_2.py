"""
Synthetic AID-like (Activation-Induced Deaminase) mutation generator:
- Hidden states encode only binding: 0=unbound/background, 1=bound/scanning
- "Firing" is an emission/event while in state 1, with context-dependent probability 
        
        (hotspot_bias/coldspot_bias: Context-dependent multipliers for WRC (hotspot)
        and SYC (coldspot) sequence)

- Mutations happen on C with probability p_fire * mutation_eff 
        (mutation_eff: Repair failure rate (80% of deaminated bases become mutations))
- Background noise mutations happen in state 0 with probability noise
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd


## Config
OUTPUT_FILENAME = "synthetic_aid_data_2.csv"
NUM_SEQUENCES = 100
SEQ_LENGTH = 1000

# init params for AID activity on synthesized data
params = {
    # Prob that AID initiates binding at any given base (state 0 -> state 1)
    "alpha_recruit": 0.03,

    # Processivity: Prob AID stays bound and moves to next base (State 1 -> State 1)
    "lambda_scan": 0.95,

    # While scanning (state 1), probability of a catalytic "fire" at a neutral C
    # in Markov model -> emission
    "base_activation": 0.04,

    # Context multipliers for catalytic "fire"
    # WRC (hotspot) and SYC (coldspot), where C is the target base at position t
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
    Two-state HMM:
      state 0: unbound/background
      state 1: bound/scanning

    Emissions:
      - if state 1: AID "fires" with context-dependent p_fire; if fires on C then mutate with mutation_eff
      - if state 0: random substitution noise with probability noise
    """

    def __init__(self, params: Dict[str, float], rng: np.random.Generator | None = None):
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_context_fire_prob(self, seq: str, pos: int) -> float:
        """
        Return p_fire at position pos.
        Only defined for targets where seq[pos] == 'C' and pos>=2 (needs two upstream bases).
        Motifs use IUPAC codes:
          W = A/T
          R = A/G
          S = G/C
          Y = C/T
        Hotspot: W R C
        Coldspot: S Y C
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
            return _clamp01(self.params["base_activation"] * self.params["hotspot_bias"])

        is_S = b2 in ("G", "C")
        is_Y = b1 in ("C", "T")
        if is_S and is_Y:
            return _clamp01(self.params["base_activation"] * self.params["coldspot_bias"])

        return _clamp01(self.params["base_activation"])

    def run(self, length: int = 1000) -> Dict[str, Any]:
        bases = ["A", "C", "G", "T"]

        # generate random germline seq (single strand representation)
        original_seq = "".join(self.rng.choice(bases, size=length))

        hidden_states = np.zeros(length, dtype=int)  # 0/1 only
        fire_events = np.zeros(length, dtype=int)    # 1 if an AID fire event happens at pos t
        
        # mutable list of single-character strings, initially copied from original_seq.
        mutated_seq: List[str] = list(original_seq)

        current_state = 0  # start unbound

        for t in range(length):
            # ---- Transition (binding only) ----
            if t > 0:
                prev = hidden_states[t - 1]
                if prev == 0:
                    # recruit (0->1) with alpha_recruit
                    current_state = 1 if self.rng.random() < self.params["alpha_recruit"] else 0
                else:
                    # stay bound (1->1) with lambda_scan else fall off (1->0)
                    current_state = 1 if self.rng.random() < self.params["lambda_scan"] else 0

                hidden_states[t] = current_state
            else:
                hidden_states[0] = current_state

            # ---- Emissions ----
            if current_state == 1:
                # AID "fire" event is context-dependent
                p_fire = self.get_context_fire_prob(original_seq, t)
                did_fire = self.rng.random() < p_fire
                fire_events[t] = 1 if did_fire else 0

                # If it fired on a C, mutate with probability mutation_eff
                if did_fire and original_seq[t] == "C" and self.rng.random() < self.params["mutation_eff"]:
                    mutated_seq[t] = "T"

            else:
                # background noise substitution
                if self.rng.random() < self.params["noise"]:
                    mutated_seq[t] = self.rng.choice([b for b in bases if b != original_seq[t]])

        return {
            "original_seq": original_seq,
            "mutated_seq": "".join(mutated_seq),
            "hidden_states": hidden_states,
            "fire_events": fire_events,
        }


def main() -> None:
    print("--- Generating Synthetic Data (Option 1: fire-as-emission) ---")
    print(f"Sequences: {NUM_SEQUENCES} | Length: {SEQ_LENGTH}")

    sim = AIDSim(params)
    dataset: List[Dict[str, Any]] = []

    total_mutations = 0
    total_fire_events = 0
    total_bound_positions = 0

    for _ in range(NUM_SEQUENCES):
        result = sim.run(length=SEQ_LENGTH)

        orig = result["original_seq"]
        mut = result["mutated_seq"]
        states = result["hidden_states"]
        fires = result["fire_events"]

        total_mutations += sum(1 for a, b in zip(orig, mut) if a != b)
        total_fire_events += int(fires.sum())
        total_bound_positions += int(states.sum())

        dataset.append(result)

    df = pd.DataFrame(dataset)
    df["hidden_states"] = df["hidden_states"].apply(lambda x: "".join(map(str, x.tolist())))
    df["fire_events"] = df["fire_events"].apply(lambda x: "".join(map(str, x.tolist())))
    df.to_csv(OUTPUT_FILENAME, index=False)

    print("\n--- Generation Complete ---")
    print(f"Saved to: {os.path.abspath(OUTPUT_FILENAME)}")
    print(f"Total mutations across dataset: {total_mutations}")
    print(f"Total fire events: {total_fire_events}")
    print(f"Total bound positions (state=1): {total_bound_positions}")
    print(f"Avg mutations per sequence: {total_mutations / NUM_SEQUENCES:.4f}")


if __name__ == "__main__":
    main()