import numpy as np
import random
import pandas as pd
import json
import os


## Config
OUTPUT_FILENAME = "synthetic_aid_data.csv"
NUM_SEQUENCES = 100
SEQ_LENGTH = 1000


# init params for AID activity on synthesized data
params = {

    # Prob that AID initiates binding at any given base (state 0 -> state 1)
    # this should be used to normalize the avg mut / genration at ~0.3
    "alpha_recruit": 0.03,

    # Processivity: Prob AID stays bound and moves to next base (State 1 -> State 1)
    # 1 - (1 / average_patch_size) for patch_size ~20bp
    "lambda_scan": 0.95,
    
    # State 1 -> State 2, base catalytic activity (prob of firing at a neutral spot)
    "base_activation": 0.04,
    
    ## context: multipliers for catalytic activity (state 1 -> state 2)
    # WRC (Hotspot) SYC (coldspot)
    "hotspot_bias": 10.0,
    "coldspot_bias": 0.1,
    

    # Emission: if active (state 2) how likely is a mutation
    # this accounts for repair failure/success
    "mutation_eff": 0.8,

    # noise: sequencing error rate (state 0 emission)
    "noise": 0.0001 
    
}






class AIDSim:
    def __init__(self, params):
        self.params = params
    
    def get_context_prob(self, seq, pos):
        """Calculate P(state 1 -> state 2) given seq"""
        
        # check boundaries (wrc has 2 bases upstream)
        if pos < 2 or pos >= len(seq):
            return self.params["base_activation"]
        
        # only C
        if seq[pos] != "C":
            return 0.0
        
        ## WRC
        base_minus_2 = seq[pos-2]
        base_minus_1 = seq[pos-1]
        
        is_W = base_minus_2 in ["A", "T"]
        is_R = base_minus_1 in ["A", "G"]
        
        if is_W and is_R:
            # hotspot
            prob = self.params["base_activation"] * self.params["hotspot_bias"]
            return min(prob, 1.0) # cap
        
        ## SYC
        is_S = base_minus_2 in ["G", "C"]
        is_Y = base_minus_1 in ["C", "T"]
        
        if is_S and is_Y:
            # coldspot
            return self.params["base_activation"] * self.params["coldspot_bias"]
        
        ## neutral
        return self.params["base_activation"]
        
                
    
        
    def run(self, length = 1000):
        bases = ["A", "C", "G", "T"]
        
        # generate random germline seq
        original_seq = "".join(random.choices(bases, k = length))
        
        hidden_states = np.zeros(length, dtype=int)
        mutated_seq = list(original_seq)
        
        # start in background (State 0)
        current_state = 0
        
        for t in range(length):
            ### Transition
            # look at transition prob from the last step
            
            if t > 0:
                prev_state = hidden_states[t-1]
                
                if prev_state == 0:
                    # background -> scan or background
                    p_recruit = self.params["alpha_recruit"]
                    probs = [1 - p_recruit, p_recruit, 0.0]
                    
                elif prev_state ==1:
                    # Scan -> active, scan or background
                    
                    # calculate the context dependent activation
                    p_active = self.get_context_prob(original_seq, t)
                    
                    # remaining prob
                    p_stay_scan = self.params["lambda_scan"] * (1-p_active)
                    p_fall_off = 1.0 - p_active - p_stay_scan
                    
                    probs = [p_fall_off, p_stay_scan, p_active]
                    
                elif prev_state == 2:
                    # Active -> scan or background
                    p_return_scan = self.params["lambda_scan"]
                    p_fall_off = 1.0 - p_return_scan
                    
                    probs = [p_fall_off, p_return_scan, 0.0]
                
                # now sample the state for pos t with probs    
                current_state = np.random.choice([0,1,2], p=probs)
                hidden_states[t] = current_state
            
            
            ### Emission (mutation)
            if current_state ==2: # acitve
                # mutation if aid deaminates and then repair efficiency fails
                if original_seq[t] =="C" and random.random() < self.params["mutation_eff"]:
                    mutated_seq[t] = "T"
                    
            elif current_state ==0: #background
                if random.random() < self.params["noise"]:
                    # put any random base
                    mutated_seq[t] = random.choice([b for b in bases if b!= original_seq[t]])
        
        return{
            "original_seq": original_seq,
            "hidden_states": hidden_states,
            "mutated_seq":mutated_seq
            
        }
        
        
        
def main():
    print(f"--- Generating Synthetic Data ---")
    print(f"Sequences: {NUM_SEQUENCES} | Length: {SEQ_LENGTH}")
    
    sim = AIDSim(params)
    dataset = []
    
    total_mutations = 0
    active_states_count = 0
    
    for i in range(NUM_SEQUENCES):
        result = sim.run(length=SEQ_LENGTH)
        result["mutated_seq"] = "".join(result["mutated_seq"])
        dataset.append(result)
        
        #stats
        orig = result["original_seq"]
        mut = result["mutated_seq"]
        states = result["hidden_states"]
        
        muts = sum(1 for a, b in zip(orig, mut) if a != b)
        total_mutations += muts
        active_states_count += sum(1 for s in states if s == 2)
        
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{NUM_SEQUENCES} sequences")

    df = pd.DataFrame(dataset)
    df["hidden_states"] = df["hidden_states"].apply(lambda x: "".join(map(str, x)))
    df.to_csv(OUTPUT_FILENAME, index = False)
    
        
    print(f"\n--- Generation Complete ---")
    print(f"Saved to: {os.path.abspath(OUTPUT_FILENAME)}")
    print(f"Total Mutations across dataset: {total_mutations}")
    print(f"Total Enzyme 'Fire' Events (State 2): {active_states_count}")
    print(f"Avg Mutations per Sequence: {total_mutations / NUM_SEQUENCES}")

if __name__ == "__main__":
    main()