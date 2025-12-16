import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys
import time

INPUT_FILE = "synthetic_aid_data.csv"

noise = 0.0001 # sequencing error is known normally, so we can just assume it given


class AID_HMM:
    def __init__(self, noise):
        self.noise = noise
        self.bases = ["A", "C", "G", "T"]
        
    
    def get_context_prob(self, seq, pos, base_activation, hotspot_bias, coldspot_bias):
        """
        This calculates the prob of entering "active" state (state 2)
        given the sequence motif. Its basically the same as in the data synth script
        """
        
        # check if you are at the boundaries
        if pos < 2 or pos >= len(seq):
            return base_activation
        
        # recuitment check (only C recruits)
        if seq[pos] != "C":
            return 0.0
        
        # check motif
        base_minus_2 = seq[pos-2]
        base_minus_1 = seq[pos-1]
        
        # WRC 
        is_W = base_minus_2 in ["A", "T"]
        is_R = base_minus_1 in ["A", "G"]
        if is_W and is_R:
            # cap probability at 1.0 to prevent errors
            return min(base_activation * hotspot_bias, 1.0)
        
        # SCY (cold)
        is_S = base_minus_2 in ["G", "C"]
        is_Y = base_minus_1 in ["C", "T"]
        if is_S and is_Y:
            return base_activation * coldspot_bias
        
        #neutral context (no motif)
        return base_activation


    def forward_loglik(self, orig_seq, mut_seq, params):
        """
        Compute the log-likelihood of the observed mutation seq
        given a set of 6 params
        """
        
        # unpack params
        alpha_rec, lambda_scan, base_act, hot_bias, cold_bias, mut_eff = params
        
        T = len(orig_seq)
        n_states = 3
        
        # init alpha (log_space)
        # alpha matrix: alpha[t,j] = log P(obs_0...obs_t, State_t=j)
        alpha = np.full((T, n_states), -np.inf)
        
        # force background (state 0) as start
        alpha[0,0] = 0.0
        
        # iterate through the seq (Time t)
        for t in range(1, T):
            
            ## First Transition probs (how likely is it to move to state j at time t given t-1)
            # context dep activation of this position
            p_act = self.get_context_prob(orig_seq, t, base_act, hot_bias, cold_bias)
            
            # State 0 transition
            t0 = [1-alpha_rec, alpha_rec, 0.0]
            
            # state 0 transition
            p_scan_stay = lambda_scan * (1-p_act)
            p_fall = 1.0 - p_act - p_scan_stay
            if p_fall < 0:
                p_fall = 0
            t1 = [p_fall, p_scan_stay, p_act]
            
            #state 2
            t2 = [1-lambda_scan, lambda_scan, 0.0]
            
            trans_mat = np.array([t0,t1,t2])
            
            
            ## Emission probability (how likely is it to observe nuc j at t)
            obs = mut_seq[t] # get current nuc
            orig = orig_seq[t]
            emit = np.zeros(3)
            
            # state 0 and 1 (noise)
            if obs == orig:
                p_emit_bg = 1 - self.noise
            else:
                p_emit_bg = self.noise / 3.0
            
            emit[0] = p_emit_bg
            emit[1] = p_emit_bg
            
            
        
        
        
