Abstract—Activation-Induced Cytidine Deaminase (AID) is the key enzyme in B-cell receptor maturation during adaptive immune response. Although its biological role is well established, its kinetic mechanisms remain mostly elusive. In this project, we propose a latent probabilistic modeling framework that treats AID activity as a hidden stochastic process along the B-cell receptor gene sequence. We model the enzyme using a Markovian model, with two enzyme activity states (bound/ unbound) for each base as hidden states, assuming observable mutations happen dependent on enzyme state, base state, and motif bias of enzyme activity. Using likelihood maximization, we estimate biologically interpretable parameters, calculate the posterior probabilities for enzyme states and use the Viterbi algorithm to reconstruct the most likely hidden enzyme activity path from observed mutation data. This project helps to better understand the probabilistic nature of AID dynamics during immune responses.

Please refer to the project report for detailed context.

- Use generate_snythetic_data_2.py to generate synthetic data
- inference_2.py and inference_with_evaluation.py serve as recovering the model parameters using likelihood maximization (forward). At the core both scripts do the same, inference_with_evaluation.py is an extended version with evaluation metrics for the project repor
- posterior_state_probabilites.py is the file for calcualating the posterior probabilities for the hidden states
- plots_posterior_probabilites.py creates nice plots
-viterby_algorithm_and_plot.py creates the corresponding Viterbi paths
