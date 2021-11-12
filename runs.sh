
###################################
# plots for Sepsis-MDP experiment #
###################################

# non-cf, non generalized pq
python run_mdp_experiment.py --noise_scale 0.0 --gadget2_jax

# non-cf, generalized pq
python run_mdp_experiment.py --noise_scale 1.0 --gadget2_jax

# cf, non generalized pq
python run_mdp_experiment.py --counterfactual --noise_scale 0.0 --gadget2_jax

# cf, generalized pq
python run_mdp_experiment.py --counterfactual --noise_scale 1.0 --gadget2_jax